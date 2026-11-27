from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd

from app.schemas import production_strategy_config
from app.services.data_provider import Provider, create_provider
from app.services.bars_cache import with_data_quality
from app.services.backtest import aggregate_reports, evaluate_bars
from app.services.company import CompanyService, empty_company_payload
from app.services.detail import build_full_chart, build_market_snapshot
from app.services.industry import IndustryContext, IndustryService, apply_industry_context
from app.services.market_data import DEFAULT_CACHE_DIR, MarketDataService
from app.services.market_calendar import CalendarUnavailable, china_now, last_completed_session
from app.services.market_regime import MarketRegimeService, apply_market_regime, regime_on
from app.services.related_board import BoardIndexRepository, rank_related_boards
from app.services.strategy import (
    SignalCategory,
    SignalResult,
    StrategyConfig,
    STRATEGY_VERSION,
    build_strategy_config,
    version_for_engine,
    _run_pipeline,
    _normalize_bars,
    analyze_bars,
    diagnose_bars,
)

OFFLINE_PROVIDERS = {"sample", "fake", "static", "demo"}


def scan_sort_key(item: SignalResult) -> tuple:
    """买入预警在前，再按结构分；同分时距线更近者在前（止损更紧），最后按代码。

    结构分对已确认的买入预警几乎没有区分度，距离线价只是可解释的并列排序，不代表胜率。
    """
    distance = abs(item.distance_to_line_pct) if item.distance_to_line_pct is not None else float("inf")
    return (item.category != SignalCategory.BUY_ALERT, -item.score, distance, item.stock_code)

SCAN_WORKERS = 4


class StaleMarketData(ValueError):
    pass


def _require_updated_bars(bars: pd.DataFrame) -> None:
    quality = bars.attrs.get("data_quality", {})
    if quality and quality.get("status") != "fresh":
        raise StaleMarketData(
            f"{quality.get('error') or '行情尚未核验'}；仅有截至 {quality.get('last_bar_date') or '未知日期'} 的缓存；"
            "不纳入本次筛选"
        )


@dataclass
class Scanner:
    provider_name: str = "auto"
    provider: Provider | None = None
    industry_service: IndustryService | None = None
    company_service: CompanyService | None = None
    market_data_service: MarketDataService | None = None
    board_repository: BoardIndexRepository | None = None
    market_regime_service: MarketRegimeService | None = None
    # 最近一次扫描用的策略配置。详情与诊断复用它，保证点进去看到的结论
    # 跟列表里的一致——否则改了量比门槛，列表按新值筛、详情按默认值判。
    last_strategy_config: StrategyConfig | None = None
    full_bars_cache: dict[str, tuple[float, int, Any]] = field(default_factory=dict, repr=False)
    full_bars_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    detail_cache: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)
    detail_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    detail_inflight: dict[str, threading.Event] = field(default_factory=dict, repr=False)
    backtest_cache: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)
    backtest_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    backtest_inflight: dict[str, threading.Event] = field(default_factory=dict, repr=False)
    interactive_condition: threading.Condition = field(default_factory=threading.Condition, repr=False)
    interactive_requests: int = field(default=0, repr=False)
    last_provider_name: str = "auto"
    last_scan_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "universe_count": 0,
            "target_count": 0,
            "attempted_count": 0,
            "analyzed_count": 0,
            "failed_count": 0,
            "stale_count": 0,
            "current_stock": "",
        }
    )
    last_scan_errors: list[str] = field(default_factory=list)
    last_backtest_stats: dict[str, Any] = field(default_factory=dict)

    def _provider(self, requested: str | None = None) -> Provider:
        if self.provider is not None and requested in {None, "", self.provider.name, "fake"}:
            return self.provider
        return create_provider(requested or self.provider_name)

    @contextmanager
    def _interactive_request(self):
        with self.interactive_condition:
            self.interactive_requests += 1
            self.interactive_condition.notify_all()
        try:
            yield
        finally:
            with self.interactive_condition:
                self.interactive_requests = max(0, self.interactive_requests - 1)
                self.interactive_condition.notify_all()

    def _wait_for_interactive_requests(self) -> None:
        with self.interactive_condition:
            while self.interactive_requests > 0:
                self.interactive_condition.wait(timeout=0.25)

    def _build_config(self, params: dict[str, Any]) -> StrategyConfig:
        """把扫描请求里的筛选条件翻译成策略配置。"""
        return build_strategy_config(params)

    def _market_table(self, provider: Provider, cfg: StrategyConfig):
        """本次分析要用的沪深300逐日环境；关闭或离线数据源时不取。"""
        if cfg.market_filter == "off":
            return None
        if self.market_regime_service is None:
            if provider.name in OFFLINE_PROVIDERS:
                return None
            self.market_regime_service = MarketRegimeService()
        return self.market_regime_service.table()

    def _backtest_market_table(self, provider: Provider, cfg: StrategyConfig):
        """回测用：取不到指数时给空表——逐日判为「未评估、不降级」，与实盘缺数据时的处理一致。

        回测结果里的 market_regime_available=False 会如实标出这一点，而不是静默当作已过滤。
        """
        table = self._market_table(provider, cfg)
        if table is None and cfg.market_filter == "downgrade":
            return pd.DataFrame()
        return table

    def _apply_market(self, result: SignalResult | None, table, cfg: StrategyConfig) -> None:
        if result is not None:
            apply_market_regime(result, regime_on(table, result.signal_date), cfg.market_filter)

    def strategy_config(self, scan_params: dict[str, Any] | None = None) -> StrategyConfig:
        """详情/诊断用的配置。

        优先用"打开它的那次筛选"的参数（调用方从数据库按 scan_id 取出后传入），
        其次沿用本进程最近一次扫描，最后才是默认值。只靠进程内的全局变量不够：
        后端重启、开始新扫描、或改完参数看旧批次时都会对不上。
        """
        if scan_params:
            return self._build_config(scan_params)
        # 没有批次上下文时与新建扫描同一配置，而不是早期的 offset=1、无新规则的默认值。
        return self.last_strategy_config or production_strategy_config()

    def _detail_cache_key(self, provider_name: str, code: str, years: int,
                          scan_params: dict[str, Any] | None = None) -> str:
        """缓存键带上配置指纹——否则改完筛选条件十分钟内打开会拿到旧结论。"""
        cfg = self.strategy_config(scan_params)
        fingerprint = "|".join(str(getattr(cfg, name)) for name in (
            "signal_offset", "require_breakout", "breakout_vol_mult", "require_breakout_volume",
            "min_days_since_breakout", "max_days_since_breakout", "min_line_strength",
            "require_pullback", "min_pullback_days", "max_pullback_days",
            "strategy_mode", "flag_direction",
            "precision_engine", "event_reversal_atr", "event_min_reversal_pct",
            "candidate_event_count", "confirmed_event_count",
            "max_prebreakout_extension_pct", "prebreakout_reject_above_days",
            "prebreakout_above_buffer_pct", "pullback_max_normal_volume_ratio",
            "max_post_breakout_runup_pct", "max_event_age_bars", "market_filter",
        ))
        return f"{version_for_engine(cfg.precision_engine)}:{provider_name}:{code}:{years}:{fingerprint}:{(scan_params or {}).get('_analysis_as_of', '')}"

    def run_scan(self, params: dict[str, Any], on_progress: Any | None = None,
                 should_stop: Any | None = None) -> list[SignalResult]:
        provider = self._provider(params.get("provider"))
        self.last_provider_name = provider.name
        stock_list = provider.get_stock_list()
        # 先清空上一次的统计再判空：提前 return 时如果残留旧数字，
        # 调用方（ScanJobManager._finish）会把上一轮的 total/processed 当成本轮的。
        self.last_scan_stats = {
            "universe_count": 0,
            "target_count": 0,
            "attempted_count": 0,
            "analyzed_count": 0,
            "failed_count": 0,
            "stale_count": 0,
            "current_stock": "",
        }
        self.last_scan_errors = []
        if stock_list.empty:
            return []
        stock_codes = params.get("stock_codes")
        stock_list = stock_list.copy()
        stock_list["stock_code"] = stock_list["stock_code"].astype(str).str.zfill(6)
        if stock_codes:
            wanted = {str(code).zfill(6) for code in stock_codes}
            stock_list = stock_list[stock_list["stock_code"].isin(wanted)]
        stock_list = stock_list.sort_values("stock_code").reset_index(drop=True)
        universe_count = len(stock_list)
        # 默认 0 = 全量，与 ScanRequest.limit / /api/config/defaults 保持一致
        limit = int(params.get("limit", 0))
        if 0 < limit < universe_count:
            if limit == 1:
                positions = [0]
            else:
                positions = [round(idx * (universe_count - 1) / (limit - 1)) for idx in range(limit)]
            stock_list = stock_list.iloc[positions].reset_index(drop=True)
        cfg = self._build_config(params)
        self.last_strategy_config = cfg
        market_table = self._market_table(provider, cfg)
        industry_contexts: dict[str, IndustryContext] = {}
        if self.industry_service is not None or provider.name not in {"sample", "fake", "static"}:
            service = self.industry_service or IndustryService()
            self.industry_service = service
            industry_contexts = service.prepare()

        self.last_scan_stats["universe_count"] = universe_count
        # 抽样之后的实际扫描数量，用作进度分母（universe_count 是抽样之前的全市场总数）
        self.last_scan_stats["target_count"] = len(stock_list)
        results: list[SignalResult] = []
        stats_lock = threading.Lock()

        def analyze_one(code: str, name: str) -> SignalResult | None:
            # 用户点了「停止筛选」：剩下的股票直接跳过。已扫出来的结果由上层落库。
            if should_stop is not None and should_stop():
                return None
            try:
                self._wait_for_interactive_requests()
                bars = provider.get_daily_bars(code)
                if bars is None or bars.empty:
                    raise ValueError("未获取到有效日线数据")
                _require_updated_bars(bars)
                # A detail request may have arrived while this worker was fetching data.
                self._wait_for_interactive_requests()
                result = analyze_bars(code, name, bars, cfg)
                if result is not None:
                    apply_industry_context(result, industry_contexts.get(code))
                    self._apply_market(result, market_table, cfg)
                succeeded, error = True, None
            except Exception as exc:
                result, succeeded, error = None, False, exc

            with stats_lock:
                self.last_scan_stats["attempted_count"] += 1
                if succeeded:
                    self.last_scan_stats["analyzed_count"] += 1
                else:
                    self.last_scan_stats["failed_count"] += 1
                    if isinstance(error, StaleMarketData):
                        self.last_scan_stats["stale_count"] += 1
                    if len(self.last_scan_errors) < 20:
                        self.last_scan_errors.append(f"{code} {name}: {error}")
                self.last_scan_stats["current_stock"] = f"{code} {name}"

                hit = None
                if result and result.category in {SignalCategory.BUY_ALERT, SignalCategory.WATCH}:
                    if "技术预警不构成投资建议" not in result.risks:
                        result.risks.append("技术预警不构成投资建议")
                    results.append(result)
                    hit = result

                if on_progress is not None:
                    on_progress(dict(self.last_scan_stats), hit)

            return result

        tasks = [
            (str(row.stock_code).zfill(6), str(row.short_name))
            for row in stock_list.itertuples(index=False)
        ]
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
            list(executor.map(lambda item: analyze_one(*item), tasks))

        return sorted(results, key=scan_sort_key)

    def analyze_stock(self, stock_code: str, stock_name: str | None = None) -> SignalResult | None:
        provider = self._provider()
        code = str(stock_code).zfill(6)
        name = stock_name or code
        if not stock_name:
            stocks = provider.get_stock_list()
            if not stocks.empty:
                match = stocks[stocks["stock_code"].astype(str).str.zfill(6) == code]
                if not match.empty:
                    name = str(match.iloc[0]["short_name"])
        bars = provider.get_daily_bars(code)
        if bars is not None:
            _require_updated_bars(bars)
        cfg = self.strategy_config()
        result = analyze_bars(code, name, bars, cfg)
        if result is not None:
            apply_industry_context(result, self._industry_context(provider, code))
            self._apply_market(result, self._market_table(provider, cfg), cfg)
        return result

    def _industry_context(self, provider: Provider, code: str) -> IndustryContext | None:
        if self.industry_service is None and provider.name in {"sample", "fake", "static"}:
            return None
        service = self.industry_service or IndustryService()
        self.industry_service = service
        return service.get_context(code)

    @staticmethod
    def _resolve_name(provider: Provider, code: str, stock_name: str | None) -> str:
        if stock_name:
            return stock_name
        for loader in (getattr(provider, "get_cached_stock_list", None), provider.get_stock_list):
            if loader is None:
                continue
            try:
                stocks = loader()
                match = stocks[stocks["stock_code"].astype(str).str.zfill(6) == code]
                if not match.empty:
                    return str(match.iloc[0]["short_name"])
            except Exception:
                continue
        return code

    def _get_full_bars(self, provider: Provider, code: str, years: int) -> Any:
        cache_key = f"{provider.name}:{code}"
        with self.full_bars_cache_lock:
            cached = self.full_bars_cache.get(cache_key)
            cache_covers_request = cached is not None and (
                cached[1] == 0 or (years > 0 and cached[1] >= years)
            )
            if cached is not None and time.monotonic() - cached[0] < 10 * 60 and cache_covers_request:
                previous = cached[2].attrs.get("data_quality", {})
                current = (
                    with_data_quality(cached[2], previous.get("checked_at") or "", verify=True)
                    if previous.get("calendar_source") else cached[2].copy()
                )
                if current.attrs.get("data_quality", {}).get("status") != "stale":
                    return current
        fetch_full = getattr(provider, "get_full_daily_bars", provider.get_daily_bars)
        try:
            bars = fetch_full(code, years=years)
        except TypeError:
            bars = fetch_full(code)
        if bars is not None and not bars.empty and bars.attrs.get("data_quality", {}).get("status") != "stale":
            with self.full_bars_cache_lock:
                self.full_bars_cache[cache_key] = (time.monotonic(), years, bars.copy())
        elif cached is not None and cache_covers_request:
            # 短时断网时优先展示最近一次成功行情。详情页会标出行情日期，旧缓存
            # 比整页 503 更有价值；下一次刷新仍会重新尝试实时数据源。
            previous = cached[2].attrs.get("data_quality", {})
            return with_data_quality(cached[2], previous.get("checked_at") or "", "最新行情获取失败")
        return bars

    def _get_analysis_bars(self, provider: Provider, code: str, years: int,
                           scan_params: dict[str, Any] | None = None):
        as_of = (scan_params or {}).get("_analysis_as_of")
        if as_of and pd.Timestamp(as_of).date() < china_now().date():
            cached = None
            with self.full_bars_cache_lock:
                memory = self.full_bars_cache.get(f"{provider.name}:{code}")
                if memory and memory[1] == 0:
                    cached = memory[2].copy()
            if cached is None:
                loader = getattr(provider, "get_cached_daily_bars", None)
                if loader is not None:
                    cached = loader(code)
            if cached is not None and not cached.empty:
                dates = pd.to_datetime(cached.trade_date, errors="coerce")
                if dates.max() >= pd.Timestamp(as_of):
                    cached = cached.copy()
                    cached.attrs["data_quality"] = {
                        **cached.attrs.get("data_quality", {}), "status": "historical_cache",
                        "last_bar_date": dates.max().strftime("%Y-%m-%d"),
                        "analysis_as_of": as_of, "error": None,
                    }
                    return cached
        return self._get_full_bars(provider, code, years)

    def latest_prices(self, stock_codes: list[str]) -> list[dict[str, Any]]:
        """刷新观察池价格；单只失败不影响其他股票。"""
        provider = self._provider()
        codes = list(dict.fromkeys(str(code).zfill(6) for code in stock_codes))

        def fetch_one(code: str) -> dict[str, Any]:
            try:
                bars = self._get_full_bars(provider, code, 0)
                _require_updated_bars(bars)
                snapshot = build_market_snapshot(bars)
                if not snapshot or snapshot.get("close") is None:
                    raise ValueError("未获取到最新行情")
                return {
                    "stock_code": code,
                    "latest_price": snapshot["close"],
                    "latest_price_at": snapshot.get("trade_date"),
                    "data_source": provider.name,
                    "error": None,
                }
            except Exception as exc:
                return {
                    "stock_code": code,
                    "latest_price": None,
                    "latest_price_at": None,
                    "data_source": provider.name,
                    "error": str(exc),
                }

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(codes)))) as executor:
            return list(executor.map(fetch_one, codes))

    def stock_detail(self, stock_code: str, stock_name: str | None = None, years: int = 0,
                     scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        # 沿用该批次选的数据源：扫描指定了 akshare 而详情走 auto/adata 的话，
        # 价格与结论都可能对不上。
        provider = self._provider((scan_params or {}).get("provider"))
        code = str(stock_code).zfill(6)
        cache_key = self._detail_cache_key(provider.name, code, years, scan_params)
        owns_request = False
        with self.detail_cache_lock:
            cached_detail = self.detail_cache.get(cache_key)
            cache_ttl = 5 if cached_detail and cached_detail[1].get("data_quality", {}).get("status") == "stale" else 10 * 60
            if cached_detail and not (scan_params or {}).get("_analysis_as_of"):
                quality = cached_detail[1].get("data_quality", {})
                if quality.get("calendar_source"):
                    try:
                        if quality.get("expected_bar_date") != last_completed_session():
                            cache_ttl = 0
                    except CalendarUnavailable:
                        cache_ttl = 0
            if cached_detail is not None and time.monotonic() - cached_detail[0] < cache_ttl:
                return copy.deepcopy(cached_detail[1])
            request_done = self.detail_inflight.get(cache_key)
            if request_done is None:
                request_done = threading.Event()
                self.detail_inflight[cache_key] = request_done
                owns_request = True

        if not owns_request:
            if not request_done.wait(timeout=120):
                raise TimeoutError("个股详情计算超时，请稍后重试")
            with self.detail_cache_lock:
                cached_detail = self.detail_cache.get(cache_key)
                if cached_detail is not None:
                    return copy.deepcopy(cached_detail[1])
            # The first request failed. Allow this caller to make one fresh attempt.
            # 重试要继续带上批次参数，否则会退回最近配置或默认配置。
            return self.stock_detail(code, stock_name, years, scan_params)

        try:
            with self._interactive_request():
                name = self._resolve_name(provider, code, stock_name)
                bars = self._get_analysis_bars(provider, code, years, scan_params)
                if bars is None or bars.empty:
                    raise ValueError("未获取到完整日线数据，请稍后重试")

                cfg = self.strategy_config(scan_params)
                as_of = (scan_params or {}).get("_analysis_as_of")
                if as_of:
                    attrs = dict(bars.attrs)
                    bars = bars[pd.to_datetime(bars.trade_date) <= pd.Timestamp(as_of)].copy()
                    bars.attrs = attrs
                    if bars.empty:
                        raise ValueError("指定日期之前没有该股可用行情")
                    cfg = replace(cfg, signal_offset=0)
                stage, message, result = _run_pipeline(code, name, bars, cfg)
                context = None if as_of else self._industry_context(provider, code)
                if result is not None:
                    apply_industry_context(result, context)
                    # 大盘环境按信号日逐日取，历史视角同样适用，详情与列表结论一致。
                    self._apply_market(result, self._market_table(provider, cfg), cfg)
                diagnosis = {"stock_code": code, "stock_name": name, "found": stage == "ok",
                             "stage": stage, "message": message, "result": None, "chart": {}}
                quality = bars.attrs.get("data_quality", {})
                if quality and quality.get("status") not in {"fresh", "historical_cache"}:
                    message = f"{quality.get('error') or '行情尚未更新'}；以下仅为截至 {quality.get('last_bar_date') or '未知日期'} 的历史结构"
                    diagnosis.update(found=False, stage="stale_data", message=message)
                    if result is not None:
                        result.category = SignalCategory.WATCH
                        result.suggestion = message
                        result.risks.append("暂停交易预警，需先更新行情")
                        for match in result.pattern_matches:
                            match.update(category="watch", data_stale=True, suggestion=message)
                if result is not None:
                    diagnosis["result"] = result.to_dict()
                    diagnosis["chart"] = result.chart

                chart = build_full_chart(bars, result, [])
                normalized_dates = chart.get("range", {})
                payload = {
                    "stock": {"stock_code": code, "stock_name": name},
                    "analysis": diagnosis,
                    "chart": chart,
                    "market": build_market_snapshot(bars),
                    "company": {},
                    "shareholders": empty_company_payload("公司与股东数据加载中")["shareholders"],
                    "company_data": {
                        "source": None,
                        "updated_at": None,
                        "stale": False,
                        "message": "公司与股东数据加载中",
                    },
                    "industry": asdict(context) if context is not None else None,
                    "industry_market": {"name": context.name if context else None, "mapped_name": None, "chart": {"bars": [], "markers": []}, "message": "板块行情加载中"},
                    "news": {"items": [], "source": None, "updated_at": None, "message": "新闻加载中"},
                    "backtest": {"signals": [], "summary": {"signal_count": 0}, "methodology": "点击加载全历史预警复盘"},
                    "data_source": provider.name,
                    "strategy_mode": cfg.strategy_mode,
                    "precision_engine": cfg.precision_engine,
                    "strategy_version": version_for_engine(cfg.precision_engine),
                    "analysis_as_of": as_of,
                    "price_basis_note": "当前缓存前复权口径；历史模式只使用分析日及以前日线",
                    "flag_direction": cfg.flag_direction,
                    "data_quality": quality,
                    "bar_count": normalized_dates.get("count", 0),
                    "start_date": normalized_dates.get("start"),
                    "end_date": normalized_dates.get("end"),
                    "disclaimer": "技术预警与股东趋势标记仅用于研究和复盘，不构成投资建议。",
                }
            with self.detail_cache_lock:
                if len(self.detail_cache) >= 64:
                    oldest_key = min(self.detail_cache, key=lambda key: self.detail_cache[key][0])
                    self.detail_cache.pop(oldest_key, None)
                self.detail_cache[cache_key] = (time.monotonic(), copy.deepcopy(payload))
            return payload
        finally:
            with self.detail_cache_lock:
                completed = self.detail_inflight.pop(cache_key, None)
                if completed is not None:
                    completed.set()

    def stock_company(self, stock_code: str) -> dict[str, Any]:
        provider = self._provider()
        if provider.name in {"sample", "fake", "static"}:
            return empty_company_payload("示例数据不包含公司及股东资料")
        self.company_service = self.company_service or CompanyService()
        return self.company_service.get(str(stock_code).zfill(6))

    def compare_stock_strategies(self, stock_code: str, stock_name: str | None = None,
                                scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        provider = self._provider((scan_params or {}).get("provider"))
        code = str(stock_code).zfill(6)
        name = self._resolve_name(provider, code, stock_name)
        frame = self._get_analysis_bars(provider, code, 0, scan_params)
        cfg = self.strategy_config(scan_params)
        as_of = (scan_params or {}).get("_analysis_as_of")
        if as_of:
            frame = frame[pd.to_datetime(frame.trade_date) <= pd.Timestamp(as_of)].copy()
            cfg = replace(cfg, signal_offset=0)
        results = []
        for engine in ("legacy", "pivot_fixed", "platform_events"):
            selected = replace(cfg, precision_engine=engine, strategy_mode="precision_pullback", include_chart=True)
            result = diagnose_bars(code, name, frame, selected)
            results.append({"engine": engine, "strategy_version": version_for_engine(engine), **result})
        return {
            "stock_code": code, "stock_name": name, "analysis_as_of": as_of,
            "results": results, "strategy_mode": "precision_pullback",
            "method": "同一行情、同一截止日和其余过滤条件；只比较精准线引擎，不叠加行业分",
        }

    def stock_news(self, stock_code: str) -> dict[str, Any]:
        provider = self._provider()
        if provider.name in {"sample", "fake", "static"}:
            return {"items": [], "source": None, "updated_at": None, "message": "示例数据不包含新闻"}
        self.market_data_service = self.market_data_service or MarketDataService()
        return self.market_data_service.get_news(str(stock_code).zfill(6), 20)

    def stock_industry_market(self, stock_code: str, years: int = 10) -> dict[str, Any]:
        provider = self._provider()
        code = str(stock_code).zfill(6)
        context = self._industry_context(provider, code)
        if provider.name in {"sample", "fake", "static"}:
            return {"name": None, "mapped_name": None, "chart": {"bars": [], "markers": []}, "message": "示例数据不包含板块行情"}
        self.market_data_service = self.market_data_service or MarketDataService()
        return self.market_data_service.get_industry_chart(context.name if context else None, years)

    def stock_related_board(self, stock_code: str, top_n: int = 1, years: int = 3) -> dict[str, Any]:
        """按走势相关性找最相关的板块/概念，并返回它的 K 线。

        行业缓存只覆盖约 6 成股票，`stock_industry_market` 对其余股票必然返回空图。
        这里不依赖那份映射：任何有足够历史的股票都能算出结果。
        """
        provider = self._provider()
        code = str(stock_code).zfill(6)
        if provider.name in {"sample", "fake", "static"}:
            return {"board": None, "chart": {"bars": [], "markers": []}, "message": "示例数据不支持相关板块"}
        bars = self._get_full_bars(provider, code, years)
        if bars is None or len(bars) == 0:
            return {"board": None, "chart": {"bars": [], "markers": []}, "message": "未取到个股行情，无法计算相关板块"}
        frame = _normalize_bars(bars)
        closes = pd.Series(
            frame["close"].to_numpy(dtype=float),
            index=pd.to_datetime(frame["trade_date"]),
        )
        self.board_repository = self.board_repository or BoardIndexRepository(cache_dir=DEFAULT_CACHE_DIR)
        if not self.board_repository.ensure_ready(background=True):
            return {
                "board": None,
                "chart": {"bars": [], "markers": []},
                "building": True,
                "message": "正在后台拉取全部行业与概念板块指数（约 4 分钟），完成后刷新即可",
            }
        universe = self.board_repository.closes()
        if not universe:
            return {"board": None, "chart": {"bars": [], "markers": []}, "message": "板块指数暂不可用"}
        ranked = rank_related_boards(closes, universe, top_n=top_n, kinds=self.board_repository.kinds())
        if not ranked:
            return {"board": None, "chart": {"bars": [], "markers": []}, "message": "与个股重叠的交易日不足，无法判断相关板块"}
        best = ranked[0]
        try:
            board_bars = self.board_repository.chart_bars(best["name"], best["kind"], years)
            chart = build_full_chart(board_bars) if len(board_bars) else {"bars": [], "markers": []}
        except Exception as exc:
            return {"board": best, "chart": {"bars": [], "markers": []}, "message": f"板块 K 线取数失败：{exc}"}
        return {
            "board": best,
            "candidates": ranked,
            "chart": chart,
            "universe_size": len(universe),
            "message": "",
        }

    def backtest_stock(self, stock_code: str, stock_name: str | None = None, lookback_days: int = 0,
                       scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        provider = self._provider((scan_params or {}).get("provider"))
        code = str(stock_code).zfill(6)
        history_days = max(0, int(lookback_days))
        cfg = self.strategy_config(scan_params)
        real_provider = provider.name in {"auto", "adata", "akshare"}
        cutoff = (scan_params or {}).get("_analysis_as_of") or (last_completed_session() if real_provider else None)
        cache_key = f"{self._detail_cache_key(provider.name, code, history_days, scan_params)}:{cutoff}"
        owns_request = False
        with self.backtest_cache_lock:
            cached = self.backtest_cache.get(cache_key)
            cache_ttl = 5 if cached and cached[1].get("data_quality", {}).get("status") == "stale" else 30 * 60
            if cached is not None and time.monotonic() - cached[0] < cache_ttl:
                return copy.deepcopy(cached[1])
            request_done = self.backtest_inflight.get(cache_key)
            if request_done is None:
                request_done = threading.Event()
                self.backtest_inflight[cache_key] = request_done
                owns_request = True

        if not owns_request:
            if not request_done.wait(timeout=10 * 60):
                raise TimeoutError("历史预警复盘计算超时，请稍后重试")
            with self.backtest_cache_lock:
                cached = self.backtest_cache.get(cache_key)
                if cached is not None:
                    return copy.deepcopy(cached[1])
            return self.backtest_stock(code, stock_name, history_days, scan_params)

        try:
            name = self._resolve_name(provider, code, stock_name)
            bars = self._get_analysis_bars(provider, code, 0, scan_params)
            as_of = (scan_params or {}).get("_analysis_as_of")
            if as_of:
                bars = bars[pd.to_datetime(bars.trade_date) <= pd.Timestamp(as_of)].copy()
            evaluation_end = cutoff or bars.attrs.get("data_quality", {}).get("expected_bar_date")
            payload = evaluate_bars(code, name, bars, lookback_days=history_days, config=cfg, as_of=evaluation_end,
                                    market_regimes=self._backtest_market_table(provider, cfg))
            payload["analysis_as_of"] = as_of
            payload["data_quality"] = bars.attrs.get("data_quality", {})
            with self.backtest_cache_lock:
                if len(self.backtest_cache) >= 64:
                    oldest_key = min(self.backtest_cache, key=lambda key: self.backtest_cache[key][0])
                    self.backtest_cache.pop(oldest_key, None)
                self.backtest_cache[cache_key] = (time.monotonic(), copy.deepcopy(payload))
            return payload
        finally:
            with self.backtest_cache_lock:
                completed = self.backtest_inflight.pop(cache_key, None)
                if completed is not None:
                    completed.set()

    def run_backtest(self, params: dict[str, Any], on_progress: Any | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        provider = self._provider(params.get("provider"))
        cutoff = params.get("_analysis_as_of") or (
            last_completed_session() if provider.name in {"auto", "adata", "akshare"} else None
        )
        cfg = self._build_config(params)
        market_table = self._backtest_market_table(provider, cfg)
        self.last_provider_name = provider.name
        stocks = provider.get_stock_list()
        stocks = stocks.copy()
        stocks["stock_code"] = stocks["stock_code"].astype(str).str.zfill(6)
        stock_codes = params.get("stock_codes")
        if stock_codes:
            wanted = {str(code).zfill(6) for code in stock_codes}
            stocks = stocks[stocks["stock_code"].isin(wanted)].reset_index(drop=True)
        universe_count = len(stocks)
        limit = max(0, int(params.get("limit", 0) or 0))
        if limit and limit < universe_count:
            positions = [round(idx * (universe_count - 1) / (limit - 1)) for idx in range(limit)] if limit > 1 else [0]
            stocks = stocks.iloc[positions].reset_index(drop=True)
        lookback_days = int(params.get("lookback_days", 500))
        reports: list[dict[str, Any]] = []
        lock = threading.Lock()
        self.last_backtest_stats = {
            "universe_count": universe_count,
            "target_count": len(stocks),
            "processed_count": 0,
            "failed_count": 0,
            "signal_count": 0,
            "current_stock": "",
        }

        def evaluate_one(code: str, name: str) -> None:
            report = None
            error = None
            try:
                fetch_full = getattr(provider, "get_full_daily_bars", provider.get_daily_bars)
                try:
                    bars = fetch_full(code, years=0)
                except TypeError:
                    bars = fetch_full(code)
                if bars is None or bars.empty:
                    raise ValueError("未获取到历史行情")
                report = evaluate_bars(code, name, bars, lookback_days=lookback_days, config=cfg, as_of=cutoff,
                                       market_regimes=market_table)
                report["data_quality"] = bars.attrs.get("data_quality", {})
            except Exception as exc:
                error = str(exc)
            with lock:
                self.last_backtest_stats["processed_count"] += 1
                self.last_backtest_stats["current_stock"] = f"{code} {name}"
                if report is None:
                    self.last_backtest_stats["failed_count"] += 1
                else:
                    reports.append(report)
                    self.last_backtest_stats["signal_count"] += int(report["summary"]["signal_count"])
                if on_progress is not None:
                    on_progress(dict(self.last_backtest_stats), report, error)

        tasks = [(str(row.stock_code).zfill(6), str(row.short_name)) for row in stocks.itertuples(index=False)]
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
            list(executor.map(lambda item: evaluate_one(*item), tasks))
        reports.sort(key=lambda item: (-item["summary"]["signal_count"], item["stock_code"]))
        return reports, aggregate_reports(reports)

    def diagnose_stock(self, stock_code: str, stock_name: str | None = None,
                       scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        provider = self._provider((scan_params or {}).get("provider"))
        code = str(stock_code).zfill(6)
        name = stock_name or code
        if not stock_name:
            stocks = provider.get_stock_list()
            if not stocks.empty:
                match = stocks[stocks["stock_code"].astype(str).str.zfill(6) == code]
                if not match.empty:
                    name = str(match.iloc[0]["short_name"])
        def fetch_failed(reason: str) -> dict[str, Any]:
            return {
                "stock_code": code,
                "stock_name": name,
                "found": False,
                "stage": "fetch_failed",
                "message": reason,
                "result": None,
                "chart": {},
            }

        try:
            bars = provider.get_daily_bars(code)
        except Exception as exc:
            return fetch_failed(f"行情获取失败：{exc}")

        # provider 层的重试会把网络异常吞成空表。此时若交给 diagnose_bars，
        # 用户会看到「历史数据不足…可能是新股或长期停牌」——对老股是误导。
        if bars is None or bars.empty:
            return fetch_failed("未获取到行情数据，可能是股票代码不存在，或数据源暂时不可用")
        try:
            _require_updated_bars(bars)
        except StaleMarketData as exc:
            diagnosis = fetch_failed(str(exc))
            diagnosis["stage"] = "stale_data"
            return diagnosis

        return diagnose_bars(code, name, bars, self.strategy_config(scan_params))


class StaticScanner:
    provider_name = "static"

    def __init__(self, results: list[SignalResult]):
        self.results = results

    def run_scan(self, params: dict[str, Any], on_progress: Any | None = None,
                 should_stop: Any | None = None) -> list[SignalResult]:
        return self.results

    def analyze_stock(self, stock_code: str, stock_name: str | None = None) -> SignalResult | None:
        for item in self.results:
            if item.stock_code == stock_code:
                return item
        return None

    def diagnose_stock(self, stock_code: str, stock_name: str | None = None,
                       scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.analyze_stock(stock_code, stock_name)
        return {
            "stock_code": stock_code,
            "stock_name": stock_name or stock_code,
            "found": result is not None,
            "stage": "ok" if result is not None else "no_key_line",
            "message": result.suggestion if result is not None else "未找到精准线",
            "result": result.to_dict() if result is not None else None,
            "chart": result.chart if result is not None else {},
        }

    def stock_detail(self, stock_code: str, stock_name: str | None = None, years: int = 10,
                     scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.analyze_stock(stock_code, stock_name)
        chart = result.chart if result is not None else {"bars": [], "markers": []}
        return {
            "stock": {"stock_code": stock_code, "stock_name": stock_name or stock_code},
            "analysis": self.diagnose_stock(stock_code, stock_name),
            "chart": chart,
            "market": {},
            "company": {},
            "shareholders": empty_company_payload()["shareholders"],
            "company_data": {"source": None, "updated_at": None, "stale": False, "message": "静态数据"},
            "industry": None,
            "industry_market": {"name": None, "mapped_name": None, "chart": {"bars": [], "markers": []}, "message": "静态数据"},
            "news": {"items": [], "source": None, "updated_at": None, "message": "静态数据"},
            "backtest": {"signals": [], "summary": {"signal_count": 0}},
            "data_source": "static",
            "bar_count": len(chart.get("bars", [])),
            "start_date": None,
            "end_date": None,
            "disclaimer": "技术预警仅用于研究和复盘，不构成投资建议。",
        }

    def backtest_stock(self, stock_code: str, stock_name: str | None = None, lookback_days: int = 500,
                       scan_params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.stock_detail(stock_code, stock_name)["backtest"]

    def stock_company(self, stock_code: str) -> dict[str, Any]:
        return empty_company_payload("静态数据不包含公司及股东资料")

    def stock_news(self, stock_code: str) -> dict[str, Any]:
        return {"items": [], "source": None, "updated_at": None, "message": "静态数据不包含新闻"}

    def stock_industry_market(self, stock_code: str, years: int = 10) -> dict[str, Any]:
        return {"name": None, "mapped_name": None, "chart": {"bars": [], "markers": []}, "message": "静态数据不包含板块行情"}

    def stock_related_board(self, stock_code: str, top_n: int = 1, years: int = 3) -> dict[str, Any]:
        return {"board": None, "chart": {"bars": [], "markers": []}, "message": "静态数据不支持相关板块"}

    def run_backtest(self, params: dict[str, Any], on_progress: Any | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        reports = [self.backtest_stock(item.stock_code, item.stock_name) for item in self.results]
        return reports, aggregate_reports(reports)
