from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from app.services.scanner import Scanner, StaticScanner
from app.services.strategy import SignalCategory, SignalResult


class RecordingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.requested: list[str] = []
        self._lock = threading.Lock()

    def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stock_code": f"{idx:06d}", "short_name": f"股票{idx}", "exchange": "SZ"}
                for idx in range(10)
            ]
        )

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        with self._lock:
            self.requested.append(stock_code)
        if stock_code == "000006":
            raise RuntimeError("data source failed")
        closes = [8.0 + idx * 0.01 for idx in range(130)]
        return pd.DataFrame(
            {
                "trade_date": [date(2026, 1, 1) + timedelta(days=idx) for idx in range(len(closes))],
                "open": [close * 0.995 for close in closes],
                "high": [close * 1.01 for close in closes],
                "low": [close * 0.99 for close in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            }
        )


def test_limited_scan_samples_across_whole_market_and_reports_failures() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)

    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1})

    assert sorted(provider.requested) == ["000000", "000003", "000006", "000009"]
    assert scanner.last_scan_stats["universe_count"] == 10
    assert scanner.last_scan_stats["attempted_count"] == 4
    assert scanner.last_scan_stats["analyzed_count"] == 3
    assert scanner.last_scan_stats["failed_count"] == 1


def test_progress_callback_fires_once_per_stock() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    events: list[dict] = []
    lock = threading.Lock()

    def on_progress(stats, result):
        with lock:
            events.append({"processed": stats["attempted_count"], "code": stats["current_stock"]})

    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1}, on_progress=on_progress)

    assert len(events) == 4
    assert events[-1]["processed"] == 4


def test_scan_without_callback_behaves_as_before() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)

    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1})

    assert scanner.last_scan_stats["attempted_count"] == 4
    assert scanner.last_scan_stats["analyzed_count"] == 3
    assert scanner.last_scan_stats["failed_count"] == 1


def test_stock_detail_reuses_completed_payload() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)

    first = scanner.stock_detail("000001", "平安银行")
    second = scanner.stock_detail("000001", "平安银行")

    assert first["stock"] == second["stock"]
    assert provider.requested == ["000001"], "重复打开详情不应再次取数和计算全部历史"


def test_concurrent_stock_detail_requests_share_one_calculation() -> None:
    class SlowProvider(RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
            self.started.set()
            self.release.wait(timeout=2)
            return super().get_daily_bars(stock_code)

    provider = SlowProvider()
    scanner = Scanner(provider=provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(scanner.stock_detail, "000001", "平安银行")
        assert provider.started.wait(timeout=1)
        second = executor.submit(scanner.stock_detail, "000001", "平安银行")
        time.sleep(0.03)
        provider.release.set()
        assert first.result()["stock"] == second.result()["stock"]

    assert provider.requested == ["000001"]


def test_scan_worker_waits_while_interactive_detail_has_priority() -> None:
    scanner = Scanner(provider=RecordingProvider())
    with ThreadPoolExecutor(max_workers=1) as executor:
        with scanner._interactive_request():
            waiter = executor.submit(scanner._wait_for_interactive_requests)
            time.sleep(0.03)
            assert waiter.done() is False
        waiter.result(timeout=1)


def test_stock_backtest_reuses_completed_all_history_report(monkeypatch) -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    calls: list[int] = []

    def fake_evaluate(code, name, bars, lookback_days=0, config=None, as_of=None, **_):
        calls.append((lookback_days, config.breakout_vol_mult))
        return {"stock_code": code, "stock_name": name, "signals": [], "summary": {"signal_count": 0}}

    monkeypatch.setattr("app.services.scanner.evaluate_bars", fake_evaluate)

    params = {"provider": "fake", "breakout_vol_mult": 1.2}
    first = scanner.backtest_stock("000001", "平安银行", 0, params)
    second = scanner.backtest_stock("000001", "平安银行", 0, params)
    scanner.backtest_stock(
        "000001", "平安银行", 0, {"provider": "fake", "breakout_vol_mult": 1.8}
    )

    assert first == second
    assert calls == [(0, 1.2), (0, 1.8)], "相同配置复用缓存，不同配置必须重新复盘"


def test_target_count_reflects_sampled_size_not_universe() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)

    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1})

    assert scanner.last_scan_stats["universe_count"] == 10
    assert scanner.last_scan_stats["target_count"] == 4


class WideProvider:
    """股票数超过旧默认值 80 的假数据源。

    RecordingProvider 只有 10 只，撞不到抽样分支（0 < 80 < 10 为假），
    区分不出默认 limit 是 80 还是 0，所以这里单独造一个 100 只的。
    日线故意给得很短，analyze_bars 会直接判定历史不足返回 None，扫得快。
    """

    name = "fake"

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.requested: list[str] = []
        self._lock = threading.Lock()

    def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stock_code": f"{idx:06d}", "short_name": f"股票{idx}", "exchange": "SZ"}
                for idx in range(self.size)
            ]
        )

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        with self._lock:
            self.requested.append(stock_code)
        closes = [9.5 + idx * 0.01 for idx in range(80)]
        return pd.DataFrame(
            {
                "trade_date": [date(2026, 1, 1) + timedelta(days=idx) for idx in range(len(closes))],
                "open": [close * 0.995 for close in closes],
                "high": [close * 1.01 for close in closes],
                "low": [close * 0.99 for close in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            }
        )


def test_run_scan_without_limit_param_scans_everything() -> None:
    provider = WideProvider(size=100)
    scanner = Scanner(provider=provider)

    scanner.run_scan({"provider": "fake", "signal_offset": 1})

    assert scanner.last_scan_stats["target_count"] == 100, "不传 limit 时应扫描全部股票"
    assert scanner.last_scan_stats["universe_count"] == 100
    assert len(provider.requested) == 100


def test_empty_universe_resets_previous_stats() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1})
    assert scanner.last_scan_stats["attempted_count"] == 4

    class EmptyProvider:
        name = "fake"

        def get_stock_list(self):
            return pd.DataFrame(columns=["stock_code", "short_name", "exchange"])

        def get_daily_bars(self, stock_code):
            return pd.DataFrame()

    scanner.provider = EmptyProvider()
    scanner.run_scan({"provider": "fake", "limit": 4, "signal_offset": 1})

    assert scanner.last_scan_stats["attempted_count"] == 0
    assert scanner.last_scan_stats["target_count"] == 0
    assert scanner.last_scan_stats["universe_count"] == 0


DIAGNOSIS_KEYS = {"stock_code", "stock_name", "found", "stage", "message", "result", "chart"}


def _sample_result(stock_code: str = "000001") -> SignalResult:
    return SignalResult(
        stock_code=stock_code,
        stock_name="平安银行",
        category=SignalCategory.BUY_ALERT,
        score=88.0,
        suggestion="买入预警：测试信号",
        key_line_price=10.1,
        line_strength=4,
        signal_date="2026-05-13",
    )


def test_static_scanner_diagnose_reports_found_and_missing() -> None:
    scanner = StaticScanner([_sample_result("000001")])

    hit = scanner.diagnose_stock("000001")
    miss = scanner.diagnose_stock("000002")

    assert hit["found"] is True
    assert hit["stage"] == "ok"
    assert hit["result"]["key_line_price"] == 10.1
    assert miss["found"] is False
    assert miss["stage"] == "no_key_line"
    assert miss["result"] is None


def test_diagnose_stock_reports_fetch_failure() -> None:
    # RecordingProvider 对 000006 抛 RuntimeError，走接口层特有的 fetch_failed 分支
    scanner = Scanner(provider=RecordingProvider())

    diagnosis = scanner.diagnose_stock("000006")

    assert diagnosis["found"] is False
    assert diagnosis["stage"] == "fetch_failed"
    assert "data source failed" in diagnosis["message"]
    assert diagnosis["result"] is None


def test_diagnose_contract_keys_match_across_implementations() -> None:
    # 前端只写一套渲染逻辑，三个实现的键名必须完全一致
    live = Scanner(provider=RecordingProvider()).diagnose_stock("000000")
    failed = Scanner(provider=RecordingProvider()).diagnose_stock("000006")
    static_hit = StaticScanner([_sample_result("000001")]).diagnose_stock("000001")
    static_miss = StaticScanner([]).diagnose_stock("000002")

    assert set(live) == DIAGNOSIS_KEYS
    assert set(failed) == DIAGNOSIS_KEYS
    assert set(static_hit) == DIAGNOSIS_KEYS
    assert set(static_miss) == DIAGNOSIS_KEYS


def test_diagnose_stock_reports_fetch_failure_when_bars_are_empty() -> None:
    """取数返回空表时必须说是数据源问题，不能误报成「新股或长期停牌」。

    provider 层的重试会把网络异常吞成空 DataFrame，诊断层若直接交给
    diagnose_bars，用户会看到「历史数据不足 120 根（当前 0 根），可能是新股」——
    对贵州茅台这种老股是完全误导的解释。
    """

    class EmptyBarsProvider:
        name = "fake"

        def get_stock_list(self) -> pd.DataFrame:
            return pd.DataFrame(
                [{"stock_code": "600519", "short_name": "贵州茅台", "exchange": "SH"}]
            )

        def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
            return pd.DataFrame()

    scanner = Scanner(provider=EmptyBarsProvider())

    payload = scanner.diagnose_stock("600519")

    assert payload["found"] is False
    assert payload["stage"] == "fetch_failed"
    assert "新股" not in payload["message"]
    assert payload["stock_name"] == "贵州茅台"


def test_backtest_can_be_limited_to_selected_stock_codes(monkeypatch) -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    monkeypatch.setattr(
        "app.services.scanner.evaluate_bars",
        lambda code, name, bars, lookback_days=500, config=None, as_of=None, **_: {
            "stock_code": code,
            "stock_name": name,
            "signals": [],
            "summary": {"signal_count": 0},
            "config_volume": config.breakout_vol_mult,
        },
    )

    reports, summary = scanner.run_backtest(
        {"provider": "fake", "stock_codes": ["000001", "000003"], "lookback_days": 120,
         "breakout_vol_mult": 1.2}
    )

    assert [item["stock_code"] for item in reports] == ["000001", "000003"]
    assert scanner.last_backtest_stats["target_count"] == 2
    assert summary["stock_count"] == 2
    assert all(item["config_volume"] == 1.2 for item in reports)


def test_breakout_age_window_reaches_the_strategy_config() -> None:
    """人工设的天数窗口必须一路传到 StrategyConfig，否则页面上调了等于没调。"""
    from unittest.mock import patch

    from app.services.strategy import StrategyConfig

    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    seen: list[StrategyConfig] = []
    real = StrategyConfig

    def spy(**kwargs):
        cfg = real(**kwargs)
        seen.append(cfg)
        return cfg

    with patch("app.services.strategy.StrategyConfig", side_effect=spy):
        scanner.run_scan({
            "provider": "fake", "limit": 2, "signal_offset": 1,
            "min_days_since_breakout": 3, "max_days_since_breakout": 7,
        })

    assert seen, "未构造 StrategyConfig"
    assert seen[0].min_days_since_breakout == 3
    assert seen[0].max_days_since_breakout == 7


def test_breakout_age_window_defaults_to_off_when_not_supplied() -> None:
    from unittest.mock import patch

    from app.services.strategy import StrategyConfig

    scanner = Scanner(provider=RecordingProvider())
    seen: list[StrategyConfig] = []
    real = StrategyConfig

    def spy(**kwargs):
        cfg = real(**kwargs)
        seen.append(cfg)
        return cfg

    with patch("app.services.strategy.StrategyConfig", side_effect=spy):
        scanner.run_scan({"provider": "fake", "limit": 2, "signal_offset": 1})

    assert seen[0].min_days_since_breakout == 0
    assert seen[0].max_days_since_breakout == 0


def test_require_breakout_flag_reaches_the_strategy_config() -> None:
    from unittest.mock import patch

    from app.services.strategy import StrategyConfig

    scanner = Scanner(provider=RecordingProvider())
    seen: list[StrategyConfig] = []
    real = StrategyConfig

    def spy(**kwargs):
        cfg = real(**kwargs)
        seen.append(cfg)
        return cfg

    with patch("app.services.strategy.StrategyConfig", side_effect=spy):
        scanner.run_scan({"provider": "fake", "limit": 2, "signal_offset": 1, "require_breakout": True})

    assert seen[0].require_breakout is True


def test_breakout_volume_threshold_reaches_the_strategy_config() -> None:
    """量能门槛是"算不算突破"的决定性条件，必须可以人工调。"""
    from unittest.mock import patch

    from app.services.strategy import StrategyConfig

    scanner = Scanner(provider=RecordingProvider())
    seen: list[StrategyConfig] = []
    real = StrategyConfig

    def spy(**kwargs):
        cfg = real(**kwargs)
        seen.append(cfg)
        return cfg

    with patch("app.services.strategy.StrategyConfig", side_effect=spy):
        scanner.run_scan({"provider": "fake", "limit": 2, "breakout_vol_mult": 1.3})
    assert seen[0].breakout_vol_mult == 1.3

    seen.clear()
    with patch("app.services.strategy.StrategyConfig", side_effect=spy):
        scanner.run_scan({"provider": "fake", "limit": 2})
    assert seen[0].breakout_vol_mult == 1.5, "未指定时保持原默认"


def test_detail_and_diagnosis_reuse_the_scan_strategy_config() -> None:
    """列表和详情必须用同一套筛选条件，否则点进去会看到不同的结论。

    扫描用的是用户填的量比门槛/突破天数窗口，详情却写死默认配置——
    把量比调到 1.3 筛出一批股票，点进详情看到的却是 1.5 的判定。
    """
    scanner = Scanner(provider=RecordingProvider())
    scanner.run_scan({"provider": "fake", "limit": 2, "breakout_vol_mult": 1.3,
                      "require_breakout": True, "min_days_since_breakout": 3,
                      "signal_offset": 0})

    cfg = scanner.strategy_config()
    assert cfg.breakout_vol_mult == 1.3
    assert cfg.require_breakout is True
    assert cfg.min_days_since_breakout == 3
    assert cfg.signal_offset == 0


def test_strategy_config_falls_back_to_defaults_before_any_scan() -> None:
    """还没扫过时与新建扫描同一配置（最近已收盘交易日 + 2026-11-27 规则），不能报错。"""
    from app.schemas import production_strategy_config

    cfg = Scanner(provider=RecordingProvider()).strategy_config()
    assert cfg == production_strategy_config()
    assert cfg.signal_offset == 0
    assert cfg.pullback_max_normal_volume_ratio == 1.0


def test_detail_paths_do_not_hardcode_a_default_config() -> None:
    """防回归：三处取数路径都不该再写死 StrategyConfig(signal_offset=1)。"""
    import inspect

    from app.services import scanner as module

    source = inspect.getsource(module)
    for call in ("analyze_bars(code, name, bars, StrategyConfig(",
                 "diagnose_bars(code, name, bars, StrategyConfig("):
        assert call not in source, f"仍在硬编码配置：{call}"
    # 扫描器里不允许手写配置；唯一的默认值来源是 strategy_config() 兜底调用的生产配置
    assert "StrategyConfig(" not in source.replace("-> StrategyConfig:", ""), "扫描器不应手写 StrategyConfig"
    assert source.count("production_strategy_config()") == 1, "默认配置只应出现在 strategy_config() 的兜底分支"


def test_detail_uses_the_params_of_the_batch_it_was_opened_from() -> None:
    """详情要绑定"打开它的那次筛选"，而不是进程里最后一次扫描。

    只记全局 last_strategy_config 时，以下几种情况列表和详情仍会不一致：
    后端重启后看历史批次、新扫描开始后看旧批次、改完参数十分钟内打开缓存过的股票。
    """
    scanner = Scanner(provider=RecordingProvider())
    scanner.run_scan({"provider": "fake", "limit": 2, "breakout_vol_mult": 1.9})

    # 传入某个历史批次的参数时，应以它为准，而不是最近这次的 1.9
    cfg = scanner.strategy_config({"breakout_vol_mult": 1.2, "require_breakout": True})
    assert cfg.breakout_vol_mult == 1.2
    assert cfg.require_breakout is True

    # 不传时仍沿用最近一次扫描
    assert scanner.strategy_config().breakout_vol_mult == 1.9


def test_detail_cache_key_separates_different_filter_settings() -> None:
    """缓存键必须含配置指纹，否则改完参数十分钟内打开会拿到旧结论。"""
    scanner = Scanner(provider=RecordingProvider())
    loose = scanner._detail_cache_key("fake", "000001", 0, {"breakout_vol_mult": 1.2})
    strict = scanner._detail_cache_key("fake", "000001", 0, {"breakout_vol_mult": 1.9})

    assert loose != strict, "不同筛选条件必须落在不同的缓存键上"
    assert loose == scanner._detail_cache_key("fake", "000001", 0, {"breakout_vol_mult": 1.2})


def test_detail_uses_the_data_source_of_that_batch() -> None:
    """扫描指定了 akshare，详情不该改用 auto/adata——价格和结论会对不上。"""
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    names = []

    def spy(name=None):
        names.append(name)
        return provider

    scanner._provider = spy
    scanner.stock_detail("000001", "平安银行", 0, {"provider": "akshare"})
    assert names and names[0] == "akshare", f"应沿用批次的数据源，实得 {names}"
    assert provider.requested == ["000001"]


def test_a_retry_after_a_failed_detail_keeps_the_batch_params() -> None:
    """并发等待方在首个请求失败后重试时，不能把批次参数丢掉。"""
    import inspect

    from app.services import scanner as module

    body = inspect.getsource(module.Scanner.stock_detail)
    assert "return self.stock_detail(code, stock_name, years)" not in body, \
        "重试时丢了 scan_params，会退回最近配置或默认配置"
    assert "scan_params" in body.split("Allow this caller")[-1], "重试要继续传批次参数"


def test_diagnosis_uses_the_data_source_of_that_batch() -> None:
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    requested: list[str | None] = []

    def spy(name=None):
        requested.append(name)
        return provider

    scanner._provider = spy
    scanner.diagnose_stock("000001", "平安银行", {"provider": "akshare"})

    assert requested == ["akshare"]


def test_backtest_uses_the_data_source_of_that_batch() -> None:
    """回测是详情/诊断的第三个入口，同样不能悄悄换数据源。

    详情和诊断各有一条测试守着，回测这条之前是漏的：把
    `self._provider(scan_params.get("provider"))` 改回 `self._provider()`
    全量 195 条测试仍然全绿。
    """
    provider = RecordingProvider()
    scanner = Scanner(provider=provider)
    requested: list[str | None] = []

    def spy(name=None):
        requested.append(name)
        return provider

    scanner._provider = spy
    scanner.backtest_stock("000001", "平安银行", 0, {"provider": "akshare"})

    assert requested == ["akshare"], f"应沿用批次的数据源，实得 {requested}"


class StaleProvider(RecordingProvider):
    def get_daily_bars(self, stock_code):
        from app.services.bars_cache import with_data_quality
        return with_data_quality(
            super().get_daily_bars(stock_code), "2026-05-10T16:00:00", "offline",
        )


def test_stale_bars_never_produce_new_scan_hits(monkeypatch) -> None:
    scanner = Scanner(provider=StaleProvider())

    def must_not_analyze(*args):
        raise AssertionError("旧行情不应进入实时筛选算法")

    monkeypatch.setattr("app.services.scanner.analyze_bars", must_not_analyze)
    assert scanner.run_scan({"provider": "fake", "stock_codes": ["000001"]}) == []
    assert scanner.last_scan_stats["analyzed_count"] == 0
    assert scanner.last_scan_stats["failed_count"] == scanner.last_scan_stats["stale_count"] == 1


def test_stale_detail_remains_viewable_but_is_not_an_actionable_alert(monkeypatch) -> None:
    scanner = Scanner(provider=StaleProvider())
    monkeypatch.setattr("app.services.scanner._run_pipeline", lambda *args: ("ok", "test", _sample_result()))
    detail = scanner.stock_detail("000001", "测试")
    assert detail["bar_count"] > 0
    assert detail["data_quality"]["status"] == "stale"
    assert detail["analysis"]["found"] is False
    assert detail["analysis"]["stage"] == "stale_data"
    assert detail["analysis"]["result"]["category"] != "buy_alert"
    assert scanner.full_bars_cache == {}, "不缓存失败回退，下一次必须有机会恢复"


def test_stale_watch_prices_report_failure_instead_of_success() -> None:
    result = Scanner(provider=StaleProvider()).latest_prices(["000001"])[0]
    assert result["error"]
    assert result["latest_price"] is None


def test_stale_diagnosis_is_explicit() -> None:
    result = Scanner(provider=StaleProvider()).diagnose_stock("000001")
    assert result["stage"] == "stale_data"
    assert result["found"] is False
    assert set(result) == DIAGNOSIS_KEYS


def test_fresh_watch_prices_do_not_depend_on_strategy_config() -> None:
    quote = Scanner(provider=RecordingProvider()).latest_prices(["000001"])[0]
    assert quote["error"] is None
    assert quote["latest_price"] is not None


def test_flag_mode_and_direction_separate_scan_and_detail_caches() -> None:
    scanner = Scanner(provider=RecordingProvider())
    params = {"strategy_mode": "flag_breakout", "flag_direction": "ascending", "signal_offset": 0}
    cfg = scanner.strategy_config(params)
    assert cfg.strategy_mode == "flag_breakout" and cfg.flag_direction == "ascending"
    assert scanner._detail_cache_key("fake", "000001", 0, params) != scanner._detail_cache_key("fake", "000001", 0, {})
    assert scanner._detail_cache_key("fake", "000001", 0, params) != scanner._detail_cache_key("fake", "000001", 0, {**params, "flag_direction": "horizontal"})
    detail = scanner.stock_detail("000001", "测试", scan_params=params)
    assert detail["strategy_mode"] == "flag_breakout"
    assert detail["flag_direction"] == "ascending"


def test_prebreakout_threshold_is_batch_scoped_and_part_of_cache_identity():
    scanner = Scanner(provider=RecordingProvider())
    assert scanner.strategy_config({"provider": "fake"}).max_prebreakout_extension_pct is None
    assert scanner.strategy_config({"max_prebreakout_extension_pct": .2}).max_prebreakout_extension_pct == .2
    assert scanner.strategy_config({"max_prebreakout_extension_pct": 0}).max_prebreakout_extension_pct == 0
    keys = [scanner._detail_cache_key("fake", "000001", 0, {"max_prebreakout_extension_pct": value}) for value in (None, 0, .2, .3)]
    assert len(set(keys)) == 4


def test_prebreakout_duration_is_batch_scoped_and_separates_caches():
    scanner = Scanner(provider=RecordingProvider())
    assert scanner.strategy_config().prebreakout_reject_above_days == 30
    assert scanner.strategy_config({"provider": "fake"}).prebreakout_reject_above_days is None
    assert scanner.strategy_config({"max_prebreakout_extension_pct": .2}).prebreakout_reject_above_days is None
    params = {"prebreakout_reject_above_days": 10, "prebreakout_above_buffer_pct": 0}
    cfg = scanner.strategy_config(params)
    assert cfg.prebreakout_reject_above_days == 10
    assert cfg.prebreakout_above_buffer_pct == 0
    variants = [
        params, {**params, "prebreakout_reject_above_days": None},
        {**params, "prebreakout_reject_above_days": 9},
        {**params, "prebreakout_reject_above_days": 30},
        {**params, "prebreakout_above_buffer_pct": .02},
    ]
    assert len({scanner._detail_cache_key("fake", "000001", 0, v) for v in variants}) == 5


def test_path_parameters_reach_real_scanner_scan_detail_diagnosis_and_backtest(monkeypatch):
    from app.services import scanner as module
    seen = []

    def recording(original):
        def wrapped(*args, **kwargs):
            config = kwargs["config"] if "config" in kwargs else args[3]
            seen.append((config.max_prebreakout_extension_pct, config.prebreakout_reject_above_days,
                         config.prebreakout_above_buffer_pct))
            return original(*args, **kwargs)
        return wrapped

    for name in ("analyze_bars", "diagnose_bars", "evaluate_bars", "_run_pipeline"):
        monkeypatch.setattr(module, name, recording(getattr(module, name)))
    scanner = Scanner(provider=RecordingProvider())
    new = {"provider": "fake", "limit": 1, "max_prebreakout_extension_pct": .5,
           "prebreakout_reject_above_days": 10, "prebreakout_above_buffer_pct": .02}
    old = {"provider": "fake", "max_prebreakout_extension_pct": .2}
    scanner.run_scan(new)
    assert seen == [(.5, 10, .02)]
    for params, expected in ((old, (.2, None, .02)), (new, (.5, 10, .02))):
        seen.clear()
        scanner.stock_detail("000001", "测试", scan_params=params)
        assert seen == [expected], "详情应只运行一次策略，同时生成结果和诊断"
        seen.clear()
        scanner.diagnose_stock("000001", "测试", scan_params=params)
        assert seen == [expected]
        seen.clear()
        scanner.backtest_stock("000001", "测试", lookback_days=1, scan_params=params)
        assert seen == [expected]
