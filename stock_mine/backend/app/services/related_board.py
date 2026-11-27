"""按走势相关性找出与个股最相关的板块指数。

为什么用相关性而不是"所属行业"：`industry_cache.json` 只覆盖 2975/5008 只股票
（59%），中国核建、宁德时代、中芯国际这类都查不到行业，板块 K 线必然是空的。
相关性不依赖那份映射，任何股票都能给出结果。

关键：必须对**日收益率**求相关，而不是收盘价。任意两条上升趋势的价格序列
相关系数都接近 1，那样选出来的只是"同样在涨的板块"，没有信息量。
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.akshare_guard import call_akshare, get_akshare

MIN_OVERLAP_DAYS = 60


def _prices(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric > 0].sort_index()


def _aligned_returns(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """先按共同交易日对齐价格，再各自算收益。

    顺序不能反：先算收益再取日期交集的话，个股停牌 10 天后复牌那日的收益跨了
    整个停牌区间，而板块那天只跨一天——两个不同跨度的收益被直接相比。
    """
    a, b = _prices(left), _prices(right)
    shared = a.index.intersection(b.index)
    if len(shared) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    a, b = a.loc[shared].sort_index(), b.loc[shared].sort_index()
    return a.pct_change().dropna(), b.pct_change().dropna()


def correlation(
    stock_closes: pd.Series,
    board_closes: pd.Series,
    min_overlap: int = MIN_OVERLAP_DAYS,
) -> float | None:
    """两条收盘价序列日收益率的皮尔逊相关系数。

    只比较共同的交易日（个股停牌、板块指数起始日不同都会造成缺口）。
    重叠不足或任一方没有波动时返回 None——宁可不给答案，也不要把 NaN
    混进排序里冒充一个名次。
    """
    left, right = _aligned_returns(stock_closes, board_closes)
    if len(left) < max(2, min_overlap):
        return None
    a = left.to_numpy(dtype=float)
    b = right.to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return None if not np.isfinite(value) else round(value, 4)


def rank_related_boards(
    stock_closes: pd.Series,
    boards: dict[str, pd.Series],
    top_n: int = 1,
    min_overlap: int = MIN_OVERLAP_DAYS,
    kinds: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """把候选板块按与个股的相关性从高到低排序。"""
    scored: list[dict[str, Any]] = []
    for name, closes in (boards or {}).items():
        value = correlation(stock_closes, closes, min_overlap)
        if value is None:
            continue
        overlap = len(_aligned_returns(stock_closes, closes)[0])
        scored.append({
            "name": name,
            "kind": (kinds or {}).get(name, "industry"),
            "correlation": value,
            "overlap_days": overlap,
        })
    scored.sort(key=lambda item: (-item["correlation"], item["name"]))
    return scored[: max(1, top_n)]


CACHE_TTL = timedelta(hours=24)
# 至少要成功抓到这个比例的板块，才认为这次构建是有效的。全失败时若把空结果
# 落盘，ensure_ready() 会以为已经建好，24 小时内都不再重试。
MIN_BUILD_SUCCESS_RATIO = 0.5
UNIVERSE_YEARS = 3


def _close_column(frame: pd.DataFrame) -> str | None:
    for name in ("收盘价", "收盘", "close"):
        if name in frame.columns:
            return name
    return None


def _date_column(frame: pd.DataFrame) -> str | None:
    for name in ("日期", "date", "trade_date"):
        if name in frame.columns:
            return name
    return None


class BoardIndexRepository:
    """同花顺板块指数收盘价仓库。

    候选池是全部行业(90) + 概念(375) = 465 个板块。首次构建约 90 秒
    （akshare 有全局锁，实际是串行取数），之后 24 小时内直接读盘。
    这份缓存与个股无关，所有股票共用。
    """

    def __init__(
        self,
        cache_dir: str | Path,
        client: Any | None = None,
        years: int = UNIVERSE_YEARS,
        ttl: timedelta = CACHE_TTL,
    ):
        self.cache_dir = Path(cache_dir)
        self.client = client
        self.years = years
        self.ttl = ttl
        self._closes: dict[str, pd.Series] | None = None
        self._kinds: dict[str, str] = {}
        self._build_lock = threading.Lock()
        self._build_thread: threading.Thread | None = None

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"board_index_{self.years}y.json"

    def _ak(self) -> Any:
        if self.client is None:
            self.client = get_akshare()
        return self.client

    def _universe(self) -> tuple[list[tuple[str, str]], set[str]]:
        ak = self._ak()
        out: list[tuple[str, str]] = []
        loaded_kinds: set[str] = set()
        for kind, loader in (
            ("industry", lambda: ak.stock_board_industry_name_ths()),
            ("concept", lambda: ak.stock_board_concept_name_ths()),
        ):
            try:
                frame = call_akshare(loader)
            except Exception:
                continue
            if frame is None or frame.empty or "name" not in frame.columns:
                continue
            loaded_kinds.add(kind)
            out.extend((str(name), kind) for name in frame["name"])
        return out, loaded_kinds

    def _fetch_one(self, name: str, kind: str) -> pd.Series | None:
        ak = self._ak()
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=max(1, self.years) * 366)).strftime("%Y%m%d")
        loader = (
            ak.stock_board_industry_index_ths if kind == "industry" else ak.stock_board_concept_index_ths
        )
        try:
            frame = call_akshare(lambda: loader(symbol=name, start_date=start, end_date=end))
        except Exception:
            return None
        if frame is None or frame.empty:
            return None
        close_col, date_col = _close_column(frame), _date_column(frame)
        if not close_col or not date_col:
            return None
        series = pd.Series(
            pd.to_numeric(frame[close_col], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame[date_col], errors="coerce"),
            dtype=float,
        ).dropna()
        series = series[~series.index.isna()].sort_index()
        return series if len(series) else None

    def closes(self) -> dict[str, pd.Series]:
        if self._closes is not None:
            return self._closes
        cached = self._load()
        if cached is not None:
            self._closes, self._kinds = cached
            return self._closes

        # 名单只取一次：早先循环用一次、算总数又请求一次，既多一趟网络，
        # 第二次若失败返回 0 还会绕过成功率门槛、把残缺结果缓存 24 小时。
        universe, loaded_kinds = self._universe()
        if loaded_kinds != {"industry", "concept"}:
            return {}
        closes: dict[str, pd.Series] = {}
        kinds: dict[str, str] = {}
        for name, kind in universe:
            series = self._fetch_one(name, kind)
            # 465 个板块里挂掉一两个是常态，跳过即可，不要让整个功能不可用
            if series is None:
                continue
            closes[name] = series
            kinds[name] = kind
        universe_size = len(universe)
        if not closes or (universe_size and len(closes) < universe_size * MIN_BUILD_SUCCESS_RATIO):
            # 成功比例太低说明是数据源在抽风，不是板块真的不存在——保持"未就绪"，
            # 让下次调用重试，而不是把残缺结果当成定稿缓存起来。
            self._closes = None
            return {}
        self._closes, self._kinds = closes, kinds
        self._save(closes, kinds)
        return closes

    @property
    def is_building(self) -> bool:
        thread = self._build_thread
        return bool(thread and thread.is_alive())

    def ensure_ready(self, background: bool = True) -> bool:
        """缓存可用就返回 True；否则在后台开始构建并返回 False。

        首次构建要串行拉 465 个板块指数（akshare 有全局锁），实测约 4.5 分钟。
        HTTP 请求不能挂在那里等，所以先答"还没好"，让前端稍后再问。
        """
        if self._closes is not None:
            return True
        cached = self._load()
        if cached is not None:
            self._closes, self._kinds = cached
            return True
        if not background:
            self.closes()
            return True
        with self._build_lock:
            if self._build_thread is None or not self._build_thread.is_alive():
                self._build_thread = threading.Thread(
                    target=self.closes, name="board-index-build", daemon=True
                )
                self._build_thread.start()
        return False

    def wait_for_build(self, timeout: float | None = None) -> None:
        thread = self._build_thread
        if thread is not None:
            thread.join(timeout)

    def kinds(self) -> dict[str, str]:
        self.closes()
        return self._kinds

    def _load(self) -> tuple[dict[str, pd.Series], dict[str, str]] | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            updated = datetime.fromisoformat(payload["updated_at"])
            if datetime.now() - updated > self.ttl:
                return None
            closes = {
                name: pd.Series(
                    item["closes"], index=pd.to_datetime(item["dates"]), dtype=float
                )
                for name, item in payload["boards"].items()
            }
            kinds = {name: item["kind"] for name, item in payload["boards"].items()}
            if not closes or set(kinds.values()) != {"industry", "concept"}:
                return None
            return closes, kinds
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save(self, closes: dict[str, pd.Series], kinds: dict[str, str]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "boards": {
                            name: {
                                "kind": kinds.get(name, "industry"),
                                "dates": [d.strftime("%Y-%m-%d") for d in series.index],
                                "closes": [float(v) for v in series.to_numpy()],
                            }
                            for name, series in closes.items()
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass


    def chart_bars(self, name: str, kind: str, years: int | None = None) -> pd.DataFrame:
        """取某个板块指数的 OHLCV，列名规整成 build_full_chart 认得的形式。"""
        ak = self._ak()
        span = self.years if years is None else years
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=max(1, span) * 366)).strftime("%Y%m%d")
        loader = (
            ak.stock_board_industry_index_ths if kind == "industry" else ak.stock_board_concept_index_ths
        )
        frame = call_akshare(lambda: loader(symbol=name, start_date=start, end_date=end))
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume"])
        renamed = frame.rename(columns={
            "日期": "trade_date", "date": "trade_date",
            "开盘价": "open", "开盘": "open",
            "最高价": "high", "最高": "high",
            "最低价": "low", "最低": "low",
            "收盘价": "close", "收盘": "close",
            "成交量": "volume", "volume": "volume",
        })
        for column in ("trade_date", "open", "high", "low", "close"):
            if column not in renamed.columns:
                return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume"])
        if "volume" not in renamed.columns:
            renamed["volume"] = 0.0
        return renamed[["trade_date", "open", "high", "low", "close", "volume"]].copy()
