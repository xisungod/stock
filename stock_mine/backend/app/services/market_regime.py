"""沪深300市场环境：只用截至信号日（含）的指数日线判断强弱。

「偏弱」沿用 scripts/analyze_signal_failures.py 的 market_weak_regime 定义：
近20日收益为负，或 MA20 近5日下行。规则不是按收益挑出来的阈值，也尚未被前向验证，
因此默认只作风险标注；`market_filter="downgrade"` 才会把买入预警降为观察。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.services.market_calendar import CalendarUnavailable, china_now, last_completed_session
from app.services.market_data import DEFAULT_CACHE_DIR
from app.services.tencent_bars import fetch_tencent_bars

BENCHMARK_SYMBOL = "sh000300"
BENCHMARK_NAME = "沪深300"
WEAK_RULE = "沪深300近20日收益为负，或MA20近5日下行"


def regime_table(index_bars: pd.DataFrame | None) -> pd.DataFrame | None:
    """逐日指标；每一行只依赖当日及以前的指数收盘价。"""
    if index_bars is None or index_bars.empty:
        return None
    frame = index_bars[["trade_date", "close"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame = frame[frame["close"] > 0].reset_index(drop=True)
    if frame.empty:
        return None
    close = frame["close"]
    ma20 = close.rolling(20).mean()
    frame["return_20_pct"] = close.pct_change(20) * 100
    frame["ma20_slope_5_pct"] = (ma20 / ma20.shift(5) - 1) * 100
    frame["distance_to_ma20_pct"] = (close / ma20 - 1) * 100
    frame["date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    return frame.set_index("date")


def regime_on(table: pd.DataFrame | None, date: str | None) -> dict[str, Any] | None:
    """信号日当天的环境；指数缺这一天（数据没更新或停市）就返回 None，绝不拿旧日子顶替。"""
    if table is None or not date or date not in table.index:
        return None
    row = table.loc[date]
    values = [row["return_20_pct"], row["ma20_slope_5_pct"]]
    if not all(np.isfinite(float(value)) for value in values):
        return None
    return_20 = float(row["return_20_pct"])
    slope = float(row["ma20_slope_5_pct"])
    return {
        "benchmark": BENCHMARK_NAME, "symbol": BENCHMARK_SYMBOL, "date": date, "status": "evaluated",
        "weak": bool(return_20 < 0 or slope < 0), "rule": WEAK_RULE,
        "return_20_pct": round(return_20, 3), "ma20_slope_5_pct": round(slope, 3),
        "distance_to_ma20_pct": round(float(row["distance_to_ma20_pct"]), 3),
    }


def regime_message(regime: dict[str, Any]) -> str:
    return (
        f"{regime['benchmark']}近20日 {regime['return_20_pct']:+.2f}%，"
        f"MA20近5日 {regime['ma20_slope_5_pct']:+.2f}%"
    )


def apply_market_regime(result: Any, regime: dict[str, Any] | None, mode: str) -> Any:
    from app.services.strategy import SignalCategory

    if result is None or mode == "off":
        return result
    if regime is None:
        result.market_regime = {"benchmark": BENCHMARK_NAME, "symbol": BENCHMARK_SYMBOL,
                                "date": result.signal_date, "status": "unavailable", "rule": WEAK_RULE}
        if result.category != SignalCategory.NONE:
            result.risks.append(f"{BENCHMARK_NAME}缺少信号日行情，未评估大盘环境")
        return result
    result.market_regime = {**regime, "mode": mode}
    if not regime["weak"]:
        if result.category != SignalCategory.NONE:
            result.reasons.append(f"大盘环境：{regime_message(regime)}")
        return result
    result.risks.append(f"大盘偏弱：{regime_message(regime)}")
    if mode == "downgrade" and result.category == SignalCategory.BUY_ALERT:
        result.category = SignalCategory.WATCH
        result.suggestion = "观察：形态已确认，但大盘环境偏弱，买入预警降为观察"
        result.market_regime["downgraded"] = True
        for match in result.pattern_matches:
            if match.get("category") == SignalCategory.BUY_ALERT.value:
                match.update(category=SignalCategory.WATCH.value, market_downgraded=True)
    return result


class MarketRegimeService:
    """按「最近已收盘交易日」缓存一份沪深300日线；取数失败时退回磁盘缓存，再失败返回 None。"""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 fetch: Callable[..., pd.DataFrame] | None = None,
                 now: Callable[[], datetime] = china_now):
        self.cache_path = Path(cache_dir) / f"{BENCHMARK_SYMBOL}.json"
        self.fetch = fetch or (lambda: fetch_tencent_bars(BENCHMARK_SYMBOL, start_date="20050101"))
        self.now = now
        self._lock = threading.Lock()
        self._memory: tuple[str, pd.DataFrame | None] | None = None
        self.last_error: str | None = None

    def table(self) -> pd.DataFrame | None:
        try:
            session = last_completed_session(self.now())
        except CalendarUnavailable as exc:
            self.last_error = str(exc)
            session = None
        with self._lock:
            if self._memory is not None and session is not None and self._memory[0] == session:
                return self._memory[1]
            bars = self._load_cache()
            if bars is None or session is None or bars["trade_date"].max() < session:
                try:
                    fetched = self.fetch()
                    if fetched is not None and not fetched.empty:
                        bars = fetched[["trade_date", "close"]].copy()
                        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.strftime("%Y-%m-%d")
                        if session is not None:
                            # 盘中取到的当日指数还没收盘，不能进缓存，否则收盘后不会再刷新。
                            bars = bars[bars["trade_date"] <= session]
                        self._save_cache(bars)
                        self.last_error = None
                except Exception as exc:  # 网络或数据源异常：保留旧缓存，信号日缺数据时自然判为未评估
                    self.last_error = f"{BENCHMARK_NAME}行情获取失败：{exc}"
            table = regime_table(bars)
            if session is not None:
                self._memory = (session, table)
            return table

    def _load_cache(self) -> pd.DataFrame | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload.get("bars", []))
            return frame if {"trade_date", "close"} <= set(frame.columns) and not frame.empty else None
        except (OSError, ValueError, TypeError):
            return None

    def _save_cache(self, bars: pd.DataFrame) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"symbol": BENCHMARK_SYMBOL, "checked_at": self.now().isoformat(timespec="seconds"),
                       "bars": bars.to_dict("records")}
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
