from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app.services.chart_patterns import detect_chart_patterns
from app.services.market_calendar import SETTLED_AFTER, china_time
from app.services.strategy import (
    SignalResult,
    StrategyConfig,
    _completed_session_bars,
    _normalize_bars,
    _swings,
)


def _swing_points(df: pd.DataFrame, chart_bars: list[dict[str, Any]]) -> tuple[list, list, dict | None]:
    """图上大级别趋势峰谷；平台事件引擎的精准点另由冻结线证据提供。

    波峰取当日最高价、波谷取当日最低价，标记才会贴在 K 线的上下影线端点上。
    """
    cfg = StrategyConfig()
    if len(df) < 3:
        return [], [], None
    highs, lows, pending = _swings(df, cfg)
    peaks = [{"date": chart_bars[i]["date"], "price": chart_bars[i]["high"]} for i in highs]
    valleys = [{"date": chart_bars[i]["date"], "price": chart_bars[i]["low"]} for i in lows]
    # 最右侧尚未回撤确认的极值只是候选，图上要与已确认的区分开
    candidate = None
    if pending is not None and 0 <= pending["index"] < len(chart_bars):
        bar = chart_bars[pending["index"]]
        candidate = {"date": bar["date"], "kind": pending["kind"],
                     "price": bar["high"] if pending["kind"] == "peak" else bar["low"]}
    return peaks, valleys, candidate


def _settled_pattern_bars(df: pd.DataFrame, quality: dict[str, Any]) -> pd.DataFrame:
    settled = _completed_session_bars(df, quality, StrategyConfig(signal_offset=0))
    checked_at = quality.get("checked_at")
    if not checked_at:
        return settled
    try:
        checked = china_time(datetime.fromisoformat(checked_at))
    except (ValueError, TypeError):
        return settled
    # 收盘后请求失败时，盘中保存的最后一根 K 线仍未完成，直至再次成功获取。
    cutoff = checked.date()
    if checked.time() < SETTLED_AFTER:
        cutoff -= timedelta(days=1)
    return settled[settled["trade_date"] <= pd.Timestamp(cutoff)]


def build_full_chart(
    bars: pd.DataFrame,
    result: SignalResult | None = None,
    historical_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    df = _normalize_bars(bars)
    for period in (5, 10, 20, 60):
        df[f"ma{period}"] = df["close"].rolling(period).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2
    chart_bars = []
    for row in df.itertuples(index=False):
        item = {
            "date": pd.to_datetime(row.trade_date).strftime("%Y-%m-%d"),
            "open": round(float(row.open), 3),
            "close": round(float(row.close), 3),
            "low": round(float(row.low), 3),
            "high": round(float(row.high), 3),
            "volume": float(row.volume),
        }
        for key in ("ma5", "ma10", "ma20", "ma60", "dif", "dea", "macd"):
            value = getattr(row, key)
            item[key] = round(float(value), 4) if pd.notna(value) else None
        chart_bars.append(item)

    markers = [
        {"date": item["signal_date"], "type": "alert", "label": "历史预警", "price": item["signal_price"]}
        for item in historical_signals or []
    ]
    if result is not None:
        if result.breakout_date:
            markers.append({"date": result.breakout_date, "type": "breakout", "label": "突破"})
        if result.pullback_date:
            markers.append({"date": result.pullback_date, "type": "pullback", "label": "回踩"})
    peaks, valleys, pending_swing = _swing_points(df, chart_bars)
    # 形态只用已收盘 K 线识别：盘中未定型的最后一根会让「突破」在收盘前反复出现又消失。
    settled = _settled_pattern_bars(df, bars.attrs.get("data_quality") or {})
    chart_patterns = detect_chart_patterns(
        settled.reset_index(drop=True), [item["date"] for item in chart_bars[:len(settled)]]
    )
    return {
        "bars": chart_bars,
        "peaks": peaks,
        "valleys": valleys,
        "pending_swing": pending_swing,
        "key_line": result.key_line_price if result is not None else None,
        "key_lines": result.key_lines if result is not None else [],
        "markers": markers,
        "flag_pattern": result.flag_pattern if result is not None else None,
        "chart_patterns": chart_patterns,
        "range": {
            "start": chart_bars[0]["date"] if chart_bars else None,
            "end": chart_bars[-1]["date"] if chart_bars else None,
            "count": len(chart_bars),
        },
    }


def build_market_snapshot(bars: pd.DataFrame) -> dict[str, Any]:
    df = _normalize_bars(bars)
    if df.empty:
        return {}
    latest = df.iloc[-1]
    previous_close = float(df["close"].iloc[-2]) if len(df) > 1 else None
    close = float(latest["close"])
    recent_20 = df.tail(20)
    recent_250 = df.tail(250)
    first_year_close = float(recent_250["close"].iloc[0]) if len(recent_250) > 1 else None
    returns = df["close"].pct_change().tail(20)
    return {
        "trade_date": pd.to_datetime(latest["trade_date"]).strftime("%Y-%m-%d"),
        "open": round(float(latest["open"]), 3),
        "high": round(float(latest["high"]), 3),
        "low": round(float(latest["low"]), 3),
        "close": round(close, 3),
        "previous_close": round(previous_close, 3) if previous_close is not None else None,
        "change_pct": round((close / previous_close - 1) * 100, 2) if previous_close else None,
        "volume": float(latest["volume"]),
        "average_volume_20": round(float(recent_20["volume"].mean()), 0),
        "volume_ratio_20": round(float(latest["volume"]) / float(recent_20["volume"].mean()), 2) if float(recent_20["volume"].mean()) > 0 else None,
        "high_52w": round(float(recent_250["high"].max()), 3),
        "low_52w": round(float(recent_250["low"].min()), 3),
        "return_1y_pct": round((close / first_year_close - 1) * 100, 2) if first_year_close else None,
        "volatility_20_pct": round(float(returns.std()) * (252**0.5) * 100, 2) if returns.notna().any() else None,
    }
