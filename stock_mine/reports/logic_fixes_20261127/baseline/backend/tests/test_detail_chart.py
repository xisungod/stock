from __future__ import annotations

from datetime import date, timedelta

import math

import pandas as pd

from app.services.detail import build_full_chart


def _wavy_bars(count: int = 260) -> pd.DataFrame:
    """带清晰摆动的行情，足以产生多个波峰波谷。"""
    start = date(2025, 1, 1)
    closes = [10.0 + 1.4 * math.sin(i / 7.0) + i * 0.004 for i in range(count)]
    return pd.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(count)],
            "open": [c * 0.995 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [1000 + i for i in range(count)],
        }
    )


def test_full_chart_exposes_swing_points_for_the_toggle() -> None:
    """图上要能开关波峰波谷，接口就得把它们送出来。"""
    chart = build_full_chart(_wavy_bars())

    assert chart["peaks"], "应给出波峰"
    assert chart["valleys"], "应给出波谷"
    for point in chart["peaks"] + chart["valleys"]:
        assert set(point) >= {"date", "price"}, f"每个点要带日期和价格，实得 {point}"


def test_swing_points_line_up_with_the_bars_they_mark() -> None:
    """波峰的价格必须等于那一天的最高价，波谷等于最低价——否则标记会飘在K线外。"""
    bars = _wavy_bars()
    chart = build_full_chart(bars)
    by_date = {b["date"]: b for b in chart["bars"]}

    for peak in chart["peaks"]:
        bar = by_date[peak["date"]]
        assert peak["price"] == bar["high"], f"{peak['date']} 波峰价 {peak['price']} != 当日高 {bar['high']}"
    for valley in chart["valleys"]:
        bar = by_date[valley["date"]]
        assert valley["price"] == bar["low"], f"{valley['date']} 波谷价 {valley['price']} != 当日低 {bar['low']}"


def test_swing_points_are_absent_when_history_is_too_short() -> None:
    """数据太少时不要硬凑摆动点。"""
    chart = build_full_chart(_wavy_bars(6))
    assert chart["peaks"] == []
    assert chart["valleys"] == []
