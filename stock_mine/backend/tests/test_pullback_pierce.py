"""回踩最低价跌破精准线的上限：默认 3%，收到 1% 后跌破更深的当天不算踩到。

不改确认日收盘容差，也不要求最低价必须碰到线。
"""
import pandas as pd

from app.services.strategy import StrategyConfig, _pullback_conditions


def _frame(low: float) -> tuple[pd.DataFrame, dict]:
    line = 10.0
    rows = []
    for index in range(13):
        rows.append({
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
            "open": line, "high": line * 1.01, "low": line * 0.995, "close": line, "volume": 1000.0,
        })
    rows[10].update(high=line * 1.02, low=line * 0.998, close=line * 1.012, volume=3000.0)
    rows[12].update(high=line * 1.01, low=low, close=line * 0.998, volume=500.0)
    return pd.DataFrame(rows), {"index": 10, "volume": 3000.0, "avg_volume": 1000.0}


def _config(**overrides) -> StrategyConfig:
    base = dict(
        pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
        max_pullback_days=10, min_pullback_days=1,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_default_allows_a_two_percent_pierce():
    frame, breakout = _frame(9.80)
    default = _config()
    assert default.max_intraday_undercut_pct == 0.03
    assert default.allow_undercut_pct == 0.003
    assert default.pullback_touch_pct == 0.012
    conditions = _pullback_conditions(frame, 10.0, breakout, default)
    assert conditions["touched"] is True
    assert conditions["recovered"] is True


def test_one_percent_cap_rejects_a_two_percent_pierce_and_keeps_a_shallow_one():
    deep, breakout = _frame(9.80)
    strict = _config(max_intraday_undercut_pct=0.01)
    assert _pullback_conditions(deep, 10.0, breakout, strict)["touched"] is False
    shallow, shallow_breakout = _frame(9.95)
    assert _pullback_conditions(shallow, 10.0, shallow_breakout, strict)["touched"] is True
