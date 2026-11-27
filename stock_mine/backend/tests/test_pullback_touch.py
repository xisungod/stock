"""回踩是否碰到精准线：默认允许最低价停在线上方 1.2% 内，收到 0 才必须碰到线。"""
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
    rows[12].update(high=line * 1.02, low=low, close=line * 1.005, volume=500.0)
    frame = pd.DataFrame(rows)
    breakout = {"index": 10, "volume": 3000.0, "avg_volume": 1000.0}
    return frame, breakout


def test_default_touch_allows_a_low_one_percent_above_the_line():
    frame, breakout = _frame(10.10)
    default = StrategyConfig(pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
                             max_pullback_days=10, min_pullback_days=1)
    assert default.pullback_touch_pct == 0.012
    assert _pullback_conditions(frame, 10.0, breakout, default)["touched"] is True
    strict = StrategyConfig(pullback_touch_pct=0.0, pullback_max_normal_volume_ratio=1.0,
                            max_post_breakout_runup_pct=0.10, max_pullback_days=10, min_pullback_days=1)
    assert _pullback_conditions(frame, 10.0, breakout, strict)["touched"] is False


def test_a_low_on_the_line_counts_as_a_touch_with_zero_slack():
    frame, breakout = _frame(10.0)
    strict = StrategyConfig(pullback_touch_pct=0.0, pullback_max_normal_volume_ratio=1.0,
                            max_post_breakout_runup_pct=0.10, max_pullback_days=10, min_pullback_days=1)
    assert _pullback_conditions(frame, 10.0, breakout, strict)["touched"] is True
