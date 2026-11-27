"""确认日收盘不得高出 20 日均线 10%。默认不设这条上限。"""
import pandas as pd

from app.services.strategy import StrategyConfig, _pullback_conditions


def _frame(last_close: float) -> tuple[pd.DataFrame, dict]:
    line = 10.0
    rows = []
    for index in range(25):
        rows.append({
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
            "open": line, "high": line * 1.01, "low": line * 0.99, "close": line, "volume": 1000.0,
        })
    rows[22].update(high=line * 1.02, close=line * 1.012, volume=3000.0)
    rows[24].update(high=max(line * 1.01, last_close), low=line * 0.995, close=last_close, volume=500.0)
    return pd.DataFrame(rows), {"index": 22, "volume": 3000.0, "avg_volume": 1000.0}


def _config(**overrides) -> StrategyConfig:
    base = dict(
        pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
        max_pullback_days=10, min_pullback_days=1,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_default_does_not_cap_distance_above_the_moving_average():
    frame, breakout = _frame(11.2)
    default = _config()
    assert default.max_close_above_ma_pct is None
    assert _pullback_conditions(frame, 10.0, breakout, default)["ma_extension_ok"] is True


def test_ten_percent_cap_rejects_a_close_far_above_the_average():
    far, breakout = _frame(11.2)
    strict = _config(max_close_above_ma_pct=0.10)
    assert _pullback_conditions(far, 10.0, breakout, strict)["ma_extension_ok"] is False
    near, near_breakout = _frame(10.5)
    assert _pullback_conditions(near, 10.0, near_breakout, strict)["ma_extension_ok"] is True
