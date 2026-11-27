"""回踩确认必须在突破后 5 个交易日内完成。默认窗口是 1～10 日。"""
import pandas as pd

from app.services.strategy import StrategyConfig, _pullback_conditions


def _frame(age: int) -> tuple[pd.DataFrame, dict]:
    line = 10.0
    rows = []
    for index in range(11 + age):
        rows.append({
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
            "open": line, "high": line * 1.01, "low": line * 0.99, "close": line, "volume": 1000.0,
        })
    rows[10].update(close=line * 1.012, volume=3000.0)
    return pd.DataFrame(rows), {"index": 10, "volume": 3000.0, "avg_volume": 1000.0}


def _config(maximum: int) -> StrategyConfig:
    return StrategyConfig(
        max_pullback_days=maximum, min_pullback_days=1,
        pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
    )


def test_default_window_still_allows_day_eight():
    frame, breakout = _frame(8)
    default = _config(10)
    assert default.max_pullback_days == 10
    assert _pullback_conditions(frame, 10.0, breakout, default)["in_window"] is True


def test_five_day_cap_rejects_day_eight_and_keeps_day_four():
    late, breakout = _frame(8)
    strict = _config(5)
    assert _pullback_conditions(late, 10.0, breakout, strict)["in_window"] is False
    early, early_breakout = _frame(4)
    assert _pullback_conditions(early, 10.0, early_breakout, strict)["in_window"] is True
