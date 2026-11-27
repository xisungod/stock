"""回踩可以碰到或略穿精准线，但确认日收盘必须站回线上。

默认允许收盘低于线 0.3%。收到 0 之后，最低价仍可低于线，收盘不得低于线。
这不是把最低价必须碰到线（pullback_touch_pct=0）打开。
"""
import pandas as pd

from app.services.strategy import StrategyConfig, _pullback_conditions


def _frame(close: float) -> tuple[pd.DataFrame, dict]:
    line = 10.0
    rows = []
    for index in range(13):
        rows.append({
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
            "open": line, "high": line * 1.01, "low": line * 0.995, "close": line, "volume": 1000.0,
        })
    rows[10].update(high=line * 1.02, low=line * 0.998, close=line * 1.012, volume=3000.0)
    rows[12].update(high=line * 1.01, low=line * 0.98, close=close, volume=500.0)
    return pd.DataFrame(rows), {"index": 10, "volume": 3000.0, "avg_volume": 1000.0}


def _config(**overrides) -> StrategyConfig:
    base = dict(pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
                max_pullback_days=10, min_pullback_days=1)
    base.update(overrides)
    return StrategyConfig(**base)


def test_default_allows_a_close_slightly_under_the_line():
    frame, breakout = _frame(9.98)
    default = _config()
    assert default.allow_undercut_pct == 0.003
    assert default.pullback_touch_pct == 0.012
    conditions = _pullback_conditions(frame, 10.0, breakout, default)
    assert conditions["touched"] is True
    assert conditions["recovered"] is True


def test_zero_undercut_rejects_a_close_under_the_line_but_keeps_the_pierce():
    frame, breakout = _frame(9.98)
    strict = _config(allow_undercut_pct=0.0)
    conditions = _pullback_conditions(frame, 10.0, breakout, strict)
    assert conditions["touched"] is True
    assert conditions["recovered"] is False
    held = _frame(10.0)
    assert _pullback_conditions(held[0], 10.0, held[1], strict)["recovered"] is True
