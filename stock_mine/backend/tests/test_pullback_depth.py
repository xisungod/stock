"""回踩必须从突破后最高价至少回落 3%。轻微回落不算确认。

默认不设这条下限。最低价仍可低于线 3%、停在线上方 1.2% 内，收盘仍可低于线 0.3%。
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
    # 突破日最高 10.20，之后没有更高。
    rows[10].update(high=line * 1.02, low=line * 0.998, close=line * 1.012, volume=3000.0)
    rows[12].update(high=line * 1.01, low=low, close=line, volume=500.0)
    return pd.DataFrame(rows), {"index": 10, "volume": 3000.0, "avg_volume": 1000.0}


def _config(**overrides) -> StrategyConfig:
    base = dict(
        pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10,
        max_pullback_days=10, min_pullback_days=1,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_default_accepts_a_pullback_that_only_gives_back_about_one_percent():
    # 最高 10.20，最低 10.05，回落约 1.5%。
    frame, breakout = _frame(10.05)
    default = _config()
    assert default.min_pullback_depth_pct is None
    conditions = _pullback_conditions(frame, 10.0, breakout, default)
    assert conditions["depth_ok"] is True
    assert conditions["touched"] is True


def test_three_percent_depth_rejects_a_shallow_dip_and_keeps_a_deeper_one():
    shallow, breakout = _frame(10.05)
    strict = _config(min_pullback_depth_pct=0.03)
    assert strict.pullback_touch_pct == 0.012
    assert strict.max_intraday_undercut_pct == 0.03
    assert strict.allow_undercut_pct == 0.003
    shallow_conditions = _pullback_conditions(shallow, 10.0, breakout, strict)
    assert shallow_conditions["touched"] is True
    assert shallow_conditions["depth_ok"] is False
    # 最高 10.20，最低 9.85，回落约 3.4%，仍在线下方 3% 以内。
    deep, deep_breakout = _frame(9.85)
    deep_conditions = _pullback_conditions(deep, 10.0, deep_breakout, strict)
    assert deep_conditions["touched"] is True
    assert deep_conditions["depth_ok"] is True
