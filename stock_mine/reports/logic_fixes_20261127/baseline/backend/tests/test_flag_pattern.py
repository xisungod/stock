from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services.flag_pattern import FlagConfig, _fit_channel, detect_flag


def flag_bars(direction="horizontal", terminal="breakout"):
    slope = {"ascending": .04, "horizontal": 0, "descending": -.04}[direction]
    close = [10.0] * 120 + [10.4, 10.8, 11.2, 11.6, 12.0, 12.4]
    close += [12.2 + slope * i + (.18 if i % 2 == 0 else -.18) for i in range(12)]
    upper, lower = 12.2 + slope * 12 + .23, 12.2 + slope * 12 - .23
    close += [upper + .25 if terminal == "breakout" else lower - .3 if terminal == "failed" else (upper + lower) / 2]
    close = np.array(close)
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2025-01-02", periods=len(close)),
        "open": close - .01, "close": close, "high": close + .05, "low": close - .05,
        "volume": [1000] * 120 + [2000] * 6 + np.linspace(1300, 500, 12).tolist() + [4000 if terminal == "breakout" else 500],
    })


@pytest.mark.parametrize("direction", ["ascending", "horizontal", "descending"])
def test_all_three_flag_directions_can_break_out(direction):
    pattern = detect_flag(flag_bars(direction))
    assert pattern is not None
    assert pattern["direction"] == direction
    assert pattern["status"] == "breakout"
    assert pattern["fit_end_date"] < pattern["signal_date"]
    assert pattern["upper_touch_count"] >= 2 and pattern["lower_touch_count"] >= 2


def test_an_uphill_flag_can_exceed_the_pole_end_price():
    bars = flag_bars("ascending")
    result = detect_flag(bars)
    flag = bars[bars.trade_date >= result["flag_start_date"]].iloc[:-1]
    assert flag.high.max() > result["pole"][-1]["price"]
    assert result["direction"] == "ascending"


@pytest.mark.parametrize("terminal,status", [("forming", "forming"), ("failed", "failed")])
def test_forming_and_failed_are_not_breakouts(terminal, status):
    assert detect_flag(flag_bars(terminal=terminal))["status"] == status


def test_signal_bar_cannot_redraw_the_channel():
    bars = flag_bars()
    first = detect_flag(bars)
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] *= 1.3
    second = detect_flag(bars)
    for key in ("channel", "pole", "id", "direction", "fit_end_date", "upper_price", "lower_price"):
        assert second[key] == first[key]


def test_breakout_volume_uses_only_prior_bars():
    bars = flag_bars()
    bars.loc[bars.index[-1], "volume"] = 100
    result = detect_flag(bars)
    assert result["status"] == "unconfirmed_breakout"
    assert result["breakout_volume_ratio"] == pytest.approx(100 / bars.volume.iloc[-21:-1].mean(), abs=1e-4)


def test_no_strong_pole_and_no_volume_contraction_are_rejected():
    bars = flag_bars()
    assert detect_flag(bars, replace(FlagConfig(), pole_min_gain=.5)) is None
    bars.loc[bars.index[-13:-1], "volume"] = 1500
    assert detect_flag(bars) is None


def test_obvious_wedge_is_not_forced_into_a_parallel_flag():
    x = np.arange(20)
    close = 12 + x * .02 + np.where(x % 2 == 0, 1, -1) * (.6 - x * .02)
    assert _fit_channel(close + .01, close - .01, close, close, .3, FlagConfig()) is None


def test_scale_invariance_of_direction_and_breakout():
    bars = flag_bars("ascending")
    expected = detect_flag(bars)
    bars[["open", "close", "high", "low"]] *= 100
    actual = detect_flag(bars)
    assert actual["direction"] == expected["direction"]
    assert actual["status"] == expected["status"]
    assert actual["width_to_pole_pct"] == pytest.approx(expected["width_to_pole_pct"])


def test_zero_volume_and_nonfinite_prices_cannot_trigger():
    bars = flag_bars()
    bars.loc[bars.index[-1], "volume"] = 0
    assert detect_flag(bars)["status"] == "untradable"
    bars.loc[bars.index[-1], "close"] = np.inf
    assert detect_flag(bars) is None


@pytest.mark.parametrize("direction", ["ascending", "horizontal", "descending"])
def test_public_pipeline_and_replay_preserve_the_flag_type(direction):
    from app.services.strategy import StrategyConfig, SignalCategory, analyze_bars
    from app.services.backtest import evaluate_bars
    bars = flag_bars(direction)
    cfg = StrategyConfig(strategy_mode="flag_breakout", signal_offset=0)
    result = analyze_bars("000001", "测试", bars, cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.key_line_price is None
    assert result.chart["flag_pattern"]["direction"] == direction
    report = evaluate_bars("000001", "测试", bars, lookback_days=30, config=cfg)
    assert report["signals"][-1]["flag_direction"] == direction
    assert report["summary"]["signal_count"] == sum(row["signal_count"] for row in report["summary"]["flag_directions"])
    assert report["pattern_events"], "记录形成中和突破事件，而不只是事后赢家"


def test_future_bars_cannot_change_a_past_flag_verdict():
    from app.services.strategy import StrategyConfig, analyze_bars
    bars = flag_bars("ascending")
    cfg = StrategyConfig(strategy_mode="flag_breakout", signal_offset=0)
    before = analyze_bars("000001", "测试", bars, cfg)
    future = bars.tail(5).copy()
    future["trade_date"] = pd.bdate_range(bars.trade_date.iloc[-1] + pd.Timedelta(days=1), periods=5)
    future[["open", "close", "high", "low"]] *= 2
    combined = pd.concat([bars, future], ignore_index=True)
    after = analyze_bars("000001", "测试", combined, replace(cfg, signal_offset=5))
    assert before.to_dict() == after.to_dict()


def test_flags_do_not_require_a_precision_line_or_ma20_filter(monkeypatch):
    from app.services.strategy import StrategyConfig, analyze_bars
    def unexpected(*args, **kwargs):
        raise AssertionError("旗形不应借道精准线聚类")
    monkeypatch.setattr("app.services.strategy._cluster_key_lines", unexpected)
    assert analyze_bars("000001", "测试", flag_bars(), StrategyConfig(strategy_mode="flag_breakout", signal_offset=0))


def test_direction_and_confirmed_only_filters_are_explicit():
    from app.services.strategy import StrategyConfig, analyze_bars, SignalCategory
    cfg = StrategyConfig(strategy_mode="flag_breakout", signal_offset=0, flag_direction="descending")
    assert analyze_bars("000001", "测试", flag_bars("ascending"), cfg) is None
    cfg = replace(cfg, flag_direction="all", require_breakout=True)
    assert analyze_bars("000001", "测试", flag_bars(terminal="forming"), cfg).category == SignalCategory.NONE
    with pytest.raises(ValueError):
        analyze_bars("000001", "测试", flag_bars(), replace(cfg, min_days_since_breakout=2))
