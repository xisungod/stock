from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services.strategy import (
    KeyLine, SignalCategory, SignalResult, StrategyConfig,
    _check_prebreakout_extension, _find_breakout, _select_best_signal, analyze_bars,
)


def fixture():
    close = np.full(25, 9.9)
    close[20], close[24] = 10.3, 10.1
    volume = np.full(25, 1000.0)
    volume[20] = 2500
    bars = pd.DataFrame({
        "trade_date": pd.bdate_range("2026-01-01", periods=25),
        "open": close - .02, "close": close, "high": close + .1,
        "low": close - .1, "volume": volume,
    })
    bars.loc[24, "low"] = 10
    bars["ma"] = bars.close.rolling(5).mean()
    line = KeyLine(
        price=10, strength=3, touch_indices=[2, 6, 10],
        last_touch_index=10, formation_index=11,
        resistance_confirmed_at={2: 3, 6: 7, 10: 11},
    )
    cfg = StrategyConfig(signal_offset=0, min_history_bars=20, ma_period=5,
                         ma_slope_days=3, include_chart=False, max_prebreakout_extension_pct=.20,
                         prebreakout_reject_above_days=None, require_pullback=False,
                         max_pullback_days=None, require_breakout_volume=True)
    return bars, line, cfg


@pytest.mark.parametrize("high,passes", [(11.99, True), (12, True), (12.0001, False), (13, False)])
def test_twenty_percent_is_inclusive_and_uses_highs_not_closes(high, passes):
    bars, line, cfg = fixture()
    bars.loc[14, "high"] = high
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.prebreakout_check["passed"] is passes
    assert result.prebreakout_check["threshold_price"] == 12
    assert result.prebreakout_check["limit_pct"] == 20
    assert result.category == (SignalCategory.BUY_ALERT if passes else SignalCategory.NONE)


def test_breakout_day_future_bars_and_history_before_anchor_are_excluded():
    bars, line, cfg = fixture()
    bars.loc[[1, 20, 22], "high"] = 30
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.prebreakout_check["max_high"] == 10
    assert result.prebreakout_check["checked_bars"] == 9


def test_later_support_touch_does_not_reset_the_window():
    bars, line, cfg = fixture()
    bars.loc[14, "high"] = 13
    line.support_indices = [18]
    check = _check_prebreakout_extension(bars, line, _find_breakout(bars, line, cfg), cfg)
    assert check["passed"] is False
    assert check["anchor_date"] == str(bars.trade_date.iloc[10].date())


@pytest.mark.parametrize("confirmed_at", [20, 22])
def test_a_peak_confirmed_on_or_after_breakout_cannot_hide_the_earlier_overshoot(confirmed_at):
    bars, line, cfg = fixture()
    bars.loc[14, "high"] = 13
    line.touch_indices.append(17)
    line.resistance_confirmed_at[17] = confirmed_at
    check = _check_prebreakout_extension(bars, line, _find_breakout(bars, line, cfg), cfg)
    assert check["anchor_date"] == str(bars.trade_date.iloc[10].date())
    assert check["passed"] is False


def test_missing_confirmation_metadata_is_not_assumed_to_be_known():
    bars, line, cfg = fixture()
    line.resistance_confirmed_at = {}
    check = _check_prebreakout_extension(bars, line, _find_breakout(bars, line, cfg), cfg)
    assert check["status"] == "no_prior_point" and check["passed"] is False


def test_rejected_latest_breakout_does_not_fall_back_to_an_older_one():
    bars, line, cfg = fixture()
    bars.loc[15, "close"] = 10.3
    bars.loc[15, "high"] = 10.4
    bars.loc[15, "volume"] = 2500
    bars.loc[18, "high"] = 13
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.breakout_date == str(bars.trade_date.iloc[20].date())
    assert result.category == SignalCategory.NONE
    assert result.prebreakout_check["status"] == "exceeded"


def test_disabled_and_not_yet_broken_out_are_not_falsely_reported_as_passed():
    bars, line, cfg = fixture()
    assert _check_prebreakout_extension(bars, line, None, cfg)["passed"] is None
    assert _check_prebreakout_extension(bars, line, None, cfg)["status"] == "no_breakout"
    disabled = _check_prebreakout_extension(bars, line, _find_breakout(bars, line, cfg),
                                           replace(cfg, max_prebreakout_extension_pct=None))
    assert disabled["status"] == "disabled" and disabled["passed"] is None


def test_zero_means_zero_tolerance_not_disabled():
    bars, line, cfg = fixture()
    bars.loc[14, "high"] = 10.01
    cfg = replace(cfg, max_prebreakout_extension_pct=0)
    check = _check_prebreakout_extension(bars, line, _find_breakout(bars, line, cfg), cfg)
    assert check["passed"] is False


@pytest.mark.parametrize("failed_rule", ["amplitude", "duration"])
def test_combined_mode_can_still_select_a_flag_after_precision_is_rejected(monkeypatch, failed_rule):
    from app.services import strategy
    if failed_rule == "amplitude":
        bars, line, cfg = fixture()
        cfg = replace(cfg, prebreakout_reject_above_days=10)
        bars.loc[14, "high"] = 13
    else:
        bars, line, cfg = duration_fixture(10)
    def precision(code, name, frame, config):
        frame["ma"] = frame.close.rolling(config.ma_period).mean()
        result = _select_best_signal(code, name, frame, [line], config)
        assert result.category == SignalCategory.NONE
        return "no_signal", result.suggestion, result
    def flag(code, name, frame, config):
        assert config.max_prebreakout_extension_pct is None
        assert config.prebreakout_reject_above_days is None
        result = SignalResult(code, name, SignalCategory.BUY_ALERT, 80, "旗形突破", None, 0,
                              str(frame.trade_date.iloc[-1].date()), pattern_type="flag_breakout",
                              flag_pattern={"id": "flag:test", "direction": "ascending", "status": "breakout"})
        return "ok", result.suggestion, result
    monkeypatch.setattr(strategy, "_run_precision_pipeline", precision)
    monkeypatch.setattr(strategy, "_run_flag_pipeline", flag)
    result = analyze_bars("000001", "测试", bars, replace(cfg, strategy_mode="combined"))
    assert result.category == SignalCategory.BUY_ALERT
    assert [m["pattern_type"] for m in result.pattern_matches] == ["flag_breakout"]


def duration_fixture(run_length=10):
    bars, _, cfg = fixture()
    line = KeyLine(
        price=10, strength=3, touch_indices=[1, 3, 6], last_touch_index=6,
        formation_index=7, resistance_confirmed_at={1: 2, 3: 4, 6: 7},
    )
    if run_length:
        bars.loc[7:6 + run_length, ["open", "close", "high", "low"]] = [10.28, 10.3, 10.4, 10.2]
    bars["ma"] = bars.close.rolling(5).mean()
    return bars, line, replace(cfg, max_prebreakout_extension_pct=.50,
                               prebreakout_reject_above_days=10)


@pytest.mark.parametrize("high,passes", [(14.99, True), (15, True), (15.0001, False)])
def test_fifty_percent_ceiling_is_inclusive(high, passes):
    bars, line, cfg = duration_fixture(9)
    bars.loc[12, "high"] = high
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    check = result.prebreakout_check
    assert check["threshold_price"] == 15
    assert check["limit_pct"] == 50
    assert check["extension_passed"] is passes
    assert check["consecutive_passed"] is True
    assert result.category == (SignalCategory.BUY_ALERT if passes else SignalCategory.NONE)


@pytest.mark.parametrize("run_length,passes", [(0, True), (9, True), (10, False), (11, False)])
def test_ten_consecutive_effective_closes_rejects_but_nine_passes(run_length, passes):
    bars, line, cfg = duration_fixture(run_length)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    check = result.prebreakout_check
    assert check["max_consecutive_above_days"] == run_length
    assert check["reject_above_days"] == 10
    assert check["effective_close_threshold"] == 10.2
    assert check["above_buffer_pct"] == 2
    assert check["extension_passed"] is True
    assert check["consecutive_passed"] is passes
    assert check["passed"] is passes
    assert check["status"] == ("passed" if passes else "duration_exceeded")
    assert check["longest_run_start_date"] == (str(bars.trade_date.iloc[7].date()) if run_length else None)
    assert check["longest_run_end_date"] == (str(bars.trade_date.iloc[6 + run_length].date()) if run_length else None)
    assert result.category == (SignalCategory.BUY_ALERT if passes else SignalCategory.NONE)
    if not passes:
        assert "10 日排除阈值" in result.suggestion


@pytest.mark.parametrize("close,expected", [(10.19, 6), (10.2, 6), (10.2001, 12)])
def test_buffer_boundary_resets_the_run_and_scattered_days_are_not_added(close, expected):
    bars, line, cfg = duration_fixture(12)
    bars.loc[12, "close"] = close
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert check["max_consecutive_above_days"] == expected
    assert check["passed"] is (expected < 10)


def test_highs_and_opens_above_the_line_are_not_effective_closes():
    bars, line, cfg = duration_fixture(0)
    bars.loc[7:19, ["high", "open"]] = [14.0, 13.0]
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert check["max_consecutive_above_days"] == 0
    assert check["passed"] is True


def test_duration_uses_trading_rows_including_weekends_between_dates():
    bars, line, cfg = duration_fixture(10)
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    span = pd.Timestamp(check["longest_run_end_date"]) - pd.Timestamp(check["longest_run_start_date"])
    assert span.days > 9
    assert check["max_consecutive_above_days"] == 10
    assert check["passed"] is False


def test_duration_ignores_anchor_breakout_and_subsequent_closes():
    bars, line, cfg = duration_fixture(0)
    bars.loc[:6, "close"] = 11
    bars.loc[20:, "close"] = 11
    bars.loc[11:19, "close"] = 10.3
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert check["max_consecutive_above_days"] == 9
    assert check["longest_run_end_date"] == str(bars.trade_date.iloc[19].date())
    assert check["passed"] is True


@pytest.mark.parametrize("confirmation", [20, 22])
def test_duration_anchor_cannot_be_reset_by_support_or_late_confirmed_peaks(confirmation):
    bars, line, cfg = duration_fixture(10)
    line.support_indices = [18]
    line.touch_indices.append(17)
    line.resistance_confirmed_at[17] = confirmation
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert check["anchor_date"] == str(bars.trade_date.iloc[6].date())
    assert check["max_consecutive_above_days"] == 10
    assert check["passed"] is False


def test_both_filters_are_independent_and_both_failure_reasons_are_retained():
    bars, line, cfg = duration_fixture(10)
    bars.loc[12, "high"] = 15.1
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.prebreakout_check["status"] == "extension_and_duration_exceeded"
    assert "超过 50% 上限" in result.suggestion
    assert "10 日排除阈值" in result.suggestion
    assert result.category == SignalCategory.NONE
    duration_only = _check_prebreakout_extension(
        bars, line, {"index": 20}, replace(cfg, max_prebreakout_extension_pct=None))
    assert duration_only["status"] == "duration_exceeded"
    assert duration_only["extension_passed"] is None
    amplitude_only = _check_prebreakout_extension(
        bars, line, {"index": 20}, replace(cfg, prebreakout_reject_above_days=None))
    assert amplitude_only["status"] == "exceeded"
    assert amplitude_only["consecutive_passed"] is None
    disabled = _check_prebreakout_extension(bars, line, {"index": 20}, replace(
        cfg, max_prebreakout_extension_pct=None, prebreakout_reject_above_days=None))
    assert disabled["status"] == "disabled"
    assert disabled["passed"] is None


def test_duration_without_breakout_stays_pending_and_invalid_closes_fail_closed():
    bars, line, cfg = duration_fixture()
    pending = _check_prebreakout_extension(bars, line, None, cfg)
    assert pending["status"] == "no_breakout"
    assert pending["passed"] is pending["consecutive_passed"] is None
    assert pending["max_consecutive_above_days"] is None
    bars.loc[12, "close"] = float("nan")
    invalid = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert invalid["status"] == "invalid_data"
    assert invalid["passed"] is False


def test_duration_rejection_does_not_fall_back_to_an_older_breakout():
    bars, line, cfg = duration_fixture(0)
    bars.loc[9:18, ["open", "close", "high", "low"]] = [10.28, 10.3, 10.4, 10.2]
    bars.loc[9, "volume"] = 2500
    cfg = replace(cfg, breakout_window=20)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.breakout_date == str(bars.trade_date.iloc[20].date())
    assert result.prebreakout_check["status"] == "duration_exceeded"
    assert result.category == SignalCategory.NONE


def test_buffer_is_configurable_and_zero_is_not_replaced_by_default():
    bars, line, cfg = duration_fixture(10)
    assert _check_prebreakout_extension(bars, line, {"index": 20}, replace(
        cfg, prebreakout_above_buffer_pct=.04))["passed"] is True
    bars.loc[7:16, "close"] = 10.01
    check = _check_prebreakout_extension(bars, line, {"index": 20}, replace(
        cfg, prebreakout_above_buffer_pct=0))
    assert check["effective_close_threshold"] == 10
    assert check["max_consecutive_above_days"] == 10
    assert check["passed"] is False


@pytest.mark.parametrize("high,passes", [(15.00055, True), (15.0007, False)])
def test_filter_uses_unrounded_line_price(high, passes):
    bars, line, cfg = duration_fixture(0)
    line.price = 10.0004
    bars.loc[12, "high"] = high
    check = _check_prebreakout_extension(bars, line, {"index": 20}, cfg)
    assert check["threshold_price"] == 15.0006
    assert check["passed"] is passes


def test_run_detection_matches_an_independent_loop_on_long_intervals():
    rng = np.random.default_rng(8341)
    for size in (15, 45, 300):
        closes = rng.choice([9.9, 10.2, 10.21], size)
        closes[20:31] = 10.21
        bars = pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", periods=size),
                             "close": closes, "high": closes + .1})
        line = KeyLine(10, 3, [0, 2, 4], 4, 5, resistance_confirmed_at={0: 1, 2: 3, 4: 5})
        cfg = StrategyConfig()
        check = _check_prebreakout_extension(bars, line, {"index": size - 1}, cfg)
        longest = current = 0
        for close in closes[5:-1]:
            current = current + 1 if close > 10.2 else 0
            longest = max(longest, current)
        assert check["checked_bars"] == size - 6
        assert check["max_consecutive_above_days"] == longest
        assert check["passed"] is (longest < cfg.prebreakout_reject_above_days)


@pytest.mark.parametrize("run_length,passes", [(14, True), (29, True), (30, False), (31, False)])
def test_thirty_day_default_uses_closes_and_includes_the_rejection_boundary(run_length, passes):
    closes = np.full(45, 9.9)
    closes[6:6 + run_length] = 10.3
    bars = pd.DataFrame({
        "trade_date": pd.bdate_range("2026-01-01", periods=len(closes)),
        "close": closes, "high": closes + .1,
    })
    line = KeyLine(10, 3, [0, 2, 5], 5, 6, resistance_confirmed_at={0: 1, 2: 3, 5: 6})
    cfg = StrategyConfig()
    check = _check_prebreakout_extension(bars, line, {"index": 40}, cfg)
    assert check["reject_above_days"] == 30
    assert check["effective_close_threshold"] == 10.2
    assert check["max_consecutive_above_days"] == run_length
    assert check["passed"] is passes
    assert check["status"] == ("passed" if passes else "duration_exceeded")
    assert check["extension_passed"] is True
    assert cfg.breakout_vol_mult == 1.5
    assert cfg.max_prebreakout_extension_pct == .5
    assert (cfg.min_pullback_days, cfg.max_pullback_days) == (1, 10)
    old = _check_prebreakout_extension(bars, line, {"index": 40}, replace(cfg, prebreakout_reject_above_days=10))
    assert old["passed"] is False


@pytest.mark.parametrize("days", [0, -1, 251, 9.5, True, float("nan")])
def test_direct_duration_config_rejects_invalid_days(days):
    bars, line, cfg = duration_fixture()
    with pytest.raises(ValueError, match="prebreakout_reject_above_days"):
        _check_prebreakout_extension(bars, line, {"index": 20}, replace(cfg, prebreakout_reject_above_days=days))


@pytest.mark.parametrize("buffer", [-.01, 1.01, float("nan"), float("inf")])
def test_direct_duration_config_rejects_invalid_buffer(buffer):
    bars, line, cfg = duration_fixture()
    with pytest.raises(ValueError, match="prebreakout_above_buffer_pct"):
        _check_prebreakout_extension(bars, line, {"index": 20}, replace(cfg, prebreakout_above_buffer_pct=buffer))


def test_duration_check_is_frozen_in_pattern_snapshot_and_past_replay(monkeypatch):
    from app.services import strategy
    from app.services.backtest import evaluate_bars
    bars, line, cfg = duration_fixture(9)
    monkeypatch.setattr(strategy, "_swings", lambda *a, **kw: ([], [], None, {}))
    monkeypatch.setattr(strategy, "_cluster_key_lines", lambda *a, **kw: [line])
    result = analyze_bars("000001", "测试", bars, cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.pattern_matches[0]["prebreakout_check"] == result.prebreakout_check
    future = bars.iloc[-1:].copy()
    future["trade_date"] += pd.Timedelta(days=3)
    future[["open", "high", "low", "close"]] *= 5
    extended = pd.concat([bars, future], ignore_index=True)
    past = analyze_bars("000001", "测试", extended, replace(cfg, signal_offset=1))
    assert past.to_dict() == result.to_dict()
    report = evaluate_bars("000001", "测试", bars, lookback_days=1, config=cfg)
    assert report["prebreakout_reject_above_days"] == 10
    assert report["prebreakout_above_buffer_pct"] == .02
    assert report["max_prebreakout_extension_pct"] == .5
    assert report["signals"][0]["prebreakout_check"] == result.prebreakout_check
    assert report["signals"][0]["pattern_matches"][0]["prebreakout_check"] == result.prebreakout_check
