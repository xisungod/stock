from dataclasses import replace

import pandas as pd
import pytest

from app.services import strategy
from app.services.backtest import evaluate_bars
from app.services.strategy import (
    KeyLine, SignalCategory, SignalResult, _select_best_signal, _serialize_key_line,
)
from test_prebreakout_extension import fixture


@pytest.mark.parametrize("include_chart", [False, True])
def test_anchor_records_actual_high_and_survives_pattern_and_result_serialization(include_chart):
    bars, line, cfg = fixture()
    bars.loc[10, "high"] = 10.147891
    result = _select_best_signal("000001", "test", bars, [line], replace(cfg, include_chart=include_chart))
    expected = {
        "date": str(bars.trade_date.iloc[10].date()),
        "price": 10.147891, "price_basis": "high",
        "confirmed_date": str(bars.trade_date.iloc[11].date()),
        "role": "before_breakout",
    }
    assert result.resistance_anchor == expected
    assert result.to_dict()["resistance_anchor"] == expected
    assert result.to_dict()["pattern_matches"][0]["resistance_anchor"] == expected
    assert result.prebreakout_check["anchor_price"] == 10.147891
    assert result.key_line_price == 10
    assert result.category == SignalCategory.BUY_ALERT
    if include_chart:
        assert result.key_lines[0]["resistance_points"][-1]["price"] == 10.147891


@pytest.mark.parametrize("confirmation", [20, 22, None])
def test_anchor_excludes_support_and_peaks_not_confirmed_before_breakout(confirmation):
    bars, line, cfg = fixture()
    line.support_indices = [18]
    line.touch_indices.append(17)
    if confirmation is not None:
        line.resistance_confirmed_at[17] = confirmation
    result = _select_best_signal("000001", "test", bars, [line], cfg)
    assert result.resistance_anchor["date"] == str(bars.trade_date.iloc[10].date())


def test_disabled_path_filters_still_expose_the_resistance_anchor():
    bars, line, cfg = fixture()
    result = _select_best_signal("000001", "test", bars, [line], replace(
        cfg, max_prebreakout_extension_pct=None, prebreakout_reject_above_days=None))
    assert result.prebreakout_check["status"] == "disabled"
    assert result.resistance_anchor["role"] == "before_breakout"
    assert result.resistance_anchor["date"] == str(bars.trade_date.iloc[10].date())


def test_no_breakout_is_labeled_latest_confirmed_and_missing_confirmation_is_not_invented():
    bars, line, cfg = fixture()
    bars.loc[20, "volume"] = 1000
    result = _select_best_signal("000001", "test", bars, [line], cfg)
    assert result.breakout_date is None
    assert result.resistance_anchor["role"] == "latest_confirmed"
    line.resistance_confirmed_at = {}
    result = _select_best_signal("000001", "test", bars, [line], cfg)
    assert result.resistance_anchor is None


def test_all_historical_point_prices_are_independent_of_the_compact_chart_window():
    bars, _, _ = fixture()
    bars = pd.concat([bars] * 10, ignore_index=True)
    bars["trade_date"] = pd.bdate_range("2025-01-01", periods=len(bars))
    bars.loc[2, "high"] = 10.125678
    line = KeyLine(
        10, 3, [2, 130, 220], 220, 221,
        resistance_confirmed_at={2: 3, 130: 131, 220: 221},
    )
    payload = _serialize_key_line(bars, line, True, 10.1)
    assert len(payload["resistance_points"]) == payload["resistance_point_count"] == 3
    first = payload["resistance_points"][0]
    assert first["date"] < str(bars.tail(140).trade_date.iloc[0].date())
    assert first["price"] == 10.125678
    assert payload["resistance_price_basis"] == first["price_basis"] == "high"


def test_primary_flag_and_historical_replay_retain_the_precision_anchor(monkeypatch):
    bars, line, cfg = fixture()
    precision = _select_best_signal("000001", "test", bars, [line], cfg)
    flag = SignalResult(
        "000001", "test", SignalCategory.BUY_ALERT, 100, "flag", None, 0,
        precision.signal_date, pattern_type="flag_breakout",
        flag_pattern={"id": "test", "direction": "ascending", "status": "breakout"},
    )
    monkeypatch.setattr(strategy, "_run_precision_pipeline", lambda *args: ("ok", "", precision))
    monkeypatch.setattr(strategy, "_run_flag_pipeline", lambda *args: ("ok", "", flag))
    combined = strategy.analyze_bars("000001", "test", bars, replace(cfg, strategy_mode="combined"))
    assert combined.pattern_type == "flag_breakout"
    assert combined.pattern_matches[0]["resistance_anchor"] == precision.resistance_anchor
    assert combined.resistance_anchor is None

    replay = evaluate_bars("000001", "test", bars, lookback_days=1, config=cfg)
    assert len(replay["signals"]) == 1
    assert replay["signals"][0]["resistance_anchor"] == precision.resistance_anchor
    assert replay["signals"][0]["pattern_matches"][0]["resistance_anchor"] == precision.resistance_anchor
