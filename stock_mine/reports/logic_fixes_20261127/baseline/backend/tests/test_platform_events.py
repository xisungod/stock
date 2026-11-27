from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services.platform_events import (
    LocalTurnDetector, PlatformEngine, ValidationEvent, iter_platform_results,
)
from app.services.strategy import StrategyConfig, SignalCategory, analyze_bars


def config(**changes):
    return replace(StrategyConfig(
        precision_engine="platform_events", signal_offset=0, min_history_bars=10,
        ma_period=5, ma_slope_days=2, max_line_distance_pct=0,
        max_prebreakout_extension_pct=None, prebreakout_reject_above_days=None,
    ), **changes)


def bars(closes):
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2020-01-01", periods=len(values)),
        "open": values - .02, "high": values + .08, "low": values - .08,
        "close": values, "volume": np.full(len(values), 1000.0),
    })


def event(kind, pivot, confirmed, price, start=None, origins=("local_turn",)):
    return ValidationEvent(kind, pivot, confirmed, pivot - 1 if start is None else start,
                           price, .2, price if kind == "resistance" else price + .2,
                           price if kind == "support" else price - .2, origins)


def test_local_turn_uses_a_frozen_atr_and_close_not_the_body_high():
    detector = LocalTurnDetector("resistance")
    cfg = config()
    detector.feed(0, 6, 5.8, 5.9, .2, cfg)
    detector.feed(1, 6.67, 5.87, 6.67, .325, cfg)
    assert detector.feed(2, 6.51, 6.21, 6.35, .9, cfg) is None
    found = detector.feed(3, 6.19, 6.0, 6.11, 1.5, cfg)
    assert found.price == 6.67 and found.pivot == 1 and found.confirmed == 3
    assert found.atr == .325


def test_no_same_candle_confirmation_and_no_repeated_points_while_grinding():
    detector = LocalTurnDetector("resistance")
    cfg = config()
    detector.feed(0, 10, 9, 9.5, .3, cfg)
    assert detector.feed(1, 11, 9, 10, .3, cfg) is None
    first = detector.feed(2, 10.7, 10.1, 10.2, .3, cfg)
    assert first and first.price == 11
    rest = [detector.feed(i, 10.25, 10.1, 10.2, .3, cfg) for i in range(3, 15)]
    assert all(item is None for item in rest)


def test_confirmation_is_immutable_when_later_prices_make_a_larger_peak():
    detector = LocalTurnDetector("resistance")
    cfg = config()
    result = []
    for i, (high, low, close) in enumerate([(9, 8, 8.8), (10, 8.9, 9.9), (9.7, 9.2, 9.4), (11, 9.3, 10.8), (12, 10.6, 11.9), (11.5, 11, 11.1)]):
        found = detector.feed(i, high, low, close, .3, cfg)
        if found:
            result.append(found)
    assert result[0].price == 10
    assert result[-1].price == 12
    assert result[0].confirmed < result[-1].confirmed


def test_wick_prominence_records_a_new_candidate_but_only_a_later_close_confirms_it():
    detector = LocalTurnDetector("resistance")
    cfg = config()
    assert detector.feed(0, 9.2, 9, 9.1, .3, cfg) is None
    assert detector.feed(1, 10, 9.1, 9.9, .3, cfg) is None
    first = detector.feed(2, 9.8, 9.4, 9.5, .3, cfg)
    assert first.pivot == 1
    assert detector.feed(3, 10.2, 9.5, 9.6, .3, cfg) is None
    second = detector.feed(4, 9.8, 9.2, 9.3, .3, cfg)
    assert second.pivot == 3 and second.price == 10.2 and second.confirmed == 4
    assert first.price == 10


def test_support_and_resistance_both_form_a_candidate_without_a_preexisting_line():
    frame = bars([9.8] * 20)
    frame.loc[4, ["open", "high", "low", "close"]] = [10.8, 10.9, 10.7, 10.8]
    engine = PlatformEngine("000001", "test", frame, config())
    engine._register(event("resistance", 2, 4, 10.1), 4)
    engine._register(event("support", 5, 7, 9.9), 7)
    line = engine.platforms[0].snapshot
    assert line["platform_state"] == "candidate"
    assert line["point_count"] == 2
    assert line["resistance_point_count"] == line["support_point_count"] == 1
    assert line["formation_date"] == engine.dates[7]


def test_same_turn_and_band_test_share_one_event_and_snapshot_is_frozen():
    frame = bars([9.8] * 20)
    frame.loc[4, ["open", "high", "low", "close"]] = [10.8, 10.9, 10.7, 10.8]
    engine = PlatformEngine("000001", "test", frame, config())
    engine._register(event("resistance", 2, 4, 10), 4)
    engine._register(event("support", 5, 7, 9.9), 7)
    first = engine.platforms[0].snapshot
    engine._register(event("resistance", 2, 8, 10, origins=("resistance_test",)), 8)
    second = engine.platforms[0].snapshot
    assert first["point_count"] == second["point_count"] == 2
    assert first["resistance_points"][0]["origins"] == ["local_turn"]
    assert second["resistance_points"][0]["origins"] == ["local_turn", "resistance_test"]
    assert first["version_id"] != second["version_id"]


def test_all_versions_have_bounded_contributions_and_do_not_backdate_birth():
    frame = bars([6.7] * 30)
    frame.loc[4, ["open", "high", "low", "close"]] = [7.4, 7.5, 7.3, 7.4]
    frame.loc[9, ["open", "high", "low", "close"]] = [6, 6.1, 5.9, 6]
    engine = PlatformEngine("000001", "test", frame, config())
    for item in [event("resistance", 1, 5, 6.67), event("support", 6, 10, 6.81)]:
        engine._register(item, item.confirmed)
    third = replace(event("resistance", 11, 15, 6.67, origins=("resistance_test",)), episode_id="test:third")
    engine._register(third, third.confirmed, owner=engine.platforms[0])
    for platform in engine.platforms:
        for version in platform.versions:
            points = version["resistance_points"] + version["support_points"]
            assert all(abs(point["price"] / version["price"] - 1) <= .02 + 1e-9 for point in points)
            assert all(point["confirmed_date"] <= version["version_created_at"] for point in points)
    assert engine.platforms[0].snapshot["confirmation_date"] == engine.dates[15]


def seeded_engine(closes):
    frame = bars(closes)
    frame.loc[2, ["open", "high", "low", "close"]] = [10.6, 10.7, 10.5, 10.6]
    frame.loc[4, ["open", "high", "low", "close"]] = [9.4, 9.5, 9.3, 9.4]
    cfg = config()
    engine = PlatformEngine("000001", "test", frame, cfg)
    for item in [event("resistance", 1, 2, 10), event("support", 3, 4, 10)]:
        engine._register(item, item.confirmed)
    third = replace(event("resistance", 5, 6, 10, origins=("resistance_test",)), episode_id="seed:third")
    engine._register(third, third.confirmed, owner=engine.platforms[0])
    engine.index = 7
    return engine


def test_plain_cross_enters_watch_without_pullback_and_does_not_use_today_evidence():
    engine = seeded_engine([9.8] * 8 + [10.02, 10.2, 10.8, 10.02, 10.8])
    engine.feed(8)
    result = engine.result(8)
    assert result.category == SignalCategory.WATCH
    assert result.platform_state == "breakout_observation"
    assert result.breakout_date is None
    engine.feed(9)
    frozen = engine.platforms[0].observation.snapshot
    assert frozen["point_count"] == 3
    assert all(point["confirmed_index"] < 9 for point in frozen["resistance_points"] + frozen["support_points"])
    qualified = replace(event("support", 11, 12, 10.02, origins=("support_test",)), episode_id="test:fourth")
    engine._register(qualified, 12, owner=engine.platforms[0])
    assert engine.platforms[0].observation.snapshot["point_count"] == 3
    assert frozen["version_id"] != engine.platforms[0].snapshot["version_id"]


def test_pullback_alert_has_a_stable_breakout_event_id():
    engine = seeded_engine([9.8] * 8 + [10.2, 10.07, 10.05])
    engine.volume = engine.volume.copy()
    engine.volume[8] = 2500
    engine.frame.loc[8, "volume"] = 2500
    engine.feed(8)
    assert engine.result(8).category == SignalCategory.WATCH
    engine.feed(9)
    first = engine.result(9)
    engine.feed(10)
    second = engine.result(10)
    assert first.category == second.category == SignalCategory.BUY_ALERT
    assert first.signal_event_id == second.signal_event_id
    assert first.line_snapshot == second.line_snapshot


def test_breakdown_invalidates_the_attempt_without_erasing_price_history():
    engine = seeded_engine([9.8] * 8 + [10.2, 9.6, 10.2])
    engine.feed(8)
    platform = engine.platforms[0]
    original = platform.observation.snapshot
    engine.feed(9)
    assert platform.observation is None and platform.invalidated == 9
    assert len(platform.events) == 3
    assert original["point_count"] == 3
    assert engine.result(9).platform_state == "invalid"
    engine._register(event("resistance", 8, 10, 10.2), 10)
    assert len(platform.events) == 3
    assert platform.observation is None


def test_band_probe_keeps_the_whole_extreme_instead_of_picking_a_closer_candle():
    frame = bars([9.5] * 12)
    frame.loc[2, ["open", "high", "low", "close"]] = [10.6, 10.7, 10.5, 10.6]
    engine = PlatformEngine("000001", "test", frame, config())
    for item in [event("resistance", 1, 2, 10), event("support", 3, 4, 10)]:
        engine._register(item, item.confirmed)
    platform = engine.platforms[0]
    engine.high = engine.high.copy()
    engine.low = engine.low.copy()
    engine.close = engine.close.copy()
    engine.high[5], engine.low[5], engine.close[5] = 10.7, 9.8, 10.0
    engine.high[6], engine.low[6], engine.close[6] = 10.05, 9.0, 9.2
    platform.armed_side = "below"
    assert engine._probe(platform, 5) is None
    assert engine._probe(platform, 6) is None
    assert platform.rejected[-1]["reason"] == "event_extreme_outside_band"
    assert platform.rejected[-1]["max_high"] == 10.7
    assert platform.snapshot["point_count"] == 2


def test_local_oscillations_inside_one_band_are_not_independent_validation_events():
    engine = PlatformEngine("000001", "test", bars([10.0] * 30), config())
    engine._register(event("resistance", 1, 3, 10.1), 3)
    engine._register(event("resistance", 5, 7, 10.12), 7)
    engine._register(event("support", 10, 12, 9.9), 12)
    assert len(engine.platforms) == 1
    assert len(engine.platforms[0].events) == 1
    assert engine.platforms[0].snapshot is None
    assert engine.platforms[0].rejected[-1]["reason"] == "same_trial_without_price_departure"
    engine.low = engine.low.copy()
    engine.low[14] = 9.3
    engine._register(event("resistance", 16, 18, 10.1), 18)
    assert engine.platforms[0].snapshot["point_count"] == 2


def test_excursion_uses_a_real_intervening_extreme_not_the_candidate_candle_itself():
    engine = PlatformEngine("000001", "test", bars([10.0] * 20), config())
    engine._register(event("resistance", 1, 3, 10.1), 3)
    engine.low = engine.low.copy()
    engine.low[5] = 9
    engine._register(event("resistance", 5, 7, 10.1), 7)
    assert len(engine.platforms[0].events) == 1
    engine._register(event("resistance", 8, 10, 10.1), 10)
    assert len(engine.platforms[0].events) == 2


def test_zero_volume_candle_cannot_confirm_an_event_or_trigger_a_buy_alert():
    engine = seeded_engine([9.8] * 8 + [10.2, 10.07])
    engine.volume = engine.volume.copy()
    engine.volume[8], engine.volume[9] = 2500, 0
    engine.frame.loc[8, "volume"], engine.frame.loc[9, "volume"] = 2500, 0
    engine.feed(8)
    count = len(engine.events)
    engine.feed(9)
    assert len(engine.events) == count
    assert engine.result(9).category == SignalCategory.NONE
    assert engine.result(9).platform_state == "no_trading"


def test_single_day_replay_matches_stream_and_future_bars_cannot_change_the_past():
    rng = np.random.default_rng(22)
    frame = bars(10 * np.exp(np.cumsum(rng.normal(0, .012, 140))))
    cfg = config()
    streamed = dict(iter_platform_results("000001", "test", frame, cfg))
    for index in [40, 70, 100, 139]:
        direct = analyze_bars("000001", "test", frame.iloc[:index + 1], cfg)
        result = streamed[index]
        assert (direct.to_dict() if direct else None) == (result.to_dict() if result else None)
        offset = analyze_bars("000001", "test", frame, replace(cfg, signal_offset=len(frame) - 1 - index))
        assert (direct.to_dict() if direct else None) == (offset.to_dict() if offset else None)


def overshooting_trial_frame():
    frame = bars([9.0] * 110 + [9.0, 9.5, 9.92, 9.5, 9.0] * 2
                 + [10.0, 10.0, 9.85, 9.45, 9.0, 10.2, 10.05])
    for index, values in {
        120: [10, 10.7, 9.75, 10], 121: [10, 10.1, 9.7, 10],
        122: [9.8, 10.18, 9.45, 9.85], 123: [9.8, 10, 9.4, 9.45],
        124: [9.1, 9.2, 8.8, 9],
    }.items():
        frame.loc[index, ["open", "high", "low", "close"]] = values
    frame.loc[125, "volume"] = 2500
    return frame


@pytest.mark.parametrize("local_high", [10.18, 10.21])
def test_local_turn_cannot_make_a_third_point_inside_an_overshooting_trial(local_high):
    frame = overshooting_trial_frame()
    frame.loc[122, "high"] = local_high
    cfg = StrategyConfig(precision_engine="platform_events", signal_offset=0)
    engine = PlatformEngine("000001", "test", frame, cfg)
    for index in range(len(frame)):
        engine.feed(index)
    level = next(item for item in engine.platforms if item.band.price == 10)
    assert any(item.pivot == 122 for item in engine.events.values()), "Keep the raw local turn"
    assert len(level.events) == 2, "The 10.18 local turn belongs to the rejected 10.70 trial"
    assert level.confirmed is None
    result = analyze_bars("000001", "test", frame, cfg)
    assert result is None or result.category != SignalCategory.BUY_ALERT


def test_late_local_confirmation_cannot_escape_a_completed_rejected_trial():
    frame = overshooting_trial_frame()
    engine = PlatformEngine("000001", "test", frame, config())
    for index in range(125):
        engine.feed(index)
    level = next(item for item in engine.platforms if item.band.price == 10)
    engine._register(event("resistance", 121, 125, 10.1), 125)
    assert len(level.events) == 2
    assert level.rejected


def test_valid_trial_is_not_credited_until_its_complete_reaction_is_confirmed():
    frame = bars([9.0] * 110 + [9.0, 9.5, 9.92, 9.5, 9.0] * 2
                 + [9.96, 9.7, 9.4, 9.0])
    cfg = StrategyConfig(precision_engine="platform_events", signal_offset=0)
    engine = PlatformEngine("000001", "test", frame, cfg)
    for index in range(122):
        engine.feed(index)
    level = next(item for item in engine.platforms if item.band.price == 10)
    assert level.probe is not None
    assert len(level.events) == 2
    for index in range(122, len(frame)):
        engine.feed(index)
    assert len(level.events) == 3
    contribution = max(level.events.values(), key=lambda item: item.pivot)
    assert contribution.pivot == 120
    assert contribution.confirmed >= 122
    assert contribution.episode_id
    assert len({item.episode_id for item in level.events.values() if item.episode_id}) == 1


def test_established_candidate_requires_a_complete_trial_even_without_a_touching_candle():
    frame = bars([10.0] * 20)
    frame.loc[4, ["open", "high", "low", "close"]] = [10.8, 10.9, 10.7, 10.8]
    frame.loc[9, ["open", "high", "low", "close"]] = [9, 9.1, 8.9, 9]
    engine = PlatformEngine("000001", "test", frame, config())
    engine._register(event("resistance", 2, 4, 10.1), 4)
    engine._register(event("support", 5, 7, 9.9), 7)
    assert engine.platforms[0].snapshot["point_count"] == 2
    engine._register(event("resistance", 12, 14, 10.1), 14)
    assert engine.platforms[0].snapshot["point_count"] == 2
    assert "resistance:12" in engine.events
