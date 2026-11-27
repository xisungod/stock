from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.services.backtest import (
    EXECUTION_MODEL, compute_forward_metrics, compute_signal_features,
    compute_trade_metrics, dedupe_signal_indices, evaluate_bars, summarize_signals,
)
from app.services.strategy import SignalCategory, SignalResult, StrategyConfig


def _bars(size: int = 190) -> pd.DataFrame:
    closes = [10 + idx * 0.01 for idx in range(size)]
    return pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 1) + timedelta(days=idx) for idx in range(size)],
            "open": [value * 0.995 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [1000] * size,
        }
    )


def _synthetic_calendar(bars):
    # These fixtures deliberately include weekends; their clock is explicitly synthetic.
    return pd.to_datetime(bars.trade_date).dt.strftime("%Y-%m-%d").tolist()


def test_dedupe_signal_indices_keeps_first_signal_in_each_cluster() -> None:
    assert dedupe_signal_indices([100, 101, 105, 121, 140], min_gap=20) == [100, 121]


def test_event_deduplication_is_not_a_fixed_twenty_bar_cooldown():
    from app.services.backtest import dedupe_event_signals
    from types import SimpleNamespace
    def signal(key):
        return SimpleNamespace(signal_event_id=key, flag_pattern=None, pattern_type="precision_pullback")
    assert dedupe_event_signals({1: signal("one"), 5: signal("two"), 30: signal("one")}) == [1, 5]


def test_entry_feasibility_excludes_pending_entries_and_counts_blocked_ones():
    signals = [
        {"execution_model": EXECUTION_MODEL, "entry_status": "filled", "net_return_5": 2},
        {"execution_model": EXECUTION_MODEL, "entry_status": "blocked"},
        {"execution_model": EXECUTION_MODEL, "entry_status": "pending"},
    ]
    summary = summarize_signals(signals)
    assert summary["entry_feasibility_sample_count"] == 2
    assert summary["entry_feasibility_rate"] == 50
    assert summary["net_t5_profit_factor"] is None
    assert summary["net_t5_profit_factor_status"] == "no_losses"


def test_forward_metrics_report_returns_target_and_stop() -> None:
    bars = _bars(40)
    bars.loc[12, "high"] = bars.loc[10, "close"] * 1.06
    bars.loc[14, "low"] = bars.loc[10, "close"] * 0.94

    metrics = compute_forward_metrics(bars, 10, calendar=_synthetic_calendar(bars))

    assert metrics["target_5pct_hit"] is True
    assert metrics["stop_5pct_hit"] is True
    assert metrics["forward_return_5"] is not None
    assert metrics["forward_return_5_from_next_open"] is not None
    assert metrics["mfe_5"] is not None
    assert metrics["mae_5"] is not None


def test_signal_features_use_only_signal_day_and_prior_bars() -> None:
    bars = _bars(80)
    result = SignalResult(
        stock_code="000001",
        stock_name="测试",
        category=SignalCategory.BUY_ALERT,
        score=90,
        suggestion="测试信号",
        key_line_price=10,
        line_strength=4,
        signal_date=str(bars.iloc[69]["trade_date"]),
        line_formation_date=str(bars.iloc[30]["trade_date"]),
        breakout_date=str(bars.iloc[66]["trade_date"]),
        pullback_date=str(bars.iloc[69]["trade_date"]),
        distance_to_line_pct=1.2,
        breakout_volume_ratio=1.8,
        pullback_volume_ratio=0.5,
    )

    features = compute_signal_features(bars, 69, result)

    assert features["bars_since_breakout"] == 3
    assert features["line_age_bars"] == 39
    assert features["distance_to_line_pct"] == 1.2
    assert features["prior_return_20_pct"] is not None


def test_evaluate_bars_is_point_in_time_and_deduplicates(monkeypatch) -> None:
    seen_lengths: list[int] = []
    seen_last_dates: list[str] = []

    def fake_analyze(code, name, visible, config):
        seen_lengths.append(len(visible))
        signal_date = pd.Timestamp(visible.iloc[-1]["trade_date"])
        seen_last_dates.append(signal_date.strftime("%Y-%m-%d"))
        signal_number = (signal_date.date() - date(2025, 1, 1)).days + 1
        if signal_number in {130, 131, 132, 155}:
            return SignalResult(
                stock_code=code,
                stock_name=name,
                category=SignalCategory.BUY_ALERT,
                score=80,
                suggestion="测试信号",
                key_line_price=10,
                line_strength=3,
                signal_date=str(visible.iloc[-1]["trade_date"]),
            )
        return None

    monkeypatch.setattr("app.services.backtest._analyze_normalized_bars", fake_analyze)
    bars = _bars()
    report = evaluate_bars(
        "000001",
        "测试",
        bars,
        lookback_days=80,
        min_signal_gap=20,
        config=StrategyConfig(signal_offset=0, min_history_bars=120, lookback_days=125),
    )

    assert max(seen_lengths) == 125
    assert seen_last_dates[-1] == "2025-07-09"
    assert [item["signal_index"] for item in report["signals"]] == [129, 154]
    assert report["summary"]["signal_count"] == 2


def test_zero_lookback_evaluates_every_eligible_historical_day(monkeypatch) -> None:
    seen_dates: list[str] = []

    def fake_analyze(code, name, visible, config):
        seen_dates.append(pd.Timestamp(visible.iloc[-1]["trade_date"]).strftime("%Y-%m-%d"))
        return None

    monkeypatch.setattr("app.services.backtest._analyze_normalized_bars", fake_analyze)
    bars = _bars()

    report = evaluate_bars(
        "000001",
        "测试",
        bars,
        lookback_days=0,
        config=StrategyConfig(signal_offset=0, min_history_bars=120, lookback_days=125),
    )

    assert len(seen_dates) == len(bars) - 119
    assert seen_dates[0] == "2025-04-30"
    assert seen_dates[-1] == "2025-07-09"
    assert report["history_scope"] == "all"


def test_net_metrics_use_next_open_and_do_not_overwrite_legacy_returns() -> None:
    bars = _bars(30)
    bars.loc[0, "close"] = 10
    bars.loc[1, "open"] = 10.8
    bars.loc[5, "close"] = 10.5
    metrics = compute_forward_metrics(bars, 0, calendar=_synthetic_calendar(bars))
    assert metrics["forward_return_5"] == 5
    assert metrics["net_return_5"] == pytest.approx(-2.977778)
    summary = summarize_signals([metrics])
    assert summary["net_t5_positive_rate"] == 0
    assert summary["t5_positive_rate"] == 100
    assert summary["execution_model"] == EXECUTION_MODEL


def test_cost_can_turn_a_gross_winner_into_a_net_loser() -> None:
    bars = _bars(30)
    bars.loc[1, "open"] = 10
    bars.loc[5, "close"] = 10.01
    metrics = compute_trade_metrics(bars, 0, calendar=_synthetic_calendar(bars))
    assert metrics["gross_return_5"] == pytest.approx(.1)
    assert metrics["net_return_5"] == pytest.approx(-.1)
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            compute_trade_metrics(bars, 0, roundtrip_cost_bps=invalid, calendar=_synthetic_calendar(bars))


def test_unbuyable_bar_is_excluded_and_counted_not_assumed_filled() -> None:
    bars = _bars(30)
    bars.loc[1, ["open", "high", "low", "close"]] = 11
    metrics = compute_forward_metrics(bars, 0, calendar=_synthetic_calendar(bars))
    assert metrics["entry_status"] == "blocked"
    assert metrics["net_return_5"] is None
    summary = summarize_signals([metrics])
    assert summary["blocked_entry_count"] == 1
    assert summary["net_t5_sample_count"] == 0
    assert summary["net_t5_positive_rate"] is None


def test_unsellable_bar_delays_exit_and_includes_later_losses() -> None:
    bars = _bars(30)
    bars.loc[1, "open"] = 10
    bars.loc[5, ["open", "high", "low", "close"]] = 9
    bars.loc[6, "close"] = 8
    metrics = compute_trade_metrics(bars, 0, calendar=_synthetic_calendar(bars))
    assert metrics["exit_delay_5"] == 1
    assert metrics["net_return_5"] == pytest.approx(-20.2)
    assert metrics["exit_date_5"] == str(bars.loc[6, "trade_date"])


def test_t_plus_one_never_exits_on_the_entry_day() -> None:
    bars = _bars(10)
    metrics = compute_trade_metrics(bars, 0, windows=(1,), calendar=_synthetic_calendar(bars))
    assert metrics["exit_date_1"] == str(bars.loc[2, "trade_date"])


def test_unfinished_trades_and_legacy_reports_are_not_net_winners() -> None:
    bars = _bars(3)
    pending = compute_forward_metrics(bars, 2, calendar=_synthetic_calendar(bars))
    assert pending["entry_status"] == "pending"
    assert summarize_signals([pending])["net_t5_sample_count"] == 0
    legacy = summarize_signals([{"forward_return_5": 10}])
    assert "net_t5_positive_rate" not in legacy
    assert "execution_model" not in legacy


def test_candidate_filters_never_read_future_bars() -> None:
    bars = _bars(100)
    result = SignalResult("000001", "测试", SignalCategory.BUY_ALERT, 90, "测试", 10, 3, str(bars.loc[69, "trade_date"]),
                          breakout_date=str(bars.loc[65, "trade_date"]))
    before = compute_signal_features(bars, 69, result)
    bars.loc[70:, ["open", "close", "high", "low", "volume"]] *= 1000
    after = compute_signal_features(bars, 69, result)
    assert before == after
    assert before["pullback_vs_prebreakout_volume"] == 1
    assert isinstance(before["macd_hist_rising"], bool)


def test_app_and_report_use_the_same_next_market_session_even_when_stock_bars_are_missing():
    from app.services.validation import trade_on_calendar
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    rows = [
        dict(trade_date=day, open=10., high=10.5, low=9.5, close=10., volume=100.)
        for day in [calendar[0], calendar[2], calendar[3]]
    ]
    app = compute_trade_metrics(pd.DataFrame(rows), 0, windows=(2,), calendar=calendar)
    report = trade_on_calendar("000001", {"signal_date": calendar[0]},
                               {row["trade_date"]: row for row in rows}, calendar, 2)
    assert app["entry_status"] == report["entry_status"] == "blocked"
    assert app["entry_date"] == report["entry_date"] == "2026-01-06"
    assert app["net_return_2"] is None


def test_default_execution_calendar_skips_chinese_exchange_holidays():
    rows = [
        dict(trade_date=day, open=10., high=10.5, low=9.5, close=10., volume=100.)
        for day in ["2026-02-13", "2026-02-24", "2026-02-25", "2026-02-26"]
    ]
    metrics = compute_trade_metrics(pd.DataFrame(rows), 0, windows=(2,))
    assert metrics["entry_date"] == "2026-02-24"
    assert metrics["exit_date_2"] == "2026-02-25"


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_app_and_report_agree_on_holiday_gaps_and_delayed_exits(horizon):
    from app.services.validation import trade_on_calendar
    calendar = ["2026-02-13", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-03-02"]
    rows = [
        dict(trade_date=day, open=10., high=11., low=9., close=10.2, volume=100.)
        for day in calendar
    ]
    rows[2].update(open=9, high=9, low=9, close=9)
    prices = {row["trade_date"]: row for row in rows}
    actual = compute_trade_metrics(pd.DataFrame(rows), 0, windows=(horizon,), calendar=calendar)
    expected = trade_on_calendar("000001", {"signal_date": calendar[0]}, prices, calendar, horizon)
    assert actual["entry_status"] == expected["entry_status"]
    assert actual[f"exit_date_{horizon}"] == expected["exit_date"]
    assert actual[f"exit_delay_{horizon}"] == expected.get("exit_delay_sessions")
    assert actual[f"net_return_{horizon}"] == pytest.approx(expected["net_return_pct"])


def test_known_evaluation_date_marks_missing_next_day_as_blocked_not_pending():
    frame = pd.DataFrame([dict(trade_date="2026-01-05", open=10., high=11., low=9., close=10., volume=100.)])
    pending = compute_trade_metrics(frame, 0)
    blocked = compute_trade_metrics(frame, 0, as_of="2026-01-06")
    assert pending["entry_status"] == "pending"
    assert blocked["entry_status"] == "blocked"
    assert blocked["entry_date"] == "2026-01-06"
    explicit = compute_trade_metrics(frame, 0, calendar=["2026-01-05", "2026-01-06"])
    assert explicit["entry_status"] == "blocked"
    assert explicit["entry_date"] == "2026-01-06"
