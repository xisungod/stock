from app.services.validation import (
    block_mean_difference, portfolio_curve, trade_on_calendar, trade_statistics,
)
import pytest


CALENDAR = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


def quote(close, opening=None, flat=False):
    return {"open": close if opening is None else opening, "close": close, "high": close if flat else close + 1,
            "low": close if flat else min(close - 1, opening if opening is not None else close), "volume": 100}


def test_missing_next_market_session_does_not_delay_an_entry_until_resumption():
    prices = {CALENDAR[0]: quote(10), CALENDAR[2]: quote(10), CALENDAR[3]: quote(11)}
    trade = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 2)
    assert trade["entry_status"] == "blocked"
    assert trade["entry_date"] == CALENDAR[1]


def test_delayed_sale_and_pending_exit_have_explicit_statuses():
    prices = {day: quote(10) for day in CALENDAR}
    prices[CALENDAR[2]] = quote(9, flat=True)
    prices[CALENDAR[3]] = quote(11)
    trade = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 2)
    assert trade["exit_date"] == CALENDAR[3]
    assert trade["exit_delay_sessions"] == 1
    assert round(trade["net_return_pct"], 2) == 9.8
    pending = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 20)
    assert pending["entry_status"] == "filled" and pending["net_return_pct"] is None


def test_feasibility_counts_are_not_all_labeled_as_executable():
    prices = {day: quote(10) for day in CALENDAR}
    filled = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 2)
    blocked = {**filled, "entry_status": "blocked", "net_return_pct": None}
    pending = {**filled, "entry_status": "pending", "net_return_pct": None}
    result = trade_statistics([filled, blocked, pending])
    assert result["entry_feasibility_pct"] == 50
    assert result["closed_trade_count"] == 1
    assert result["mean_net_return_pct"] == -.2


def test_drawdown_is_marked_daily_not_computed_from_sorted_trade_outcomes():
    prices = {day: quote(close) for day, close in zip(CALENDAR, [10, 10, 8, 9, 11])}
    trade = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 4, 0)
    result = portfolio_curve([trade], {"one": prices}, CALENDAR, CALENDAR[0], CALENDAR[-1], 1, 0)
    assert round(result["max_drawdown_pct"], 6) == 20
    assert round(result["net_return_pct"], 6) == 10


def test_same_day_closing_proceeds_cannot_fund_opening_entries():
    prices = {day: quote(10) for day in CALENDAR}
    one = trade_on_calendar("one", {"signal_date": CALENDAR[0]}, prices, CALENDAR, 2, 0)
    two = trade_on_calendar("two", {"signal_date": CALENDAR[1]}, prices, CALENDAR, 2, 0)
    result = portfolio_curve([one, two], {"one": prices, "two": prices}, CALENDAR,
                             CALENDAR[0], CALENDAR[-1], 1, 0)
    assert result["executed_entries"] == 1
    assert result["orders_skipped_for_capital_or_existing_position"] == 1


def test_resampling_handles_empty_arms_and_is_deterministic():
    assert block_mean_difference([], [], CALENDAR) is None
    a = [{"entry_date": CALENDAR[0], "net_return_pct": 1}]
    b = [{"entry_date": CALENDAR[0], "net_return_pct": 0}]
    assert block_mean_difference(a, b, CALENDAR, repetitions=50) == [1, 1]


def test_non_session_signal_is_not_moved_to_a_different_market_day():
    prices = {day: quote(10) for day in CALENDAR}
    result = trade_on_calendar("one", {"signal_date": "2026-01-04"}, prices, CALENDAR, 2)
    assert result["entry_status"] == "blocked"
    assert result["entry_reason"] == "signal_is_not_a_calendar_session"


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan")])
def test_invalid_cost_assumptions_are_rejected(bad):
    with pytest.raises(ValueError):
        trade_on_calendar("one", {"signal_date": CALENDAR[0]}, {}, CALENDAR, 5, bad)
