from datetime import datetime, timezone

import pandas as pd
import pytest

from app.services.market_calendar import (
    CalendarUnavailable, last_completed_session, quote_quality, session_dates,
)


def test_chinese_holidays_and_weekend_makeup_workdays_are_not_exchange_sessions():
    assert session_dates("2026-02-13", "2026-02-24") == ["2026-02-13", "2026-02-24"]
    assert session_dates("2026-01-01", "2026-01-05") == ["2026-01-05"]


def test_completed_day_respects_china_time_and_settlement_grace():
    assert last_completed_session(datetime(2026, 9, 10, 15, 4)) == "2026-09-09"
    assert last_completed_session(datetime(2026, 9, 10, 15, 5)) == "2026-09-10"
    assert last_completed_session(datetime(2026, 9, 10, 7, 6, tzinfo=timezone.utc)) == "2026-09-10"
    assert last_completed_session(datetime(2026, 9, 13, 12)) == "2026-09-11"
    assert last_completed_session(datetime(2026, 2, 23, 16)) == "2026-02-13"


def test_unknown_calendar_year_is_explicit_not_a_weekday_guess():
    with pytest.raises(CalendarUnavailable):
        session_dates("2035-01-01", "2035-01-31")
    with pytest.raises(CalendarUnavailable):
        last_completed_session(datetime(2035, 1, 3, 16))


def test_provisional_today_row_cannot_hide_missing_yesterdays_complete_bar():
    frame = pd.DataFrame([dict(trade_date="2026-09-10", open=10, high=11, low=9, close=10, volume=10)])
    quality = quote_quality(frame, "2026-09-10T10:00:00", datetime(2026, 9, 10, 10))
    assert quality["status"] == "stale"
    assert quality["expected_bar_date"] == "2026-09-09"


def test_bad_latest_ohlc_is_not_a_fresh_quote():
    frame = pd.DataFrame([dict(trade_date="2026-09-10", open=10, high=11, low=9, close=20, volume=10)])
    assert quote_quality(frame, "2026-09-10T16:00:00", datetime(2026, 9, 10, 16))["status"] == "stale"
