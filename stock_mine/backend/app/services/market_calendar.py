"""Bounded mainland exchange sessions and quote readiness, always in China time."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHINA = ZoneInfo("Asia/Shanghai")
SETTLED_AFTER = time(15, 5)


class CalendarUnavailable(ValueError):
    pass


def china_now() -> datetime:
    return datetime.now(CHINA)


def china_time(value: datetime) -> datetime:
    return value.replace(tzinfo=CHINA) if value.tzinfo is None else value.astimezone(CHINA)


@lru_cache(maxsize=1)
def exchange_calendar():
    try:
        import exchange_calendars as calendars
        from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar
        return calendars.get_calendar(
            "XSHG", start=XSHGExchangeCalendar.bound_min(), end=XSHGExchangeCalendar.bound_max(),
        )
    except (ImportError, ValueError) as exc:
        raise CalendarUnavailable(f"交易日历不可用：{exc}") from exc


def calendar_info() -> dict:
    import exchange_calendars
    calendar = exchange_calendar()
    return {
        "source": f"XSHG / exchange_calendars {exchange_calendars.__version__}",
        "start": calendar.first_session.strftime("%Y-%m-%d"),
        "end": calendar.last_session.strftime("%Y-%m-%d"),
    }


def _calendar_date(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(CHINA).tz_localize(None)
    stamp = stamp.normalize()
    calendar = exchange_calendar()
    if pd.isna(stamp) or stamp < calendar.first_session or stamp > calendar.last_session:
        raise CalendarUnavailable(f"交易日历未覆盖 {value}，不能按普通工作日推算")
    return stamp


def session_dates(start, end) -> list[str]:
    start, end = _calendar_date(start), _calendar_date(end)
    if end < start:
        return []
    return exchange_calendar().sessions_in_range(start, end).strftime("%Y-%m-%d").tolist()


def last_completed_session(now: datetime | None = None) -> str:
    clock = china_time(now or china_now())
    day = _calendar_date(clock.date())
    calendar = exchange_calendar()
    if clock.time().replace(tzinfo=None) < SETTLED_AFTER:
        day = _calendar_date(day - pd.Timedelta(days=1))
    return calendar.date_to_session(day, direction="previous").strftime("%Y-%m-%d")


def quote_quality(frame: pd.DataFrame, checked_at: str, now: datetime | None = None,
                  error: str | None = None) -> dict:
    clock = china_time(now or china_now())
    dates = pd.to_datetime(frame.get("trade_date", pd.Series(dtype=str)), errors="coerce")
    actual = dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None
    quality = {"status": "stale", "checked_at": checked_at or None, "last_bar_date": actual,
               "expected_bar_date": None, "error": error}
    try:
        quality["calendar_source"] = calendar_info()["source"]
        expected = last_completed_session(clock)
        quality["expected_bar_date"] = expected
        stamp = china_time(datetime.fromisoformat(checked_at))
        numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        valid = (
            np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
            & (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (numeric.volume >= 0)
            & (numeric.high >= numeric[["open", "close", "low"]].max(axis=1))
            & (numeric.low <= numeric[["open", "close", "high"]].min(axis=1))
        )
        valid_dates = dates[valid].dropna().dt.strftime("%Y-%m-%d")
        finalization = datetime.combine(pd.Timestamp(expected).date(), SETTLED_AFTER, CHINA)
        if error:
            return quality
        if stamp > clock or actual and actual > clock.date().isoformat():
            quality["error"] = "行情或检查时间晚于当前中国时间"
        elif expected not in set(valid_dates):
            quality["error"] = f"缺少 {expected} 的有效完整日线，可能停牌或数据源尚未更新"
        elif stamp < finalization:
            quality["error"] = f"{expected} 日线只在收盘前检查过，尚未重新确认"
        else:
            quality["status"] = "fresh"
        quality["provisional_last_bar"] = bool(actual and actual > expected)
    except (CalendarUnavailable, ValueError, KeyError, TypeError, ImportError) as exc:
        quality["error"] = error or f"无法核验行情新鲜度：{exc}"
    if frame.empty:
        quality["status"] = "unavailable"
    return quality


def can_reuse_quotes(frame: pd.DataFrame, checked_at: str, now: datetime,
                     recheck: timedelta) -> bool:
    clock = china_time(now)
    try:
        stamp = china_time(datetime.fromisoformat(checked_at))
        if not timedelta(0) <= clock - stamp:
            return False
        quality = quote_quality(frame, checked_at, clock)
        day = _calendar_date(clock.date())
        intraday = exchange_calendar().is_session(day) and clock.time().replace(tzinfo=None) < SETTLED_AFTER
        if quality["status"] == "fresh" and not intraday:
            return True
        return clock - stamp < recheck
    except (CalendarUnavailable, ValueError, TypeError):
        return False
