from datetime import datetime, timezone
import json
import sqlite3
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.db import Database
from app.main import create_app
from app.services.alert_monitor import AlertMonitor, MonitorFilters, MonitorStore, UnadjustedQuotes, valid_closes
from app.services.market_calendar import CHINA


def clock(day="2026-09-14", hour=16):
    return datetime.fromisoformat(f"{day}T{hour:02}:00:00").replace(tzinfo=CHINA)


def save_scan(db, selected="2026-09-10", codes=("000001",), provider="akshare", category="buy_alert"):
    scan_id = db.create_scan_run(provider, {"provider": provider})
    payloads = [{
        "stock_code": code, "stock_name": f"股票{code}", "category": category,
        "score": 80, "signal_date": "2026-09-08", "current_price": 999,
        "industry": "银行" if code == "000001" else "制造",
    } for code in codes]
    db.complete_scan_run(scan_id, provider, {"provider": provider}, payloads)
    with db.connect() as conn:
        conn.execute("UPDATE scan_runs SET started_at = ?, finished_at = ? WHERE id = ?",
                     (f"{selected}T08:00:00", f"{selected}T08:05:00", scan_id))
    return scan_id


def frame(values, basis="unadjusted"):
    result = pd.DataFrame([{
        "trade_date": day, "open": price, "close": price,
        "high": price + .1, "low": price - .1, "volume": 100,
    } for day, price in values.items()])
    result.attrs.update(price_basis=basis, source="test-raw")
    return result


class Quotes:
    def __init__(self, values=None):
        self.values = values or {"2026-09-10": 10, "2026-09-11": 10.5, "2026-09-14": 11}
        self.calls = []

    def fetch(self, code, start, end):
        self.calls.append((code, start, end))
        return frame(self.values)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "monitor.db")


def test_imports_saved_real_buy_alerts_without_modifying_scans(db):
    first = save_scan(db)
    save_scan(db)
    save_scan(db, category="watch", codes=("000002",))
    save_scan(db, provider="sample", codes=("000003",))
    save_scan(db, selected="2026-09-11")
    before = db.get_scan_results(first)
    store = MonitorStore(db)
    assert store.sync_alerts(clock()) == 2
    assert store.sync_alerts(clock()) == 0
    rows = store.entries()
    assert len(rows) == 2
    older = next(row for row in rows if row["selected_date"] == "2026-09-10")
    assert older["first_scan_id"] == first and older["scan_count"] == 2
    assert older["signal_date"] == "2026-09-08"
    assert older["baseline_date"] == "2026-09-10"
    assert older["baseline_close"] is None
    assert db.get_scan_results(first) == before


def test_weekend_selection_has_an_explicit_previous_session_baseline(db):
    save_scan(db, selected="2026-09-12")
    monitor = AlertMonitor(db, Quotes(), clock=lambda: clock())
    monitor.submit(background=False)
    row = monitor.listing(MonitorFilters())["items"][0]
    assert row["selected_date"] == "2026-09-12"
    assert row["baseline_date"] == "2026-09-11"
    assert row["baseline_close"] == 10.5
    assert row["trading_days"] == 1


def test_cumulative_return_uses_actual_selection_close_and_exchange_sessions(db):
    save_scan(db)
    quotes = Quotes()
    monitor = AlertMonitor(db, quotes, clock=lambda: clock())
    monitor.submit(background=False)
    row = monitor.listing(MonitorFilters())["items"][0]
    assert row["baseline_close"] == 10 and row["latest_close"] == 11
    assert row["return_pct"] == 10
    assert row["trading_days"] == 2  # Friday and Monday, not four calendar days.
    assert row["data_status"] == "fresh"
    assert monitor.store.latest_run()["status"] == "completed"
    assert quotes.calls == [("000001", "2026-09-10", "2026-09-14")]
    history = monitor.store.history(row["id"])
    assert len(history) == 1 and history[0]["market_date"] == "2026-09-14"


def test_same_stock_is_fetched_once_for_multiple_selection_dates(db):
    save_scan(db)
    save_scan(db, selected="2026-09-11")
    quotes = Quotes()
    monitor = AlertMonitor(db, quotes, clock=lambda: clock())
    monitor.submit(background=False)
    assert len(quotes.calls) == 1
    assert monitor.store.latest_run()["updated"] == 2
    assert monitor.listing(MonitorFilters())["total_count"] == 1
    assert monitor.listing(MonitorFilters(scope="all"))["total_count"] == 2


def test_frozen_entry_price_and_daily_upsert_survive_restart(db):
    save_scan(db)
    first = AlertMonitor(db, Quotes(), clock=lambda: clock())
    first.submit(background=False)
    second = AlertMonitor(db, Quotes({"2026-09-10": 8, "2026-09-14": 11}), clock=lambda: clock())
    second.submit(background=False)
    row = second.store.entries()[0]
    assert row["baseline_close"] == 10
    assert row["return_pct"] == 10
    assert len(second.store.history(row["id"])) == 1


def test_stale_prices_do_not_become_todays_return_or_pass_return_filters(db):
    save_scan(db)
    quotes = Quotes({"2026-09-10": 10, "2026-09-11": 10.5})
    monitor = AlertMonitor(db, quotes, clock=lambda: clock("2026-09-11"))
    monitor.submit(background=False)
    monitor.clock = lambda: clock()
    monitor.submit(background=False)
    row = monitor.listing(MonitorFilters())["items"][0]
    assert row["quote_date"] == "2026-09-11"
    assert row["latest_close"] == 10.5 and row["last_return_pct"] == 5
    assert row["return_pct"] is None and row["data_status"] == "stale"
    assert "2026-09-14" in row["last_error"]
    assert monitor.listing(MonitorFilters(min_return=0))["total_count"] == 0
    assert len(monitor.store.history(row["id"])) == 1
    assert monitor.store.latest_run()["status"] == "partial"


def test_transient_error_does_not_erase_a_valid_same_day_snapshot(db):
    save_scan(db)
    quotes = Quotes()
    monitor = AlertMonitor(db, quotes, clock=lambda: clock())
    monitor.submit(background=False)
    quotes.values = {"2026-09-10": 10}
    monitor.submit(background=False)
    row = monitor.listing(MonitorFilters())["items"][0]
    assert row["return_pct"] == 10 and row["data_status"] == "fresh"
    assert row["last_error"]


def test_missing_base_and_future_only_quotes_never_manufacture_performance(db):
    save_scan(db)
    monitor = AlertMonitor(db, Quotes({"2026-09-14": 11}), clock=lambda: clock())
    monitor.submit(background=False)
    row = monitor.listing(MonitorFilters())["items"][0]
    assert row["baseline_close"] is None and row["return_pct"] is None
    assert "基准日" in row["last_error"]
    monitor.quotes = Quotes({"2026-09-10": 10, "2026-09-15": 12})
    monitor.submit(background=False)
    assert monitor.store.entries()[0]["quote_date"] is None


def test_before_close_uses_last_completed_session_and_new_entries_wait(db):
    save_scan(db, selected="2026-09-14")
    quotes = Quotes()
    monitor = AlertMonitor(db, quotes, clock=lambda: clock(hour=14))
    monitor.submit(background=False)
    result = monitor.listing(MonitorFilters())
    assert result["as_of"] == "2026-09-11"
    assert result["items"][0]["baseline_close"] is None
    assert result["items"][0]["trading_days"] == 0
    assert quotes.calls == []


def test_inclusive_equal_bounds_zero_and_negative_return_filters(db):
    save_scan(db)
    monitor = AlertMonitor(db, Quotes(), clock=lambda: clock())
    monitor.submit(background=False)
    assert monitor.listing(MonitorFilters(min_days=2, max_days=2, min_return=10, max_return=10))["total_count"] == 1
    assert monitor.listing(MonitorFilters(max_return=0))["total_count"] == 0
    monitor.quotes.values["2026-09-14"] = 9
    monitor.submit(background=False)
    assert monitor.listing(MonitorFilters(min_return=-10, max_return=-10))["total_count"] == 1
    assert monitor.listing(MonitorFilters(min_return=0))["total_count"] == 0


def test_search_dates_pagination_and_sort(db):
    save_scan(db, codes=("000001", "000002", "000003"))
    monitor = AlertMonitor(db, Quotes(), clock=lambda: clock())
    monitor.submit(background=False)
    assert monitor.listing(MonitorFilters(q="银行"))["total_count"] == 1
    assert monitor.listing(MonitorFilters(selected_from=date_value("2026-09-11")))["total_count"] == 0
    first = monitor.listing(MonitorFilters(page_size=1, page=1, sort="return_desc"))
    second = monitor.listing(MonitorFilters(page_size=1, page=2, sort="return_desc"))
    assert first["total_count"] == 3 and first["items"][0]["id"] != second["items"][0]["id"]


def date_value(value):
    return datetime.fromisoformat(value).date()


@pytest.mark.parametrize("parameters", [
    {"min_days": 3, "max_days": 2}, {"min_days": -1}, {"min_days": 1.5},
    {"min_return": 5, "max_return": 4}, {"min_return": float("nan")},
    {"selected_from": "2026-09-14", "selected_to": "2026-09-11"},
    {"page_size": 201}, {"sort": "DROP TABLE scan_runs"},
])
def test_filter_validation(parameters):
    with pytest.raises(ValidationError):
        MonitorFilters(**parameters)


def test_adjusted_and_invalid_prices_are_rejected():
    with pytest.raises(ValueError, match="未复权"):
        valid_closes(frame({"2026-09-14": 10}, basis="qfq"))
    bars = frame({"2026-09-10": 10, "2026-09-11": 10, "2026-09-14": 10})
    bars.loc[0, "volume"] = 0
    bars.loc[1, "close"] = float("nan")
    bars.loc[2, "high"] = 8
    assert valid_closes(bars) == {}


def test_raw_provider_requests_no_adjustment_and_clips_future_rows(monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(kwargs)
        return frame({"2026-09-10": 10, "2026-09-14": 11, "2026-09-15": 12})

    monkeypatch.setattr("app.services.tencent_bars.fetch_tencent_bars", fetch)
    result = UnadjustedQuotes().fetch("000001", "2026-09-10", "2026-09-14")
    assert calls[0]["adjust"] == "" and calls[0]["end_date"] == "20260915"
    assert result.attrs["price_basis"] == "unadjusted"
    assert result.trade_date.tolist() == ["2026-09-10", "2026-09-14"]


def test_repeated_submit_does_not_start_duplicate_jobs(db):
    save_scan(db)
    release = threading.Event()
    entered = threading.Event()

    class SlowQuotes(Quotes):
        def fetch(self, *args):
            entered.set()
            release.wait(3)
            return super().fetch(*args)

    monitor = AlertMonitor(db, SlowQuotes(), clock=lambda: clock())
    try:
        first = monitor.submit()
        assert entered.wait(2)
        second = monitor.submit()
        assert first["id"] == second["id"]
    finally:
        release.set()
        monitor._thread.join(5)
    assert monitor.store.latest_run()["status"] == "completed"


def test_synced_listing_does_not_require_a_write_lock(db):
    save_scan(db)
    monitor = AlertMonitor(db, Quotes(), clock=lambda: clock())
    monitor.store.sync_alerts(clock())
    writer = sqlite3.connect(db.path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        assert monitor.listing(MonitorFilters())["total_count"] == 1
    finally:
        writer.rollback()
        writer.close()


def test_monitor_connections_are_closed_after_use(db):
    store = MonitorStore(db)
    with store.connect() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


@pytest.mark.parametrize("now,should_run", [
    (clock(hour=15), False), (clock(hour=16), True),
    (clock("2026-09-12", hour=16), False),
    (datetime(2026, 9, 14, 8, tzinfo=timezone.utc), True),
])
def test_schedule_uses_china_time_and_exchange_calendar(db, now, should_run):
    monitor = AlertMonitor(db, clock=lambda: now)
    calls = []
    monitor.submit = lambda **kwargs: calls.append(kwargs)
    monitor.run_due()
    assert bool(calls) is should_run
    if calls:
        assert calls[0]["trigger"] == "scheduled"


def test_daily_success_is_idempotent_and_failures_have_a_retry_cap(db):
    monitor = AlertMonitor(db, clock=lambda: clock())
    for _ in range(4):
        run = monitor.store.create_run("2026-09-14", "scheduled", clock())
        monitor.store.update_run(run["id"], status="partial")
    calls = []
    monitor.submit = lambda **kwargs: calls.append(kwargs)
    monitor.run_due()
    assert calls == []
    with db.connect() as conn:
        conn.execute("DELETE FROM alert_monitor_runs")
    run = monitor.store.create_run("2026-09-14", "scheduled", clock())
    monitor.store.update_run(run["id"], status="completed")
    monitor.run_due()
    assert calls == []


def test_manual_updates_do_not_consume_or_suppress_the_scheduled_run(db):
    monitor = AlertMonitor(db, clock=lambda: clock())
    for status in ("failed", "failed", "partial", "completed"):
        run = monitor.store.create_run("2026-09-14", "manual", clock(hour=15))
        monitor.store.update_run(run["id"], status=status)
    calls = []
    monitor.submit = lambda **kwargs: calls.append(kwargs)
    monitor.run_due()
    assert calls == [{"trigger": "scheduled", "market_date": "2026-09-14"}]


def test_startup_marks_only_monitor_jobs_interrupted_and_catches_up(db):
    monitor = AlertMonitor(db, clock=lambda: clock(), schedule_enabled=True)
    monitor.store.create_run("2026-09-14", "scheduled", clock())
    calls = []
    monitor.submit = lambda **kwargs: calls.append(kwargs)
    try:
        monitor.start()
        assert monitor.store.latest_run()["status"] == "failed"
        assert calls == [{"trigger": "scheduled", "market_date": "2026-09-14"}]
        job = monitor.scheduler.get_job("daily-close")
        assert str(job.trigger.timezone) == "Asia/Shanghai"
        assert "hour='16'" in str(job.trigger) and "minute='0'" in str(job.trigger)
        assert monitor.status()["schedule"]["next_run_at"]
    finally:
        monitor.stop()


def test_disabled_schedule_never_starts_timers(db):
    monitor = AlertMonitor(db, clock=lambda: clock())
    monitor.start()
    assert monitor.scheduler is None
    assert monitor.store.latest_run() is None
    monitor.stop()


def test_restart_before_todays_16_only_catches_up_the_previous_due_close(db):
    monitor = AlertMonitor(db, clock=lambda: clock(hour=15))
    calls = []
    monitor.submit = lambda **kwargs: calls.append(kwargs)
    monitor.run_due(catch_up=True)
    assert calls == [{"trigger": "scheduled", "market_date": "2026-09-11"}]
    run = monitor.store.create_run("2026-09-11", "scheduled", clock())
    monitor.store.update_run(run["id"], status="completed")
    monitor.run_due(catch_up=True)
    assert len(calls) == 1


def test_http_routes_return_progress_filters_and_history(tmp_path):
    path = tmp_path / "api.db"
    db = Database(path)
    save_scan(db)
    app = create_app(path, monitor_quotes=Quotes(), monitor_clock=lambda: clock(), monitor_schedule_enabled=False)
    with TestClient(app) as client:
        assert client.get("/api/alert-monitor").json()["total_count"] == 1
        assert client.get("/api/alert-monitor?min_days=5&max_days=2").status_code == 422
        assert client.get("/api/alert-monitor?min_return=nan").status_code == 422
        assert client.get("/api/alert-monitor/999/history").status_code == 404
        response = client.post("/api/alert-monitor/refresh")
        assert response.status_code == 202 and response.json()["id"]
        for _ in range(100):
            run = client.get("/api/alert-monitor/status").json()["run"]
            if run["status"] != "running":
                break
            time.sleep(.01)
        assert run["status"] == "completed"
        data = client.get("/api/alert-monitor?min_days=2&max_days=2&min_return=10&max_return=10").json()
        assert data["total_count"] == 1 and data["price_basis"] == "unadjusted"
        row = data["items"][0]
        history = client.get(f"/api/alert-monitor/{row['id']}/history").json()["items"]
        assert len(history) == 1 and history[0]["return_pct"] == 10
        assert db.latest_scan()["id"] == 1
