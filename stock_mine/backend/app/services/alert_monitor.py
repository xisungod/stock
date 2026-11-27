"""Persistent, close-to-close tracking of saved real-data buy alerts."""
from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from app.services.market_calendar import (
    CHINA, china_now, china_time, exchange_calendar, last_completed_session, session_dates,
)

LOG = logging.getLogger(__name__)
MAX_DAILY_ATTEMPTS = 4


class MonitorFilters(BaseModel):
    q: str = Field(default="", max_length=100)
    scope: Literal["latest", "all"] = "latest"
    min_days: int | None = Field(default=None, ge=0, le=20000)
    max_days: int | None = Field(default=None, ge=0, le=20000)
    min_return: float | None = Field(default=None, allow_inf_nan=False)
    max_return: float | None = Field(default=None, allow_inf_nan=False)
    selected_from: date | None = None
    selected_to: date | None = None
    data_status: Literal["all", "fresh", "stale", "pending"] = "all"
    sort: Literal["return_desc", "return_asc", "days_desc", "days_asc", "selected_desc"] = "selected_desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)

    @model_validator(mode="after")
    def ranges(self):
        for lower, upper, label in (
            (self.min_days, self.max_days, "交易日数"),
            (self.min_return, self.max_return, "涨跌幅"),
            (self.selected_from, self.selected_to, "入选日期"),
        ):
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{label}下限不能大于上限")
        return self


def _stamp(now: datetime | None = None) -> str:
    return china_time(now or china_now()).isoformat(timespec="seconds")


def _baseline_date(selected: date) -> str:
    # A weekend selection references the previous session's close, explicitly dated.
    session_dates(selected, selected)
    return exchange_calendar().date_to_session(
        pd.Timestamp(selected), direction="previous",
    ).strftime("%Y-%m-%d")


def _positive(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


class MonitorStore:
    def __init__(self, db):
        self.db = db
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alert_monitor_imports (
                    scan_id INTEGER PRIMARY KEY, imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alert_monitor_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL, stock_name TEXT NOT NULL,
                    first_scan_id INTEGER NOT NULL, selected_at TEXT NOT NULL,
                    selected_date TEXT NOT NULL, baseline_date TEXT NOT NULL,
                    signal_date TEXT, industry TEXT NOT NULL DEFAULT '',
                    baseline_close REAL, baseline_source TEXT,
                    latest_close REAL, quote_date TEXT, return_pct REAL,
                    quote_source TEXT, updated_at TEXT,
                    last_attempt_at TEXT, last_error TEXT,
                    UNIQUE(stock_code, selected_date)
                );
                CREATE TABLE IF NOT EXISTS alert_monitor_sources (
                    entry_id INTEGER NOT NULL, scan_id INTEGER NOT NULL,
                    PRIMARY KEY(entry_id, scan_id)
                );
                CREATE TABLE IF NOT EXISTS alert_monitor_daily (
                    entry_id INTEGER NOT NULL, market_date TEXT NOT NULL,
                    close REAL NOT NULL, return_pct REAL NOT NULL,
                    trading_days INTEGER NOT NULL, source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(entry_id, market_date)
                );
                CREATE TABLE IF NOT EXISTS alert_monitor_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_date TEXT NOT NULL, trigger TEXT NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL,
                    finished_at TEXT, total INTEGER NOT NULL DEFAULT 0,
                    processed INTEGER NOT NULL DEFAULT 0,
                    updated INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    current_stock TEXT NOT NULL DEFAULT '', error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS alert_monitor_one_running
                    ON alert_monitor_runs(status) WHERE status = 'running';
                CREATE INDEX IF NOT EXISTS alert_monitor_dates
                    ON alert_monitor_entries(selected_date, stock_code);
                CREATE INDEX IF NOT EXISTS alert_monitor_run_date
                    ON alert_monitor_runs(market_date, id);
            """)

    @contextmanager
    def connect(self):
        conn = self.db.connect()
        try:
            conn.execute("PRAGMA busy_timeout = 15000")
            with conn:
                yield conn
        finally:
            conn.close()

    def sync_alerts(self, now: datetime | None = None) -> int:
        clock = china_time(now or china_now())
        added = 0
        # An ordinary listing must remain readable while another job holds a write lock.
        with self.connect() as conn:
            pending = conn.execute("""
                SELECT 1 FROM scan_runs s
                WHERE s.status IN ('completed', 'cancelled')
                AND NOT EXISTS (SELECT 1 FROM alert_monitor_imports i WHERE i.scan_id = s.id)
                LIMIT 1
            """).fetchone()
        if pending is None:
            return 0
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            runs = conn.execute("""
                SELECT s.* FROM scan_runs s
                WHERE s.status IN ('completed', 'cancelled')
                AND NOT EXISTS (SELECT 1 FROM alert_monitor_imports i WHERE i.scan_id = s.id)
                ORDER BY s.id
            """).fetchall()
            for run in runs:
                params = json.loads(run["parameters_json"])
                providers = {str(run["provider"]).lower(), str(params.get("provider", "")).lower()}
                if providers & {"sample", "demo", "fake"}:
                    conn.execute("INSERT INTO alert_monitor_imports VALUES (?, ?)", (run["id"], _stamp(clock)))
                    continue
                selected = china_time(datetime.fromisoformat(run["finished_at"] or run["started_at"]))
                if selected > clock:
                    continue
                baseline = _baseline_date(selected.date())
                results = conn.execute("""
                    SELECT stock_code, stock_name, payload_json FROM scan_results
                    WHERE scan_id = ? AND category = 'buy_alert' ORDER BY id
                """, (run["id"],)).fetchall()
                for result in results:
                    code = str(result["stock_code"]).zfill(6)
                    if len(code) != 6 or not code.isdigit():
                        continue
                    payload = json.loads(result["payload_json"])
                    cur = conn.execute("""
                        INSERT OR IGNORE INTO alert_monitor_entries
                        (stock_code, stock_name, first_scan_id, selected_at, selected_date,
                         baseline_date, signal_date, industry)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        code, result["stock_name"], run["id"], _stamp(selected),
                        selected.date().isoformat(), baseline, payload.get("signal_date"),
                        str(payload.get("industry") or ""),
                    ))
                    added += cur.rowcount
                    entry = conn.execute(
                        "SELECT id FROM alert_monitor_entries WHERE stock_code = ? AND selected_date = ?",
                        (code, selected.date().isoformat()),
                    ).fetchone()
                    conn.execute("INSERT OR IGNORE INTO alert_monitor_sources VALUES (?, ?)",
                                 (entry["id"], run["id"]))
                conn.execute("INSERT INTO alert_monitor_imports VALUES (?, ?)", (run["id"], _stamp(clock)))
        return added

    def entries(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT e.*, (SELECT COUNT(*) FROM alert_monitor_sources s
                             WHERE s.entry_id = e.id) AS scan_count
                FROM alert_monitor_entries e ORDER BY selected_date DESC, id DESC
            """).fetchall()]

    def history(self, entry_id: int) -> list[dict] | None:
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM alert_monitor_entries WHERE id = ?", (entry_id,)).fetchone() is None:
                return None
            return [dict(row) for row in conn.execute("""
                SELECT * FROM alert_monitor_daily WHERE entry_id = ? ORDER BY market_date
            """, (entry_id,)).fetchall()]

    def fail_interrupted(self):
        with self.connect() as conn:
            conn.execute("""
                UPDATE alert_monitor_runs SET status = 'failed', finished_at = ?,
                error = '跟踪更新被服务重启中断，将在下次更新时重试'
                WHERE status = 'running'
            """, (_stamp(),))

    def active_run(self):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM alert_monitor_runs WHERE status = 'running'").fetchone()
            return dict(row) if row else None

    def latest_run(self):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM alert_monitor_runs ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def daily_runs(self, market_date: str) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM alert_monitor_runs WHERE market_date = ? ORDER BY id",
                (market_date,),
            ).fetchall()]

    def create_run(self, market_date, trigger, now) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute("SELECT * FROM alert_monitor_runs WHERE status = 'running'").fetchone()
            if active:
                return {**dict(active), "created": False}
            cur = conn.execute("""
                INSERT INTO alert_monitor_runs (market_date, trigger, status, started_at)
                VALUES (?, ?, 'running', ?)
            """, (market_date, trigger, _stamp(now)))
            row = conn.execute("SELECT * FROM alert_monitor_runs WHERE id = ?", (cur.lastrowid,)).fetchone()
            return {**dict(row), "created": True}

    def update_run(self, run_id, **values):
        allowed = {"status", "finished_at", "total", "processed", "updated", "failed", "current_stock", "error"}
        if not values or set(values) - allowed:
            raise ValueError("Invalid monitor progress fields")
        fields = ", ".join(f"{name} = ?" for name in values)
        with self.connect() as conn:
            conn.execute(f"UPDATE alert_monitor_runs SET {fields} WHERE id = ?",
                         (*values.values(), run_id))

    def save_quote(self, entry, market_date, baseline_close, latest_close, days, source, now):
        baseline = entry["baseline_close"] or baseline_close
        change = round((latest_close / baseline - 1) * 100, 6)
        with self.connect() as conn:
            conn.execute("""
                UPDATE alert_monitor_entries SET
                    baseline_close = COALESCE(baseline_close, ?),
                    baseline_source = COALESCE(baseline_source, ?),
                    latest_close = ?, quote_date = ?, return_pct = ?, quote_source = ?,
                    updated_at = ?, last_attempt_at = ?, last_error = NULL
                WHERE id = ? AND (quote_date IS NULL OR quote_date <= ?)
            """, (baseline, source, latest_close, market_date, change, source,
                  _stamp(now), _stamp(now), entry["id"], market_date))
            conn.execute("""
                INSERT INTO alert_monitor_daily VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id, market_date) DO UPDATE SET
                    close = excluded.close, return_pct = excluded.return_pct,
                    trading_days = excluded.trading_days, source = excluded.source,
                    updated_at = excluded.updated_at
            """, (entry["id"], market_date, latest_close, change, days, source, _stamp(now)))

    def save_error(self, entry_id, error, now):
        with self.connect() as conn:
            conn.execute("UPDATE alert_monitor_entries SET last_error = ?, last_attempt_at = ? WHERE id = ?",
                         (str(error)[:500], _stamp(now), entry_id))


class UnadjustedQuotes:
    """Do not reuse the strategy's forward-adjusted price cache as an entry price."""

    def fetch(self, code: str, start: str, end: str) -> pd.DataFrame:
        from app.services.tencent_bars import fetch_tencent_bars
        from app.services.akshare_guard import call_akshare_parallel, get_akshare

        errors = []
        # Tencent's upper bound can be exclusive; clip the response back to the target day.
        until = (date.fromisoformat(end) + timedelta(days=1)).strftime("%Y%m%d")
        try:
            frame = fetch_tencent_bars(
                ("sh" if code.startswith("6") else "sz") + code,
                start_date=start.replace("-", ""), end_date=until, adjust="", timeout=12,
            )
            frame = self.normalize(frame, start, end)
            if not frame.empty and end in set(frame.trade_date):
                frame.attrs.update(price_basis="unadjusted", source="tencent")
                return frame
        except Exception as exc:
            errors.append(f"腾讯: {str(exc)[:150]}")
        try:
            frame = call_akshare_parallel(lambda: get_akshare().stock_zh_a_hist(
                symbol=code, period="daily", start_date=start.replace("-", ""),
                end_date=end.replace("-", ""), adjust="", timeout=12,
            ))
            frame = self.normalize(frame, start, end)
            if not frame.empty:
                frame.attrs.update(price_basis="unadjusted", source="eastmoney")
                return frame
        except Exception as exc:
            errors.append(f"东方财富: {str(exc)[:150]}")
        raise RuntimeError("; ".join(errors) or "未返回未复权收盘行情")

    @staticmethod
    def normalize(frame, start, end):
        if frame is None or frame.empty:
            return pd.DataFrame()
        frame = frame.rename(columns={
            "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume",
        }).copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame[(frame.trade_date >= start) & (frame.trade_date <= end)].drop_duplicates(
            "trade_date", keep="last",
        ).sort_values("trade_date").reset_index(drop=True)


def valid_closes(frame: pd.DataFrame) -> dict[str, float]:
    if frame.attrs.get("price_basis") != "unadjusted":
        raise ValueError("跟踪只接受未复权价格，不能混用策略的前复权缓存")
    quotes = {}
    for row in frame.to_dict("records"):
        prices = [_positive(row.get(key)) for key in ("open", "high", "low", "close")]
        if any(value is None for value in prices) or _positive(row.get("volume")) is None:
            continue
        opening, high, low, close = prices
        if low > min(opening, close) or high < max(opening, close) or high < low:
            continue
        stamp = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        quotes[stamp] = close
    return quotes


class AlertMonitor:
    def __init__(self, db, quotes=None, clock=china_now, schedule_enabled=False):
        self.store = MonitorStore(db)
        self.quotes = quotes or UnadjustedQuotes()
        self.clock = clock
        self.schedule_enabled = schedule_enabled
        self.scheduler = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.store.fail_interrupted()
        if not self.schedule_enabled:
            return
        from apscheduler.schedulers.background import BackgroundScheduler

        self.scheduler = BackgroundScheduler(timezone=CHINA)
        self.scheduler.add_job(self.run_due, "cron", hour=16, minute=0, id="daily-close",
                               coalesce=True, max_instances=1, misfire_grace_time=3600)
        self.scheduler.add_job(self.run_due, "cron", hour="16-18", minute="15,30,45",
                               id="close-retry", coalesce=True, max_instances=1, misfire_grace_time=600)
        self.scheduler.start()
        self.run_due(catch_up=True)

    def stop(self):
        self._stop.set()
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None
        if self._thread:
            self._thread.join(timeout=25)

    def run_due(self, catch_up=False):
        clock = china_time(self.clock())
        if not catch_up and clock.time().replace(tzinfo=None) < time(16):
            return
        try:
            if catch_up:
                cutoff = clock
                if clock.time().replace(tzinfo=None) < time(16):
                    cutoff = datetime.combine(clock.date() - timedelta(days=1), time(16), CHINA)
                day = last_completed_session(cutoff)
            else:
                day = clock.date().isoformat()
                if not session_dates(day, day):
                    return
            runs = [run for run in self.store.daily_runs(day) if run["trigger"] == "scheduled"]
            if any(run["status"] in {"running", "completed"} for run in runs):
                return
            if len(runs) >= MAX_DAILY_ATTEMPTS:
                return
            self.submit(trigger="scheduled", market_date=day)
        except Exception:
            LOG.exception("Automatic alert tracking could not start")

    def submit(self, trigger="manual", background=True, market_date=None):
        with self._lock:
            existing = self.store.active_run()
            if existing:
                return existing
            clock = china_time(self.clock())
            market_date = market_date or last_completed_session(clock)
            run = self.store.create_run(market_date, trigger, clock)
            if not run.pop("created"):
                return run
            if background:
                self._thread = threading.Thread(target=self._run, args=(run,), daemon=True)
                self._thread.start()
            else:
                self._run(run)
            return run

    def _run(self, run):
        run_id, target = run["id"], run["market_date"]
        processed = updated = failed = 0
        try:
            self.store.sync_alerts(self.clock())
            groups = {}
            for entry in self.store.entries():
                if entry["baseline_date"] <= target:
                    groups.setdefault(entry["stock_code"], []).append(entry)
            self.store.update_run(run_id, total=len(groups))
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = {executor.submit(self._update_stock, code, rows, target): code
                           for code, rows in groups.items()}
                for future in as_completed(pending):
                    count, errors = future.result()
                    processed += 1
                    updated += count
                    failed += int(errors > 0)
                    self.store.update_run(run_id, processed=processed, updated=updated, failed=failed,
                                          current_stock=pending[future])
            status = "failed" if self._stop.is_set() else ("partial" if failed else "completed")
            self.store.update_run(run_id, status=status, finished_at=_stamp(self.clock()), current_stock="")
        except Exception as exc:
            LOG.exception("Alert monitor run failed")
            self.store.update_run(run_id, status="failed", finished_at=_stamp(self.clock()), error=str(exc)[:500])

    def _update_stock(self, code, entries, target):
        if self._stop.is_set():
            return 0, len(entries)
        stamp = self.clock()
        try:
            missing_bases = [entry["baseline_date"] for entry in entries if entry["baseline_close"] is None]
            frame = self.quotes.fetch(code, min(missing_bases + [target]), target)
            prices = valid_closes(frame)
            source = str(frame.attrs.get("source", "unknown"))
            if target not in prices:
                raise ValueError(f"缺少 {target} 的有效收盘行情，可能停牌或数据源尚未更新")
        except Exception as exc:
            for entry in entries:
                self.store.save_error(entry["id"], exc, stamp)
            return 0, len(entries)
        updated = failed = 0
        for entry in entries:
            baseline = entry["baseline_close"] or prices.get(entry["baseline_date"])
            if baseline is None:
                self.store.save_error(entry["id"], f"缺少入选基准日 {entry['baseline_date']} 的有效收盘价", stamp)
                failed += 1
                continue
            days = len(session_dates(entry["baseline_date"], target)) - 1
            self.store.save_quote(entry, target, baseline, prices[target], days, source, stamp)
            updated += 1
        return updated, failed

    def status(self):
        next_run = None
        if self.scheduler:
            job = self.scheduler.get_job("daily-close")
            next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
        return {
            "run": self.store.active_run() or self.store.latest_run(),
            "schedule": {"enabled": self.schedule_enabled, "time": "16:00", "timezone": "Asia/Shanghai",
                         "next_run_at": next_run, "max_daily_attempts": MAX_DAILY_ATTEMPTS},
        }

    def listing(self, filters: MonitorFilters):
        self.store.sync_alerts(self.clock())
        target = last_completed_session(self.clock())
        entries = self.store.entries()
        earliest = min((entry["baseline_date"] for entry in entries), default=target)
        sessions = {stamp: index for index, stamp in enumerate(session_dates(earliest, target))}
        rows, seen = [], set()
        for entry in entries:
            code = entry["stock_code"]
            if filters.scope == "latest" and code in seen:
                continue
            seen.add(code)
            row = dict(entry)
            row["trading_days"] = (
                sessions[target] - sessions[row["baseline_date"]] if row["baseline_date"] in sessions else 0
            )
            fresh = row["quote_date"] == target and row["baseline_close"] is not None
            row["data_status"] = "fresh" if fresh else ("pending" if row["baseline_close"] is None else "stale")
            row["last_return_pct"] = row["return_pct"]
            if not fresh:
                row["return_pct"] = None
            rows.append(row)
        tracked_count = len(rows)
        query = filters.q.strip().casefold()

        def matches(row):
            if query and query not in f"{row['stock_code']} {row['stock_name']} {row['industry']}".casefold():
                return False
            if filters.data_status != "all" and row["data_status"] != filters.data_status:
                return False
            for value, lower, upper in (
                (row["trading_days"], filters.min_days, filters.max_days),
                (row["return_pct"], filters.min_return, filters.max_return),
                (row["selected_date"], filters.selected_from.isoformat() if filters.selected_from else None,
                 filters.selected_to.isoformat() if filters.selected_to else None),
            ):
                if (lower is not None or upper is not None) and value is None:
                    return False
                if lower is not None and value < lower or upper is not None and value > upper:
                    return False
            return True

        rows = [row for row in rows if matches(row)]
        if filters.sort.startswith("return_"):
            direction = -1 if filters.sort == "return_desc" else 1
            rows.sort(key=lambda row: (row["return_pct"] is None, direction * (row["return_pct"] or 0), row["id"]))
        elif filters.sort.startswith("days_"):
            rows.sort(key=lambda row: (row["trading_days"], row["id"]), reverse=filters.sort == "days_desc")
        summary = {"tracked_count": tracked_count, "filtered_count": len(rows),
                   "fresh_count": sum(row["data_status"] == "fresh" for row in rows),
                   "up_count": sum(row["return_pct"] is not None and row["return_pct"] > 0 for row in rows),
                   "down_count": sum(row["return_pct"] is not None and row["return_pct"] < 0 for row in rows),
                   "pending_count": sum(row["data_status"] != "fresh" for row in rows)}
        offset = (filters.page - 1) * filters.page_size
        return {**self.status(), "items": rows[offset:offset + filters.page_size], "total_count": len(rows),
                "page": filters.page, "page_size": filters.page_size, "as_of": target,
                "price_basis": "unadjusted", "summary": summary}
