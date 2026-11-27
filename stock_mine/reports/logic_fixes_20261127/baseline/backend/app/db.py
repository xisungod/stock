from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scan_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    buy_alert_count INTEGER NOT NULL DEFAULT 0,
                    watch_count INTEGER NOT NULL DEFAULT 0,
                    universe_count INTEGER NOT NULL DEFAULT 0,
                    attempted_count INTEGER NOT NULL DEFAULT 0,
                    analyzed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(scan_id) REFERENCES scan_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS watchlist (
                    stock_code TEXT PRIMARY KEY,
                    stock_name TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_signal_json TEXT,
                    entry_price REAL,
                    entry_date TEXT,
                    entry_price_source TEXT NOT NULL DEFAULT 'pending',
                    latest_price REAL,
                    latest_price_at TEXT
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    stock_count INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    summary_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    signal_count INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
                );
                """
            )
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(scan_runs)").fetchall()}
            for column in ("universe_count", "attempted_count", "analyzed_count", "failed_count", "stale_count"):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE scan_runs ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            watch_columns = {row[1] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
            watch_migrations = {
                "entry_price": "REAL",
                "entry_date": "TEXT",
                "entry_price_source": "TEXT NOT NULL DEFAULT 'legacy'",
                "latest_price": "REAL",
                "latest_price_at": "TEXT",
            }
            for column, declaration in watch_migrations.items():
                if column not in watch_columns:
                    conn.execute(f"ALTER TABLE watchlist ADD COLUMN {column} {declaration}")
            conn.execute(
                "UPDATE backtest_runs SET status = 'failed', finished_at = ?, error = ? WHERE status = 'running'",
                (now_iso(), "复盘被中断（后端重启或进程退出）"),
            )

    def create_scan_run(self, provider: str, parameters: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scan_runs (status, provider, parameters_json, started_at)
                VALUES (?, ?, ?, ?)
                """,
                ("running", provider, json.dumps(parameters, ensure_ascii=False), now_iso()),
            )
            return int(cur.lastrowid)

    def complete_scan_run(
        self,
        scan_id: int,
        provider: str,
        parameters: dict[str, Any],
        results: list[dict[str, Any]],
        stats: dict[str, int] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        buy_count = sum(1 for item in results if item.get("category") == "buy_alert")
        watch_count = sum(1 for item in results if item.get("category") == "watch")
        scan_stats = stats or {}
        with self.connect() as conn:
            conn.execute("DELETE FROM scan_results WHERE scan_id = ?", (scan_id,))
            conn.executemany(
                """
                INSERT INTO scan_results (scan_id, stock_code, stock_name, category, score, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_id,
                        item["stock_code"],
                        item["stock_name"],
                        item["category"],
                        float(item["score"]),
                        json.dumps(item, ensure_ascii=False),
                    )
                    for item in results
                ],
            )
            conn.execute(
                """
                UPDATE scan_runs
                SET status = ?, provider = ?, parameters_json = ?, finished_at = ?,
                    total_count = ?, buy_alert_count = ?, watch_count = ?,
                    universe_count = ?, attempted_count = ?, analyzed_count = ?, failed_count = ?, stale_count = ?,
                    error = NULL
                WHERE id = ?
                """,
                (
                    status,
                    provider,
                    json.dumps(parameters, ensure_ascii=False),
                    now_iso(),
                    len(results),
                    buy_count,
                    watch_count,
                    int(scan_stats.get("universe_count", 0)),
                    int(scan_stats.get("attempted_count", 0)),
                    int(scan_stats.get("analyzed_count", 0)),
                    int(scan_stats.get("failed_count", 0)),
                    int(scan_stats.get("stale_count", 0)),
                    scan_id,
                ),
            )
        return self.get_scan(scan_id, include_results=True) or {}

    def fail_stale_running_scans(self) -> int:
        """把残留的 running 记录标记为失败，返回处理条数。

        任务进度只存在进程内存中，后端一旦重启或崩溃，上次中断的扫描会永远停在
        running。而 latest_scan 取的是 id 最大的一条，前端刷新后就会看到一个
        没有任何结果的「上次筛选」。启动时清理掉。
        """
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE scan_runs SET status = ?, finished_at = ?, error = ? WHERE status = ?",
                ("failed", now_iso(), "扫描被中断（后端重启或进程退出），本次结果未保存", "running"),
            )
            return int(cur.rowcount)

    def fail_scan_run(self, scan_id: int, error: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                "UPDATE scan_runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                ("failed", now_iso(), error, scan_id),
            )
        return self.get_scan(scan_id, include_results=True) or {}

    def get_scan(self, scan_id: int, include_results: bool = False) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM scan_runs WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return None
            payload = _scan_row(row)
            if include_results:
                payload["results"] = self.get_scan_results(scan_id)
            return payload

    def latest_scan(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            # 页面展示的是最近一次有效结果。失败或被重启中断的任务没有结果，
            # 不能因为 id 更新就把上一批完整预警从页面上覆盖掉。
            row = conn.execute(
                # 被取消的扫描同样有真实结果（只是不完整），页面该看得到
                "SELECT * FROM scan_runs WHERE status IN ('completed', 'cancelled') "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            payload = _scan_row(row)
            payload["results"] = self.get_scan_results(int(row["id"]))
            return payload

    def get_scan_results(self, scan_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM scan_results WHERE scan_id = ? ORDER BY category, score DESC",
                (scan_id,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_watchlist(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM watchlist ORDER BY updated_at DESC").fetchall()
        return [_watch_row(row) for row in rows]

    def upsert_watch(
        self,
        stock_code: str,
        stock_name: str,
        note: str = "",
        status: str = "active",
        entry_price: float | None = None,
        entry_date: str | None = None,
    ) -> dict[str, Any]:
        ts = now_iso()
        normalized_price = float(entry_price) if entry_price is not None and float(entry_price) > 0 else None
        normalized_date = entry_date or (ts[:10] if normalized_price is not None else None)
        entry_source = "recorded" if normalized_price is not None else "pending"
        with self.connect() as conn:
            existing = conn.execute("SELECT created_at FROM watchlist WHERE stock_code = ?", (stock_code,)).fetchone()
            created_at = existing["created_at"] if existing else ts
            conn.execute(
                """
                INSERT INTO watchlist (
                    stock_code, stock_name, note, status, created_at, updated_at,
                    entry_price, entry_date, entry_price_source, latest_price, latest_price_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock_code) DO UPDATE SET
                    stock_name = excluded.stock_name,
                    note = excluded.note,
                    status = excluded.status,
                    entry_price = COALESCE(watchlist.entry_price, excluded.entry_price),
                    entry_date = COALESCE(watchlist.entry_date, excluded.entry_date),
                    entry_price_source = CASE
                        WHEN watchlist.entry_price IS NULL AND excluded.entry_price IS NOT NULL THEN excluded.entry_price_source
                        ELSE watchlist.entry_price_source
                    END,
                    latest_price = COALESCE(watchlist.latest_price, excluded.latest_price),
                    latest_price_at = COALESCE(watchlist.latest_price_at, excluded.latest_price_at),
                    updated_at = excluded.updated_at
                """,
                (
                    stock_code,
                    stock_name,
                    note,
                    status,
                    created_at,
                    ts,
                    normalized_price,
                    normalized_date,
                    entry_source,
                    normalized_price,
                    normalized_date,
                ),
            )
        return self.get_watch(stock_code) or {}

    def update_watch(self, stock_code: str, note: str | None = None, status: str | None = None, last_signal: dict[str, Any] | None = None) -> dict[str, Any] | None:
        current = self.get_watch(stock_code)
        if current is None:
            return None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE watchlist
                SET note = ?, status = ?,
                    last_signal_json = COALESCE(?, last_signal_json), updated_at = ?
                WHERE stock_code = ?
                """,
                (
                    current["note"] if note is None else note,
                    current["status"] if status is None else status,
                    json.dumps(last_signal, ensure_ascii=False) if last_signal is not None else None,
                    now_iso(),
                    stock_code,
                ),
            )
        return self.get_watch(stock_code)

    def update_watch_price(
        self,
        stock_code: str,
        latest_price: float,
        latest_price_at: str | None,
    ) -> dict[str, Any] | None:
        current = self.get_watch(stock_code)
        if current is None:
            return None
        price = float(latest_price)
        quote_date = latest_price_at or now_iso()[:10]
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE watchlist
                SET entry_price = COALESCE(entry_price, ?),
                    entry_date = COALESCE(entry_date, ?),
                    entry_price_source = CASE
                        WHEN entry_price IS NULL THEN 'first_refresh'
                        ELSE entry_price_source
                    END,
                    latest_price = ?, latest_price_at = ?, updated_at = ?
                WHERE stock_code = ?
                """,
                (price, quote_date, price, quote_date, now_iso(), stock_code),
            )
        return self.get_watch(stock_code)

    def get_watch(self, stock_code: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM watchlist WHERE stock_code = ?", (stock_code,)).fetchone()
        return _watch_row(row) if row else None

    def delete_watch(self, stock_code: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM watchlist WHERE stock_code = ?", (stock_code,))
            return cur.rowcount > 0

    def create_backtest_run(self, provider: str, parameters: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO backtest_runs (status, provider, parameters_json, started_at) VALUES (?, ?, ?, ?)",
                ("running", provider, json.dumps(parameters, ensure_ascii=False), now_iso()),
            )
            return int(cur.lastrowid)

    def complete_backtest_run(
        self,
        run_id: int,
        provider: str,
        parameters: dict[str, Any],
        reports: list[dict[str, Any]],
        stats: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO backtest_results (run_id, stock_code, stock_name, signal_count, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item["stock_code"],
                        item["stock_name"],
                        int(item.get("summary", {}).get("signal_count", 0)),
                        json.dumps(item, ensure_ascii=False),
                    )
                    for item in reports
                ],
            )
            conn.execute(
                """
                UPDATE backtest_runs
                SET status = 'completed', provider = ?, parameters_json = ?, finished_at = ?,
                    stock_count = ?, processed_count = ?, failed_count = ?, summary_json = ?, error = NULL
                WHERE id = ?
                """,
                (
                    provider,
                    json.dumps(parameters, ensure_ascii=False),
                    now_iso(),
                    int(stats.get("target_count", len(reports))),
                    int(stats.get("processed_count", len(reports))),
                    int(stats.get("failed_count", 0)),
                    json.dumps(summary, ensure_ascii=False),
                    run_id,
                ),
            )
        return self.get_backtest_run(run_id, include_results=True) or {}

    def fail_backtest_run(self, run_id: int, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE backtest_runs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
                (now_iso(), error, run_id),
            )

    def get_backtest_run(self, run_id: int, include_results: bool = False) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        payload = _backtest_row(row)
        if include_results:
            payload["results"] = self.get_backtest_results(run_id)
        return payload

    def latest_backtest(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM backtest_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        payload = _backtest_row(row)
        payload["results"] = self.get_backtest_results(int(row["id"]), only_with_signals=True)
        return payload

    def get_backtest_results(self, run_id: int, only_with_signals: bool = False) -> list[dict[str, Any]]:
        query = "SELECT payload_json FROM backtest_results WHERE run_id = ?"
        if only_with_signals:
            query += " AND signal_count > 0"
        query += " ORDER BY signal_count DESC, stock_code"
        with self.connect() as conn:
            rows = conn.execute(query, (run_id,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


def _scan_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "status": row["status"],
        "provider": row["provider"],
        "parameters": json.loads(row["parameters_json"]),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "total_count": int(row["total_count"]),
        "buy_alert_count": int(row["buy_alert_count"]),
        "watch_count": int(row["watch_count"]),
        "universe_count": int(row["universe_count"]),
        "attempted_count": int(row["attempted_count"]),
        "analyzed_count": int(row["analyzed_count"]),
        "failed_count": int(row["failed_count"]),
        "stale_count": int(row["stale_count"]),
        "error": row["error"],
    }


def _watch_row(row: sqlite3.Row) -> dict[str, Any]:
    last_signal_raw = row["last_signal_json"]
    entry_price = float(row["entry_price"]) if row["entry_price"] is not None else None
    latest_price = float(row["latest_price"]) if row["latest_price"] is not None else None
    price_change = latest_price - entry_price if latest_price is not None and entry_price else None
    return_pct = price_change / entry_price * 100 if price_change is not None and entry_price else None
    return {
        "stock_code": row["stock_code"],
        "stock_name": row["stock_name"],
        "note": row["note"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_signal": json.loads(last_signal_raw) if last_signal_raw else None,
        "entry_price": round(entry_price, 3) if entry_price is not None else None,
        "entry_date": row["entry_date"],
        "entry_price_source": row["entry_price_source"],
        "latest_price": round(latest_price, 3) if latest_price is not None else None,
        "latest_price_at": row["latest_price_at"],
        "price_change": round(price_change, 3) if price_change is not None else None,
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
    }


def _backtest_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "status": row["status"],
        "provider": row["provider"],
        "parameters": json.loads(row["parameters_json"]),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "stock_count": int(row["stock_count"]),
        "processed_count": int(row["processed_count"]),
        "failed_count": int(row["failed_count"]),
        "summary": json.loads(row["summary_json"]) if row["summary_json"] else {},
        "error": row["error"],
    }
