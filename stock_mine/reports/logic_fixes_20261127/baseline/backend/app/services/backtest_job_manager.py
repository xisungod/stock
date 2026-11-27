from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestJobState:
    run_id: int
    status: str = "running"
    total: int = 0
    processed: int = 0
    failed: int = 0
    signal_count: int = 0
    current_stock: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "failed": self.failed,
            "signal_count": self.signal_count,
            "current_stock": self.current_stock,
            "results": list(self.results),
            "summary": self.summary,
            "error": self.error,
        }


class BacktestJobManager:
    def __init__(self, db: Any, scanner: Any):
        self.db = db
        self.scanner = scanner
        self.jobs: dict[int, BacktestJobState] = {}
        self.running_id: int | None = None
        self.lock = threading.Lock()

    def submit(self, params: dict[str, Any]) -> int:
        with self.lock:
            if self.running_id is not None:
                return self.running_id
            run_id = self.db.create_backtest_run(params.get("provider", "auto"), params)
            self.jobs[run_id] = BacktestJobState(run_id)
            self.running_id = run_id
        threading.Thread(target=self._run, args=(run_id, params), daemon=True).start()
        return run_id

    def get(self, run_id: int) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(run_id)
            return state.to_dict() if state else self.db.get_backtest_run(run_id, include_results=True)

    def active(self) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(self.running_id) if self.running_id is not None else None
            return state.to_dict() if state else None

    def _run(self, run_id: int, params: dict[str, Any]) -> None:
        try:
            reports, summary = self.scanner.run_backtest(
                params,
                on_progress=lambda stats, report, error: self._progress(run_id, stats, report),
            )
            provider = getattr(self.scanner, "last_provider_name", params.get("provider", "auto"))
            stats = getattr(self.scanner, "last_backtest_stats", {})
            self.db.complete_backtest_run(run_id, provider, params, reports, stats, summary)
            with self.lock:
                state = self.jobs[run_id]
                state.status = "completed"
                state.results = [item for item in reports if item.get("signals")]
                state.summary = summary
        except Exception as exc:
            self.db.fail_backtest_run(run_id, str(exc))
            with self.lock:
                state = self.jobs[run_id]
                state.status = "failed"
                state.error = str(exc)
        finally:
            with self.lock:
                if self.running_id == run_id:
                    self.running_id = None

    def _progress(self, run_id: int, stats: dict[str, Any], report: dict[str, Any] | None) -> None:
        with self.lock:
            state = self.jobs.get(run_id)
            if state is None:
                return
            state.total = int(stats.get("target_count", 0))
            state.processed = int(stats.get("processed_count", 0))
            state.failed = int(stats.get("failed_count", 0))
            state.signal_count = int(stats.get("signal_count", 0))
            state.current_stock = str(stats.get("current_stock", ""))
            if report and report.get("signals"):
                state.results.append(report)
