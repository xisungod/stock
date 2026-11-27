from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

RETAIN_SECONDS = 30 * 60


@dataclass
class JobState:
    scan_id: int
    status: str = "running"
    # 本次实际要扫描的股票数量（抽样之后），即进度条分母；不是全市场股票总数
    total: int = 0
    processed: int = 0
    hits: int = 0
    failed: int = 0
    stale: int = 0
    current_stock: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "hits": self.hits,
            "failed": self.failed,
            "stale": self.stale,
            "current_stock": self.current_stock,
            "results": list(self.results),
            "error": self.error,
        }


class ScanJobManager:
    """管理扫描任务的后台执行与进度状态（进程内内存，不落额外的表）。"""

    def __init__(self, db: Any, scanner: Any) -> None:
        self._db = db
        self._scanner = scanner
        self._jobs: dict[int, JobState] = {}
        self._lock = threading.Lock()
        self._running_id: int | None = None

    def submit(self, params: dict[str, Any]) -> int:
        with self._lock:
            self._purge_expired()
            if self._running_id is not None:
                return self._running_id
            scan_id = self._db.create_scan_run(params.get("provider", "auto"), params)
            self._jobs[scan_id] = JobState(scan_id=scan_id)
            self._running_id = scan_id

        thread = threading.Thread(target=self._run, args=(scan_id, params), daemon=True)
        thread.start()
        return scan_id

    def cancel(self, scan_id: int) -> bool:
        """请求停止扫描。扫描循环每只股票检查一次，收到后跳出并保留已命中的结果。

        返回是否真的发出了取消（任务不存在或已结束时返回 False）。
        """
        with self._lock:
            state = self._jobs.get(scan_id)
            if state is None or state.status != "running":
                return False
            state.cancel.set()
            return True

    def get(self, scan_id: int) -> dict[str, Any] | None:
        with self._lock:
            state = self._jobs.get(scan_id)
            return state.to_dict() if state else None

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            if self._running_id is None:
                return None
            state = self._jobs.get(self._running_id)
            return state.to_dict() if state else None

    def _cancelled(self, scan_id: int) -> bool:
        with self._lock:
            state = self._jobs.get(scan_id)
            return bool(state and state.cancel.is_set())

    def _run(self, scan_id: int, params: dict[str, Any]) -> None:
        status = "completed"
        payload: list[dict[str, Any]] = []
        error: str | None = None
        try:
            results = self._scanner.run_scan(
                params,
                on_progress=lambda stats, hit: self._on_progress(scan_id, stats, hit),
                should_stop=lambda: self._cancelled(scan_id),
            )
            payload = [item.to_dict() for item in results]
            if self._cancelled(scan_id):
                status = "cancelled"
            provider_name = getattr(self._scanner, "last_provider_name", params.get("provider", "auto"))
            stats = getattr(self._scanner, "last_scan_stats", {})
            # 中途停止也把已命中的结果落库——白跑几百只再丢掉没有道理
            self._db.complete_scan_run(scan_id, provider_name, params, payload, stats, status=status)
        except Exception as exc:
            status, payload, error = "failed", [], str(exc)
            self._db.fail_scan_run(scan_id, error)
        finally:
            # 放在 finally 里：即便上面写库再次抛错，也必须释放 _running_id，
            # 否则之后每一次 submit 都会被误判为"已有任务在跑"。
            self._finish(scan_id, status, payload, error)

    def _finish(self, scan_id: int, status: str, payload: list[dict[str, Any]], error: str | None) -> None:
        with self._lock:
            state = self._jobs.get(scan_id)
            if state is not None:
                state.status = status
                state.error = error
                if status in {"completed", "cancelled"}:
                    # 用扫描器返回的完整（已排序）结果覆盖边扫边出的增量列表
                    state.results = payload
                    state.hits = len(payload)
                    # 只给出最终统计、全程不回调 on_progress 的扫描器走不到 _on_progress，
                    # 这里从最终统计里补齐分母/已处理/失败数，避免显示成 0/0 却有 N 条结果。
                    # `or state.<field>` 兜底：扫描器没有统计时不要把已收集的正确值清零。
                    stats = getattr(self._scanner, "last_scan_stats", {}) or {}
                    if stats:
                        state.total = int(stats.get("target_count") or stats.get("universe_count", 0)) or state.total
                        state.processed = int(stats.get("attempted_count", 0)) or state.processed
                        state.failed = int(stats.get("failed_count", 0)) or state.failed
                        state.stale = int(stats.get("stale_count", 0)) or state.stale
                state.finished_at = time.time()
            if self._running_id == scan_id:
                self._running_id = None

    def _on_progress(self, scan_id: int, stats: dict[str, Any], hit: Any | None) -> None:
        with self._lock:
            state = self._jobs.get(scan_id)
            if state is None:
                return
            state.total = int(stats.get("target_count") or stats.get("universe_count", 0))
            state.processed = int(stats.get("attempted_count", 0))
            state.failed = int(stats.get("failed_count", 0))
            state.stale = int(stats.get("stale_count", 0))
            state.current_stock = str(stats.get("current_stock", ""))
            if hit is not None:
                state.results.append(hit.to_dict())
                state.hits = len(state.results)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            scan_id
            for scan_id, state in self._jobs.items()
            if state.finished_at is not None and now - state.finished_at > RETAIN_SECONDS
        ]
        for scan_id in expired:
            del self._jobs[scan_id]
