from __future__ import annotations

import threading
import time
from pathlib import Path

from app.db import Database
from app.services.job_manager import ScanJobManager
from app.services.strategy import SignalCategory, SignalResult


class StubScanner:
    provider_name = "fake"

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.last_provider_name = "fake"
        self.last_scan_stats = {
            "universe_count": 2,
            "target_count": 2,
            "attempted_count": 2,
            "analyzed_count": 2,
            "failed_count": 0,
            "current_stock": "000002 万科A",
        }

    def run_scan(self, params, on_progress=None, should_stop=None):
        if self.fail:
            raise RuntimeError("数据源不可用")
        time.sleep(self.delay)
        result = SignalResult(
            stock_code="000001",
            stock_name="平安银行",
            category=SignalCategory.BUY_ALERT,
            score=88.0,
            suggestion="买入预警：测试",
            key_line_price=10.1,
            line_strength=4,
            signal_date="2026-08-03",
        )
        if on_progress is not None:
            on_progress(dict(self.last_scan_stats), result)
        return [result]


class SilentScanner:
    """只给出最终统计、全程不回调 on_progress 的扫描器。"""

    provider_name = "fake"

    def __init__(self) -> None:
        self.last_provider_name = "fake"
        self.last_scan_stats = {
            "universe_count": 4997,
            "target_count": 80,
            "attempted_count": 80,
            "analyzed_count": 78,
            "failed_count": 2,
            "current_stock": "",
        }

    def run_scan(self, params, on_progress=None, should_stop=None):
        return [
            SignalResult(
                stock_code="000001",
                stock_name="平安银行",
                category=SignalCategory.BUY_ALERT,
                score=88.0,
                suggestion="买入预警：测试",
                key_line_price=10.1,
                line_strength=4,
                signal_date="2026-08-03",
            )
        ]


def wait_until_done(manager: ScanJobManager, scan_id: int, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = manager.get(scan_id)
        if state and state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.02)
    raise AssertionError("任务未在超时时间内结束")


def test_submit_runs_in_background_and_completes(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=StubScanner())

    scan_id = manager.submit({"provider": "fake", "limit": 2, "signal_offset": 1})
    state = wait_until_done(manager, scan_id)

    assert state["status"] == "completed"
    assert state["processed"] == 2
    assert state["hits"] == 1
    assert state["results"][0]["stock_code"] == "000001"


def test_duplicate_submit_returns_running_job(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=StubScanner(delay=0.3))

    first = manager.submit({"provider": "fake", "limit": 2, "signal_offset": 1})
    second = manager.submit({"provider": "fake", "limit": 2, "signal_offset": 1})

    assert first == second
    wait_until_done(manager, first)


def test_failed_scan_reports_error(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=StubScanner(fail=True))

    scan_id = manager.submit({"provider": "fake", "limit": 2, "signal_offset": 1})
    state = wait_until_done(manager, scan_id)

    assert state["status"] == "failed"
    assert "数据源不可用" in state["error"]


def test_progress_total_uses_target_count_while_running(tmp_path: Path) -> None:
    """飞行中的进度分母必须是 target_count，不是 universe_count。

    这是用户可见的核心 bug：limit=80 时进度会显示 80/4997，永远到不了 100%。
    必须在扫描运行中断言——完成时 _finish 的回填会把这个错误掩盖掉。
    """
    reported = threading.Event()
    release = threading.Event()

    class BlockingScanner:
        provider_name = "fake"
        last_provider_name = "fake"

        def __init__(self) -> None:
            self.last_scan_stats = {
                "universe_count": 5000,
                "target_count": 3,
                "attempted_count": 1,
                "analyzed_count": 1,
                "failed_count": 0,
                "current_stock": "000001 测试",
            }

        def run_scan(self, params, on_progress=None, should_stop=None):
            if on_progress is not None:
                on_progress(dict(self.last_scan_stats), None)
            reported.set()
            release.wait(timeout=5.0)
            return []

    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=BlockingScanner())

    scan_id = manager.submit({"provider": "fake", "limit": 3, "signal_offset": 1})
    assert reported.wait(timeout=5.0), "扫描器未上报进度"

    state = manager.get(scan_id)
    assert state["status"] == "running"
    assert state["total"] == 3, "运行中的分母应是 target_count(3)，不是 universe_count(5000)"

    release.set()
    wait_until_done(manager, scan_id)


def test_finish_backfills_counters_when_scanner_never_reports_progress(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=SilentScanner())

    scan_id = manager.submit({"provider": "fake", "limit": 80, "signal_offset": 1})
    state = wait_until_done(manager, scan_id)

    assert state["status"] == "completed"
    assert state["total"] == 80, "完成后分母不能还是 0"
    assert state["processed"] == 80
    assert state["failed"] == 2
    assert state["hits"] == 1


def test_active_returns_none_when_idle(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    manager = ScanJobManager(db=db, scanner=StubScanner())

    assert manager.active() is None

    scan_id = manager.submit({"provider": "fake", "limit": 2, "signal_offset": 1})
    wait_until_done(manager, scan_id)

    assert manager.active() is None


class SlowScanner:
    """按 should_stop 逐只让路的扫描器，用来验证取消。"""

    def __init__(self, total: int = 50):
        self.total = total
        self.analyzed = 0
        self.last_scan_stats: dict[str, int] = {}
        self.last_provider_name = "fake"

    def run_scan(self, params, on_progress=None, should_stop=None):
        results = []
        for i in range(self.total):
            if should_stop is not None and should_stop():
                break
            self.analyzed += 1
            hit = SignalResult(
                stock_code=f"{i:06d}", stock_name=f"股票{i}",
                category=SignalCategory.BUY_ALERT, score=80.0,
                suggestion="测试", key_line_price=10.0, line_strength=3,
                signal_date="2026-08-31",
            )
            results.append(hit)
            if on_progress:
                on_progress({"target_count": self.total, "attempted_count": i + 1, "failed_count": 0,
                             "current_stock": f"{i:06d}"}, hit)
            time.sleep(0.01)
        self.last_scan_stats = {"target_count": self.total, "attempted_count": self.analyzed,
                                "analyzed_count": self.analyzed, "failed_count": 0}
        return results


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_cancel_stops_the_scan_early(tmp_path: Path) -> None:
    db = Database(tmp_path / "s.db")
    scanner = SlowScanner(total=200)
    manager = ScanJobManager(db=db, scanner=scanner)

    scan_id = manager.submit({"provider": "fake"})
    assert _wait_until(lambda: (manager.get(scan_id) or {}).get("processed", 0) >= 3)
    manager.cancel(scan_id)

    assert _wait_until(lambda: (manager.get(scan_id) or {}).get("status") != "running")
    assert manager.get(scan_id)["status"] == "cancelled"
    assert scanner.analyzed < 200, f"取消后不该扫完全部，实得 {scanner.analyzed}"


def test_cancelled_scan_keeps_the_stocks_it_already_found(tmp_path: Path) -> None:
    """中途停止不该白跑——已命中的结果要落库，页面刷新后还看得见。"""
    db = Database(tmp_path / "s.db")
    scanner = SlowScanner(total=200)
    manager = ScanJobManager(db=db, scanner=scanner)

    scan_id = manager.submit({"provider": "fake"})
    assert _wait_until(lambda: (manager.get(scan_id) or {}).get("processed", 0) >= 3)
    manager.cancel(scan_id)
    assert _wait_until(lambda: (manager.get(scan_id) or {}).get("status") == "cancelled")

    saved = db.get_scan_results(scan_id)
    assert saved, "已命中的结果必须落库"
    latest = db.latest_scan()
    assert latest is not None and latest["id"] == scan_id, "页面应能看到这批部分结果"
    assert latest["status"] == "cancelled", "状态要如实标成已取消，而不是伪装成完成"


def test_cancel_releases_the_slot_so_a_new_scan_can_start(tmp_path: Path) -> None:
    db = Database(tmp_path / "s.db")
    manager = ScanJobManager(db=db, scanner=SlowScanner(total=200))

    first = manager.submit({"provider": "fake"})
    assert _wait_until(lambda: (manager.get(first) or {}).get("processed", 0) >= 2)
    manager.cancel(first)
    assert _wait_until(lambda: (manager.get(first) or {}).get("status") == "cancelled")

    second = manager.submit({"provider": "fake"})
    assert second != first, "取消后必须能提交新扫描，而不是被判为'已有任务在跑'"


def test_cancelling_an_unknown_or_finished_scan_is_harmless(tmp_path: Path) -> None:
    manager = ScanJobManager(db=Database(tmp_path / "s.db"), scanner=SlowScanner(total=1))
    assert manager.cancel(99999) is False
