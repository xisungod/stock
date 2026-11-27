from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.akshare_guard import call_akshare


class ConcurrentProbeClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def fetch(self, value: int) -> int:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return value
        finally:
            with self.lock:
                self.active -= 1


def test_akshare_calls_are_serialized_across_worker_threads() -> None:
    client = ConcurrentProbeClient()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda value: call_akshare(lambda: client.fetch(value)),
                range(16),
            )
        )

    assert results == list(range(16))
    assert client.max_active == 1
