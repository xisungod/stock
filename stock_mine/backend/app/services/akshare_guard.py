from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

# Some AkShare endpoints initialize libmini_racer internally. Its native runtime is
# not safe to initialize or use concurrently from FastAPI/scanner worker threads.
AKSHARE_LOCK = threading.RLock()


def get_akshare() -> Any:
    with AKSHARE_LOCK:
        import akshare as ak  # type: ignore

        return ak


def call_akshare(call: Callable[[], T]) -> T:
    with AKSHARE_LOCK:
        return call()


def call_akshare_parallel(call: Callable[[], T]) -> T:
    """行情/历史类接口的并发路径：不加全局锁。

    锁存在的理由是某些 AkShare 接口内部会初始化 libmini_racer，它的原生运行时
    不能并发初始化。但日线/历史接口只做 HTTP 请求和 pandas 组装，碰不到它。

    实测（2026-08-28，4 线程拉腾讯日线）：带锁加速比 1.21x —— SCAN_WORKERS=4
    被压成实际并发 1；去锁后 2.16x，8 只股票数据全部正确。

    调用前必须先经 `get_akshare()` 完成过一次锁内初始化。
    """
    return call()
