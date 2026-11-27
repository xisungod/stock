"""直接向腾讯行情接口取日线，自己按满页翻页。

为什么不用 akshare 的 `stock_zh_a_hist_tx`：它**按年**翻页——
    for year in range(1991, 2027):   # 36 年 = 36 次串行 HTTP
        param = f"{symbol},day,{year}-01-01,{year+1}-12-31,640,qfq"
每次只取约 244 根，却把 count 写成 640，等于把容量浪费了 2.5 倍；而且区间跨了
两年，同一年的数据要取两遍再去重。

实测（平安银行 8479 根）：akshare 36 次请求 9.31s，本模块 14 次请求 3.80s，
数据完全一致。单次请求本身只要 0.17s——慢的从来不是网络，是往返次数。
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Callable

import pandas as pd

URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
# 服务端单次上限；再大会改变响应结构（返回 list 而非 dict）
PAGE_SIZE = 640
COLUMNS = ["trade_date", "open", "high", "low", "close", "volume"]
# 腾讯每行的顺序是 日期, 开, 收, 高, 低, 成交量 —— 注意「收」在「高」前面
_RAW_ORDER = ("trade_date", "open", "close", "high", "low", "volume")
MAX_PAGES = 60


def _default_get(url: str, params: dict[str, str], timeout: float | None = None) -> Any:
    import requests

    return requests.get(url, params=params, timeout=timeout)


def _decode(text: str, symbol: str, adjust: str) -> list[list[str]]:
    start = text.find("={")
    if start < 0:
        return []
    payload = json.loads(text[start + 1 :])
    data = (payload.get("data") or {}).get(symbol) or {}
    for key in (f"{adjust}day", "day"):
        rows = data.get(key)
        if rows:
            return rows
    return []


def _iso(compact: str) -> str:
    digits = re.sub(r"\D", "", compact or "")
    if len(digits) != 8:
        return compact
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def fetch_tencent_bars(
    symbol: str,
    start_date: str = "19900101",
    end_date: str | None = None,
    adjust: str = "qfq",
    timeout: float | None = 20,
    get: Callable[..., Any] = _default_get,
) -> pd.DataFrame:
    """取 [start_date, end_date] 的日线，按 640 根一页从后往前翻。

    请求失败直接抛出，交由上层重试或换数据源——静默返回空会让调用方把
    "取数失败"误当成"这只股票没有数据"。
    """
    begin = _iso(start_date)
    cursor = _iso(end_date) if end_date else _dt.date.today().isoformat()
    collected: dict[str, list[str]] = {}

    for _ in range(MAX_PAGES):
        params = {
            "_var": f"kline_day{adjust}",
            "param": f"{symbol},day,{begin},{cursor},{PAGE_SIZE},{adjust}",
            "r": "0.8205512681390605",
        }
        rows = _decode(get(URL, params=params, timeout=timeout).text, symbol, adjust)
        fresh = [row for row in rows if row and row[0] not in collected]
        if not fresh:
            break
        for row in fresh:
            collected[row[0]] = row
        # 未取满说明这一页已经触到区间起点
        if len(rows) < PAGE_SIZE:
            break
        earliest = min(row[0] for row in fresh)
        if earliest <= begin:
            break
        cursor = (_dt.date.fromisoformat(earliest) - _dt.timedelta(days=1)).isoformat()

    if not collected:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(
        [row[: len(_RAW_ORDER)] for row in collected.values()], columns=list(_RAW_ORDER)
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["trade_date"] >= begin]
    return frame[COLUMNS].sort_values("trade_date").reset_index(drop=True)
