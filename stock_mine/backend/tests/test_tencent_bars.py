from __future__ import annotations

import json

import pandas as pd
import pytest

from app.services.tencent_bars import PAGE_SIZE, fetch_tencent_bars


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeTencent:
    """按腾讯接口的真实语义作答：返回区间内**最后** count 根。"""

    def __init__(self, bars: list[list[str]], symbol: str = "sz000001"):
        self.bars = bars
        self.symbol = symbol
        self.calls: list[dict[str, str]] = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        _sym, _period, start, end, count, _adj = params["param"].split(",")
        window = [row for row in self.bars if start <= row[0] <= end]
        window = window[-int(count):]
        body = {"code": 0, "data": {self.symbol: {"qfqday": window}}}
        return _FakeResponse(f"kline_dayqfq={json.dumps(body)}")


def _rows(n: int, start_year: int = 2000) -> list[list[str]]:
    dates = pd.bdate_range(f"{start_year}-01-03", periods=n)
    return [[d.strftime("%Y-%m-%d"), "10.0", "10.5", "10.8", "9.9", "1000"] for d in dates]


def test_one_page_is_enough_for_a_short_history() -> None:
    fake = _FakeTencent(_rows(120))
    got = fetch_tencent_bars("sz000001", get=fake)

    assert len(got) == 120
    assert len(fake.calls) == 1, f"一页装得下就不该翻页，实得 {len(fake.calls)} 次"


def test_long_history_is_paged_backwards_by_full_page_size() -> None:
    """akshare 按年翻页（每次只拿约 244 根），把 640 的容量浪费了 2.5 倍。"""
    total = PAGE_SIZE * 3 + 100
    fake = _FakeTencent(_rows(total, start_year=1995))

    got = fetch_tencent_bars("sz000001", get=fake)

    assert len(got) == total
    assert len(fake.calls) == 4, f"{total} 根应翻 4 页，实得 {len(fake.calls)}"
    assert all(call["param"].split(",")[4] == str(PAGE_SIZE) for call in fake.calls), \
        "每页都应按满额 640 请求"


def test_every_request_asks_for_forward_adjusted_prices() -> None:
    """前复权是全系统的口径，缺了会让所有百分比比较失真。"""
    fake = _FakeTencent(_rows(PAGE_SIZE * 2))
    fetch_tencent_bars("sz000001", get=fake)

    assert fake.calls, "未发出请求"
    for call in fake.calls:
        assert call["param"].endswith(",qfq"), f"未要求前复权：{call['param']}"


def test_result_is_sorted_by_date_without_duplicates() -> None:
    fake = _FakeTencent(_rows(PAGE_SIZE * 2 + 50))
    got = fetch_tencent_bars("sz000001", get=fake)

    dates = got["trade_date"].tolist()
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates), "翻页重叠处必须去重"


def test_columns_match_what_the_rest_of_the_system_expects() -> None:
    got = fetch_tencent_bars("sz000001", get=_FakeTencent(_rows(30)))
    assert list(got.columns) == ["trade_date", "open", "high", "low", "close", "volume"]
    assert got["close"].dtype.kind == "f", "价格必须是数值，不能留字符串"


def test_start_date_is_respected() -> None:
    fake = _FakeTencent(_rows(400, start_year=2000))
    got = fetch_tencent_bars("sz000001", start_date="20010101", get=fake)

    assert got["trade_date"].min() >= "2001-01-01"


def test_empty_payload_yields_an_empty_frame_not_a_crash() -> None:
    got = fetch_tencent_bars("sz000001", get=_FakeTencent([]))
    assert got.empty
    assert list(got.columns) == ["trade_date", "open", "high", "low", "close", "volume"]


def test_a_failing_request_propagates_so_the_caller_can_fall_back() -> None:
    def boom(url, params=None, timeout=None):
        raise RuntimeError("网络炸了")

    with pytest.raises(RuntimeError):
        fetch_tencent_bars("sz000001", get=boom)
