from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from app.services.bars_cache import BarsCache

# 用例默认取收盘后的时刻——那是数据定稿的时段
_AFTER_CLOSE = __import__('datetime').time(16, 0)


def _bars(start: str, days: int, base: float = 10.0) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=days)
    return pd.DataFrame({
        "trade_date": dates.strftime("%Y-%m-%d"),
        "open": [base + i * 0.1 for i in range(days)],
        "high": [base + i * 0.1 + 0.2 for i in range(days)],
        "low": [base + i * 0.1 - 0.2 for i in range(days)],
        "close": [base + i * 0.1 for i in range(days)],
        "volume": [1000 + i for i in range(days)],
    })


class _Fetcher:
    """记录每次取数的起始日，便于断言"只取了增量"。"""

    def __init__(self, full: pd.DataFrame):
        self.full = full
        self.calls: list[str] = []

    def __call__(self, code: str, start_date: str) -> pd.DataFrame:
        # 真实的 akshare 行情接口收的是紧凑日期（YYYYMMDD），替身要照此约定
        self.calls.append(start_date)
        iso = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        return self.full[self.full["trade_date"] >= iso].reset_index(drop=True)


def test_first_fetch_goes_to_network_and_lands_on_disk(tmp_path) -> None:
    full = _bars("2026-01-01", 40)
    fetcher = _Fetcher(full)
    cache = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(date(2026, 2, 25), _AFTER_CLOSE))

    got = cache.get("000001")

    assert len(got) == 40
    assert len(fetcher.calls) == 1
    assert (tmp_path / "000001.json").exists()


def test_cache_hit_serves_without_fetching(tmp_path) -> None:
    full = _bars("2026-01-01", 40)
    last_day = full["trade_date"].iloc[-1]
    fetcher = _Fetcher(full)
    today = date.fromisoformat(last_day)
    cache = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(today, _AFTER_CLOSE))

    cache.get("000001")
    fetcher.calls.clear()
    again = cache.get("000001")

    assert fetcher.calls == [], "缓存已到最新交易日，不该再取数"
    assert len(again) == 40


def test_stale_cache_only_fetches_the_increment(tmp_path) -> None:
    """这是提速的关键：隔天再扫只取新增的几根，而不是重拉 33 年。"""
    full = _bars("2026-01-01", 60)
    fetcher = _Fetcher(full)
    cache = BarsCache(tmp_path, _Fetcher(full.iloc[:50].reset_index(drop=True)),
                      now=lambda: datetime.combine(date.fromisoformat(full["trade_date"].iloc[49]), _AFTER_CLOSE))
    cache.get("000001")

    cache = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(date.fromisoformat(full["trade_date"].iloc[-1]), _AFTER_CLOSE))
    merged = cache.get("000001")

    assert len(merged) == 60, f"应补齐到 60 根，实得 {len(merged)}"
    assert merged["trade_date"].is_unique, "拼接不能产生重复交易日"
    assert fetcher.calls, "应发生一次增量取数"
    assert fetcher.calls[0] != "19900101", f"增量取数不该从头开始，实得 {fetcher.calls}"


def test_dividend_rescaling_is_detected_and_forces_a_full_refetch(tmp_path) -> None:
    """前复权价格在除权日会被整体重算。

    若直接把新K线拼到旧缓存后面，旧段仍是老的复权基准，接缝处价格会突变——
    后面所有的精准线、涨跌幅都会算错。必须靠重叠段比对发现并重新全量取数。
    """
    old = _bars("2026-01-01", 50, base=10.0)
    cache = BarsCache(tmp_path, _Fetcher(old),
                      now=lambda: datetime.combine(date.fromisoformat(old["trade_date"].iloc[-1]), _AFTER_CLOSE))
    cache.get("000001")

    # 除权后：同样的交易日，价格整体缩放到 0.8
    rescaled = _bars("2026-01-01", 60, base=8.0)
    fetcher = _Fetcher(rescaled)
    cache = BarsCache(tmp_path, fetcher,
                      now=lambda: datetime.combine(date.fromisoformat(rescaled["trade_date"].iloc[-1]), _AFTER_CLOSE))
    merged = cache.get("000001")

    assert len(merged) == 60
    assert fetcher.calls[-1] == "19900101", f"应回退到全量取数，实得 {fetcher.calls}"
    first_close = float(merged["close"].iloc[0])
    assert first_close == pytest.approx(8.0), f"整段都应是新基准，实得 {first_close}"


def test_incremental_fetch_failure_falls_back_to_the_cached_bars(tmp_path) -> None:
    """增量取数失败不该让整只股票分析不了——旧数据比没有数据强。"""
    full = _bars("2026-01-01", 40)
    cache = BarsCache(tmp_path, _Fetcher(full),
                      now=lambda: datetime.combine(date.fromisoformat(full["trade_date"].iloc[-1]), _AFTER_CLOSE))
    cache.get("000001")

    def boom(code, start_date):
        raise RuntimeError("网络炸了")

    later = date.fromisoformat(full["trade_date"].iloc[-1]) + timedelta(days=7)
    cache = BarsCache(tmp_path, boom, now=lambda: datetime.combine(later, _AFTER_CLOSE))

    got = cache.get("000001")
    assert len(got) == 40
    assert got.attrs["data_quality"]["status"] == "stale"
    assert got.attrs["data_quality"]["last_bar_date"] == full["trade_date"].iloc[-1]
    assert got.attrs["data_quality"]["error"]


def test_missing_cache_and_failing_network_returns_empty_not_an_exception(tmp_path) -> None:
    def boom(code, start_date):
        raise RuntimeError("网络炸了")

    got = BarsCache(tmp_path, boom, now=lambda: datetime.combine(date(2026, 2, 25), _AFTER_CLOSE)).get("000001")
    assert got is not None and got.empty


def test_second_run_on_the_same_day_does_not_re_ask_the_network(tmp_path) -> None:
    """同一天问过一次没有新数据，再问还是没有——不该每次扫描都再发一轮请求。

    非交易日或收盘前，缓存的最后一根永远追不上"今天"，若不记录"今天已问过"，
    每只股票每轮都要白跑一次增量取数（实测东财不通时单只要 6.4 秒）。
    """
    full = _bars("2026-01-01", 40)
    last_day = full["trade_date"].iloc[-1]
    later = date.fromisoformat(last_day) + timedelta(days=3)   # 数据停在上个交易日

    fetcher = _Fetcher(full)
    cache = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(later, _AFTER_CLOSE))
    cache.get("000001")
    first_round = len(fetcher.calls)
    assert first_round >= 1, "第一次仍应尝试增量"

    fetcher.calls.clear()
    again = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(later, _AFTER_CLOSE)).get("000001")

    assert fetcher.calls == [], "当天已确认过没有新数据，不该再取"
    assert len(again) == 40


def test_a_new_day_does_trigger_a_fresh_check(tmp_path) -> None:
    """但换了一天必须重新问，否则永远拿不到新K线。"""
    full = _bars("2026-01-01", 41)
    cache = BarsCache(tmp_path, _Fetcher(full.iloc[:40].reset_index(drop=True)),
                      now=lambda: datetime.combine(date.fromisoformat(full["trade_date"].iloc[39]), _AFTER_CLOSE) + timedelta(days=1))
    cache.get("000001")

    fetcher = _Fetcher(full)
    tomorrow = date.fromisoformat(full["trade_date"].iloc[-1]) + timedelta(days=1)
    got = BarsCache(tmp_path, fetcher, now=lambda: datetime.combine(tomorrow, _AFTER_CLOSE)).get("000001")

    assert fetcher.calls, "新的一天必须重新检查"
    assert len(got) == 41


# ---- 盘中数据不能当定稿缓存 ----

def _dt(y, m, d, hh, mm):
    from datetime import datetime
    return datetime(y, m, d, hh, mm)


def test_an_intraday_check_is_not_treated_as_final(tmp_path) -> None:
    """上午取过一次就锁死当天，晚上会拿到盘中价当收盘价。

    只记日期不记时间时，`checked_at >= today` 会把盘中那次检查当成"今天已确认"，
    收盘后的真实收盘价再也取不回来——signal_offset=0 与回测用的正是最新一根。
    """
    full = _bars("2026-01-01", 11)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    intraday = full.copy()
    intraday.loc[intraday.index[-1], "close"] = 11.0     # 盘中
    final = full.copy()
    final.loc[final.index[-1], "close"] = 12.5           # 收盘后

    seq = [intraday, final]
    calls = []

    def fetch(code, start_date):
        calls.append(start_date)
        return seq[min(len(calls) - 1, 1)]

    morning = BarsCache(tmp_path, fetch, now=lambda: _dt(day.year, day.month, day.day, 10, 30))
    assert float(morning.get("000001")["close"].iloc[-1]) == 11.0

    evening = BarsCache(tmp_path, fetch, now=lambda: _dt(day.year, day.month, day.day, 20, 0))
    assert float(evening.get("000001")["close"].iloc[-1]) == 12.5, "收盘后必须重新确认"


def test_a_post_close_check_is_final_and_not_repeated(tmp_path) -> None:
    """收盘之后确认过的当天数据才是定稿，同一天不必再问。"""
    full = _bars("2026-01-01", 11)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    fetcher = _Fetcher(full)

    BarsCache(tmp_path, fetcher, now=lambda: _dt(day.year, day.month, day.day, 16, 0)).get("000001")
    fetcher.calls.clear()
    BarsCache(tmp_path, fetcher, now=lambda: _dt(day.year, day.month, day.day, 22, 0)).get("000001")

    assert fetcher.calls == [], "收盘后已确认过，同一天不该再取"


def test_repeated_intraday_calls_are_throttled_not_hammered(tmp_path) -> None:
    """盘中数据是临时的，必须重查；但一分钟内连问两次没有意义。"""
    full = _bars("2026-01-01", 11)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    fetcher = _Fetcher(full)

    BarsCache(tmp_path, fetcher, now=lambda: _dt(day.year, day.month, day.day, 10, 0)).get("000001")
    fetcher.calls.clear()
    BarsCache(tmp_path, fetcher, now=lambda: _dt(day.year, day.month, day.day, 10, 5)).get("000001")
    assert fetcher.calls == [], "5 分钟内不重复取"

    BarsCache(tmp_path, fetcher, now=lambda: _dt(day.year, day.month, day.day, 11, 30)).get("000001")
    assert fetcher.calls, "隔了足够久应重新取"


def test_a_failed_fetch_is_not_recorded_as_settled(tmp_path) -> None:
    """"请求成功但没有新K线"和"数据源全挂了"必须区分开。

    provider 在东财、腾讯都失败时返回空表；若把空表当成"今天没有新数据"并写入
    收盘后时间戳，当天就再也不会重试——晚上那次筛选会一直用着昨天的数据。
    """
    full = _bars("2026-01-01", 10)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    calls = []
    served = {"n": 0}          # 独立计数：calls 会被 clear，不能用它判断第几次

    def flaky(code, start_date):
        calls.append(start_date)
        served["n"] += 1
        if served["n"] == 1:
            return full
        raise RuntimeError("两个数据源都连不上")

    BarsCache(tmp_path, flaky, now=lambda: _dt(day.year, day.month, day.day, 10, 0)).get("000001")
    calls.clear()

    later = day + timedelta(days=1)
    BarsCache(tmp_path, flaky, now=lambda: _dt(later.year, later.month, later.day, 16, 0)).get("000001")
    assert len(calls) == 1, "收盘后应尝试一次"
    calls.clear()

    BarsCache(tmp_path, flaky, now=lambda: _dt(later.year, later.month, later.day, 19, 0)).get("000001")
    assert calls, "上一次是取数失败，不能算定稿，同一天必须再试"


def test_an_empty_successful_response_on_a_trading_day_does_not_settle_missing_data(tmp_path) -> None:
    """空响应不能证明下一交易日已经定稿。"""
    full = _bars("2026-01-01", 10)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    calls = []
    served = {"n": 0}

    def no_new_data(code, start_date):
        calls.append(start_date)
        served["n"] += 1
        return full if served["n"] == 1 else full.iloc[0:0]

    BarsCache(tmp_path, no_new_data, now=lambda: _dt(day.year, day.month, day.day, 10, 0)).get("000001")
    calls.clear()
    later = day + timedelta(days=1)
    BarsCache(tmp_path, no_new_data, now=lambda: _dt(later.year, later.month, later.day, 16, 0)).get("000001")
    calls.clear()
    BarsCache(tmp_path, no_new_data, now=lambda: _dt(later.year, later.month, later.day, 19, 0)).get("000001")

    assert calls, "最新完整交易日缺失，同一天仍须重试"


def test_provider_that_swallows_errors_is_still_treated_as_a_failure(tmp_path) -> None:
    """真实 provider 的 `_fetch_with_retry` 会吞掉异常、返回空表。

    缓存层若只看"是否抛异常"，这条集成路径上的失败仍会被当成"今天没有新K线"
    并盖上定稿时间戳——上一版测试只覆盖了直接抛异常的情形，漏掉了这个。
    取数方要能告诉缓存"这次是失败还是确实没有新数据"。
    """
    from app.services.bars_cache import FetchFailed

    full = _bars("2026-01-01", 10)
    day = date.fromisoformat(full["trade_date"].iloc[-1])
    calls = []
    served = {"n": 0}

    def swallowing(code, start_date):
        calls.append(start_date)
        served["n"] += 1
        if served["n"] == 1:
            return full
        raise FetchFailed("两个数据源都失败了")   # provider 侧翻译过的失败信号

    BarsCache(tmp_path, swallowing, now=lambda: _dt(day.year, day.month, day.day, 10, 0)).get("000001")
    calls.clear()
    later = day + timedelta(days=1)
    BarsCache(tmp_path, swallowing, now=lambda: _dt(later.year, later.month, later.day, 16, 0)).get("000001")
    calls.clear()
    BarsCache(tmp_path, swallowing, now=lambda: _dt(later.year, later.month, later.day, 19, 0)).get("000001")

    assert calls, "取数失败不能算定稿，同一天必须再试"


def test_empty_rescale_refetch_preserves_history_as_stale(tmp_path) -> None:
    old = _bars("2026-01-01", 40)
    day = date.fromisoformat(old.trade_date.iloc[-1])
    BarsCache(tmp_path, _Fetcher(old), now=lambda: datetime.combine(day, _AFTER_CLOSE)).get("000001")
    changed = old.copy()
    changed["close"] *= .8

    def fetch(code, start):
        return changed.iloc[0:0] if start == "19900101" else changed

    got = BarsCache(
        tmp_path, fetch, now=lambda: datetime.combine(day + timedelta(days=1), _AFTER_CLOSE),
    ).get("000001")
    assert got.close.tolist() == old.close.tolist()
    assert got.attrs["data_quality"]["status"] == "stale"


def test_successful_recovery_clears_stale_status(tmp_path) -> None:
    full = _bars("2026-01-01", 40)
    day = date.fromisoformat(full.trade_date.iloc[-1])
    clock = lambda: datetime.combine(day, _AFTER_CLOSE)
    first = BarsCache(tmp_path, _Fetcher(full), now=clock).get("000001")
    cached = BarsCache(tmp_path, _Fetcher(full), now=clock).get("000001")
    assert first.attrs["data_quality"]["status"] == cached.attrs["data_quality"]["status"] == "fresh"
    assert cached.attrs["data_quality"]["checked_at"] == clock().isoformat(timespec="seconds")


def test_delayed_post_close_quotes_are_not_fresh_and_do_not_settle_the_whole_day(tmp_path):
    from app.services.scanner import _require_updated_bars, StaleMarketData
    old = _bars("2026-09-08", 2)
    new = _bars("2026-09-08", 3)
    clock = [datetime(2026, 9, 9, 16)]
    calls = []

    def fetch(code, start):
        calls.append(clock[0])
        return old if len(calls) == 1 else new

    cache = BarsCache(tmp_path, fetch, now=lambda: clock[0])
    cache._save("000001", old)
    clock[0] = datetime(2026, 9, 10, 15, 6)
    delayed = cache.get("000001")
    assert delayed.attrs["data_quality"]["status"] == "stale"
    with pytest.raises(StaleMarketData):
        _require_updated_bars(delayed)
    clock[0] = datetime(2026, 9, 10, 18)
    recovered = cache.get("000001")
    assert len(calls) == 2
    assert recovered.attrs["data_quality"]["status"] == "fresh"
    assert recovered.trade_date.iloc[-1] == "2026-09-10"


def test_empty_holiday_response_can_reuse_already_completed_preholiday_prices(tmp_path):
    frame = _bars("2026-02-12", 2)
    cache = BarsCache(tmp_path, _Fetcher(frame), now=lambda: datetime(2026, 2, 13, 16))
    assert cache.get("000001").attrs["data_quality"]["status"] == "fresh"
    calls = []

    def empty(code, start):
        calls.append(code)
        return frame.iloc[0:0]

    holiday = BarsCache(tmp_path, empty, now=lambda: datetime(2026, 2, 23, 16)).get("000001")
    assert holiday.attrs["data_quality"]["status"] == "fresh"
    assert holiday.attrs["data_quality"]["expected_bar_date"] == "2026-02-13"
    assert not calls
