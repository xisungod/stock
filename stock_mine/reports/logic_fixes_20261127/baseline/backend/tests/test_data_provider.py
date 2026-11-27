from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.data_provider import AkshareProvider, AutoProvider


def test_auto_provider_prefers_fresh_fallback_over_stale_primary():
    from types import SimpleNamespace
    from app.services.bars_cache import with_data_quality
    bars = pd.DataFrame({
        "trade_date": ["2026-09-04"], "open": [10], "high": [11],
        "low": [9], "close": [10.5], "volume": [1000],
    })
    stale = with_data_quality(bars, "2026-09-03", "offline")
    fresh = with_data_quality(bars, "2026-09-05")
    provider = AutoProvider(
        adata=SimpleNamespace(name="adata", get_daily_bars=lambda code: stale),
        akshare=SimpleNamespace(name="akshare", get_daily_bars=lambda code: fresh),
    )
    assert provider.get_daily_bars("000001").attrs["data_quality"]["status"] == "fresh"
    provider.akshare.get_daily_bars = lambda code: pd.DataFrame()
    assert provider.get_daily_bars("000001").attrs["data_quality"]["status"] == "stale"


class FallbackAkClient:
    def stock_info_a_code_name(self):
        raise RuntimeError("primary unavailable")

    def stock_zh_a_spot_em(self):
        return pd.DataFrame(
            [
                {"代码": "000001", "名称": "平安银行"},
                {"代码": "600000", "名称": "浦发银行"},
                {"代码": "000001", "名称": "平安银行"},
                {"代码": "000003", "名称": "测试退"},
            ]
        )


class OfflineAkClient:
    def stock_info_a_code_name(self):
        raise RuntimeError("primary unavailable")

    def stock_zh_a_spot_em(self):
        raise RuntimeError("fallback unavailable")


def test_akshare_stock_list_uses_secondary_source_and_caches_clean_data(tmp_path: Path) -> None:
    cache_path = tmp_path / "stocks.json"
    provider = AkshareProvider(cache_path=cache_path, legacy_cache_path=tmp_path / "missing.json")
    provider._ak = FallbackAkClient()

    result = provider.get_stock_list()

    assert result[["stock_code", "exchange"]].to_dict("records") == [
        {"stock_code": "000001", "exchange": "SZ"},
        {"stock_code": "600000", "exchange": "SH"},
    ]
    assert cache_path.exists()


def test_akshare_stock_list_uses_local_cache_when_all_remote_sources_fail(tmp_path: Path) -> None:
    cache_path = tmp_path / "stocks.json"
    cache_path.write_text(
        json.dumps([{"stock_code": "000001", "short_name": "平安银行", "exchange": "SZ"}], ensure_ascii=False),
        encoding="utf-8",
    )
    provider = AkshareProvider(cache_path=cache_path, legacy_cache_path=tmp_path / "missing.json")
    provider._ak = OfflineAkClient()

    result = provider.get_stock_list()

    assert result.iloc[0]["stock_code"] == "000001"


def test_akshare_provider_retries_until_success(monkeypatch) -> None:
    from app.services import data_provider as dp

    calls = {"n": 0}

    class FlakyAk:
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("ProxyError")
            return pd.DataFrame(
                {
                    "日期": ["2026-01-01"],
                    "开盘": [10.0],
                    "最高": [10.5],
                    "最低": [9.8],
                    "收盘": [10.2],
                    "成交量": [1000],
                }
            )

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    provider = dp.AkshareProvider(bars_cache_dir=None)
    provider._ak = FlakyAk()

    df = provider.get_daily_bars("000001")

    assert calls["n"] == 3
    assert len(df) == 1
    assert list(df.columns) == ["trade_date", "open", "high", "low", "close", "volume"]


def test_akshare_provider_gives_up_after_max_attempts(monkeypatch) -> None:
    from app.services import data_provider as dp

    calls = {"n": 0}

    class AlwaysFailing:
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            calls["n"] += 1
            raise RuntimeError("ProxyError")

    def failing_tencent(symbol, start_date, end_date, adjust):
        raise RuntimeError("ProxyError")

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    provider = dp.AkshareProvider(bars_cache_dir=None, tencent_fetch=failing_tencent)
    provider._ak = AlwaysFailing()

    df = provider.get_daily_bars("000001")

    assert calls["n"] == dp.FETCH_MAX_ATTEMPTS
    assert df.empty, "get_daily_bars 对扫描循环的契约：失败返回空表，不抛异常"


def test_akshare_history_falls_back_to_tencent_and_opens_circuit(monkeypatch) -> None:
    """东方财富连续断连后，本轮后续股票应直接走腾讯，避免继续触发限流。"""
    from app.services import data_provider as dp

    calls = {"eastmoney": 0, "tencent": 0}

    class EastmoneyBlocked:
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            calls["eastmoney"] += 1
            raise ConnectionError("RemoteDisconnected")

    # 腾讯路径已换成自研的满页翻页模块，桩打在注入点上
    def fake_tencent(symbol, start_date, end_date, adjust):
        calls["tencent"] += 1
        assert symbol == "sz000001"
        assert start_date == "19900101"
        assert adjust == "qfq"
        return pd.DataFrame(
            {
                "trade_date": ["2026-08-06"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
            }
        )

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    provider = dp.AkshareProvider(bars_cache_dir=None, tencent_fetch=fake_tencent)
    provider._ak = EastmoneyBlocked()

    first = provider.get_daily_bars("000001")
    second = provider.get_daily_bars("000001")

    assert calls == {"eastmoney": dp.FETCH_MAX_ATTEMPTS, "tencent": 2}
    assert list(first.columns) == ["trade_date", "open", "high", "low", "close", "volume"]
    assert first.iloc[0]["volume"] == 1000
    assert second.equals(first)


def test_adata_provider_retries_until_success(monkeypatch) -> None:
    """adata 与 akshare 一样需要重试——auto 模式会优先选它，不能裸奔。"""
    from app.services import data_provider as dp

    calls = {"n": 0}

    class FlakyAdata:
        class stock:
            class market:
                @staticmethod
                def get_market(stock_code=None, k_type=1, adjust_type=1):
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise RuntimeError("ProxyError")
                    return pd.DataFrame(
                        {
                            "trade_date": ["2026-01-01"],
                            "open": [10.0],
                            "high": [10.5],
                            "low": [9.8],
                            "close": [10.2],
                            "volume": [1000],
                        }
                    )

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    # 绕过 __post_init__ 的 import adata：本用例测的是重试逻辑，与 adata 是否安装无关
    provider = dp.ADataProvider.__new__(dp.ADataProvider)
    provider._adata = FlakyAdata()

    df = provider.get_daily_bars("000001")

    assert calls["n"] == 3
    assert len(df) == 1
    assert list(df.columns) == ["trade_date", "open", "high", "low", "close", "volume"]


def test_adata_provider_gives_up_after_max_attempts(monkeypatch) -> None:
    from app.services import data_provider as dp

    calls = {"n": 0}

    class AlwaysFailing:
        class stock:
            class market:
                @staticmethod
                def get_market(stock_code=None, k_type=1, adjust_type=1):
                    calls["n"] += 1
                    raise RuntimeError("ProxyError")

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    provider = dp.ADataProvider.__new__(dp.ADataProvider)
    provider._adata = AlwaysFailing()

    df = provider.get_daily_bars("000001")

    assert calls["n"] == dp.FETCH_MAX_ATTEMPTS
    assert df.empty


def test_all_providers_use_forward_adjusted_prices(monkeypatch) -> None:
    """所有取数路径必须统一使用前复权。

    前复权保证最新价与实盘一致、历史涨跌幅比例正确；混用复权方式会让精准线的
    绝对价位在不同数据源之间对不上。AkshareProvider 有东财/腾讯两个源、
    get_daily_bars 与 get_full_daily_bars 两个入口，四条路径都要覆盖。
    """
    from app.services import data_provider as dp

    seen: dict[str, object] = {}
    frame = pd.DataFrame(
        {"日期": ["2026-01-01"], "开盘": [10.0], "最高": [10.5],
         "最低": [9.8], "收盘": [10.2], "成交量": [1000]}
    )

    class RecordingAk:
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            seen.setdefault("eastmoney", []).append(adjust) if isinstance(
                seen.get("eastmoney"), list) else seen.update(eastmoney=[adjust])
            return frame


    monkeypatch.setattr(dp.time, "sleep", lambda _: None)

    def recording_tencent(symbol, start_date, end_date, adjust):
        seen.setdefault("tencent", []).append(adjust)
        return pd.DataFrame(
            {"trade_date": ["2026-01-01"], "open": [10.0], "high": [10.5],
             "low": [9.8], "close": [10.2], "volume": [1000]}
        )

    # 东财路径（两个入口）
    ak = dp.AkshareProvider(bars_cache_dir=None)
    ak._ak = RecordingAk()
    ak.get_daily_bars("000001")
    ak.get_full_daily_bars("000001")
    assert seen["eastmoney"], "未走到东财路径"
    assert all(a == "qfq" for a in seen["eastmoney"]), \
        f"东财路径必须前复权，实得 {seen['eastmoney']}"

    # 腾讯路径（东财返回空时的兜底，同样两个入口）
    class EmptyEastmoneyAk(RecordingAk):
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            return pd.DataFrame()

    ak2 = dp.AkshareProvider(bars_cache_dir=None, tencent_fetch=recording_tencent)
    ak2._ak = EmptyEastmoneyAk()
    ak2.get_daily_bars("000001")
    ak2.get_full_daily_bars("000001")
    assert seen["tencent"], "未走到腾讯路径"
    assert all(a == "qfq" for a in seen["tencent"]), \
        f"腾讯路径必须前复权，实得 {seen['tencent']}"

    # adata
    class RecordingAdata:
        class stock:
            class market:
                @staticmethod
                def get_market(stock_code=None, k_type=1, adjust_type=1):
                    seen["adata"] = adjust_type
                    return pd.DataFrame(
                        {"trade_date": ["2026-01-01"], "open": [10.0], "high": [10.5],
                         "low": [9.8], "close": [10.2], "volume": [1000]}
                    )

    ad = dp.ADataProvider.__new__(dp.ADataProvider)
    ad._adata = RecordingAdata()
    ad.get_daily_bars("000001")
    assert seen["adata"] == 1, "adata 的 adjust_type=1 即前复权"


def test_auto_provider_falls_back_when_adata_returns_empty() -> None:
    class EmptyProvider:
        name = "adata"

        def get_stock_list(self):
            return pd.DataFrame()

        def get_daily_bars(self, stock_code):
            return pd.DataFrame()

        def get_full_daily_bars(self, stock_code, years=0):
            return pd.DataFrame()

    class WorkingProvider:
        name = "akshare"

        def get_stock_list(self):
            return pd.DataFrame(
                [{"stock_code": "000001", "short_name": "平安银行", "exchange": "SZ"}]
            )

        def get_daily_bars(self, stock_code):
            return pd.DataFrame({"close": [10.2]})

        def get_full_daily_bars(self, stock_code, years=0):
            assert years == 0
            return pd.DataFrame({"close": [10.2]})

    provider = AutoProvider(adata=EmptyProvider(), akshare=WorkingProvider())

    assert provider.get_stock_list().iloc[0]["stock_code"] == "000001"
    assert provider.get_daily_bars("000001").iloc[0]["close"] == 10.2
    assert provider.get_full_daily_bars("000001").iloc[0]["close"] == 10.2


def test_auto_provider_falls_back_when_primary_raises() -> None:
    class FailingProvider:
        name = "adata"

        def get_stock_list(self):
            raise ConnectionError("adata disconnected")

        def get_daily_bars(self, stock_code):
            raise ConnectionError("adata disconnected")

        def get_full_daily_bars(self, stock_code, years=0):
            raise ConnectionError("adata disconnected")

    class WorkingProvider:
        name = "akshare"

        def get_stock_list(self):
            return pd.DataFrame(
                [{"stock_code": "600000", "short_name": "浦发银行", "exchange": "SH"}]
            )

        def get_daily_bars(self, stock_code):
            return pd.DataFrame({"close": [9.5]})

        def get_full_daily_bars(self, stock_code, years=0):
            return pd.DataFrame({"close": [9.5]})

    provider = AutoProvider(adata=FailingProvider(), akshare=WorkingProvider())

    assert provider.get_stock_list().iloc[0]["stock_code"] == "600000"
    assert provider.get_daily_bars("600000").iloc[0]["close"] == 9.5


def test_price_endpoints_do_not_serialize_on_the_global_lock() -> None:
    """行情接口必须能并发。

    `call_akshare` 的全局锁是为了保护 libmini_racer 的初始化，但行情接口只做
    HTTP + pandas。实测带锁时 4 线程加速比只有 1.21x —— SCAN_WORKERS=4 形同虚设；
    去锁后 2.16x 且结果全部正确。
    """
    import threading
    import time

    from app.services import akshare_guard as guard

    holding = threading.Event()
    release = threading.Event()

    def hog():
        with guard.AKSHARE_LOCK:
            holding.set()
            release.wait(2)

    thread = threading.Thread(target=hog, daemon=True)
    thread.start()
    holding.wait(1)

    start = time.perf_counter()
    guard.call_akshare_parallel(lambda: "done")
    elapsed = time.perf_counter() - start
    release.set()
    thread.join(2)

    assert elapsed < 0.5, f"行情路径被全局锁挡住了 {elapsed:.2f}s"


def test_module_init_still_goes_through_the_lock() -> None:
    """并发初始化 akshare 仍然不安全，这条路径的锁不能一起去掉。"""
    import inspect

    from app.services import akshare_guard as guard

    assert "AKSHARE_LOCK" in inspect.getsource(guard.get_akshare)
    assert "AKSHARE_LOCK" in inspect.getsource(guard.call_akshare)


def test_provider_signals_failure_instead_of_returning_an_empty_frame(monkeypatch) -> None:
    """两个数据源都失败时，provider 要抛出可识别的失败，而不是返回空表。

    返回空表的话，缓存层分不清"今天没有新K线"和"数据源全挂"，会把失败
    记成当天已定稿，当天不再重试。
    """
    from app.services import data_provider as dp

    class Blocked:
        def stock_zh_a_hist(self, symbol, period, adjust, **kw):
            raise ConnectionError("RemoteDisconnected")

    def failing_tencent(symbol, start_date, end_date, adjust):
        raise ConnectionError("RemoteDisconnected")

    monkeypatch.setattr(dp.time, "sleep", lambda _: None)
    provider = dp.AkshareProvider(bars_cache_dir=None, tencent_fetch=failing_tencent)
    provider._ak = Blocked()

    with pytest.raises(dp.FetchFailed):
        provider._fetch_range("000001", "19900101")


@pytest.mark.parametrize("name", ["akshrae", "unknown", "auto-fallback", ""])
def test_unknown_provider_is_never_sample_data(name):
    from app.services.data_provider import create_provider
    from app.schemas import ScanRequest, BacktestRequest
    from pydantic import ValidationError
    for schema in (ScanRequest, BacktestRequest):
        with pytest.raises(ValidationError):
            schema(provider=name)
    with pytest.raises(ValueError):
        create_provider(name)


def test_uncached_real_provider_cannot_bypass_quote_readiness(monkeypatch):
    from datetime import datetime
    from app.services import data_provider as dp
    from types import SimpleNamespace
    frame = pd.DataFrame([dict(trade_date="2026-09-09", open=10, high=11, low=9, close=10, volume=100)])
    monkeypatch.setattr(dp, "china_now", lambda: datetime(2026, 9, 10, 16))
    adata = dp.ADataProvider.__new__(dp.ADataProvider)
    adata._adata = SimpleNamespace(stock=SimpleNamespace(market=SimpleNamespace(get_market=lambda **kw: frame.copy())))
    assert adata.get_daily_bars("000001").attrs["data_quality"]["status"] == "stale"
    ak = dp.AkshareProvider(bars_cache_dir=None)
    monkeypatch.setattr(ak, "_fetch_range", lambda code, start: frame.copy())
    assert ak.get_daily_bars("000001").attrs["data_quality"]["status"] == "stale"
