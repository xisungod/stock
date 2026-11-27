from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from app.db import Database
from app.services.scanner import Scanner
from app.services.strategy import LEGACY_STRATEGY_VERSION, STRATEGY_VERSION
from test_platform_events import bars


def client_and_frame(tmp_path):
    frame = bars(([9.2, 9.7, 10.04, 9.7, 9.2] * 32) + [9.7, 10.3, 10.18, 10.15])
    frame.loc[len(frame) - 3, "volume"] = 2500
    provider = SimpleNamespace(
        name="fake", get_daily_bars=lambda code: frame.copy(),
        get_stock_list=lambda: pd.DataFrame([{"stock_code": "000001", "short_name": "test"}]),
    )
    scanner = Scanner(provider=provider)
    return TestClient(create_app(tmp_path / "test.db", scanner)), frame, scanner


def test_new_default_engine_and_explicit_legacy_are_independent(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    assert client.get("/api/config/defaults").json()["precision_engine"] == "platform_events"
    modern = client.get("/api/stocks/000001/detail").json()
    legacy = client.get("/api/stocks/000001/detail?precision_engine=legacy").json()
    assert modern["precision_engine"] == "platform_events"
    assert modern["strategy_version"] == STRATEGY_VERSION
    assert legacy["precision_engine"] == "legacy"
    assert legacy["strategy_version"] == LEGACY_STRATEGY_VERSION
    assert len(scanner.detail_cache) == 2


def test_historical_analysis_cuts_bars_and_does_not_backfill_current_industry(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    stamp = str(frame.trade_date.iloc[-3].date())
    result = client.get(f"/api/stocks/000001/detail?as_of={stamp}").json()
    assert result["end_date"] == stamp
    assert result["analysis_as_of"] == stamp
    assert result["industry"] is None
    assert all(bar["date"] <= stamp for bar in result["chart"]["bars"])
    if result["analysis"]["result"]:
        assert result["analysis"]["result"]["signal_date"] == stamp
    assert client.get("/api/stocks/000001/detail?as_of=1990-01-01").status_code == 422


def test_old_batch_without_engine_remains_legacy_and_override_is_explicit(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    db = Database(tmp_path / "test.db")
    scan_id = db.create_scan_run("fake", {"provider": "fake", "strategy_version": LEGACY_STRATEGY_VERSION})
    old = client.get(f"/api/stocks/000001/detail?scan_id={scan_id}").json()
    changed = client.get(f"/api/stocks/000001/detail?scan_id={scan_id}&precision_engine=platform_events").json()
    assert old["precision_engine"] == "legacy"
    assert old["scan_context"]["recomputed_with_new_strategy"] is False
    assert changed["scan_context"]["engine_override"] is True
    assert "precision_engine" not in db.get_scan(scan_id)["parameters"]


def test_comparison_uses_one_price_frame_and_all_three_engines(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    stamp = str(frame.trade_date.iloc[-2].date())
    result = client.get(f"/api/stocks/000001/strategy-comparison?as_of={stamp}").json()
    assert result["analysis_as_of"] == stamp
    assert [item["engine"] for item in result["results"]] == ["legacy", "pivot_fixed", "platform_events"]
    for item in result["results"]:
        if item["result"]:
            assert item["result"]["signal_date"] == stamp
            assert item["result"]["industry_adjustment"] == 0


def test_historical_backtest_respects_cutoff_and_emits_frozen_versions(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    stamp = str(frame.trade_date.iloc[-1].date())
    result = client.get(f"/api/stocks/000001/backtest?as_of={stamp}&lookback_days=120").json()
    assert result["precision_engine"] == "platform_events"
    assert result["deduplication"] == "breakout_event"
    for item in result["signals"]:
        assert item["signal_date"] <= stamp
        assert item["line_snapshot"]["version_id"]
        assert all(point["confirmed_date"] < item["line_snapshot"]["breakout_date"]
                   for point in item["line_snapshot"]["resistance_points"] + item["line_snapshot"]["support_points"])


def test_historical_detail_comparison_and_replay_can_use_same_provider_cache_offline(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    provider = scanner._provider()
    provider.get_cached_daily_bars = lambda code: frame.copy()
    provider.get_cached_stock_list = provider.get_stock_list

    def offline(*args, **kwargs):
        raise AssertionError("Historical requests must not need today's network data")

    provider.get_stock_list = provider.get_daily_bars = offline
    stamp = str(frame.trade_date.iloc[-2].date())
    for endpoint in ["detail", "strategy-comparison", "backtest"]:
        response = client.get(f"/api/stocks/000001/{endpoint}?as_of={stamp}")
        assert response.status_code == 200, response.text
        assert response.json()["analysis_as_of"] == stamp
    assert client.get(f"/api/stocks/000001/detail?as_of={stamp}").json()["data_quality"]["status"] == "historical_cache"


def test_historical_cache_cannot_be_reused_for_live_analysis_or_beyond_its_end(tmp_path):
    client, frame, scanner = client_and_frame(tmp_path)
    provider = scanner._provider()
    provider.get_cached_daily_bars = lambda code: frame.iloc[:-10].copy()
    calls = []
    provider.get_daily_bars = lambda code: calls.append(code) or frame.copy()
    stamp = str(frame.trade_date.iloc[-2].date())
    assert client.get(f"/api/stocks/000001/detail?as_of={stamp}").status_code == 200
    assert calls == ["000001"]
    scanner.full_bars_cache.clear()
    assert client.get("/api/stocks/000001/detail").status_code == 200
    assert calls == ["000001", "000001"]
