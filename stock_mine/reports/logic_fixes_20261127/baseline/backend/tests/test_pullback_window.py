from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import BacktestRequest, ScanRequest
from app.services import strategy
from app.services.backtest import evaluate_bars
from app.services.scanner import Scanner
from app.services.strategy import (
    KeyLine, SignalCategory, StrategyConfig, _find_breakout, _select_best_signal, analyze_bars,
)
from test_api import FakeScanner, _wait_backtest_done, _wait_done
from test_combined_strategy import result as branch_result


BREAKOUT_INDEX = 140


def bars_and_line(age):
    closes = np.full(BREAKOUT_INDEX + age + 1, 9.8)
    closes[BREAKOUT_INDEX] = 10.25
    if age:
        closes[BREAKOUT_INDEX + 1:] = 10.1
        closes[-1] = 10.3
    bars = pd.DataFrame({
        "trade_date": pd.bdate_range("2025-01-01", periods=len(closes)),
        "open": closes - .02, "high": closes + .05, "low": closes - .05,
        "close": closes, "volume": np.full(len(closes), 1000.0),
    })
    bars.loc[BREAKOUT_INDEX, "volume"] = 3000
    if age:
        bars.loc[len(bars) - 1, "low"] = 10.02
    bars.loc[[20, 60, 100], "high"] = 10
    bars["ma"] = bars.close.rolling(20).mean()
    line = KeyLine(
        10, 3, [20, 60, 100], 100, 101,
        resistance_confirmed_at={20: 21, 60: 61, 100: 101},
    )
    return bars, line


def fixed_line(monkeypatch, line):
    monkeypatch.setattr(strategy, "_swings", lambda *args, **kwargs: ([], [], None, {}))
    monkeypatch.setattr(strategy, "_cluster_key_lines", lambda *args, **kwargs: [line])


@pytest.mark.parametrize("age,minimum,maximum,passes", [
    (0, 1, 10, False), (1, 1, 10, True), (2, 3, 7, False),
    (3, 3, 7, True), (5, 5, 5, True), (7, 3, 7, True), (8, 3, 7, False),
    (10, 1, 10, True), (11, 1, 10, False), (20, 12, 20, True),
    (60, 60, 60, True), (61, 1, 60, False),
])
def test_configured_window_is_inclusive_and_extends_the_real_search(age, minimum, maximum, passes):
    bars, line = bars_and_line(age)
    cfg = StrategyConfig(signal_offset=0, include_chart=False,
                         min_pullback_days=minimum, max_pullback_days=maximum)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == (SignalCategory.BUY_ALERT if passes else SignalCategory.NONE)
    assert result.pullback_check["required"] is True
    assert result.pullback_check["min_days"] == minimum
    assert result.pullback_check["max_days"] == maximum
    assert result.pullback_check["passed"] is passes
    if passes:
        assert result.pullback_check["days_since_breakout"] == age
        assert result.pullback_date == str(bars.trade_date.iloc[-1].date())
        assert result.prebreakout_check["passed"] is True


def test_new_window_does_not_depend_on_the_old_fixed_windows():
    bars, line = bars_and_line(20)
    cfg = StrategyConfig(signal_offset=0, include_chart=False,
                         breakout_window=2, pullback_window=2,
                         min_pullback_days=12, max_pullback_days=20)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.pullback_check["days_since_breakout"] == 20


def test_latest_breakout_cannot_be_skipped_to_satisfy_the_minimum():
    bars, line = bars_and_line(12)
    latest = len(bars) - 3
    bars.loc[latest - 1, ["open", "close", "high", "low"]] = [9.98, 10, 10.05, 9.95]
    bars.loc[latest, ["open", "close", "high", "low", "volume"]] = [10.1, 10.25, 10.3, 10.05, 4000]
    cfg = StrategyConfig(signal_offset=0, include_chart=False,
                         min_pullback_days=5, max_pullback_days=20)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.breakout_date == str(bars.trade_date.iloc[latest].date())
    assert result.pullback_check["days_since_breakout"] == 2
    assert result.pullback_check["status"] == "outside_window"
    assert result.category == SignalCategory.NONE


def test_near_line_without_breakout_only_remains_when_gate_is_off():
    bars, line = bars_and_line(5)
    bars.loc[BREAKOUT_INDEX:, ["open", "close", "high", "low"]] = [10, 10.02, 10.05, 9.99]
    cfg = StrategyConfig(signal_offset=0, include_chart=False)
    rejected = _select_best_signal("000001", "测试", bars, [line], cfg)
    allowed = _select_best_signal("000001", "测试", bars, [line], replace(cfg, require_pullback=False))
    assert rejected.category == SignalCategory.NONE
    assert rejected.pullback_check["status"] == "no_breakout"
    assert allowed.category == SignalCategory.WATCH


@pytest.mark.parametrize("failure", ["not_touching", "not_shrinking", "lost_line"])
def test_only_the_signal_days_valid_pullback_counts(failure):
    bars, line = bars_and_line(5)
    if failure == "not_touching":
        bars.loc[len(bars) - 1, "low"] = 10.2
    elif failure == "not_shrinking":
        bars.loc[len(bars) - 1, "volume"] = 2400
        # Keep this a high-volume retest, not a new crossing of the breakout threshold.
        bars.loc[len(bars) - 2, ["open", "close", "high", "low"]] = [10.08, 10.1, 10.15, 10.05]
    else:
        bars.loc[BREAKOUT_INDEX + 2, ["open", "close", "high", "low"]] = [9.8, 9.8, 9.82, 9.75]
        bars.loc[BREAKOUT_INDEX + 3:, ["open", "close", "high", "low"]] = [10, 10.02, 10.05, 9.99]
    cfg = StrategyConfig(signal_offset=0, include_chart=False)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == SignalCategory.NONE
    assert result.pullback_check["status"] == "awaiting_pullback"
    assert result.breakout_status["pullback_day"] is not None


def test_failed_prebreakout_path_is_still_rejected_after_a_valid_pullback():
    bars, line = bars_and_line(5)
    bars.loc[120, "high"] = 15.1
    result = _select_best_signal("000001", "测试", bars, [line], StrategyConfig(include_chart=False))
    assert result.pullback_check["passed"] is True
    assert result.prebreakout_check["passed"] is False
    assert result.category == SignalCategory.NONE


def test_pullback_days_count_trading_rows_not_calendar_days():
    bars, line = bars_and_line(5)
    result = _select_best_signal("000001", "测试", bars, [line], StrategyConfig(include_chart=False))
    assert (pd.Timestamp(result.pullback_date) - pd.Timestamp(result.breakout_date)).days == 7
    assert result.pullback_check["days_since_breakout"] == 5


def test_future_bars_and_backtests_preserve_the_signal_day_window(monkeypatch):
    bars, line = bars_and_line(5)
    fixed_line(monkeypatch, line)
    cfg = StrategyConfig(signal_offset=0, include_chart=False, min_pullback_days=5, max_pullback_days=5)
    original = analyze_bars("000001", "测试", bars, cfg)
    assert original.category == SignalCategory.BUY_ALERT
    future = bars.iloc[-1:].copy()
    future["trade_date"] += pd.Timedelta(days=3)
    future[["open", "high", "low", "close"]] *= 3
    extended = pd.concat([bars, future], ignore_index=True)
    replayed = analyze_bars("000001", "测试", extended, replace(cfg, signal_offset=1))
    assert original.to_dict() == replayed.to_dict()
    report = evaluate_bars("000001", "测试", bars, lookback_days=1, config=cfg)
    assert report["require_pullback"] is True
    assert report["min_pullback_days"] == report["max_pullback_days"] == 5
    assert report["signals"][0]["pullback_check"] == original.pullback_check
    assert report["signals"][0]["pattern_matches"][0]["pullback_check"] == original.pullback_check


def test_rejected_precision_does_not_reject_an_independent_flag(monkeypatch):
    bars, line = bars_and_line(0)
    fixed_line(monkeypatch, line)

    def flag(code, name, frame, cfg):
        assert cfg.require_pullback is False
        assert cfg.min_pullback_days == 1 and cfg.max_pullback_days is None
        return "ok", "旗形突破", branch_result("flag_breakout", SignalCategory.BUY_ALERT)

    monkeypatch.setattr(strategy, "_run_flag_pipeline", flag)
    result = analyze_bars("000001", "测试", bars, StrategyConfig(strategy_mode="combined", signal_offset=0, include_chart=False))
    assert result.category == SignalCategory.BUY_ALERT
    assert [match["pattern_type"] for match in result.pattern_matches] == ["flag_breakout"]


@pytest.mark.parametrize("model", [ScanRequest, BacktestRequest, StrategyConfig])
def test_window_defaults_validation_and_exact_day_selection(model):
    assert model().require_pullback is True
    assert (model().min_pullback_days, model().max_pullback_days) == (1, 10)
    for minimum, maximum in [(1, 60), (12, 20), (5, 5), (60, 60), (1, None)]:
        assert model(min_pullback_days=minimum, max_pullback_days=maximum).min_pullback_days == minimum
    for minimum, maximum in [(0, 10), (1, 61), (6, 5), (2, None), (1, 0), (1.5, 10), (1, True), (None, 10)]:
        with pytest.raises((ValueError, ValidationError)):
            model(min_pullback_days=minimum, max_pullback_days=maximum)


def test_old_and_new_windows_cannot_create_a_silent_empty_intersection():
    with pytest.raises(ValidationError, match="没有交集"):
        ScanRequest(min_pullback_days=12, max_pullback_days=20, max_days_since_breakout=9)
    with pytest.raises(ValidationError, match="没有交集"):
        ScanRequest(min_pullback_days=1, max_pullback_days=5, min_days_since_breakout=9)


def test_legacy_batches_keep_the_old_gate_and_nine_day_search_limit():
    scanner = Scanner()
    cfg = scanner.strategy_config({"provider": "fake", "max_prebreakout_extension_pct": .5})
    assert cfg.require_pullback is False
    assert cfg.max_pullback_days is None
    bars, line = bars_and_line(10)
    assert _find_breakout(bars, line, cfg) is None
    assert _find_breakout(bars, line, StrategyConfig())["index"] == BREAKOUT_INDEX


def test_each_pullback_parameter_separates_detail_and_backtest_caches():
    scanner = Scanner()
    variants = [
        {"require_pullback": True, "min_pullback_days": 1, "max_pullback_days": 10},
        {"require_pullback": False, "min_pullback_days": 1, "max_pullback_days": 10},
        {"require_pullback": True, "min_pullback_days": 5, "max_pullback_days": 10},
        {"require_pullback": True, "min_pullback_days": 1, "max_pullback_days": 20},
        {"require_pullback": False, "max_pullback_days": None},
    ]
    assert len({scanner._detail_cache_key("fake", "000001", 0, params) for params in variants}) == len(variants)


def test_parameters_reach_real_scanning_detail_diagnosis_and_replay(monkeypatch):
    bars, line = bars_and_line(20)
    fixed_line(monkeypatch, line)
    provider = SimpleNamespace(
        name="fake", get_daily_bars=lambda code: bars.copy(),
        get_stock_list=lambda: pd.DataFrame([{"stock_code": "000001", "short_name": "测试"}]),
    )
    scanner = Scanner(provider=provider)
    params = ScanRequest(provider="fake", strategy_mode="precision_pullback", precision_engine="legacy", signal_offset=0,
                         min_pullback_days=12, max_pullback_days=20).model_dump()
    results = scanner.run_scan(params)
    assert len(results) == 1 and results[0].pullback_check["days_since_breakout"] == 20
    detail = scanner.stock_detail("000001", "测试", scan_params=params)
    diagnostic = scanner.diagnose_stock("000001", "测试", scan_params=params)
    for result in (detail["analysis"]["result"], diagnostic["result"]):
        assert result["category"] == "buy_alert"
        assert result["pullback_check"] == results[0].pullback_check
    replay = scanner.backtest_stock("000001", "测试", lookback_days=1, scan_params=params)
    assert replay["signals"][0]["pullback_check"] == results[0].pullback_check
    narrow = {**params, "min_pullback_days": 1, "max_pullback_days": 10}
    assert scanner.stock_detail("000001", "测试", scan_params=narrow)["analysis"]["found"] is False
    assert scanner.backtest_stock("000001", "测试", lookback_days=1, scan_params=narrow)["signals"] == []


@pytest.mark.parametrize("saved", [
    {}, {"require_pullback": False, "min_pullback_days": 1, "max_pullback_days": 7},
    {"require_pullback": True, "min_pullback_days": 12, "max_pullback_days": 20},
])
def test_single_and_batch_backtests_inherit_old_or_new_window(tmp_path, saved):
    from app.db import Database
    path = tmp_path / "window.db"
    client = TestClient(create_app(db_path=path, scanner=FakeScanner()))
    scan_id = Database(path).create_scan_run("fake", {"provider": "fake", **saved})
    expected = {"require_pullback": False, "min_pullback_days": 1, "max_pullback_days": None, **saved}
    single = client.get(f"/api/stocks/000001/backtest?scan_id={scan_id}&lookback_days=120").json()
    started = client.post("/api/backtests/run", json={
        "scan_id": scan_id, "lookback_days": 120,
        "require_pullback": True, "min_pullback_days": 40, "max_pullback_days": 60,
    }).json()
    _wait_backtest_done(client, started["run_id"])
    batch = client.get("/api/backtests/latest").json()["parameters"]
    for field, value in expected.items():
        assert single["received_params"][field] == value
        assert batch[field] == value


def test_new_scans_persist_window_defaults_and_reopen_them_after_restart(tmp_path):
    path = tmp_path / "saved.db"
    client = TestClient(create_app(db_path=path, scanner=FakeScanner()))
    run = client.post("/api/scans/run", json={"provider": "fake", "limit": 2}).json()
    _wait_done(client, run["scan_id"])
    restarted = TestClient(create_app(db_path=path, scanner=FakeScanner()))
    advertised = restarted.get("/api/config/defaults").json()
    parameters = restarted.get("/api/scans/latest").json()["parameters"]
    restored = restarted.get(f"/api/stocks/000001/backtest?scan_id={run['scan_id']}").json()["received_params"]
    for field in ("require_pullback", "min_pullback_days", "max_pullback_days"):
        assert parameters[field] == advertised[field] == restored[field] == getattr(ScanRequest(), field)


@pytest.mark.parametrize("path", ["/api/scans/run", "/api/backtests/run"])
def test_invalid_window_is_rejected_before_a_job_is_submitted(tmp_path, path):
    client = TestClient(create_app(db_path=tmp_path / "invalid.db", scanner=FakeScanner()))
    response = client.post(path, json={"provider": "fake", "min_pullback_days": 12, "max_pullback_days": 10})
    assert response.status_code == 422
    assert client.get("/api/scans/active").json() is None
    assert client.get("/api/backtests/active").json() is None
