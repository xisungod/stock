import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import BacktestRequest, ScanRequest
from app.services.backtest import evaluate_bars
from app.services.scanner import Scanner
from app.services.strategy import (
    KeyLine, SignalCategory, StrategyConfig, _find_breakout, _select_best_signal, analyze_bars,
)
from test_api import FakeScanner, _wait_backtest_done
from test_flag_pattern import flag_bars
from test_pullback_window import BREAKOUT_INDEX, bars_and_line, fixed_line


def fixture(volume=1340):
    bars, line = bars_and_line(5)
    bars.loc[BREAKOUT_INDEX, "volume"] = volume
    bars.loc[len(bars) - 1, "volume"] = 200
    return bars, line, StrategyConfig(signal_offset=0, include_chart=False)


@pytest.mark.parametrize("volume", [500, 1340])
def test_low_volume_price_breakout_can_confirm_a_pullback_when_filter_is_off(volume):
    bars, line, cfg = fixture(volume)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.breakout_volume_ratio == volume / 1000
    assert result.require_breakout_volume is False
    assert result.breakout_volume_threshold is None
    assert result.pullback_check["passed"] is True
    assert "收盘突破" in result.suggestion
    assert all("放量突破" not in reason for reason in result.reasons)
    enabled = _select_best_signal("000001", "测试", bars, [line], replace(cfg, require_breakout_volume=True))
    assert enabled.category == SignalCategory.NONE
    assert enabled.breakout_date is None
    assert enabled.breakout_volume_threshold == 1.5


def test_disabled_volume_threshold_and_volume_bonus_do_not_affect_the_verdict():
    bars, line, cfg = fixture(500)
    loose = _select_best_signal("000001", "测试", bars, [line], replace(cfg, breakout_vol_mult=1.0))
    strict = _select_best_signal("000001", "测试", bars, [line], replace(cfg, breakout_vol_mult=5.0))
    assert loose.to_dict() == strict.to_dict()
    bars.loc[BREAKOUT_INDEX, "volume"] = 4000
    strong = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert strong.category == loose.category == SignalCategory.BUY_ALERT
    assert strong.score == loose.score
    assert strong.breakout_volume_ratio == 4
    enabled = _select_best_signal("000001", "测试", bars, [line], replace(cfg, require_breakout_volume=True))
    assert enabled.score > strong.score
    assert "放量突破" in enabled.suggestion


def test_wick_above_the_line_does_not_replace_a_close_crossing():
    bars, line, cfg = fixture()
    bars.loc[BREAKOUT_INDEX:, ["open", "close", "high", "low"]] = [10, 10.02, 10.5, 9.99]
    assert _find_breakout(bars, line, cfg) is None
    assert _select_best_signal("000001", "测试", bars, [line], cfg).category == SignalCategory.NONE


def test_price_only_breakout_does_not_require_a_full_volume_baseline():
    bars = pd.DataFrame({"close": [9.8, 9.9, 10.2, 10.15], "volume": [1000, 1000, 500, 400]})
    line = KeyLine(10, 3, [0], 0, 0)
    cfg = StrategyConfig()
    breakout = _find_breakout(bars, line, cfg)
    assert breakout["index"] == 2
    assert breakout["volume_ratio"] is breakout["avg_volume"] is None
    assert _find_breakout(bars, line, replace(cfg, require_breakout_volume=True)) is None


def test_zero_volume_baseline_is_unknown_not_a_zero_ratio_and_does_not_crash():
    bars, line, cfg = fixture()
    bars.loc[:BREAKOUT_INDEX - 1, "volume"] = 0
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.category == SignalCategory.BUY_ALERT
    assert result.breakout_volume_ratio is None
    assert result.pullback_normal_volume_ratio is None
    json.dumps(result.to_dict(), allow_nan=False)
    assert _find_breakout(bars, line, replace(cfg, require_breakout_volume=True)) is None


def test_zero_breakout_volume_cannot_confirm_a_shrinking_pullback():
    bars, line, cfg = fixture(0)
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.breakout_date is not None
    assert result.breakout_volume_ratio == 0
    assert result.category == SignalCategory.NONE
    assert result.pullback_check["passed"] is False
    assert result.pullback_normal_volume_ratio == .2
    json.dumps(result.to_dict(), allow_nan=False)


def test_disabling_breakout_volume_does_not_disable_pullback_shrinkage_or_path_limits():
    bars, line, cfg = fixture(500)
    bars.loc[len(bars) - 1, "volume"] = 450
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.breakout_date is not None
    assert result.category == SignalCategory.NONE
    assert result.pullback_check["passed"] is False
    bars.loc[len(bars) - 1, "volume"] = 200
    bars.loc[120, "high"] = 15.1
    result = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert result.pullback_check["passed"] is True
    assert result.prebreakout_check["passed"] is False
    assert result.category == SignalCategory.NONE
    assert cfg.prebreakout_reject_above_days == 30


def test_latest_weak_price_cross_is_not_skipped_for_an_older_strong_breakout():
    bars, line, cfg = fixture(3000)
    latest = len(bars) - 3
    bars.loc[latest - 1, ["open", "close", "high", "low"]] = [9.98, 10, 10.05, 9.95]
    bars.loc[latest, ["open", "close", "high", "low", "volume"]] = [10.1, 10.25, 10.3, 10.05, 500]
    cfg = replace(cfg, min_pullback_days=3)
    price_only = _select_best_signal("000001", "测试", bars, [line], cfg)
    assert price_only.breakout_date == str(bars.trade_date.iloc[latest].date())
    assert price_only.pullback_check["days_since_breakout"] == 2
    assert price_only.category == SignalCategory.NONE
    volume_filtered = _select_best_signal("000001", "测试", bars, [line], replace(cfg, require_breakout_volume=True))
    assert volume_filtered.breakout_date == str(bars.trade_date.iloc[BREAKOUT_INDEX].date())
    assert volume_filtered.category == SignalCategory.BUY_ALERT


def test_point_in_time_replay_and_pattern_snapshots_preserve_the_volume_setting(monkeypatch):
    bars, line, cfg = fixture()
    fixed_line(monkeypatch, line)
    original = analyze_bars("000001", "测试", bars, cfg)
    future = bars.iloc[-1:].copy()
    future["trade_date"] += pd.Timedelta(days=3)
    future["volume"] = 100000
    extended = pd.concat([bars, future], ignore_index=True)
    assert analyze_bars("000001", "测试", extended, replace(cfg, signal_offset=1)).to_dict() == original.to_dict()
    assert original.pattern_matches[0]["require_breakout_volume"] is False
    assert original.pattern_matches[0]["breakout_volume_threshold"] is None
    replay = evaluate_bars("000001", "测试", bars, lookback_days=1, config=cfg)
    assert replay["require_breakout_volume"] is False
    assert replay["signals"][0]["require_breakout_volume"] is False
    assert replay["signals"][0]["pattern_matches"][0]["require_breakout_volume"] is False
    assert evaluate_bars("000001", "测试", bars, lookback_days=1,
                         config=replace(cfg, require_breakout_volume=True))["signals"] == []


@pytest.mark.parametrize("mode", ["flag_breakout", "combined"])
def test_flags_keep_volume_confirmation_when_precision_volume_is_disabled(monkeypatch, mode):
    from app.services import strategy
    bars = flag_bars()
    bars.loc[bars.index[-1], "volume"] = 100
    if mode == "combined":
        monkeypatch.setattr(strategy, "_run_precision_pipeline", lambda *args: ("no_signal", "无精准线", None))
    cfg = StrategyConfig(strategy_mode=mode, signal_offset=0, include_chart=False)
    result = analyze_bars("000001", "测试", bars, cfg)
    assert result.pattern_type == "flag_breakout"
    assert result.category == SignalCategory.WATCH
    assert result.flag_pattern["status"] == "unconfirmed_breakout"
    assert result.require_breakout_volume is True
    assert result.breakout_volume_threshold == 1.5
    assert result.to_dict() == analyze_bars("000001", "测试", bars, replace(cfg, require_breakout_volume=True)).to_dict()


def test_defaults_legacy_parameters_and_cache_identity(tmp_path):
    client = TestClient(create_app(db_path=tmp_path / "defaults.db", scanner=FakeScanner()))
    assert client.get("/api/config/defaults").json()["require_breakout_volume"] is False
    for model in (StrategyConfig, ScanRequest, BacktestRequest):
        assert model().require_breakout_volume is False
        assert model(require_breakout_volume=True).require_breakout_volume is True
    for model in (ScanRequest, BacktestRequest):
        with pytest.raises(ValidationError):
            model(require_breakout_volume="invalid")
    scanner = Scanner()
    assert scanner.strategy_config().require_breakout_volume is False
    assert scanner.strategy_config({"provider": "fake"}).require_breakout_volume is True
    assert scanner.strategy_config({"provider": "fake", "require_breakout_volume": False}).require_breakout_volume is False
    assert scanner.strategy_config({"strategy_mode": "flag_breakout", "require_breakout_volume": False}).require_breakout_volume is True
    on = scanner._detail_cache_key("fake", "000001", 0, {"require_breakout_volume": True})
    off = scanner._detail_cache_key("fake", "000001", 0, {"require_breakout_volume": False})
    assert on != off


def test_real_scanner_scan_detail_diagnosis_and_replay_use_the_same_switch(monkeypatch):
    bars, line, _ = fixture()
    fixed_line(monkeypatch, line)
    provider = SimpleNamespace(name="fake", get_daily_bars=lambda code: bars.copy(),
        get_stock_list=lambda: pd.DataFrame([{"stock_code": "000001", "short_name": "测试"}]))
    scanner = Scanner(provider=provider)
    params = ScanRequest(provider="fake", strategy_mode="precision_pullback", precision_engine="legacy", signal_offset=0).model_dump()
    assert len(scanner.run_scan(params)) == 1
    for result in (
        scanner.stock_detail("000001", "测试", scan_params=params)["analysis"]["result"],
        scanner.diagnose_stock("000001", "测试", scan_params=params)["result"],
    ):
        assert result["category"] == "buy_alert"
        assert result["require_breakout_volume"] is False
        assert "收盘突破" in result["suggestion"]
    assert scanner.backtest_stock("000001", "测试", 1, params)["signals"][0]["require_breakout_volume"] is False
    on = {**params, "require_breakout_volume": True}
    assert scanner.run_scan(on) == []
    assert scanner.stock_detail("000001", "测试", scan_params=on)["analysis"]["found"] is False
    assert scanner.backtest_stock("000001", "测试", 1, on)["signals"] == []


@pytest.mark.parametrize("saved_gate", [None, False, True])
def test_single_and_batch_replays_inherit_the_saved_volume_gate(tmp_path, saved_gate):
    from app.db import Database
    path = tmp_path / "gate.db"
    client = TestClient(create_app(db_path=path, scanner=FakeScanner()))
    params = {"provider": "fake", "breakout_vol_mult": 1.8}
    if saved_gate is not None:
        params["require_breakout_volume"] = saved_gate
    scan_id = Database(path).create_scan_run("fake", params)
    expected = True if saved_gate is None else saved_gate
    single = client.get(f"/api/stocks/000001/backtest?scan_id={scan_id}").json()["received_params"]
    started = client.post("/api/backtests/run", json={
        "scan_id": scan_id, "require_breakout_volume": not expected, "breakout_vol_mult": 4,
    }).json()
    _wait_backtest_done(client, started["run_id"])
    batch = client.get("/api/backtests/latest").json()["parameters"]
    for payload in (single, batch):
        assert payload["require_breakout_volume"] is expected
        assert payload["breakout_vol_mult"] == 1.8
