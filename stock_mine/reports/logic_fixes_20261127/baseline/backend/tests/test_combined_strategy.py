from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services import strategy
from app.services.backtest import evaluate_bars
from app.services.scanner import Scanner
from app.services.strategy import SignalCategory as C, SignalResult, StrategyConfig, analyze_bars
from test_flag_pattern import flag_bars


def result(kind, category, score=80):
    if category is None:
        return None
    item = SignalResult(
        "000001", "测试", category, score, kind, 10 if kind == "precision_pullback" else None,
        3 if kind == "precision_pullback" else 0, "2025-07-15",
        pattern_type=kind, structure_score=score,
    )
    if kind == "flag_breakout":
        item.flag_pattern = {
            "id": "flag:1", "direction": "ascending",
            "status": "breakout" if category == C.BUY_ALERT else "failed" if category == C.NONE else "forming",
        }
    else:
        item.key_lines = [{"price": 10, "point_count": 3, "is_primary": True}]
    return item


def stub_branches(monkeypatch, precision, flag, precision_score=80, flag_score=90):
    calls = []
    def run(kind, category, score, code, name, frame, cfg):
        calls.append((kind, len(frame), cfg))
        item = result(kind, category, score)
        return "ok" if category in {C.BUY_ALERT, C.WATCH} else "no_signal", kind, item
    monkeypatch.setattr(strategy, "_run_precision_pipeline",
        lambda code, name, frame, cfg: run("precision_pullback", precision, precision_score, code, name, frame, cfg))
    monkeypatch.setattr(strategy, "_run_flag_pipeline",
        lambda code, name, frame, cfg: run("flag_breakout", flag, flag_score, code, name, frame, cfg))
    return calls


@pytest.mark.parametrize("precision", [None, C.NONE, C.WATCH, C.BUY_ALERT])
@pytest.mark.parametrize("flag", [None, C.NONE, C.WATCH, C.BUY_ALERT])
def test_combined_uses_or_and_labels_only_real_matches(monkeypatch, precision, flag):
    calls = stub_branches(monkeypatch, precision, flag)
    merged = analyze_bars("000001", "测试", flag_bars(), StrategyConfig(strategy_mode="combined", signal_offset=0, include_chart=False))
    expected = [
        (kind, category.value) for kind, category in (("precision_pullback", precision), ("flag_breakout", flag))
        if category in {C.BUY_ALERT, C.WATCH}
    ]
    actual = merged.to_dict()["pattern_matches"] if merged else []
    assert [(m["pattern_type"], m["category"]) for m in actual] == expected
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == len(flag_bars())
    if expected:
        assert merged.category == (C.BUY_ALERT if C.BUY_ALERT in {precision, flag} else C.WATCH)
        assert merged.score <= 90, "双形态不得把分数相加"
    else:
        assert merged is None or merged.category == C.NONE


def test_confirmed_pattern_beats_a_higher_scoring_watch_candidate(monkeypatch):
    stub_branches(monkeypatch, C.BUY_ALERT, C.WATCH, precision_score=75, flag_score=95)
    merged = analyze_bars("000001", "测试", flag_bars(), StrategyConfig(strategy_mode="combined", signal_offset=0))
    assert merged.pattern_type == "precision_pullback"
    assert merged.score == 75
    assert merged.pattern_matches[1]["category"] == "watch"
    assert merged.chart["flag_pattern"]["status"] == "forming"
    assert merged.chart["key_lines"][0]["price"] == 10


def test_flag_primary_does_not_discard_precision_geometry(monkeypatch):
    stub_branches(monkeypatch, C.BUY_ALERT, C.BUY_ALERT)
    merged = analyze_bars("000001", "测试", flag_bars(), StrategyConfig(strategy_mode="combined", signal_offset=0))
    assert merged.pattern_type == "flag_breakout"
    assert merged.key_lines[0]["price"] == 10
    assert merged.chart["flag_pattern"]["direction"] == "ascending"
    assert len(merged.pattern_matches) == 2


def test_precision_age_window_does_not_filter_flags(monkeypatch):
    calls = stub_branches(monkeypatch, C.NONE, C.BUY_ALERT)
    merged = analyze_bars("000001", "测试", flag_bars(), StrategyConfig(
        strategy_mode="combined", signal_offset=0, min_days_since_breakout=5, max_days_since_breakout=7,
    ))
    assert calls[0][2].min_days_since_breakout == 5
    assert calls[1][2].min_days_since_breakout == calls[1][2].max_days_since_breakout == 0
    assert merged.pattern_matches[0]["pattern_type"] == "flag_breakout"


@pytest.mark.parametrize("flag_category,expected_flags", [(C.BUY_ALERT, 1), (C.WATCH, 0), (C.NONE, 0)])
def test_replay_counts_one_trade_and_only_confirmed_flag_wins(monkeypatch, flag_category, expected_flags):
    stub_branches(monkeypatch, C.BUY_ALERT, flag_category)
    report = evaluate_bars("000001", "测试", flag_bars(), lookback_days=1, config=StrategyConfig(strategy_mode="combined"))
    assert report["summary"]["signal_count"] == 1
    assert len(report["signals"]) == 1
    assert sum(r["signal_count"] for r in report["summary"]["flag_directions"]) == expected_flags
    assert report["pattern_events"], "失效/形成中的观察事件仍应留存，但不是旗形交易"


def test_scanner_fetches_each_stock_once_and_emits_one_result(monkeypatch):
    stub_branches(monkeypatch, C.BUY_ALERT, C.BUY_ALERT)
    fetched, progress = [], []
    def fetch(code):
        fetched.append(code)
        return flag_bars()
    provider = SimpleNamespace(
        name="fake", get_daily_bars=fetch,
        get_stock_list=lambda: pd.DataFrame([{"stock_code": "000001", "short_name": "测试"}]),
    )
    scanner = Scanner(provider=provider)
    matches = scanner.run_scan({"strategy_mode": "combined", "provider": "fake", "signal_offset": 0},
                               on_progress=lambda stats, hit: progress.append(hit))
    assert fetched == ["000001"]
    assert len(matches) == len(progress) == 1
    assert len(matches[0].to_dict()["pattern_matches"]) == 2


def test_real_union_matches_single_engines_and_is_point_in_time():
    bars = flag_bars("ascending")
    cfg = StrategyConfig(strategy_mode="combined", signal_offset=0)
    merged = analyze_bars("000001", "测试", bars, cfg)
    expected = []
    for mode in ("precision_pullback", "flag_breakout"):
        single = analyze_bars("000001", "测试", bars, replace(cfg, strategy_mode=mode))
        if single and single.category != C.NONE:
            expected.append(mode)
    assert [m["pattern_type"] for m in merged.pattern_matches] == expected
    extra = bars.tail(3).copy()
    extra["trade_date"] = pd.bdate_range(bars.trade_date.iloc[-1] + pd.Timedelta(days=1), periods=3)
    extra[["open", "high", "low", "close"]] *= 2
    after = analyze_bars("000001", "测试", pd.concat([bars, extra], ignore_index=True), replace(cfg, signal_offset=3))
    assert merged.to_dict() == after.to_dict()


def test_api_default_union_and_tag_persistence_after_reload(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from app.main import create_app

    stub_branches(monkeypatch, C.BUY_ALERT, C.BUY_ALERT)
    provider = SimpleNamespace(
        name="fake", get_daily_bars=lambda code: flag_bars(),
        get_stock_list=lambda: pd.DataFrame([{"stock_code": "000001", "short_name": "测试"}]),
    )
    path = tmp_path / "combined.db"
    client = TestClient(create_app(db_path=path, scanner=Scanner(provider=provider)))
    assert client.get("/api/config/defaults").json()["strategy_mode"] == "combined"
    submitted = client.post("/api/scans/run", json={"provider": "fake", "signal_offset": 0}).json()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get(f"/api/scans/{submitted['scan_id']}/progress").json()
        if state["status"] != "running":
            break
        time.sleep(.01)
    assert state["status"] == "completed"
    reloaded = TestClient(create_app(db_path=path, scanner=Scanner(provider=provider)))
    scan = reloaded.get("/api/scans/latest").json()
    assert scan["parameters"]["strategy_mode"] == "combined"
    assert scan["parameters"]["max_prebreakout_extension_pct"] == .5
    assert scan["parameters"]["prebreakout_reject_above_days"] == 30
    assert scan["parameters"]["prebreakout_above_buffer_pct"] == .02
    assert scan["parameters"]["require_breakout_volume"] is False
    assert scan["total_count"] == scan["buy_alert_count"] == 1
    assert len(scan["results"][0]["pattern_matches"]) == 2


def test_stale_detail_does_not_keep_confirmed_match_labels(monkeypatch):
    from app.services.bars_cache import with_data_quality
    stub_branches(monkeypatch, C.BUY_ALERT, C.BUY_ALERT)
    provider = SimpleNamespace(name="fake", get_daily_bars=lambda code: with_data_quality(flag_bars(), "2025-07-15", "offline"))
    detail = Scanner(provider=provider).stock_detail("000001", "测试", scan_params={"strategy_mode": "combined", "signal_offset": 0})
    assert detail["analysis"]["found"] is False
    matches = detail["analysis"]["result"]["pattern_matches"]
    assert len(matches) == 2
    assert all(item["category"] == "watch" and item["data_stale"] for item in matches)


def test_flag_direction_does_not_exclude_a_precision_match():
    bars = flag_bars("ascending")
    # Small synthetic swings must form a precision line as well as a flag for this test.
    cfg = StrategyConfig(signal_offset=0, require_pullback=False, swing_reversal_atr_ratio=.3,
                         swing_reversal_min_pct=.4, swing_reversal_max_pct=1.5)
    precision = analyze_bars("000001", "测试", bars, cfg)
    assert precision is not None and precision.category != C.NONE
    combined = analyze_bars("000001", "测试", bars, replace(cfg, strategy_mode="combined", flag_direction="horizontal"))
    assert combined.category == precision.category
    assert [item["pattern_type"] for item in combined.pattern_matches] == ["precision_pullback"]
