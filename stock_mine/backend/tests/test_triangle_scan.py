"""三角整理扫描：买入只在向上突破的收盘确认日，而且截断到当天结果不变。"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.schemas import ScanRequest
from app.services.chart_patterns import confirmation_volume_ratio, triangle_up_breakouts
from app.services.strategy import StrategyConfig, _run_triangle_pipeline, _triangle_buy_result

TRIANGLE_KNOTS = [(0, 9.0), (15, 12.0), (30, 10.0), (42, 11.6), (54, 10.4), (75, 12.5), (85, 12.3)]


def path_bars(knots: list[tuple[int, float]]) -> pd.DataFrame:
    xs, ys = zip(*knots)
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys)
    opens = np.r_[closes[0], closes[:-1]]
    start = date(2024, 1, 1)
    return pd.DataFrame({
        "trade_date": [pd.Timestamp(start + timedelta(days=i)) for i in range(len(closes))],
        "open": opens, "high": np.maximum(opens, closes) * 1.005, "low": np.minimum(opens, closes) * 0.995,
        "close": closes, "volume": np.full(len(closes), 1000.0),
    })


def test_triangle_mode_is_explicit_and_does_not_replace_the_default_scan():
    assert ScanRequest().strategy_mode == "combined"
    assert ScanRequest(strategy_mode="triangle_breakout").strategy_mode == "triangle_breakout"


def test_upward_breakout_is_a_buy_only_on_the_confirmation_close():
    bars = path_bars(TRIANGLE_KNOTS)
    events = triangle_up_breakouts(bars)
    assert len(events) == 1
    entry = events[0]["signal_index"]
    cfg = StrategyConfig(strategy_mode="triangle_breakout", signal_offset=0)
    stage, _message, result = _run_triangle_pipeline("000001", "测试", bars.iloc[:entry + 1].reset_index(drop=True), cfg)
    assert stage == "ok" and result.category.value == "buy_alert"
    assert result.pattern_type == "triangle_breakout"
    assert result.signal_event_id.endswith(events[0]["event_id"])
    earlier = bars.iloc[:entry].reset_index(drop=True)
    stage, _message, result = _run_triangle_pipeline("000001", "测试", earlier, cfg)
    assert result is None or result.category.value != "buy_alert"


def test_breakout_does_not_use_bars_after_the_signal_day():
    bars = path_bars(TRIANGLE_KNOTS)
    (event,) = triangle_up_breakouts(bars)
    entry = event["signal_index"]
    prefix = triangle_up_breakouts(bars.iloc[:entry + 1].reset_index(drop=True))
    assert [item["event_id"] for item in prefix] == [event["event_id"]]
    before = triangle_up_breakouts(bars.iloc[:entry].reset_index(drop=True))
    assert event["event_id"] not in {item["event_id"] for item in before}
    # 确认日之后再补一段无关走势，不应改掉已经确认的这笔。
    extra = bars.copy()
    last_close = float(extra["close"].iloc[-1])
    start = extra["trade_date"].iloc[-1] + pd.Timedelta(days=1)
    tail = pd.DataFrame({
        "trade_date": [start + pd.Timedelta(days=i) for i in range(15)],
        "open": last_close, "high": last_close * 1.01, "low": last_close * 0.99,
        "close": last_close, "volume": 1000.0,
    })
    extended = pd.concat([extra, tail], ignore_index=True)
    assert any(item["event_id"] == event["event_id"] for item in triangle_up_breakouts(extended))


def test_volume_gate_uses_the_existing_breakout_multiple_and_prior_bars_only():
    bars = path_bars(TRIANGLE_KNOTS)
    (event,) = triangle_up_breakouts(bars)
    entry = event["signal_index"]
    quiet = bars.copy()
    gated = StrategyConfig(strategy_mode="triangle_breakout", signal_offset=0, require_breakout_volume=True)
    assert gated.breakout_vol_mult == 1.5 and gated.vol_lookback == 10
    stage, _message, result = _run_triangle_pipeline("000001", "测试", quiet.iloc[:entry + 1].reset_index(drop=True), gated)
    assert stage == "triangle_volume_unconfirmed" and result.category.value == "watch"
    loud = quiet.copy()
    loud.loc[loud.index[entry], "volume"] = 1000.0 * 2
    ratio = confirmation_volume_ratio(loud.iloc[:entry + 1].reset_index(drop=True), entry, gated.vol_lookback)
    assert ratio is not None and ratio >= 1.5
    stage, _message, result = _run_triangle_pipeline("000001", "测试", loud.iloc[:entry + 1].reset_index(drop=True), gated)
    assert stage == "ok" and result.category.value == "buy_alert"
    # 门槛关闭时，同样的平量突破仍是买入。这是上一轮基线规则。
    open_gate = StrategyConfig(strategy_mode="triangle_breakout", signal_offset=0, require_breakout_volume=False)
    stage, _message, result = _run_triangle_pipeline("000001", "测试", quiet.iloc[:entry + 1].reset_index(drop=True), open_gate)
    assert result.category.value == "buy_alert"


def test_close_extension_gate_uses_the_existing_near_band_and_stays_off_by_default():
    stamp = pd.Timestamp("2024-06-03")
    bars = pd.DataFrame({
        "trade_date": [stamp], "open": [10.4], "high": [10.6], "low": [10.2],
        "close": [10.5], "volume": [1000.0],
    })
    event = {"signal_index": 0, "detected_index": 0, "upper": 10.0, "width_ratio": 0.4,
             "event_id": "triangle:0:0", "stop": 9.0}
    default = StrategyConfig(strategy_mode="triangle_breakout", signal_offset=0)
    assert default.triangle_max_close_extension_pct is None
    assert default.max_extension_pct == 0.035
    assert _triangle_buy_result("000001", "测试", bars, event, default).category.value == "buy_alert"
    gated = StrategyConfig(strategy_mode="triangle_breakout", signal_offset=0, triangle_max_close_extension_pct=0.035)
    assert _triangle_buy_result("000001", "测试", bars, event, gated).category.value == "watch"
    near = bars.copy()
    near.loc[0, "close"] = 10.2
    assert _triangle_buy_result("000001", "测试", near, event, gated).category.value == "buy_alert"


def test_dates_cover_the_synthetic_path():
    bars = path_bars([(0, 10.0), (5, 11.0)])
    assert bars["trade_date"].iloc[0] == pd.Timestamp(date(2024, 1, 1))
    assert len(bars) == 6
    assert np.isfinite(bars["close"]).all()
    assert bars["trade_date"].iloc[-1] == pd.Timestamp(date(2024, 1, 1) + timedelta(days=5))
