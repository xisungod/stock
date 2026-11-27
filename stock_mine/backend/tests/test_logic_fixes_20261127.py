"""2026-11-27 选股逻辑修复的回归测试：每条规则都覆盖「新规则拦下」与「正常形态不受影响」。"""
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from app.schemas import BacktestRequest, RuleDefaults, ScanRequest
from app.services.backtest import evaluate_bars
from app.services.market_regime import (
    MarketRegimeService, apply_market_regime, regime_on, regime_table,
)
from app.services.scanner import scan_sort_key
from app.services.strategy import (
    LIMIT_UP_BREAKOUT_RISK, SignalCategory, SignalResult, StrategyConfig,
    _confirm_pullback, _pullback_conditions, _select_best_signal, analyze_bars,
    build_strategy_config, daily_price_limit, is_limit_up_close,
)
from test_platform_events import seeded_engine
from test_pullback_window import BREAKOUT_INDEX, bars_and_line, fixed_line

NEW_RULES = dict(pullback_max_normal_volume_ratio=1.0, max_post_breakout_runup_pct=0.10, max_event_age_bars=750)


def legacy(**changes):
    return StrategyConfig(**{"signal_offset": 0, "include_chart": False, **changes})


# ---- 1. 信号日：offset=0 是最近已收盘交易日 ---------------------------------------------

def with_provisional_bar(bars):
    """在已收盘的回踩日后面接一根盘中未定型、已跌破线的K线。"""
    extra = bars.iloc[[-1]].copy()
    extra["trade_date"] = bars.trade_date.iloc[-1] + pd.offsets.BDay(1)
    extra[["open", "high", "low", "close"]] = [9.5, 9.6, 8.9, 9.0]
    frame = pd.concat([bars, extra], ignore_index=True)
    frame.attrs["data_quality"] = {
        "status": "fresh", "expected_bar_date": bars.trade_date.iloc[-1].strftime("%Y-%m-%d"),
        "provisional_last_bar": True,
    }
    return frame


def test_offset_zero_drops_only_the_unsettled_intraday_bar(monkeypatch):
    bars, line = bars_and_line(3)
    fixed_line(monkeypatch, line)
    frame = with_provisional_bar(bars.drop(columns="ma"))
    result = analyze_bars("000001", "测试", frame, legacy())
    assert result.signal_date == bars.trade_date.iloc[-1].strftime("%Y-%m-%d")
    assert result.category == SignalCategory.BUY_ALERT


def test_without_freshness_metadata_offset_zero_uses_the_last_row(monkeypatch):
    bars, line = bars_and_line(3)
    fixed_line(monkeypatch, line)
    frame = with_provisional_bar(bars.drop(columns="ma"))
    frame.attrs = {}
    result = analyze_bars("000001", "测试", frame, legacy())
    assert result.signal_date == frame.trade_date.iloc[-1].strftime("%Y-%m-%d")
    assert result.category != SignalCategory.BUY_ALERT


def test_saved_batches_keep_the_literal_drop_last_n_meaning(monkeypatch):
    bars, line = bars_and_line(3)
    fixed_line(monkeypatch, line)
    settled = bars.drop(columns="ma")
    settled.attrs["data_quality"] = {"status": "fresh", "expected_bar_date": settled.trade_date.iloc[-1].strftime("%Y-%m-%d")}
    # 收盘后旧版 offset=1 丢掉的是已收盘的当天——这正是被修复的问题，但旧批次保持原样。
    old = analyze_bars("000001", "测试", settled, legacy(signal_offset=1))
    assert old.signal_date == settled.trade_date.iloc[-2].strftime("%Y-%m-%d")


def test_new_scans_default_to_the_same_signal_bar_as_backtests():
    assert ScanRequest().signal_offset == 0
    assert build_strategy_config(ScanRequest().model_dump()).signal_offset == 0
    assert build_strategy_config({}).signal_offset == 1, "旧批次缺字段时保持原值"


# ---- 2. 缩量必须相对常态量 ----------------------------------------------------------------

def test_clean_pullback_still_alerts_under_all_new_rules():
    bars, line = bars_and_line(3)
    result = _select_best_signal("000001", "测试", bars, [line], legacy(**NEW_RULES))
    assert result.category == SignalCategory.BUY_ALERT
    assert any("突破前 10 日均量的 1.00 倍" in item for item in result.reasons)


def test_pullback_above_normal_volume_is_not_a_shrinking_pullback():
    bars, line = bars_and_line(3)
    bars.loc[len(bars) - 1, "volume"] = 1500  # 突破日 3000 的 0.5 倍，但为常态量 1000 的 1.5 倍
    before = _select_best_signal("000001", "测试", bars, [line], legacy())
    after = _select_best_signal("000001", "测试", bars, [line], legacy(**NEW_RULES))
    assert before.category == SignalCategory.BUY_ALERT
    assert after.category == SignalCategory.NONE
    assert any("1.50 倍" in item and "不算缩量回踩" in item for item in after.risks)


def test_platform_engine_applies_the_same_normal_volume_rule():
    def run(**changes):
        engine = seeded_engine([9.8] * 8 + [10.2, 10.07])
        engine.cfg = replace(engine.cfg, **changes)
        engine.volume = engine.volume.copy()
        engine.volume[8], engine.volume[9] = 2500, 1600
        engine.frame.loc[8, "volume"], engine.frame.loc[9, "volume"] = 2500, 1600
        engine.feed(8)
        engine.feed(9)
        return engine.result(9)

    assert run().category == SignalCategory.BUY_ALERT
    blocked = run(pullback_max_normal_volume_ratio=1.0)
    assert blocked.category == SignalCategory.WATCH
    assert any("不算缩量回踩" in item for item in blocked.risks)


# ---- 3. 冲高回落不是回踩；涨停突破要提示 -------------------------------------------------

# 北陆药业 300016：08-21 涨停突破 8.89，冲到 10.72 后连跌到 9.02，旧规则判为「缩量回踩第5天」。
BEILU = [
    ("2026-08-07", 7.9, 8.41, 7.9, 8.41, 355605), ("2026-08-10", 8.41, 8.69, 8.41, 8.54, 351499),
    ("2026-08-11", 8.5, 8.67, 8.47, 8.52, 212223), ("2026-08-12", 8.59, 8.8, 8.5, 8.62, 277640),
    ("2026-08-13", 8.55, 8.94, 8.41, 8.75, 391024), ("2026-08-14", 8.69, 8.9, 8.59, 8.73, 223241),
    ("2026-08-17", 8.75, 8.77, 8.44, 8.71, 206964), ("2026-08-18", 8.68, 8.82, 8.57, 8.59, 151691),
    ("2026-08-19", 8.55, 8.65, 8.26, 8.3, 166495), ("2026-08-20", 8.6, 8.94, 8.6, 8.7, 340840),
    ("2026-08-21", 8.61, 10.44, 8.48, 10.44, 1273508), ("2026-08-24", 10.3, 10.72, 9.54, 9.74, 1134324),
    ("2026-08-25", 9.74, 9.79, 9.38, 9.71, 682810), ("2026-08-26", 9.79, 10.2, 9.44, 9.47, 592294),
    ("2026-08-27", 9.34, 9.63, 9.24, 9.35, 507400), ("2026-08-28", 9.28, 9.3, 8.99, 9.02, 456722),
]


def test_real_case_beilu_is_rejected_as_a_give_back_and_heavy_volume():
    frame = pd.DataFrame(BEILU, columns=["trade_date", "open", "high", "low", "close", "volume"])
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    index = 10
    breakout = {"index": index, "volume": 1273508.0, "avg_volume": float(frame.volume.iloc[:index].mean())}
    assert _confirm_pullback(frame, 8.89, breakout, legacy()) is not None, "旧规则确实放行"
    conditions = _pullback_conditions(frame, 8.89, breakout, legacy(**NEW_RULES))
    assert conditions["normal_volume_ratio"] == pytest.approx(1.706, abs=1e-3)
    assert conditions["max_runup_pct"] == pytest.approx(20.58, abs=0.01)
    assert conditions["normal_volume_ok"] is False and conditions["runup_ok"] is False
    assert _confirm_pullback(frame, 8.89, breakout, legacy(**NEW_RULES)) is None
    assert is_limit_up_close(frame, index, "300016", "北陆药业")


def test_runup_beyond_the_limit_blocks_the_pullback_alert():
    bars, line = bars_and_line(3)
    bars.loc[BREAKOUT_INDEX + 1, "high"] = 11.5
    assert _select_best_signal("000001", "测试", bars, [line], legacy()).category == SignalCategory.BUY_ALERT
    after = _select_best_signal("000001", "测试", bars, [line], legacy(**NEW_RULES))
    assert after.category == SignalCategory.NONE
    assert any("越线 15.0%" in item and "冲高回落" in item for item in after.risks)


@pytest.mark.parametrize("code,name,date,expected", [
    ("600000", "浦发银行", "2026-01-05", 0.10), ("300750", "宁德时代", "2026-01-05", 0.20),
    ("300750", "宁德时代", "2020-08-21", 0.10), ("688981", "中芯国际", "2026-01-05", 0.20),
    ("830799", "艾融软件", "2026-01-05", 0.30), ("000001", "*ST测试", "2026-01-05", 0.05),
])
def test_daily_price_limits_by_board_and_date(code, name, date, expected):
    assert daily_price_limit(code, name, date) == expected


def test_limit_up_breakout_is_flagged_without_hiding_the_signal():
    bars, line = bars_and_line(3)
    bars.loc[BREAKOUT_INDEX, ["close", "high"]] = [10.78, 10.78]  # 9.8 → 10.78，主板涨停
    result = _select_best_signal("600000", "测试", bars, [line], legacy(**NEW_RULES))
    assert result.breakout_limit_up is True
    assert LIMIT_UP_BREAKOUT_RISK in result.risks
    assert result.category == SignalCategory.BUY_ALERT
    assert _select_best_signal("300001", "测试", bars, [line], legacy(**NEW_RULES)).breakout_limit_up is False


# ---- 4. 验证证据时效 ---------------------------------------------------------------------

def test_legacy_line_needs_recent_confirmations_for_a_buy_alert():
    bars, line = bars_and_line(3)  # 阻力点 20/60/100，突破 140
    recent = _select_best_signal("000001", "测试", bars, [line], legacy(**NEW_RULES))
    stale = _select_best_signal("000001", "测试", bars, [line], legacy(**{**NEW_RULES, "max_event_age_bars": 50}))
    assert recent.category == SignalCategory.BUY_ALERT
    assert stale.category == SignalCategory.WATCH
    assert any("内阻力确认仅 1 次" in item for item in stale.risks)


def test_platform_breakout_freezes_the_recent_event_count():
    def run(age):
        engine = seeded_engine([9.8] * 8 + [10.2, 10.07])
        engine.cfg = replace(engine.cfg, max_event_age_bars=age)
        engine.volume = engine.volume.copy()
        engine.volume[8] = 2500
        engine.frame.loc[8, "volume"] = 2500
        engine.feed(8)
        engine.feed(9)
        return engine.result(9)

    assert run(10).category == SignalCategory.BUY_ALERT
    stale = run(4)  # 事件枢轴 1/3/5，突破在 8：近4根内只有 5
    assert stale.category == SignalCategory.WATCH
    assert stale.line_snapshot["recent_event_count"] == 1
    assert any("独立验证事件仅 1 个" in item for item in stale.risks)


# ---- 5. 排序不再由行业快照决定 -------------------------------------------------------------

def result(code, category, score, distance):
    return SignalResult(code, code, category, score, "", 10.0, 3, "2026-08-28", distance_to_line_pct=distance)


def test_equal_scores_are_ordered_by_distance_to_the_line():
    items = [result("000003", SignalCategory.BUY_ALERT, 95, 2.5), result("000002", SignalCategory.WATCH, 95, 0.1),
             result("000001", SignalCategory.BUY_ALERT, 95, -0.2), result("000004", SignalCategory.BUY_ALERT, 90, 0.0)]
    assert [item.stock_code for item in sorted(items, key=scan_sort_key)] == ["000001", "000003", "000004", "000002"]


# ---- 6. 大盘环境 -------------------------------------------------------------------------

def index_frame(closes, start="2025-01-01"):
    return pd.DataFrame({"trade_date": pd.bdate_range(start, periods=len(closes)), "close": closes})


def test_regime_uses_only_data_up_to_the_signal_day():
    closes = np.r_[np.linspace(100, 130, 60), np.linspace(130, 110, 20)]
    table = regime_table(index_frame(closes))
    dates = list(table.index)
    assert regime_on(table, dates[59])["weak"] is False
    assert regime_on(table, dates[79])["weak"] is True
    assert regime_on(table, dates[5]) is None, "指标未成形不判断"
    assert regime_on(table, "2030-01-02") is None, "缺当日行情绝不拿旧日子顶替"
    shorter = regime_table(index_frame(closes[:60]))
    assert regime_on(shorter, dates[59]) == regime_on(table, dates[59]), "后续行情不改写当天判断"


def test_market_modes_annotate_or_downgrade():
    weak = {"benchmark": "沪深300", "symbol": "sh000300", "date": "2026-08-28", "status": "evaluated",
            "weak": True, "rule": "", "return_20_pct": -3.2, "ma20_slope_5_pct": -0.4, "distance_to_ma20_pct": -2.0}
    off = apply_market_regime(result("000001", SignalCategory.BUY_ALERT, 95, 0), weak, "off")
    risk = apply_market_regime(result("000001", SignalCategory.BUY_ALERT, 95, 0), weak, "risk")
    down = apply_market_regime(result("000001", SignalCategory.BUY_ALERT, 95, 0), weak, "downgrade")
    missing = apply_market_regime(result("000001", SignalCategory.BUY_ALERT, 95, 0), None, "downgrade")
    assert off.market_regime is None and off.category == SignalCategory.BUY_ALERT
    assert risk.category == SignalCategory.BUY_ALERT and any("大盘偏弱" in item for item in risk.risks)
    assert down.category == SignalCategory.WATCH and down.market_regime["downgraded"] is True
    assert missing.category == SignalCategory.BUY_ALERT and missing.market_regime["status"] == "unavailable"


def test_backtest_downgrade_is_point_in_time_and_never_silent(monkeypatch):
    bars, line = bars_and_line(3)
    fixed_line(monkeypatch, line)
    frame = bars.drop(columns="ma")
    calendar = frame.trade_date.dt.strftime("%Y-%m-%d").tolist()
    falling = regime_table(index_frame(np.linspace(130, 100, len(frame))))
    base = legacy(min_history_bars=120)
    with pytest.raises(ValueError, match="沪深300"):
        evaluate_bars("000001", "测试", frame, config=replace(base, market_filter="downgrade"), calendar=calendar)
    annotated = evaluate_bars("000001", "测试", frame, config=replace(base, market_filter="risk"),
                              calendar=calendar, market_regimes=falling)
    filtered = evaluate_bars("000001", "测试", frame, config=replace(base, market_filter="downgrade"),
                             calendar=calendar, market_regimes=falling)
    assert annotated["signals"] and all(item["market_weak"] is True for item in annotated["signals"])
    assert filtered["signals"] == []


def test_regime_service_never_caches_an_unsettled_index_bar(tmp_path):
    china = ZoneInfo("Asia/Shanghai")
    clock = {"now": datetime(2026, 11, 27, 10, 0, tzinfo=china)}  # 盘中：最近已收盘为 11-26
    rows = index_frame(np.linspace(100, 120, 45), start="2026-10-01")
    rows = rows[rows.trade_date <= "2026-11-27"]
    rows["trade_date"] = rows.trade_date.dt.strftime("%Y-%m-%d")
    calls = []

    def fetch():
        calls.append(1)
        return rows

    service = MarketRegimeService(tmp_path, fetch=fetch, now=lambda: clock["now"])
    table = service.table()
    assert table.index.max() == "2026-11-26"
    service.table()
    assert len(calls) == 1, "同一交易日只取一次"

    def broken():
        raise ConnectionError("offline")

    clock["now"] = datetime(2026, 11, 27, 16, 0, tzinfo=china)  # 收盘后需要 11-27，但取数失败
    fallback = MarketRegimeService(tmp_path, fetch=broken, now=lambda: clock["now"])
    stale = fallback.table()
    assert stale.index.max() == "2026-11-26"
    assert "行情获取失败" in fallback.last_error
    assert regime_on(stale, "2026-11-27") is None, "缺当日数据判为未评估，而不是沿用昨天"


# ---- 7. 新旧批次的参数口径 ----------------------------------------------------------------

def test_scan_and_backtest_share_rule_defaults_and_old_batches_disable_them():
    shared = RuleDefaults().model_dump()
    assert {name: ScanRequest().model_dump()[name] for name in shared} == shared
    assert {name: BacktestRequest().model_dump()[name] for name in shared} == shared
    new = build_strategy_config(ScanRequest().model_dump())
    assert (new.pullback_max_normal_volume_ratio, new.max_post_breakout_runup_pct,
            new.max_event_age_bars, new.market_filter) == (1.0, 0.10, 750, "risk")
    old = build_strategy_config({"precision_engine": "platform_events"})
    assert (old.pullback_max_normal_volume_ratio, old.max_post_breakout_runup_pct,
            old.max_event_age_bars, old.market_filter) == (None, None, None, "off")


@pytest.mark.parametrize("changes", [
    {"pullback_max_normal_volume_ratio": 0}, {"max_post_breakout_runup_pct": -0.1},
    {"max_event_age_bars": 0}, {"market_filter": "always"},
])
def test_invalid_rule_values_are_rejected(changes):
    with pytest.raises(ValueError):
        StrategyConfig(**changes)


def test_downgrade_backtest_without_index_data_is_flagged_not_failed(monkeypatch):
    """离线数据源取不到沪深300：与实盘一样不降级，并在结果里标明大盘数据缺失。"""
    from types import SimpleNamespace

    from app.services.scanner import Scanner

    bars, line = bars_and_line(3)
    fixed_line(monkeypatch, line)
    frame = bars.drop(columns="ma")
    provider = SimpleNamespace(
        name="fake", get_stock_list=lambda: pd.DataFrame({"stock_code": ["000001"], "short_name": ["测试"]}),
        get_daily_bars=lambda code: frame, get_full_daily_bars=lambda code, years=0: frame,
    )
    params = {**BacktestRequest(provider="fake", lookback_days=0, market_filter="downgrade").model_dump(),
              "precision_engine": "legacy", "strategy_mode": "precision_pullback"}
    assert BacktestRequest().market_filter == "risk", "默认只标注，不降级"
    reports, summary = Scanner(provider=provider).run_backtest(params)
    assert reports[0]["market_filter"] == "downgrade"
    assert reports[0]["market_regime_available"] is False
    assert summary["signal_count"] >= 1, "缺指数时不降级，信号照常计入"
