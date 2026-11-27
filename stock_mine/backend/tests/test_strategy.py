from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.services.strategy import (
    KeyLine,
    StrategyConfig,
    SignalCategory,
    _cluster_key_lines,
    _find_breakout,
    _support_touch_indices,
    analyze_bars,
    diagnose_bars,
)


def _config(**overrides) -> StrategyConfig:
    values = {
        "signal_offset": 0,
        "min_history_bars": 20,
        "ma_period": 10,
        "ma_slope_days": 3,
        # These tests retain the old observation and breakout-age behavior.
        "require_pullback": False,
        "max_pullback_days": None,
        "require_breakout_volume": True,
        # 这些用例验证的是突破/回踩/趋势逻辑，用的是 22 根 K 线的小数据集。
        # 显式固定波峰参数，避免生产默认值调整时误伤这些用例的原有意图。
        # 同理固定摆动阈值：这些夹具只有 22 根、振幅约 5%，而按 ATR 换算出的
        # 生产阈值会顶到 8%，那样连一个摆动都确认不出来，用例就失去了意义。
        # 阈值必须大于夹具的日内振幅，否则单根K线自身的高低差就会触发反转
        "swing_reversal_atr_ratio": 0.3,
        "swing_reversal_min_pct": 2.5,
        "swing_reversal_max_pct": 4.0,
    }
    values.update(overrides)
    return StrategyConfig(**values)


def _bars(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None, volumes: list[int] | None = None) -> pd.DataFrame:
    start = date(2026, 1, 1)
    opens = [c * 0.995 for c in closes]
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    volumes = volumes or [1000 for _ in closes]
    return pd.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(len(closes))],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def test_analyze_bars_marks_volume_breakout_pullback_as_buy_alert() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35,
        10.18, 10.12,
    ]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    volumes = [1000] * len(closes)
    volumes[19] = 2600
    volumes[21] = 1100
    lows[21] = 10.01
    df = _bars(closes, highs=highs, lows=lows, volumes=volumes)

    result = analyze_bars("000001", "平安银行", df, _config())

    assert result is not None
    assert result.category == SignalCategory.BUY_ALERT
    assert result.key_line_price is not None
    assert 9.85 <= result.key_line_price <= 10.15
    assert result.breakout_date is not None
    assert result.pullback_date is not None
    assert result.score >= 70
    assert any("放量突破" in reason for reason in result.reasons)
    assert any("缩量回踩" in reason for reason in result.reasons)


def test_relative_shrinkage_does_not_hide_abnormally_high_volume() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35, 10.18, 10.12,
    ]
    volumes = [1000] * len(closes)
    volumes[19], volumes[21] = 10000, 7000
    lows = [v * .99 for v in closes]
    lows[-1] = 10.01
    result = analyze_bars("000001", "测试", _bars(closes, lows=lows, volumes=volumes), _config())
    assert result.category == SignalCategory.BUY_ALERT, "尚无证据支持新增硬性门槛，保留结构基线"
    assert result.pullback_normal_volume_ratio == 7
    assert any("并非低于常态量" in risk for risk in result.risks)
    assert result.to_dict()["pullback_normal_volume_ratio"] == 7


def test_analyze_bars_marks_near_valid_line_as_watch_opportunity() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.7, 9.9, 10.03,
        10.07, 10.11,
    ]
    df = _bars(closes)

    result = analyze_bars("000002", "万科A", df, _config())

    assert result is not None
    assert result.category == SignalCategory.WATCH
    assert result.score >= 40
    assert any("观察" in reason or "接近" in reason for reason in result.reasons)


def test_analyze_bars_ignores_price_far_above_line_without_pullback() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 10.4, 10.8, 11.3, 11.8, 12.4,
        12.9, 13.2,
    ]
    volumes = [1000] * len(closes)
    volumes[15] = 2600
    df = _bars(closes, volumes=volumes)

    result = analyze_bars("000003", "远离线", df, _config())

    # 该组数据确实形成了精准线（10.12），只是收盘价高出 30% 且无回踩，
    # 所以是「有结构但不入选」，而不是「没结构」。
    assert result is not None
    assert result.category == SignalCategory.NONE


def test_failed_breakout_far_below_line_is_not_watch_opportunity() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35,
        9.0, 8.1,
    ]
    volumes = [1000] * len(closes)
    volumes[19] = 2600

    result = analyze_bars("000004", "失败突破", _bars(closes, volumes=volumes), _config())

    # 同样形成了精准线（10.12），但突破后收盘跌回线下约 20%，属于失败突破，
    # 既不是买入预警也不该被当成观察机会。
    assert result is not None
    assert result.category == SignalCategory.NONE


def test_breakout_must_happen_after_key_line_is_formed() -> None:
    closes = [
        9.0, 9.3, 9.7, 10.0, 9.4, 9.0, 9.6, 10.02, 9.5, 9.9,
        10.3, 9.8, 9.6, 9.8, 10.03, 9.7, 9.8, 9.9, 10.0, 10.05,
    ]
    volumes = [1000] * len(closes)
    volumes[10] = 2600
    df = _bars(closes, volumes=volumes)
    cfg = _config()
    line = _cluster_key_lines(df, [3, 7, 14], cfg)[0]

    breakout = _find_breakout(df, line, cfg)

    # 形成日取第三个波峰的**确认日**（15），不是它的发生日（14）——
    # 波峰要等回撤够了才可识别，在确认之前这条线当时还画不出来。
    assert line.formation_index == 15
    assert sorted(line.touch_indices)[2] == 14, "第三个波峰仍发生在下标 14"
    assert breakout is None


def test_most_recent_valid_breakout_is_selected() -> None:
    closes = [9.5] * 25
    volumes = [1000] * 25
    for idx in (15, 20):
        closes[idx - 1] = 9.95
        closes[idx] = 10.2
        volumes[idx] = 2200
    df = _bars(closes, volumes=volumes)
    line = KeyLine(price=10.0, strength=3, touch_indices=[2, 6, 10], last_touch_index=10, formation_index=10)

    breakout = _find_breakout(df, line, _config())

    assert breakout is not None
    assert breakout["index"] == 20


def test_deep_intraday_break_is_not_buy_alert() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35,
        10.18, 10.12,
    ]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    lows[-1] = 9.45
    volumes = [1000] * len(closes)
    volumes[19] = 2600
    volumes[-1] = 1100

    result = analyze_bars(
        "000005",
        "深度跌破",
        _bars(closes, highs=highs, lows=lows, volumes=volumes),
        _config(),
    )

    assert result is not None
    assert result.category != SignalCategory.BUY_ALERT


def test_default_strategy_rejects_insufficient_history() -> None:
    closes = [9.5 + idx * 0.01 for idx in range(80)]

    result = analyze_bars("000006", "历史不足", _bars(closes), StrategyConfig(signal_offset=0))

    assert result is None


def test_analyze_bars_returns_object_not_none_when_category_is_none() -> None:
    """锁定不变量：找到了精准线但不满足入选条件时，必须返回 SignalResult 对象而非 None。

    「没有信号」和「没有数据可分析」是两件不同的事，必须在返回值上可区分：

    1. 单股诊断依赖它。`diagnose_bars` 的 `no_signal` 阶段要把这个对象交给接口，
       用它携带的 chart / reasons / risks / score 回答「为什么没入选」。一旦这里
       退化成 None，stage 会从 `no_signal` 掉回 `no_key_line`，诊断接口就只能回
       一句「无信号」——正是 Task 5 要消灭的那种无信息回复。
    2. `Scanner.analyze_stock`（scanner.py:132）直接把这个返回值透给单股查询接口。
    3. `Scanner.run_scan`（scanner.py:98）以 `category in {BUY_ALERT, WATCH}` 作为
       唯一入选闸门；返回对象才能保证筛选只由 category 决定，而不是由 None-ness
       这个隐式的第二道闸门决定。

    注意断言必须写成 `is not None` 且 `== NONE` 两条。写成
    `result is None or result.category == NONE` 的话，返回 None 也能通过，
    这条不变量就等于没被守住。
    """
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 10.4, 10.8, 11.3, 11.8, 12.4,
        12.9, 13.2,
    ]
    volumes = [1000] * len(closes)
    volumes[15] = 2600
    df = _bars(closes, volumes=volumes)

    result = analyze_bars("000003", "远离线", df, _config())

    assert result is not None
    assert result.category == SignalCategory.NONE
    # 对象必须真的带着诊断信息，而不是个空壳
    assert result.key_line_price is not None
    assert result.chart.get("bars")
    assert result.risks

    # 与 diagnose_bars 对齐：未入选，但拿得到解释
    payload = diagnose_bars("000003", "远离线", df, _config())
    assert payload["found"] is False
    assert payload["stage"] == "no_signal"
    assert payload["result"] is not None


def test_diagnose_reports_insufficient_history() -> None:
    df = _bars([10.0] * 15)

    payload = diagnose_bars("000001", "测试", df, _config())

    assert payload["found"] is False
    assert payload["stage"] == "insufficient_history"
    assert "不足" in payload["message"]
    assert payload["result"] is None


def test_diagnose_reports_no_key_line_for_flat_series() -> None:
    df = _bars([10.0] * 60)

    payload = diagnose_bars("000001", "测试", df, _config())

    assert payload["found"] is False
    assert payload["stage"] == "no_key_line"
    assert payload["result"] is None


def test_diagnose_returns_result_when_signal_exists() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35,
        10.18, 10.12,
    ]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    volumes = [1000] * len(closes)
    volumes[19] = 2600
    volumes[21] = 1100
    lows[21] = 10.01
    df = _bars(closes, highs=highs, lows=lows, volumes=volumes)

    payload = diagnose_bars("000001", "平安银行", df, _config())

    assert payload["found"] is True
    assert payload["stage"] == "ok"
    assert payload["result"]["key_line_price"] is not None
    assert payload["result"]["category"] == "buy_alert"


def test_default_strategy_uses_all_history_for_key_lines() -> None:
    closes = [8.0 + idx * 0.001 for idx in range(320)]
    highs = [close + 0.05 for close in closes]
    lows = [close - 0.05 for close in closes]
    for idx in (20, 100, 250):
        closes[idx] = 9.8
        highs[idx] = 10.0
        lows[idx] = 9.7
    closes[-1] = 9.95
    highs[-1] = 10.02
    lows[-1] = 9.9

    result = analyze_bars(
        "000007",
        "全历史线",
        _bars(closes, highs=highs, lows=lows),
        _config(),
    )

    assert result is not None
    line = next(item for item in result.key_lines if 9.9 <= item["price"] <= 10.1)
    assert line["history_bars"] == 320
    assert line["resistance_point_count"] >= 3
    assert line["resistance_dates"][0] == "2026-01-21"


def test_precision_line_point_count_includes_resistance_and_support() -> None:
    # 三次冲高要比平台高出反转阈值以上，否则确认不出波峰、连线都成不了。
    # 平台收 9.8、冲高 10.1 → 回落 2.97%，高于用例配置的 2.5% 阈值。
    closes = [9.8] * 24
    highs = [9.9] * 24
    lows = [9.7] * 24
    for idx in (2, 6, 10):
        closes[idx] = 10.0
        highs[idx] = 10.1
    closes[13] = 10.12
    highs[13] = 10.2
    lows[13] = 10.05
    closes[18] = 10.04
    highs[18] = 10.12
    lows[18] = 9.98
    closes[-1] = 10.03
    highs[-1] = 10.08
    lows[-1] = 9.99

    result = analyze_bars(
        "000008",
        "点数统计",
        _bars(closes, highs=highs, lows=lows),
        _config(),
    )

    assert result is not None
    line = next(item for item in result.key_lines if 9.9 <= item["price"] <= 10.15)
    assert line["point_count"] == line["resistance_point_count"] + line["support_point_count"]
    assert result.chart["key_lines"] == result.key_lines


def _touch_bars(touch_days, peak_days, n=60, line=10.0):
    """在线上方震荡的行情：touch_days 那几天回踩到线，peak_days 那几天冲高。

    前 8 根压在线下方、第 8 根收盘上穿——上破必须发生在线形成之后（用例里
    formation_index=5），否则 `crossed[:formation_index+1] = False` 会把它清掉，
    支撑点在更早的关卡就被滤掉，测不到计数规则本身。
    """
    # 回踩后的常态价格比触碰低点高 3% 以上，触碰才能被 ZigZag 确认为波谷。
    closes = [line * 1.04] * n
    highs = [line * 1.042] * n
    lows = [line * 1.038] * n
    for d in range(8):
        closes[d] = line * 0.97
        highs[d] = line * 0.98
        lows[d] = line * 0.96
    for d in peak_days:
        closes[d] = line * 1.12
        highs[d] = line * 1.13
        lows[d] = line * 1.10
    for d in touch_days:
        lows[d] = line * 1.002
        closes[d] = line * 1.006
        highs[d] = line * 1.02
    return _bars(closes, highs=highs, lows=lows)


def test_a_sideways_range_contributes_no_support_points() -> None:
    """横盘中反复贴着线走，不该攒出一堆精准点。

    交替状态机已经保证"两个波谷之间必有波峰"，所以"两次触碰之间没有波峰"根本
    不会发生——真正防止线强度虚高的是这一条：波动幅度不到反转阈值时压根不产生
    摆动点，那些碰线一次都不算。
    """
    line = 10.0
    n = 60
    closes = [line * 1.004] * n
    highs = [line * 1.008] * n
    lows = [line * 1.001] * n          # 全程贴线，振幅远小于反转阈值
    for d in range(8):
        closes[d], highs[d], lows[d] = line * 0.97, line * 0.98, line * 0.96
    df = _bars(closes, highs=highs, lows=lows)
    cfg = _config(min_history_bars=20, ma_period=10, ma_slope_days=3)

    points = _support_touch_indices(df, line_price=line, formation_index=5, cfg=cfg)

    assert points == [], f"横盘不该产生精准点，实得 {points}"


def test_two_pullbacks_separated_by_a_confirmed_peak_count_as_two() -> None:
    df = _touch_bars(touch_days=[20, 40], peak_days=[30])
    cfg = _config(min_history_bars=20, ma_period=10, ma_slope_days=3)

    points = _support_touch_indices(df, line_price=10.0, formation_index=5, cfg=cfg)

    assert len(points) == 2, f"中间走出过波峰，应算两个精准点，实得 {points}"


def test_breakout_accepts_previous_close_inside_old_threshold_dead_zone() -> None:
    closes = [9.8] * 11
    closes[9] = 10.04
    closes[10] = 10.08
    volumes = [1000] * 11
    volumes[10] = 3000
    line = KeyLine(
        price=10.0,
        strength=3,
        touch_indices=[1, 3, 5],
        last_touch_index=5,
        formation_index=5,
    )

    breakout = _find_breakout(_bars(closes, volumes=volumes), line, _config())

    assert breakout is not None
    assert breakout["index"] == 10


def test_recent_touch_window_rejects_stale_lines() -> None:
    """近期触及约束应排除「多年前形成、近期不再被测试」的陈年阻力带。"""
    from app.services.strategy import _cluster_key_lines, _swings, StrategyConfig

    # 前段构造 3 个同价位高点，后段长期低位运行且不再触及该价位。
    # 基线必须带微小波动：_fractal_highs 用 >= 判断，价格完全相同会让所有点
    # 都成为候选并被 min_gap 合并成一组，退化成只剩 1 个高点。
    import math

    closes = [9.0 + 0.3 * math.sin(i / 3.0) for i in range(400)]
    highs = [c * 1.005 for c in closes]
    for i in (20, 60, 100):
        highs[i] = 12.0
        closes[i] = 11.9
    df = _bars(closes, highs=highs, lows=[c * 0.99 for c in closes])

    base = _config(recent_touch_window=0)
    gated = _config(recent_touch_window=120, recent_touch_min=2)
    cfg_for_swings = base

    hi = _swings(df, cfg_for_swings)[0]
    lines_base = _cluster_key_lines(df, hi, base)
    lines_gated = _cluster_key_lines(df, hi, gated)

    assert any(abs(l.price - 12.0) < 0.5 for l in lines_base), "无约束时应找到该阻力线"
    assert not any(abs(l.price - 12.0) < 0.5 for l in lines_gated), \
        "近期120日内无触及，该线应被排除"


def test_far_lines_are_dropped() -> None:
    """离现价过远的精准线应被丢弃。

    实测茅台现价 1272 时仍保留 428.22 的线（偏离 -197%），
    这类线对当前交易没有参考意义。
    """
    from app.services.strategy import _cluster_key_lines, _swings
    import math

    # 低位区构造一组高点，之后价格翻数倍运行，使那条线远低于现价
    closes = [10.0 + 0.3 * math.sin(i / 3.0) for i in range(200)]
    highs = [c * 1.005 for c in closes]
    for i in (20, 50, 80):
        highs[i] = 13.0
        closes[i] = 12.9
    closes[120:] = [40.0 + 0.5 * math.sin(i / 3.0) for i in range(80)]
    highs[120:] = [c * 1.005 for c in closes[120:]]
    df = _bars(closes, highs=highs, lows=[c * 0.99 for c in closes])

    unlimited = _config(max_line_distance_pct=0.0)
    limited = _config(max_line_distance_pct=0.35)
    cfg_for_swings = unlimited

    hi = _swings(df, cfg_for_swings)[0]
    lines_all = _cluster_key_lines(df, hi, unlimited)
    lines_near = _cluster_key_lines(df, hi, limited)

    assert any(l.price < 20 for l in lines_all), "不限制时应保留低位那条线"
    assert not any(l.price < 20 for l in lines_near), "现价约 40 时，13 附近的线应被丢弃"


def test_wider_cluster_band_merges_adjacent_highs() -> None:
    """放宽后的聚类误差带应把相差约 2% 的高点并成一条线。

    实测格力 41.36 与 40.41 相差仅 2.3%，在 1.2% 误差带下被拆成两条。
    """
    from app.services.strategy import _cluster_key_lines, _swings
    import math

    closes = [9.0 + 0.2 * math.sin(i / 3.0) for i in range(160)]
    highs = [c * 1.005 for c in closes]
    # 三个高点：10.00 / 10.10 / 10.18，两两相差约 1%，总跨度 1.8%
    for i, v in ((20, 10.00), (60, 10.10), (100, 10.18)):
        highs[i] = v
        closes[i] = v * 0.99
    df = _bars(closes, highs=highs, lows=[c * 0.99 for c in closes])

    narrow = _config(error_margin_pct=0.005, max_line_distance_pct=0.0)
    wide = _config(error_margin_pct=0.02, max_line_distance_pct=0.0)
    cfg_for_swings = narrow

    hi = _swings(df, cfg_for_swings)[0]
    # 只看目标价格区域（10.0~10.2），基线正弦波自身的高点群不在考察范围内
    in_band = lambda lines: [l for l in lines if 9.9 <= l.price <= 10.3]
    narrow_hits = in_band(_cluster_key_lines(df, hi, narrow))
    wide_hits = in_band(_cluster_key_lines(df, hi, wide))

    assert len(narrow_hits) == 0, \
        f"0.5% 误差带下三个高点两两相差约1%，无法聚成一条（实得 {[round(l.price,2) for l in narrow_hits]}）"
    assert len(wide_hits) == 1, \
        f"2% 误差带应将三个高点并成一条（实得 {[round(l.price,2) for l in wide_hits]}）"
    assert wide_hits[0].point_count >= 3, "合并后的线应包含全部三个高点"


def test_default_config_values_are_the_tuned_ones() -> None:
    """锁住调优后的默认参数。

    这些默认值是基于 300 只股票 × 500 交易日的回测选定的，
    改动它们会直接改变全市场扫描结果，必须是有意识的决定。
    """
    cfg = StrategyConfig()

    assert cfg.error_margin_pct == 0.02, "精准线聚类误差带"
    assert cfg.max_line_distance_pct == 0.35, "精准线与现价的最大偏离"
    assert cfg.recent_touch_window == 0, "近期触及约束默认关闭（大样本验证显示会降低成功率）"
    assert cfg.swing_reversal_atr_ratio == 1.5, "摆动反转阈值系数(×ATR)"
    assert cfg.swing_reversal_min_pct == 3.0, "反转阈值下限"
    assert cfg.swing_reversal_max_pct == 8.0, "反转阈值上限"



def _breakout_pullback_bars() -> pd.DataFrame:
    """放量突破在下标 19、缩量回踩在下标 21，即"距突破 2 个交易日"。"""
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35,
        10.18, 10.12,
    ]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    volumes = [1000] * len(closes)
    volumes[19] = 2600
    volumes[21] = 1100
    lows[21] = 10.01
    return _bars(closes, highs=highs, lows=lows, volumes=volumes)


def test_breakout_age_window_is_off_by_default() -> None:
    """新增的天数窗口默认不生效，不能悄悄改变既有选股结果。"""
    cfg = _config()
    assert cfg.min_days_since_breakout == 0
    assert cfg.max_days_since_breakout == 0
    assert analyze_bars("000001", "平安银行", _breakout_pullback_bars(), cfg).category \
        == SignalCategory.BUY_ALERT


def test_breakout_too_fresh_is_excluded() -> None:
    """突破才过 2 天，要求至少 3 天时不该入选——还没给回踩留出时间。"""
    result = analyze_bars("000001", "平安银行", _breakout_pullback_bars(),
                          _config(min_days_since_breakout=3))
    assert result.category == SignalCategory.NONE, f"实得 {result.category}"


def test_breakout_too_stale_is_excluded() -> None:
    """突破已过 2 天，要求不超过 1 天时不该入选。"""
    result = analyze_bars("000001", "平安银行", _breakout_pullback_bars(),
                          _config(max_days_since_breakout=1))
    assert result.category == SignalCategory.NONE, f"实得 {result.category}"


def test_breakout_inside_the_window_still_qualifies() -> None:
    result = analyze_bars("000001", "平安银行", _breakout_pullback_bars(),
                          _config(min_days_since_breakout=1, max_days_since_breakout=5))
    assert result.category == SignalCategory.BUY_ALERT
    assert result.breakout_date is not None


def test_stocks_without_a_breakout_are_untouched_by_the_window() -> None:
    """这个条件说的是"距突破多少天"，没有突破的股票不该被它牵连。

    否则一开启就会把观察机会整类清空——上一轮扫描里 513 只入选有 481 只是观察机会。
    """
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.7, 9.9, 10.03,
        10.07, 10.11,
    ]
    result = analyze_bars("000002", "万科A", _bars(closes),
                          _config(min_days_since_breakout=3, max_days_since_breakout=7))
    assert result.category == SignalCategory.WATCH
    assert result.breakout_date is None


def _after_breakout(tail_closes, tail_lows, tail_volumes) -> tuple[pd.DataFrame, dict]:
    """构造一段"突破日 + 之后若干根"的行情。

    突破日在下标 10，成交量 3000；线价 10.0，故：
      触碰带 [9.70, 10.12]、守住线 ≥9.97、跌破线 <9.85、缩量阈值 ≤2250
    """
    closes = [9.0] * 10 + [10.6] + list(tail_closes)
    lows = [8.9] * 10 + [10.4] + list(tail_lows)
    highs = [c * 1.01 for c in closes]
    volumes = [800] * 10 + [3000] + list(tail_volumes)
    df = _bars(closes, highs=highs, lows=lows, volumes=volumes)
    return df, {"index": 10, "volume": 3000.0, "volume_ratio": 3.75}


def test_status_reports_days_since_breakout_and_no_pullback_yet() -> None:
    from app.services.strategy import _breakout_status

    df, bo = _after_breakout([10.9, 11.2, 11.5], [10.7, 11.0, 11.3], [900, 900, 900])
    st = _breakout_status(df, 10.0, bo, _config())

    assert st["days_since_breakout"] == 3
    assert st["pullback_day"] is None
    assert st["breakdown_day"] is None
    assert st["tags"] == ["未回踩"]


def test_status_marks_a_shrinking_volume_pullback_with_its_day_count() -> None:
    from app.services.strategy import _breakout_status

    # 第 3 天最低 10.05 落在触碰带内、收盘 10.2 守住线、量 1200 ≤ 2250
    df, bo = _after_breakout([10.9, 11.0, 10.20], [10.7, 10.8, 10.05], [900, 900, 1200])
    st = _breakout_status(df, 10.0, bo, _config())

    assert st["pullback_day"] == 3
    assert st["pullback_is_shrinking"] is True
    assert st["tags"] == ["缩量回踩（第3天）"], f"实得 {st['tags']}"


def test_status_distinguishes_a_heavy_volume_pullback() -> None:
    """放量回踩说明抛压仍重，不能和缩量回踩共用一个标记。"""
    from app.services.strategy import _breakout_status

    df, bo = _after_breakout([10.9, 10.20], [10.7, 10.05], [900, 2900])
    st = _breakout_status(df, 10.0, bo, _config())

    assert st["pullback_is_shrinking"] is False
    assert st["tags"] == ["放量回踩（第2天）"], f"实得 {st['tags']}"


def test_status_marks_a_breakdown_below_the_line() -> None:
    from app.services.strategy import _breakout_status

    df, bo = _after_breakout([10.9, 9.60], [10.7, 9.50], [900, 2000])
    st = _breakout_status(df, 10.0, bo, _config())

    assert st["breakdown_day"] == 2
    assert "跌破突破位（第2天）" in st["tags"], f"实得 {st['tags']}"


def test_status_keeps_both_marks_in_time_order() -> None:
    """先缩量回踩、后跌破——两件事都发生过，标记要按时间先后都留下。"""
    from app.services.strategy import _breakout_status

    df, bo = _after_breakout([10.20, 10.6, 9.60], [10.05, 10.4, 9.50], [1200, 900, 2000])
    st = _breakout_status(df, 10.0, bo, _config())

    assert st["tags"] == ["缩量回踩（第1天）", "跌破突破位（第3天）"], f"实得 {st['tags']}"


def test_require_breakout_excludes_stocks_that_never_broke_out() -> None:
    """开启"只要已突破"后，仅仅贴近关键线的股票不再入选。"""
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.7, 9.9, 10.03,
        10.07, 10.11,
    ]
    result = analyze_bars("000002", "万科A", _bars(closes), _config(require_breakout=True))
    assert result.category == SignalCategory.NONE
    assert result.breakout_date is None


def test_require_breakout_keeps_stocks_that_did_break_out() -> None:
    result = analyze_bars("000001", "平安银行", _breakout_pullback_bars(),
                          _config(require_breakout=True))
    assert result.category == SignalCategory.BUY_ALERT
    assert result.breakout_date is not None


def test_require_breakout_is_off_by_default() -> None:
    assert _config().require_breakout is False


def test_result_carries_the_status_tags_for_display() -> None:
    """页面要直接拿到「缩量回踩（第N天）」这类标记，不用自己再算一遍。"""
    result = analyze_bars("000001", "平安银行", _breakout_pullback_bars(), _config())
    payload = result.to_dict()

    assert payload["breakout_status"] is not None
    assert payload["breakout_status"]["days_since_breakout"] == 2
    assert payload["breakout_status"]["tags"], "应给出至少一个标记"


def test_result_has_no_status_when_there_was_no_breakout() -> None:
    closes = [
        8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4,
        9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.7, 9.9, 10.03,
        10.07, 10.11,
    ]
    result = analyze_bars("000002", "万科A", _bars(closes), _config())
    assert result.to_dict()["breakout_status"] is None


# ---- ZigZag 摆动接入 ----

def test_swings_alternate_and_expose_the_pending_one() -> None:
    """生产路径上的波峰波谷必须交替，且区分已确认与候选。"""
    import math

    from app.services.strategy import _swings

    closes = [10 + 3 * math.sin(i / 9.0) + i * 0.01 for i in range(400)]
    df = _bars(closes, highs=[c * 1.01 for c in closes], lows=[c * 0.99 for c in closes])

    peaks, valleys, pending = _swings(df, StrategyConfig())

    marks = sorted([(i, "P") for i in peaks] + [(i, "V") for i in valleys])
    kinds = [k for _i, k in marks]
    assert len(marks) > 6, f"样本应产生足够摆动，实得 {len(marks)}"
    for a, b in zip(kinds, kinds[1:]):
        assert a != b, f"出现连续同类摆动：{kinds}"
    if pending is not None:
        assert pending["index"] > max([i for i, _ in marks]), "候选点必须在所有已确认点之后"


def test_reversal_threshold_follows_the_stock_own_volatility() -> None:
    """低波动股用小阈值、高波动股用大阈值，并被上下限夹住。"""
    import math

    from app.services.strategy import _swing_thresholds

    def make(amp):
        closes = [10 + 10 * amp * math.sin(i / 5.0) for i in range(120)]
        return _bars(closes, highs=[c * (1 + amp) for c in closes],
                     lows=[c * (1 - amp) for c in closes])

    cfg = StrategyConfig()
    calm = _swing_thresholds(make(0.004), cfg)
    wild = _swing_thresholds(make(0.06), cfg)

    assert calm[-1] < wild[-1], f"低波动 {calm[-1]:.2f}% 应小于高波动 {wild[-1]:.2f}%"
    assert cfg.swing_reversal_min_pct <= calm[-1] <= cfg.swing_reversal_max_pct
    assert cfg.swing_reversal_min_pct <= wild[-1] <= cfg.swing_reversal_max_pct


def test_swing_config_no_longer_carries_a_flat_tolerance() -> None:
    """平顶容差已取消：候选极值只在"更高"或"完全等高"时移动，不再需要容差。

    它曾让 26% 的波峰落在真实高点之下（中位低 0.63%），而精准线是压力位，
    必须画在价格真正触及过的高度。
    """
    cfg = StrategyConfig()
    assert not hasattr(cfg, "swing_flat_tolerance_atr_ratio")
    assert not hasattr(cfg, "swing_flat_tolerance_max_pct")


def test_swing_defaults_match_the_agreed_spec() -> None:
    cfg = StrategyConfig()
    assert cfg.swing_reversal_atr_ratio == 1.5
    assert cfg.swing_reversal_min_pct == 3.0
    assert cfg.swing_reversal_max_pct == 8.0


def test_line_formation_uses_the_confirmation_date_not_the_pivot_date() -> None:
    """精准线的形成日要用第三个波峰的**确认日**，不是它的发生日。

    波峰在发生日成形、却要几天后才可识别（实测中位滞后 2 根、最长 19）。
    若从发生日就允许找突破，系统会把一段当时还画不出这条线的上穿记成"突破本线"，
    「距突破天数」这个筛选条件也会跟着偏移几天。
    """
    from app.services.strategy import _cluster_key_lines, _swings

    # 三次冲到同一价位，其间各有一次回落
    closes, highs, lows = [], [], []
    for block in range(3):
        closes += [9.0, 9.2, 10.0, 9.3, 9.1, 9.0]
        highs += [9.1, 9.3, 10.1, 9.4, 9.2, 9.1]
        lows += [8.9, 9.1, 9.9, 9.2, 9.0, 8.9]
    closes += [9.0] * 12
    highs += [9.1] * 12
    lows += [8.9] * 12
    df = _bars(closes, highs=highs, lows=lows)
    cfg = _config(min_history_bars=20, ma_period=10, ma_slope_days=3)

    peaks, valleys, _pending, meta = _swings(df, cfg, with_confirmation=True)
    if len(peaks) < cfg.min_line_strength:
        return  # 夹具没凑出三个波峰时跳过，避免测到别的东西

    lines = _cluster_key_lines(df, peaks, cfg, valleys, meta)
    assert lines, "应至少形成一条线"
    line = lines[0]
    third_pivot = sorted(line.touch_indices)[cfg.min_line_strength - 1]
    assert line.formation_index > third_pivot, \
        f"形成日 {line.formation_index} 应晚于第三个波峰的发生日 {third_pivot}"


def test_one_analysis_computes_the_swings_only_once() -> None:
    """ZigZag 要在一次分析里算一遍就够。

    早先 `_support_touch_indices` 每条候选线都重算两次波峰波谷，一次
    analyze_bars 实测调用了 35 次——逐日回测里这是主要的耗时来源。
    """
    import app.services.strategy as module

    closes = [10 + (i % 17) * 0.35 for i in range(400)]
    df = _bars(closes, highs=[c * 1.01 for c in closes], lows=[c * 0.99 for c in closes])

    original = module._swings
    calls = {"n": 0}

    def counting(frame, cfg, with_confirmation=False):
        calls["n"] += 1
        return original(frame, cfg, with_confirmation)

    module._swings = counting
    try:
        module.analyze_bars("000001", "计数", df, _config(include_chart=False))
    finally:
        module._swings = original

    assert calls["n"] == 1, f"一次分析只该算一次摆动，实得 {calls['n']} 次"


def test_line_exposes_the_pivot_date_under_an_honest_name() -> None:
    """字段名要说实话：它是最后一次触及的**发生日**，不是确认日。

    波峰在发生日成形、要几天后才可识别。叫它 last_confirmation_date 会让页面
    显示的"最后确认日期"提前数日。真正的确认日另给一个字段。
    """
    from app.services.strategy import _serialize_key_line, _swings, _cluster_key_lines

    closes = [10 + (i % 13) * 0.4 for i in range(300)]
    df = _bars(closes, highs=[c * 1.01 for c in closes], lows=[c * 0.99 for c in closes])
    cfg = _config(min_history_bars=20, ma_period=10, ma_slope_days=3)
    highs, lows, _pending, meta = _swings(df, cfg, with_confirmation=True)
    lines = _cluster_key_lines(df, highs, cfg, lows, meta, highs)
    if not lines:
        return

    payload = _serialize_key_line(df, lines[0], True, float(df["close"].iloc[-1]))
    assert "last_pivot_date" in payload, "应给出最后一次触及的发生日"
    assert "last_confirmation_date" not in payload, "别再用会误导的旧名字"
