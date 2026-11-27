from __future__ import annotations

import numpy as np

from app.services.zigzag import confirmed_swings


def _flat(n, value):
    return np.full(n, float(value))


def _swings(highs, lows, closes, reversal=5.0, opens=None):
    n = len(highs)
    return confirmed_swings(
        np.array(highs, float), np.array(lows, float), np.array(closes, float),
        _flat(n, reversal),
        opens=None if opens is None else np.array(opens, float),
    )


def _walk(seed, n=600, drift=0.0):
    rng = np.random.RandomState(seed)
    return 100 * np.exp(np.cumsum(rng.randn(n) * 0.02 + drift))


# ---- 交替 ----

def test_peaks_and_valleys_strictly_alternate() -> None:
    """核心约束：两个波峰之间必有一个波谷。

    各自独立找局部极值时会连续标出多个波峰（实测 18% 的摆动点是连续同类），
    那些只是"候选局部极值"，不是波段拐点。
    """
    c = _walk(7)
    peaks, valleys = confirmed_swings(c, c, c, _flat(len(c), 5.0))

    marks = sorted([(i, "P") for i in peaks] + [(i, "V") for i in valleys])
    kinds = [k for _i, k in marks]
    assert len(marks) > 6, f"样本太少不足以验证，实得 {len(marks)}"
    for a, b in zip(kinds, kinds[1:]):
        assert a != b, f"出现连续同类摆动点：{kinds}"


def test_a_rally_keeps_only_its_highest_point() -> None:
    peaks, _ = _swings([10, 11, 10.5, 12, 11.5, 13, 12.8, 9],
                       [10, 11, 10.5, 12, 11.5, 13, 12.8, 9],
                       [10, 11, 10.5, 12, 11.5, 13, 12.8, 9])
    assert peaks == [5], f"应只保留最高点，实得 {peaks}"


# ---- 阈值 ----

def test_a_shallow_pullback_does_not_confirm_a_swing() -> None:
    v = [10, 12, 11.9, 13, 12.9, 14]
    peaks, valleys = _swings(v, v, v)
    assert peaks == [] and valleys == [], f"实得 峰{peaks} 谷{valleys}"


def test_a_deep_enough_pullback_does_confirm_it() -> None:
    v = [10, 12, 11.3, 12.5]
    peaks, _ = _swings(v, v, v)
    assert peaks == [1], f"实得 {peaks}"


def test_a_larger_threshold_yields_fewer_swings() -> None:
    c = _walk(3, n=800)
    counts = [len(confirmed_swings(c, c, c, _flat(len(c), t))[0])
              for t in (3.0, 8.0, 15.0)]
    assert counts[0] > counts[1] > counts[2], f"实得 {counts}"


def test_threshold_may_vary_per_bar() -> None:
    v = [10, 12, 11.3, 12.5, 11.0, 13.0]
    assert len(_swings(v, v, v, reversal=5.0)[0]) > len(_swings(v, v, v, reversal=20.0)[0])


# ---- 确认必须由后续 K 线做出 ----

def test_a_single_bar_cannot_both_set_and_confirm_an_extreme() -> None:
    """平静区间里单根 K 线自身的"最高 − 实体上沿"就可能超过阈值。

    若允许同一根既设极值又确认反转，峰谷会逐根交替刷出来——实测一段全平的
    行情曾被标出波峰 [1,3,5,7]、波谷 [2,4,6]。
    """
    n = 12
    peaks, valleys = _swings(_flat(n, 9.8), _flat(n, 9.6), _flat(n, 9.7), reversal=1.0)
    assert peaks == [] and valleys == [], f"实得 峰{peaks} 谷{valleys}"


# ---- 实体价确认（阴线取开盘价）----

def test_a_bearish_candle_confirms_a_peak_by_its_open_not_its_close() -> None:
    """确认波峰用实体上沿：阴线的收盘已是当日实体最低处，用它确认偏松。"""
    highs = [10.0, 12.0, 11.7, 11.8]
    lows = [9.8, 11.8, 11.2, 11.3]
    opens = [9.9, 11.9, 11.65, 11.4]     # 第 3 根 开11.65 > 收11.35 → 阴线
    closes = [9.9, 11.95, 11.35, 11.5]
    # 自 12.0 起：收盘 11.35 跌 5.4%（够）；实体上沿 11.65 只跌 2.9%（不够）
    assert _swings(highs, lows, closes)[0] == [1]
    assert _swings(highs, lows, closes, opens=opens)[0] == []


def test_a_bullish_candle_confirms_a_valley_by_its_open() -> None:
    """对称：确认波谷用实体下沿，阳线即开盘价——反弹要更实在才算数。"""
    highs = [10.0, 9.0, 9.5]
    lows = [9.8, 8.0, 8.3]
    opens = [9.9, 8.9, 8.25]             # 第 3 根 开8.25 < 收8.45 → 阳线
    closes = [9.9, 8.05, 8.45]
    # 自 8.0 起：收盘 8.45 涨 5.6%（够）；实体下沿 8.25 只涨 3.1%（不够）
    assert _swings(highs, lows, closes)[1] == [1]
    assert _swings(highs, lows, closes, opens=opens)[1] == [], "实体口径下这一根还不该确认"


def test_body_confirmation_only_delays_a_swing_it_does_not_erase_it() -> None:
    """更严的口径是推迟确认，不是永远不确认——涨够了照样成立。"""
    highs = [10.0, 9.0, 9.5, 9.6]
    lows = [9.8, 8.0, 8.3, 8.4]
    opens = [9.9, 8.9, 8.25, 8.5]
    closes = [9.9, 8.05, 8.45, 8.5]      # 第 4 根实体下沿 8.5 → 涨 6.2%，够了
    assert _swings(highs, lows, closes, opens=opens)[1] == [1]


# ---- 平顶 ----

def test_the_highest_bar_wins_even_if_a_later_one_is_only_slightly_lower() -> None:
    """波峰取该波段真实的最高价，哪怕后面那次只低一点点。

    原来用"平顶容差"把略低的后一次也算作平顶并右移，结果 26% 的波峰落在了
    真实高点之下（中位低 0.63%，最坏 50%）。精准线是压力位，必须画在价格真正
    触及过的高度——突破 11.98 但没破 12.00 不该算突破。
    """
    highs = [10, 12.00, 11.5, 11.98, 11.0]
    lows = [9.8, 11.9, 11.4, 11.9, 10.9]
    closes = [9.9, 11.95, 11.45, 11.95, 11.0]
    assert _swings(highs, lows, closes)[0] == [1], "应取更高的 12.00，而不是略低的 11.98"


def test_an_exact_tie_moves_the_date_to_the_rightmost() -> None:
    """完全等高时才右移——那才是真正的平顶，取最后一次冲击。"""
    highs = [10, 12.00, 11.5, 12.00, 11.0]
    lows = [9.8, 11.9, 11.4, 11.9, 10.9]
    closes = [9.9, 11.95, 11.45, 11.95, 11.0]
    assert _swings(highs, lows, closes)[0] == [3]


def test_a_clearly_lower_high_does_not_move_the_peak() -> None:
    highs = [10, 12.0, 11.5, 11.0, 10.0]
    lows = [9.8, 11.9, 11.4, 10.9, 9.9]
    closes = [9.9, 11.95, 11.45, 10.95, 9.95]
    assert _swings(highs, lows, closes)[0] == [1]


def test_the_candidate_never_slides_down_a_staircase() -> None:
    """逐级走低时波峰必须停在最高那根，不能被一路拖下去。

    早先按"容差内视为平顶"链式比较，每一级都在容差内，候选会一路滑到末端。
    改成"只有更高或完全等高才移动"之后，这种滑动在结构上不可能发生。
    """
    highs = [10.0, 12.0, 11.9, 11.8, 11.7, 11.6, 11.5, 11.4]
    lows = [value - 0.2 for value in highs]
    closes = [9.9, 11.95, 11.85, 11.75, 11.65, 11.55, 11.45, 11.35]

    peaks, _ = _swings(highs, lows, closes)

    assert peaks == [1], f"应停在真实最高的 12.0，实得 {peaks}"


# ---- 时点化与边界 ----

def test_confirmed_swings_do_not_change_when_new_bars_arrive() -> None:
    c = _walk(11, n=500)
    early = confirmed_swings(c[:400], c[:400], c[:400], _flat(400, 6.0))
    late = confirmed_swings(c, c, c, _flat(len(c), 6.0))
    settled = 350
    assert [i for i in early[0] if i < settled] == [i for i in late[0] if i < settled]
    assert [i for i in early[1] if i < settled] == [i for i in late[1] if i < settled]


def test_a_monotonic_series_has_no_confirmed_swings() -> None:
    rising = np.arange(1.0, 200.0)
    assert _swings(rising, rising, rising) == ([], [])


def test_the_first_bar_is_a_data_boundary_not_a_turning_point() -> None:
    """数据开始之前价格是高是低无从得知，把第 0 根当摆动点会凭空多出一个。"""
    v = [10, 10.6, 11.2, 11.9]
    assert _swings(v, v, v)[1] == []


def test_pending_extreme_is_reported_separately_from_confirmed_ones() -> None:
    highs = [10, 12, 11.4, 13, 13.5]
    lows = [9.8, 11.8, 11.2, 12.8, 13.3]
    closes = [9.9, 11.9, 11.3, 12.9, 13.4]
    n = 5
    peaks, _v, pending = confirmed_swings(
        np.array(highs, float), np.array(lows, float), np.array(closes, float),
        _flat(n, 5.0), with_pending=True)
    assert 4 not in peaks
    assert pending == {"kind": "peak", "index": 4}, f"实得 {pending}"


def test_empty_or_tiny_input_is_handled() -> None:
    assert confirmed_swings(np.array([]), np.array([]), np.array([]),
                            np.array([])) == ([], [])
    assert _swings([1.0], [1.0], [1.0]) == ([], [])


# ---- 发生日 vs 确认日 ----

def test_confirmation_index_is_reported_alongside_the_pivot() -> None:
    """波峰在"发生日"成形，但要到"确认日"才可识别（实测中位滞后 2 根、最长 19）。

    突破只能从确认日之后算：在确认日之前，这条线在当时还画不出来，
    把那段时间里的上穿记成"突破本线"是时间错位。
    """
    highs = [10, 12, 11.6, 11.4, 11.2]
    lows = [9.8, 11.8, 11.4, 11.2, 11.0]
    closes = [9.9, 11.9, 11.5, 11.3, 11.1]   # 自 12 起：11.5 跌 4.2%、11.3 跌 5.8%
    n = 5
    peaks, _v, _p, meta = confirmed_swings(
        np.array(highs, float), np.array(lows, float), np.array(closes, float),
        _flat(n, 5.0), with_pending=True, with_confirmation=True)

    assert peaks == [1], f"实得 {peaks}"
    assert meta["peak_confirmed_at"] == [3], f"应在下标 3 才确认，实得 {meta['peak_confirmed_at']}"


def test_confirmation_index_is_always_after_the_pivot() -> None:
    c = _walk(21, n=400)
    n = len(c)
    peaks, valleys, _p, meta = confirmed_swings(
        c, c, c, _flat(n, 5.0), with_pending=True, with_confirmation=True)

    assert len(meta["peak_confirmed_at"]) == len(peaks)
    assert len(meta["valley_confirmed_at"]) == len(valleys)
    for pivot, conf in zip(peaks, meta["peak_confirmed_at"]):
        assert conf > pivot, f"确认日必须晚于发生日：{pivot} → {conf}"
    for pivot, conf in zip(valleys, meta["valley_confirmed_at"]):
        assert conf > pivot


def test_peak_confirmation_keeps_the_low_already_seen_after_the_peak() -> None:
    highs = [10, 12, 11.8, 11, 11.4]
    lows = [9.8, 11.8, 9, 10.5, 11]
    opens = [9.9, 11.9, 11.6, 10.8, 11.2]
    closes = [9.9, 11.95, 11.5, 10.8, 11.3]
    peaks, valleys = _swings(highs, lows, closes, opens=opens)
    assert peaks == [1]
    assert valleys == [2], "低点 9 在确认波峰之前已出现，不能重置为确认日的 10.5"


def test_valley_confirmation_keeps_the_high_already_seen_after_the_valley() -> None:
    highs = [10.2, 8.2, 11, 9.5, 9]
    lows = [10, 8, 8.2, 9, 8.6]
    opens = [10.1, 8.1, 8.4, 9.2, 8.8]
    closes = [10.1, 8.05, 8.5, 9.2, 8.7]
    peaks, valleys = _swings(highs, lows, closes, opens=opens, reversal=6)
    assert valleys == [1]
    assert peaks == [2], "确认波谷时不能丢掉此前已出现的高点 11"


def test_confirmation_metadata_is_present_for_short_inputs() -> None:
    for n in (0, 1):
        for pending in (False, True):
            result = confirmed_swings(
                _flat(n, 10), _flat(n, 9), _flat(n, 9.5), _flat(n, 5),
                with_pending=pending, with_confirmation=True,
            )
            assert len(result) == (4 if pending else 3)
            assert result[-1] == {"peak_confirmed_at": [], "valley_confirmed_at": []}


def test_wick_extremes_and_confirmations_are_prefix_stable() -> None:
    rng = np.random.default_rng(53)
    closes = _walk(53, 400)
    opens = closes * np.exp(rng.normal(0, .012, len(closes)))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, .08, len(closes)))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, .08, len(closes)))
    reversal = _flat(len(closes), 5)
    full = confirmed_swings(highs, lows, closes, reversal, opens=opens, with_confirmation=True)
    for end in (80, 160, 240, 320):
        early = confirmed_swings(
            highs[:end], lows[:end], closes[:end], reversal[:end],
            opens=opens[:end], with_confirmation=True,
        )
        for kind, key in ((0, "peak_confirmed_at"), (1, "valley_confirmed_at")):
            expected = [(p, c) for p, c in zip(full[kind], full[-1][key]) if c < end]
            assert list(zip(early[kind], early[-1][key])) == expected
    marks = sorted([(p, "peak") for p in full[0]] + [(v, "valley") for v in full[1]])
    assert all(a[1] != b[1] for a, b in zip(marks, marks[1:]))
