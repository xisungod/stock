"""K 线图上的 W 底与三角形标注：识别规则、时点化（不看未来）与详情接口字段。"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from app.services.bars_cache import with_data_quality
from app.services.chart_patterns import detect_chart_patterns, find_triangles, find_w_bottoms, pattern_pivots
from app.services.detail import build_full_chart


def path_bars(knots: list[tuple[int, float]]) -> pd.DataFrame:
    """按折点线性插值的收盘价；最高/最低 ±0.5%，开盘取前收，波动小，反转阈值落在 3% 下限。"""
    xs, ys = zip(*knots)
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys)
    opens = np.r_[closes[0], closes[:-1]]
    start = date(2024, 1, 1)
    return pd.DataFrame({
        "trade_date": [pd.Timestamp(start + timedelta(days=i)) for i in range(len(closes))],
        "open": opens, "high": np.maximum(opens, closes) * 1.005, "low": np.minimum(opens, closes) * 0.995,
        "close": closes, "volume": np.full(len(closes), 1000.0),
    })


W_KNOTS = [(0, 10.5), (20, 12.0), (40, 10.0), (55, 11.2), (70, 10.1), (95, 11.8), (110, 11.6)]
TRIANGLE_KNOTS = [(0, 9.0), (15, 12.0), (30, 10.0), (42, 11.6), (54, 10.4), (75, 12.5), (85, 12.3)]


def dates_of(bars: pd.DataFrame) -> list[str]:
    return bars["trade_date"].dt.strftime("%Y-%m-%d").tolist()


def test_w_bottom_is_found_with_neckline_breakout():
    bars = path_bars(W_KNOTS)
    patterns = detect_chart_patterns(bars, dates_of(bars))
    assert len(patterns["w_bottoms"]) == 1
    w = patterns["w_bottoms"][0]
    assert [p["role"] for p in w["points"]] == ["前高", "第一底", "颈线", "第二底", "突破"]
    assert w["status"] == "breakout" and w["status_label"] == "已突破颈线"
    assert w["neckline"]["price"] == round(float(bars["high"].iloc[45:65].max()), 3), "颈线画在颈线峰的真实最高价"
    first, second = w["points"][1]["price"], w["points"][3]["price"]
    assert w["stop"] == round(min(first, second) * 0.985, 3)
    breakout = w["points"][-1]
    assert breakout["price"] >= w["neckline"]["price"] and breakout["date"] == w["breakout_date"]  # 均已四舍五入到 0.001


def test_w_bottom_status_only_uses_bars_up_to_each_day():
    """截断到第二底确认日：形态已出现但尚未突破；截断到突破日：已突破。结果不随后来的行情改变。"""
    bars = path_bars(W_KNOTS)
    full = detect_chart_patterns(bars, dates_of(bars))["w_bottoms"][0]
    dates = dates_of(bars)
    cut = bars.iloc[: dates.index(full["confirmed_date"]) + 1].reset_index(drop=True)
    early = detect_chart_patterns(cut, dates_of(cut))["w_bottoms"]
    assert len(early) == 1 and early[0]["status"] == "active" and early[0]["confirmed_date"] == full["confirmed_date"]
    assert early[0]["points"][:4] == full["points"][:4]
    cut = bars.iloc[: dates.index(full["breakout_date"]) + 1].reset_index(drop=True)
    assert detect_chart_patterns(cut, dates_of(cut))["w_bottoms"][0]["status"] == "breakout"
    cut = bars.iloc[: dates.index(full["confirmed_date"])].reset_index(drop=True)
    assert detect_chart_patterns(cut, dates_of(cut))["w_bottoms"] == [], "第二底确认前不能提前画出 W 底"


def upward_triangle(bars: pd.DataFrame) -> dict:
    """起涨段本身（低点抬高、高点回落）也符合三角形定义并向下突破；它与后面的三角形重叠，图上只保留较新的那个。"""
    triangles = detect_chart_patterns(bars, dates_of(bars))["triangles"]
    assert len(triangles) == 1
    return triangles[0]


def test_converging_triangle_breaks_upward():
    bars = path_bars(TRIANGLE_KNOTS)
    t = upward_triangle(bars)
    assert t["direction"] == "up" and t["status_label"] == "向上突破"
    assert t["upper"][1]["price"] < t["upper"][0]["price"], "上沿不上升"
    assert t["lower"][1]["price"] > t["lower"][0]["price"], "下沿不下降"
    assert t["width_ratio"] <= 0.6
    end = dates_of(bars).index(t["breakout_date"])
    assert bars["close"].iloc[end] > t["upper"][1]["price"] - 1e-9


def test_triangle_draws_each_edge_from_its_first_touch():
    t = upward_triangle(path_bars(TRIANGLE_KNOTS))
    for edge, role in (("upper", "峰"), ("lower", "谷")):
        first = next(point for point in t["points"] if point["role"] == role)
        assert t[edge][0] == {"date": first["date"], "price": first["price"]}


def triangle_pivots(*touches):
    return [dict(kind=kind, idx=index, price=price, conf=index + 2)
            for kind, index, price in touches]


def test_zero_edge_slack_rejects_a_slightly_rising_upper_rail():
    rising = triangle_pivots(("P", 0, 13.0), ("V", 8, 10.0), ("P", 30, 13.05), ("V", 50, 11.6))
    close = np.full(70, 11.8)
    close[56] = 14.0
    assert find_triangles(rising, close) != []
    assert find_triangles(rising, close, edge_slack=0.0) == []


def test_triangle_rejects_a_support_anchor_from_the_prior_rally():
    pivots = triangle_pivots(("V", 0, 10.0), ("P", 40, 20.0), ("V", 45, 16.0), ("P", 50, 20.0))
    close = np.linspace(10.0, 20.0, 56)
    close[53:] = [19.0, 17.19, 17.1]
    assert find_triangles(pivots, close) == []


def test_triangle_rejects_an_early_rally_even_if_both_edges_have_enough_bars():
    bars = path_bars([(0, 12.0), (5, 10.0), (45, 20.0), (64, 16.2), (87, 19.6), (96, 17.7), (109, 19.5)])
    prefix = bars.iloc[:93]
    assert detect_chart_patterns(prefix, dates_of(prefix))["triangles"] == []


def test_triangle_ignores_a_small_close_outside_the_line():
    pivots = triangle_pivots(("P", 0, 13.0), ("V", 10, 10.0), ("P", 20, 12.6), ("V", 40, 10.9))
    close = np.full(55, 11.5)
    close[43:45] = [10.98, 10.80]  # 第 43 根只比下沿 10.99 低 0.01，第 44 根明显跌破。
    (triangle,) = find_triangles(pivots, close)
    assert triangle["direction"] == "down" and triangle["end"] == 44


def test_triangle_keeps_its_edges_after_another_confirmed_touch():
    pivots = triangle_pivots(("P", 0, 13.0), ("V", 10, 10.0), ("P", 20, 12.6), ("V", 40, 10.9),
                             ("P", 45, 12.1))
    close = np.full(55, 11.5)
    close[48] = 12.2
    (triangle,) = find_triangles(pivots, close)
    assert triangle["detected"] == 42
    assert triangle["direction"] == "up" and triangle["end"] == 48


def test_span_window_rejects_a_consolidation_longer_than_sixty_bars():
    """四个锚点跨度 80 根时，默认仍认定；收到 20～60 根后这组边界不成立。"""
    pivots = triangle_pivots(("P", 0, 13.0), ("V", 20, 10.0), ("P", 50, 12.5), ("V", 80, 11.2))
    close = np.full(100, 11.5)
    close[85] = 12.5
    assert find_triangles(pivots, close) != []
    assert find_triangles(pivots, close, min_span_bars=20, max_span_bars=60) == []
    short = triangle_pivots(("P", 0, 13.0), ("V", 10, 10.0), ("P", 20, 12.6), ("V", 40, 10.9))
    short_close = np.full(55, 11.5)
    short_close[48] = 12.2
    assert find_triangles(short, short_close, min_span_bars=20, max_span_bars=60) != []


def test_rail_deviation_rejects_a_high_above_the_upper_rail():
    """收盘留在轨道内时，默认仍认定。最高价高出上沿超过 0.5% 时，这组边界不成立。"""
    pivots = triangle_pivots(("P", 0, 13.0), ("V", 10, 10.0), ("P", 20, 12.6), ("V", 40, 10.9))
    close = np.full(55, 11.5)
    close[48] = 12.2
    high = close * 1.002
    low = close * 0.998
    high[30] = 12.4 * 1.02  # 第 30 根上沿是 12.4，最高价高出 2%。
    assert find_triangles(pivots, close, high=high, low=low) != []
    assert find_triangles(pivots, close, high=high, low=low, max_rail_deviation_pct=0.005) == []
    high[30] = 12.4 * 1.003
    assert find_triangles(pivots, close, high=high, low=low, max_rail_deviation_pct=0.005) != []


def test_triangle_rejects_a_close_outside_the_channel_during_formation():
    pivots = triangle_pivots(("P", 0, 13.0), ("V", 10, 10.0), ("P", 20, 12.6), ("V", 40, 10.9))
    close = np.full(55, 11.5)
    close[34] = 10.6
    assert find_triangles(pivots, close) == []


def test_triangle_is_not_drawn_before_its_fourth_pivot_is_confirmed():
    bars = path_bars(TRIANGLE_KNOTS)
    t = upward_triangle(bars)
    dates = dates_of(bars)
    cut = bars.iloc[: dates.index(t["detected_date"])].reset_index(drop=True)
    before = detect_chart_patterns(cut, dates_of(cut))["triangles"]
    assert all(item["detected_date"] < t["detected_date"] for item in before), "第四个峰谷点确认前不能画出这个三角形"
    cut = bars.iloc[: dates.index(t["detected_date"]) + 1].reset_index(drop=True)
    early = [item for item in detect_chart_patterns(cut, dates_of(cut))["triangles"]
             if item["detected_date"] == t["detected_date"]]
    assert len(early) == 1 and early[0]["status"] == "active" and early[0]["direction"] is None
    assert early[0]["upper"][0] == t["upper"][0] and early[0]["lower"][0] == t["lower"][0]


def test_detectors_match_the_backtest_entry_points():
    """回测脚本只把 W 底 A/B 与三角形向上突破转成入场；这里锁定这些入场日。"""
    bars = path_bars(W_KNOTS)
    piv = pattern_pivots(bars)
    close = bars["close"].to_numpy(dtype=float)
    (w,) = find_w_bottoms(piv, close)
    assert w["a_signal"] == w["confirm"] and close[w["a_signal"]] < w["neck"]
    assert close[w["b_signal"]] > w["neck"] >= close[w["b_signal"] - 1]
    tri_bars = path_bars(TRIANGLE_KNOTS)
    found = find_triangles(pattern_pivots(tri_bars), tri_bars["close"].to_numpy(dtype=float))
    assert [item["direction"] for item in found] == ["down", "up"]
    assert found[0]["entry"] is None, "向下突破不产生买入"
    assert found[1]["entry"] == found[1]["end"] and found[1]["entry"] > found[1]["detected"]


def test_full_chart_ships_patterns_and_skips_unsettled_bars():
    bars = path_bars(W_KNOTS)
    chart = build_full_chart(bars)
    assert chart["chart_patterns"]["version"] == "chart-patterns-v2"
    assert chart["chart_patterns"]["w_bottoms"] and "仅作结构参考" in chart["chart_patterns"]["note"]
    settled = bars.copy()
    settled.attrs["data_quality"] = {"expected_bar_date": dates_of(bars)[-3]}
    chart = build_full_chart(settled)
    assert chart["chart_patterns"]["as_of"] == dates_of(bars)[-3], "盘中未收盘的K线不参与形态识别"
    assert chart["range"]["count"] == len(bars), "K线图本身仍显示全部K线"


def test_after_close_patterns_wait_for_an_after_close_quote():
    bars = path_bars(TRIANGLE_KNOTS)
    triangle = upward_triangle(bars)
    dates = dates_of(bars)
    index = dates.index(triangle["breakout_date"])
    prefix = bars.iloc[:index + 1]
    check = f"{dates[index]}T14:00:00+08:00"
    clock = datetime.fromisoformat(f"{dates[index]}T16:30:00+08:00")
    for stale in (with_data_quality(prefix, check, verify=True, now=clock),
                  with_data_quality(prefix, check, "最新行情获取失败")):
        chart = build_full_chart(stale)
        assert chart["chart_patterns"]["as_of"] == dates[index - 1]
        assert chart["chart_patterns"]["triangles"][-1]["status"] == "active"
        assert chart["range"]["count"] == len(prefix)

    fresh = with_data_quality(prefix, clock.isoformat(), verify=True, now=clock)
    assert build_full_chart(fresh)["chart_patterns"]["triangles"][-1]["status"] == "up"


def test_short_history_has_no_patterns():
    bars = path_bars([(0, 10.0), (20, 11.0)])
    assert detect_chart_patterns(bars, dates_of(bars))["w_bottoms"] == []
