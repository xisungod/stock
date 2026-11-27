"""三角形整理与 W 底的时点化识别，用于 K 线图标注。

定义与 scripts/pattern_backtest.py 的回测口径是同一份代码（回测脚本直接调用这里的函数）：
- 峰谷点来自 confirmed_swings，反转阈值 clip(1.5 × ATR20%, 3%, 8%)；一个峰谷点只有在确认日之后才能使用。
- W 底：前高 P0、第一底 V1、颈线 P1、第二底 V2。两底相差 ≤ 4%、相隔 10～120 根，颈线高于底 ≥ 8%，
  P0 高于颈线且比第一底高 ≥ 15%；止损线 = 较低的底 × (1 − 1.5%)；第二底确认后 40 根内等颈线突破。
- 三角形：两峰两谷确立边界。两边都有触点后的共同区间至少 15 根且占四点跨度的一半，
  宽度收敛到 ≤ 60%。后续触点可继续验证边界，收盘越过边界 0.5% 后才标记突破。

形态仅用于结构标注。reports/logic_fixes_20261127/patterns 中的历史回测采用初版口径。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.strategy import _atr_pct_series
from app.services.zigzag import confirmed_swings

PATTERN_VERSION = "chart-patterns-v2"
BREAK_PCT = 0.015
TRIANGLE_BREAK_PCT = 0.005
TRIANGLE_MIN_SHARED_BARS = 15
TRIANGLE_MAX_FIRST_SWING_RATIO = 1.5
WAIT_BARS = 40
MAX_DRAWN = 200  # 只为防止异常数据撑大接口；正常全历史也到不了这么多
PATTERN_NOTE = "只用已确认峰谷和已收盘 K 线识别，形态仅作结构参考。"

W_STATUS = {
    "breakout": "已突破颈线",
    "stopped": "跌破底部，形态失效",
    "changed": "走势改变，形态作废",
    "expired": f"{WAIT_BARS}日内未突破颈线",
    "active": "第二底已确认，等待突破颈线",
}
TRIANGLE_STATUS = {
    "up": "向上突破",
    "down": "向下突破",
    "changed": "走势改变，形态作废",
    "expired": "未突破（超时或到达顶点）",
    "active": "收敛中，等待突破",
}


def pattern_pivots(bars: pd.DataFrame) -> list[dict[str, Any]]:
    """按 K 线顺序交替的已确认峰谷点：kind P/V、idx 极值所在根、price、conf 确认所在根。"""
    o, h, l, c = (bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close"))
    if len(c) < 3:
        return []
    threshold = np.clip(1.5 * np.nan_to_num(_atr_pct_series(bars), nan=3.0), 3.0, 8.0)
    peaks, valleys, meta = confirmed_swings(h, l, c, threshold, opens=o, with_confirmation=True)
    out = [{"kind": "P", "idx": i, "price": h[i], "conf": k} for i, k in zip(peaks, meta["peak_confirmed_at"])]
    out += [{"kind": "V", "idx": i, "price": l[i], "conf": k} for i, k in zip(valleys, meta["valley_confirmed_at"])]
    out.sort(key=lambda p: p["idx"])
    for a, b in zip(out, out[1:]):
        if a["kind"] == b["kind"] or b["conf"] < a["conf"]:
            raise AssertionError("峰谷点未交替或确认顺序颠倒")
    return out


def _next_conf(piv: list[dict[str, Any]], j: int) -> int:
    return piv[j + 1]["conf"] if j + 1 < len(piv) else 10 ** 9


def find_w_bottoms(piv: list[dict[str, Any]], close: np.ndarray) -> list[dict[str, Any]]:
    n, out = len(close), []
    for j in range(3, len(piv)):
        p0, v1, p1, v2 = piv[j - 3:j + 1]
        if v2["kind"] != "V":
            continue
        low, high = min(v1["price"], v2["price"]), max(v1["price"], v2["price"])
        if not (low > 0 and high / low - 1 <= 0.04 and 10 <= v2["idx"] - v1["idx"] <= 120
                and p1["price"] / low - 1 >= 0.08 and p0["price"] > p1["price"] and p0["price"] / v1["price"] - 1 >= 0.15):
            continue
        t0, neck, stop = v2["conf"], p1["price"], low * (1 - BREAK_PCT)
        if close[t0] < stop:
            continue
        item = {"pivots": [p0, v1, p1, v2], "confirm": t0, "neck": neck, "stop": stop, "start": v1["idx"],
                "a_signal": t0 if close[t0] < neck else None, "b_signal": None}
        dead, last = _next_conf(piv, j), min(t0 + WAIT_BARS, n - 1)
        for t in range(t0, last + 1):
            if t >= dead or close[t] < stop:
                item.update(outcome="changed" if t >= dead else "stopped", end=t)
                break
            if close[t] > neck:
                item.update(outcome="breakout", end=t, b_signal=t)
                break
        else:
            item.update(outcome="expired" if t0 + WAIT_BARS <= n - 1 else "active", end=last)
        out.append(item)
    return out


def find_triangles(piv: list[dict[str, Any]], close: np.ndarray) -> list[dict[str, Any]]:
    n, out = len(close), []
    for j in range(3, len(piv)):
        four = piv[j - 3:j + 1]
        x0, x1 = four[0]["idx"], four[-1]["idx"]
        t0 = four[-1]["conf"]
        if out and t0 <= out[-1]["end"]:
            # 活跃形态已有边界时，后续峰谷用来验证这组边界。形态作废当天可建立新形态。
            previous = out[-1]
            if t0 < previous["end"] or previous["outcome"] != "changed":
                continue
        if x1 - x0 > 100:
            continue
        a, b = [p for p in four if p["kind"] == "P"]
        d, e = [p for p in four if p["kind"] == "V"]
        shared_start = max(a["idx"], d["idx"])
        shared_span = x1 - shared_start
        if shared_span < TRIANGLE_MIN_SHARED_BARS or 2 * shared_span < x1 - x0:
            continue
        if not (b["price"] <= a["price"] * 1.01 and e["price"] >= d["price"] * 0.99):
            continue
        up_slope = (b["price"] - a["price"]) / (b["idx"] - a["idx"])
        lo_slope = (e["price"] - d["price"]) / (e["idx"] - d["idx"])
        upper = lambda x, a=a, s=up_slope: a["price"] + s * (x - a["idx"])  # noqa: E731
        lower = lambda x, d=d, s=lo_slope: d["price"] + s * (x - d["idx"])  # noqa: E731
        w0, w1 = upper(shared_start) - lower(shared_start), upper(x1) - lower(x1)
        if not (w0 > 0 and 0 < w1 <= 0.6 * w0):
            continue
        # 首峰到首谷的波幅远超两边形成后的宽度，说明更早的锚点可能属于前一段行情。
        first_swing = abs(a["price"] - d["price"])
        if first_swing > TRIANGLE_MAX_FIRST_SWING_RATIO * w0:
            continue
        apex = x1 + w1 / (lo_slope - up_slope)
        if t0 >= apex:
            continue
        forming_days = np.arange(shared_start, t0 + 1)
        forming_closes = close[shared_start:t0 + 1]
        if (np.any(forming_closes > upper(forming_days) * (1 + TRIANGLE_BREAK_PCT))
                or np.any(forming_closes < lower(forming_days) * (1 - TRIANGLE_BREAK_PCT))):
            continue
        item = {"pivots": four, "start": x0, "upper_start": a["idx"], "lower_start": d["idx"],
                "detected": t0, "stop": e["price"] * (1 - BREAK_PCT),
                "upper": upper, "lower": lower, "width_ratio": w1 / w0, "direction": None, "entry": None}
        next_pivot, bound = j + 1, min(t0 + WAIT_BARS, int(apex))
        last = min(bound, n - 1)
        for t in range(t0 + 1, last + 1):
            if close[t] > upper(t) * (1 + TRIANGLE_BREAK_PCT):
                item.update(outcome="up", end=t, direction="up", entry=t if close[t] >= item["stop"] else None)
                break
            if close[t] < lower(t) * (1 - TRIANGLE_BREAK_PCT):
                item.update(outcome="down", end=t, direction="down")
                break
            while next_pivot < len(piv) and piv[next_pivot]["conf"] <= t:
                p = piv[next_pivot]
                is_peak = p["kind"] == "P"
                edge = (upper if is_peak else lower)(p["idx"])
                gap = (p["price"] - edge) / edge
                if (gap if is_peak else -gap) > TRIANGLE_BREAK_PCT:
                    item.update(outcome="changed", end=t)
                    break
                if abs(gap) <= TRIANGLE_BREAK_PCT:
                    item["pivots"].append(p)
                next_pivot += 1
            if item.get("outcome") == "changed":
                break
        else:
            item.update(outcome="active" if bound > n - 1 else "expired", end=last)
        out.append(item)
    return out


def _non_overlapping(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从最近的开始，保留区间互不重叠的形态：相邻的四点组合大量共用峰谷点，全画出来会叠成一团。"""
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (x["end"], x["start"]), reverse=True):
        if all(item["end"] < k["start"] or item["start"] > k["end"] for k in kept):
            kept.append(item)
        if len(kept) >= MAX_DRAWN:
            break
    return sorted(kept, key=lambda x: x["start"])


def detect_chart_patterns(bars: pd.DataFrame, dates: list[str]) -> dict[str, Any]:
    """图上标注用：bars 为已清洗、只含已收盘 K 线的行情，dates 与其逐根对应。"""
    empty = {"version": PATTERN_VERSION, "note": PATTERN_NOTE, "w_bottoms": [], "triangles": []}
    if len(bars) < 30:
        return empty
    close = bars["close"].to_numpy(dtype=float)
    piv = pattern_pivots(bars)

    def point(p: dict[str, Any], role: str) -> dict[str, Any]:
        return {"date": dates[p["idx"]], "price": round(float(p["price"]), 3), "role": role}

    w_items = []
    for w in _non_overlapping(find_w_bottoms(piv, close)):
        p0, v1, p1, v2 = w["pivots"]
        points = [point(p0, "前高"), point(v1, "第一底"), point(p1, "颈线"), point(v2, "第二底")]
        if w["b_signal"] is not None:
            points.append({"date": dates[w["b_signal"]], "price": round(float(close[w["b_signal"]]), 3), "role": "突破"})
        w_items.append({
            "points": points,
            "neckline": {"price": round(float(w["neck"]), 3), "start_date": dates[p1["idx"]], "end_date": dates[w["end"]]},
            "stop": round(float(w["stop"]), 3),
            "confirmed_date": dates[w["confirm"]],
            "breakout_date": dates[w["b_signal"]] if w["b_signal"] is not None else None,
            "status": w["outcome"], "status_label": W_STATUS[w["outcome"]],
        })

    t_items = []
    for t in _non_overlapping(find_triangles(piv, close)):
        upper_start, lower_start, end = t["upper_start"], t["lower_start"], t["end"]
        t_items.append({
            "upper": [{"date": dates[upper_start], "price": round(float(t["upper"](upper_start)), 3)},
                      {"date": dates[end], "price": round(float(t["upper"](end)), 3)}],
            "lower": [{"date": dates[lower_start], "price": round(float(t["lower"](lower_start)), 3)},
                      {"date": dates[end], "price": round(float(t["lower"](end)), 3)}],
            "points": [point(p, "峰" if p["kind"] == "P" else "谷") for p in t["pivots"]],
            "detected_date": dates[t["detected"]],
            "breakout_date": dates[end] if t["direction"] else None,
            "direction": t["direction"],
            "stop": round(float(t["stop"]), 3),
            "width_ratio": round(float(t["width_ratio"]), 3),
            "status": t["outcome"], "status_label": TRIANGLE_STATUS[t["outcome"]],
        })
    return {**empty, "w_bottoms": w_items, "triangles": t_items,
            "as_of": dates[len(bars) - 1] if dates else None}


def triangle_up_breakouts(bars: pd.DataFrame) -> list[dict[str, Any]]:
    """向上突破的确认日。每个形态最多一笔，确认日是收盘越过上沿的那一根。

    认定与 `find_triangles` 是同一份。调用方必须只传入截至信号日的已收盘 K 线；
    这里不再向后看。向下突破和仍在收敛的形态不产生买入。
    """
    if len(bars) < 30:
        return []
    close = bars["close"].to_numpy(dtype=float)
    events = []
    for triangle in find_triangles(pattern_pivots(bars), close):
        entry = triangle.get("entry")
        if entry is None:
            continue
        upper = float(triangle["upper"](entry))
        events.append({
            "signal_index": int(entry),
            "detected_index": int(triangle["detected"]),
            "start_index": int(triangle["start"]),
            "upper": upper,
            "stop": float(triangle["stop"]),
            "width_ratio": float(triangle["width_ratio"]),
            "event_id": f"triangle:{int(triangle['start'])}:{int(entry)}",
        })
    return events


def active_triangle(bars: pd.DataFrame) -> dict[str, Any] | None:
    """最后一根仍落在尚未突破的三角形里时，返回该形态，供扫描标成观察而不是买入。"""
    if len(bars) < 30:
        return None
    close = bars["close"].to_numpy(dtype=float)
    last = len(close) - 1
    found = None
    for triangle in find_triangles(pattern_pivots(bars), close):
        if triangle.get("outcome") == "active" and triangle["detected"] <= last <= triangle["end"]:
            found = triangle
    return found
