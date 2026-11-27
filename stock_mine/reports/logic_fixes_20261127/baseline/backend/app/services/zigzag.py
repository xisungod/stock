"""交替出现的波峰与波谷（ZigZag 摆动识别）。

为什么不用"间隔 N 根"：那是个纯启发式的门槛，既主观又拦不住真正的问题——
一段上涨里几个相近的高点会被同时标成波峰，图上就是一串挨着的 ▲。
摆动的定义本身要求**波峰与波谷交替**：两个波峰之间必然隔着一个波谷。

规则：
    上升段中记住最高点，价格自该高点回落达到阈值时确认它为波峰，转入下降段；
    下降段对称。阈值是涨跌幅（百分比），可逐根变化以适配各时段的波动率。

时点化：摆动要等回撤真正发生才确认，用不到未来数据。最后一个极值在回撤出现
前不算数——这是正确的代价，不是缺陷。
"""
from __future__ import annotations

import numpy as np


def confirmed_swings(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    reversal_pct: np.ndarray,
    with_pending: bool = False,
    opens: np.ndarray | None = None,
    with_confirmation: bool = False,
):
    """摆动识别：交替出现的波峰与波谷。

    三条规则：

    1. **候选极值取最高/最低价**，取该波段真实的极值；只有完全等高（真正的平顶）
       才把日期右移。用"容差"把略低的后一次也当平顶会让 26% 的波峰落在真实高点
       之下（中位低 0.63%），而精准线是压力位，必须画在价格真正触及过的高度。
    2. **确认反转取 K 线实体**：判跌用实体上沿（阴线即开盘价），判涨用实体下沿
       （阳线即开盘价）。只看收盘的话，阴线的收盘已是实体最低处，确认偏松；
       长影线更不该让反转成立。
    3. **确认必须由后续 K 线做出**。否则平静区间里单根 K 线自身的"最高 − 实体上沿"
       就可能超过阈值，峰谷会逐根交替刷出来。

    时点化：摆动要等回撤真正发生才确认，用不到未来数据。最后一个极值在回撤出现
    前只是候选——`with_pending=True` 时单独返回，供图上画空心三角。
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    # 确认价取 K 线实体的相应一沿：判跌用实体上沿（阴线即开盘价），判涨用实体
    # 下沿（阳线即开盘价）。只看收盘的话，阴线的收盘已是实体最低处，用它确认
    # 波峰偏松——和"长影线不该确认反转"是同一个道理。
    if opens is not None:
        body_high = np.maximum(np.asarray(opens, dtype=float), closes)
        body_low = np.minimum(np.asarray(opens, dtype=float), closes)
    else:
        body_high = body_low = closes
    n = len(highs)
    if n < 2:
        empty = ([], [], None) if with_pending else ([], [])
        return (*empty, {"peak_confirmed_at": [], "valley_confirmed_at": []}) if with_confirmation else empty

    peaks: list[int] = []
    valleys: list[int] = []
    # 极值"发生"在 pivot 那天，但要到回撤够了的那天才可识别。突破只能从确认日
    # 之后算——在此之前这条线当时还画不出来。
    peak_confirmed_at: list[int] = []
    valley_confirmed_at: list[int] = []
    direction = 0
    hi_idx = lo_idx = 0
    hi_value = float(highs[0])
    lo_value = float(lows[0])

    for i in range(1, n):
        limit = float(reversal_pct[i])

        # 极值价格与标记日期分开保存。近似平顶只把日期推到最右，不能把
        # hi_value 一起降下来；否则每天低 0.5% 会链式右移，最终把 12.0
        # 的真实高点漂成 11.4，反转也永远无法确认。平底同理。
        # 取该波段真实的最高/最低价；只有完全等高（真正的平顶）才把日期右移。
        # 早先用"平顶容差"把略低的后一次也当平顶并右移，结果 26% 的波峰落在真实
        # 高点之下（中位低 0.63%，最坏 50%）——精准线是压力位，必须画在价格真正
        # 触及过的高度。
        if highs[i] > hi_value:
            hi_value = float(highs[i])
            hi_idx = i
        elif highs[i] == hi_value:
            hi_idx = i
        if lows[i] < lo_value:
            lo_value = float(lows[i])
            lo_idx = i
        elif lows[i] == lo_value:
            lo_idx = i

        # 确认反转看实体，但候选极值仍取影线。
        drop = (hi_value - body_high[i]) / abs(hi_value) * 100 if hi_value else 0.0
        rise = (body_low[i] - lo_value) / abs(lo_value) * 100 if lo_value else 0.0

        # i > hi_idx：反转必须由**后面的** K 线确认。否则平静区间里单根 K 线自身
        # 的"最高 − 实体上沿"就可能超过阈值，峰谷会逐根交替刷出来。
        if direction >= 0 and i > hi_idx and drop >= limit and hi_idx > (valleys[-1] if valleys else -1):
            if hi_idx > 0:
                peaks.append(hi_idx)
                peak_confirmed_at.append(i)
            direction = -1
            # 新波段从拐点后开始，不是从确认日才开始；等低点取最右。
            since_peak = lows[hi_idx + 1 : i + 1]
            lo_idx = i - int(np.argmin(since_peak[::-1]))
            lo_value = float(lows[lo_idx])
        elif direction <= 0 and i > lo_idx and rise >= limit and lo_idx > (peaks[-1] if peaks else -1):
            if lo_idx > 0:
                valleys.append(lo_idx)
                valley_confirmed_at.append(i)
            direction = 1
            since_valley = highs[lo_idx + 1 : i + 1]
            hi_idx = i - int(np.argmax(since_valley[::-1]))
            hi_value = float(highs[hi_idx])

    meta = {"peak_confirmed_at": peak_confirmed_at, "valley_confirmed_at": valley_confirmed_at}
    if not with_pending:
        return (peaks, valleys, meta) if with_confirmation else (peaks, valleys)

    # 尚未回撤确认的最新极值
    last_confirmed = max(peaks[-1] if peaks else -1, valleys[-1] if valleys else -1)
    pending = None
    if direction >= 0 and hi_idx > last_confirmed:
        pending = {"kind": "peak", "index": hi_idx}
    elif direction <= 0 and lo_idx > last_confirmed:
        pending = {"kind": "valley", "index": lo_idx}
    if with_confirmation:
        return peaks, valleys, pending, meta
    return peaks, valleys, pending
