"""三角形整理与 W 底：时点化识别 + 形态自身止损 + 同日同止损距离随机买点配对比较。

口径（先于结果固定，2026-11-27；阈值按形态的常见定义事先选定，不按收益调参）：

峰谷点
- 用生产代码的 confirmed_swings（交替波峰波谷，K 线实体确认），反转阈值 = clip(1.5 × ATR20%, 3%, 8%)，
  与旧峰谷引擎同一组默认值。一个峰谷点只有在其确认日（confirmed_at）之后才可被使用，识别不看未来。

W 底（按时间顺序的四个峰谷点：前高 P0、第一底 V1、颈线 P1、第二底 V2）
- 两底相近：max(V1, V2) / min(V1, V2) − 1 ≤ 4%；两底相隔 10～120 个交易日。
- 形态高度：颈线 P1 / min(V1, V2) − 1 ≥ 8%。
- 之前是下跌：前高 P0 > 颈线 P1，且 P0 / V1 − 1 ≥ 15%。
- 止损线 = min(V1, V2) × (1 − 1.5%)，收盘跌破即失效（与精准线回测同一 1.5% 容差）。
- A「第二底确认买」：V2 的确认日收盘仍低于颈线 → 次日开盘买入。
- B「突破颈线买」：V2 确认后 40 个交易日内，首个收盘高于颈线的交易日 → 次日开盘买入；
  在此之前收盘跌破止损线、或有新的峰谷点被确认（形态已改变），则作废。

三角形（两峰两谷确立边界）
- 上沿过两峰、下沿过两谷；上沿不上升：后峰 ≤ 前峰 × 1.01；下沿不下降：后谷 ≥ 前谷 × 0.99。
- 最后一个拐点处宽度 ≤ 共同区间起点宽度的 60%。共同区间至少跨 15 根 K 线且不少于总跨度一半，四点最多相距 100 根。
- 首次峰谷价差 ≤ 共同区间起点宽度的 1.5 倍，过滤来自前一段行情的锚点。
- 识别日是第四个拐点的确认日，截至确认日的共同区间不能已有收盘越界 0.5%。之后 40 根内且不晚于两线交点，
  先收盘超过上沿或下沿 0.5% 记为突破，向上突破次日开盘买入；若新确认的峰谷已偏离对应边界外侧 0.5%，形态作废。
- 止损线 = 形态中较晚那个谷 × (1 − 1.5%)。

交易与对照（与 line_stop_backtest.py 完全相同）
- 次日一字涨停买不到；收盘跌破止损线 → 下一个非一字跌停日开盘卖出；否则持有满 N 日按收盘卖出。
  主口径 N=60，N=20/120 只作描述。扣双边成本 20bps。
- 同一股票同一入场方式持仓期间（按 60 日口径）不重复开仓。
- 对照：每笔在同一信号日随机抽 20 只样本内其他股票，止损距离相同、规则相同；配对差 = 信号收益 − 对照平均。
- 时段：开发段 2019-01-01～2023-09-30，留出段 2024-01-01 起（按信号日）。

判定（事先声明）
- 三个主假设：W 底 A、W 底 B、三角形向上突破。各自在主口径下配对差均值 > 0，且按月分块自助法的
  Bonferroni 区间（1 − 0.05/3 ≈ 98.3%）下限 > 0，开发段与留出段同时满足才算通过。
- 描述性检验（不参与判定）：三角形识别后 20 日的绝对涨跌幅 vs 同日、同 ATR% 五分位的随机股票；
  三角形向上/向下突破的比例。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pandas as pd

from app.services.chart_patterns import find_triangles, find_w_bottoms, pattern_pivots
from app.services.strategy import _atr_pct_series
from line_stop_backtest import COST_PCT, Stock, stats  # noqa: F401  (COST_PCT 写入报告)
from validate_logic_fixes import load_settled_bars

WINDOWS = (20, 60, 120)
PRIMARY = 60
CONTROLS = 20
SEED = 20261127
EVAL_START = "2019-01-01"
PERIODS = (("开发段 2019-01～2023-09", "2019-01-01", "2023-09-30"),
           ("留出段 2024-01～2026-11", "2024-01-01", "2099-12-31"))
PRIMARY_TESTS = ("W底A：第二底确认买", "W底B：突破颈线买", "三角形向上突破")
LEVEL = 1 - 0.05 / len(PRIMARY_TESTS)


def pivots(bars: pd.DataFrame) -> list[dict]:
    return pattern_pivots(bars)


def w_bottoms(piv: list[dict], c: np.ndarray) -> list[dict]:
    """识别本身在 app.services.chart_patterns（与 K 线图标注同一份代码），这里转成入场事件。"""
    events = []
    for w in find_w_bottoms(piv, c):
        base = {"neck": w["neck"], "stop": w["stop"], "pattern_start": int(w["start"])}
        if w["a_signal"] is not None:
            events.append({**base, "type": PRIMARY_TESTS[0], "signal": w["a_signal"]})
        if w["b_signal"] is not None:
            events.append({**base, "type": PRIMARY_TESTS[1], "signal": w["b_signal"]})
    return events


def triangles(piv: list[dict], c: np.ndarray) -> tuple[list[dict], list[dict]]:
    entries, detections = [], []
    for t in find_triangles(piv, c):
        if t["entry"] is not None:
            entries.append({"type": PRIMARY_TESTS[2], "signal": t["entry"], "stop": t["stop"],
                            "pattern_start": int(t["start"]), "neck": t["upper"](t["entry"])})
        detections.append({"signal": t["detected"], "direction": t["direction"], "width_ratio": t["width_ratio"]})
    return entries, detections


def bootstrap(frame: pd.DataFrame, column: str, level: float, iterations: int = 4000) -> list[float] | None:
    """按自然月分块重采样（与 validate_logic_fixes.block_bootstrap 相同，置信水平可调）。"""
    valid = frame[["signal_date", column]].dropna()
    if valid["signal_date"].nunique() < 10:
        return None
    months = pd.to_datetime(valid["signal_date"]).dt.to_period("M")
    blocks = valid.groupby(months)[column].agg(["sum", "count"])
    values, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    means = values[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return [round(float(x), 3) for x in np.percentile(means, [tail, 100 - tail])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/logic_fixes_20261127/patterns")
    parser.add_argument("--limit-stocks", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    codes = sorted(p.stem for p in args.cache.glob("*.json") if p.stem.isdigit() and len(p.stem) == 6)
    if args.limit_stocks:
        codes = codes[:args.limit_stocks]
    stocks, events, detections, atr_by_day = {}, [], [], {}
    for code in codes:
        if not json.loads((args.cache / f"{code}.json").read_text()).get("bars"):
            continue  # 抓取失败的空缓存（如 600849），与 validate_logic_fixes 一样计为失败、不参与
        bars = load_settled_bars(args.cache / f"{code}.json", args.as_of)
        if len(bars) < 250:
            continue
        stock = Stock(bars)
        stocks[code] = stock
        atr = _atr_pct_series(bars)
        for day, value in zip(stock.dates, atr):
            if day >= EVAL_START and np.isfinite(value):
                atr_by_day.setdefault(day, []).append((code, float(value)))
        piv = pivots(bars)
        w = w_bottoms(piv, stock.c)
        tri, det = triangles(piv, stock.c)
        for item in w + tri:
            events.append({**item, "stock_code": code})
        for item in det:
            detections.append({**item, "stock_code": code})

    on_day = {day: [code for code, _ in rows] for day, rows in atr_by_day.items()}
    rng = np.random.default_rng(SEED)
    rows, examples = [], []
    events.sort(key=lambda e: (e["stock_code"], e["type"], e["signal"]))
    busy_until: dict[tuple[str, str], int] = {}
    for event in events:
        stock = stocks[event["stock_code"]]
        signal = int(event["signal"])
        day = stock.dates[signal]
        key = (event["stock_code"], event["type"])
        if day < EVAL_START or signal + 1 >= len(stock.o) or stock.o[signal + 1] <= 0 or signal < busy_until.get(key, -1):
            continue
        ratio = event["stop"] / stock.o[signal + 1]
        primary = stock.trade(signal, ratio, PRIMARY)
        if primary is not None:
            busy_until[key] = signal + primary[1]
        peers = [code for code in on_day.get(day, []) if code != event["stock_code"]]
        picked = rng.choice(peers, size=min(CONTROLS, len(peers)), replace=False) if peers else []
        if len(examples) < 400:
            examples.append({"type": event["type"], "stock_code": event["stock_code"], "signal_date": day,
                             "pattern_start": stock.dates[event["pattern_start"]], "neck_or_upper": round(event["neck"], 3),
                             "stop": round(event["stop"], 3), "entry_open": stock.o[signal + 1]})
        for window in WINDOWS:
            own = primary if window == PRIMARY else stock.trade(signal, ratio, window)
            if own is None:
                continue
            control = [stocks[code].trade(stocks[code].row[day], ratio, window) for code in picked]
            control = [x[0] for x in control if x is not None]
            rows.append({"type": event["type"], "window": window, "stock_code": event["stock_code"], "signal_date": day,
                         "ret": own[0], "days": own[1], "stopped": own[2], "risk": own[3],
                         "control_mean": float(np.mean(control)) if control else np.nan})
    trades = pd.DataFrame(rows)
    trades["paired_diff"] = trades["ret"] - trades["control_mean"]
    trades.to_csv(args.output / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(examples).to_csv(args.output / "examples.csv", index=False, encoding="utf-8-sig")

    # 描述性：三角形识别后 20 日绝对涨跌幅，对照为同日同 ATR% 五分位的随机股票。
    quintile: dict[str, dict[str, int]] = {}
    members: dict[tuple[str, int], list[str]] = {}
    for day, values in atr_by_day.items():
        ranks = pd.Series([v for _, v in values]).rank(pct=True).to_numpy()
        quintile[day] = {}
        for (code, _), rank in zip(values, ranks):
            q = min(int(rank * 5), 4)
            quintile[day][code] = q
            members.setdefault((day, q), []).append(code)

    def abs_move(code: str, day: str, horizon: int = 20) -> float | None:
        stock = stocks[code]
        i = stock.row[day]
        return abs(stock.c[i + horizon] / stock.c[i] - 1) * 100 if i + horizon < len(stock.c) else None

    vol_rows = []
    for item in detections:
        stock = stocks[item["stock_code"]]
        day = stock.dates[item["signal"]]
        if day < EVAL_START or item["stock_code"] not in quintile.get(day, {}):
            continue
        own = abs_move(item["stock_code"], day)
        if own is None:
            continue
        pool = [code for code in members[(day, quintile[day][item["stock_code"]])] if code != item["stock_code"]]
        picked = rng.choice(pool, size=min(CONTROLS, len(pool)), replace=False) if pool else []
        control = [x for x in (abs_move(code, day) for code in picked) if x is not None]
        vol_rows.append({"signal_date": day, "direction": item["direction"] or "none", "own": own,
                         "control": float(np.mean(control)) if control else np.nan})
    vol = pd.DataFrame(vol_rows)
    vol["diff"] = vol["own"] - vol["control"]

    report, verdicts = {}, {}
    for name in PRIMARY_TESTS:
        report[name] = {}
        for window in WINDOWS:
            report[name][window] = {}
            for label, start, end in PERIODS:
                part = trades[(trades["type"] == name) & (trades["window"] == window)
                              & (trades.signal_date >= start) & (trades.signal_date <= end)]
                paired = part.dropna(subset=["control_mean"])
                report[name][window][label] = {
                    "signal": stats(part),
                    "control_mean": round(float(paired["control_mean"].mean()), 3) if len(paired) else None,
                    "paired_diff": round(float(paired["paired_diff"].mean()), 3) if len(paired) else None,
                    "paired_ci": bootstrap(paired, "paired_diff", LEVEL) if len(paired) else None,
                }
        verdicts[name] = all(
            (cell := report[name][PRIMARY][label])["paired_diff"] is not None and cell["paired_diff"] > 0
            and cell["paired_ci"] is not None and cell["paired_ci"][0] > 0
            for label, _, _ in PERIODS
        )
    vol_report = {}
    for label, start, end in PERIODS:
        part = vol[(vol.signal_date >= start) & (vol.signal_date <= end)].dropna(subset=["control"])
        counts = part["direction"].value_counts()
        vol_report[label] = {
            "n": int(len(part)), "own": round(float(part["own"].mean()), 2), "control": round(float(part["control"].mean()), 2),
            "diff": round(float(part["diff"].mean()), 2), "ci95": bootstrap(part, "diff", 0.95),
            "up": int(counts.get("up", 0)), "down": int(counts.get("down", 0)), "none": int(counts.get("none", 0)),
        }

    lines = ["# 三角形整理与 W 底：时点化识别回测", "", "## 口径（先于结果固定）", ""]
    lines += [line for line in __doc__.strip().splitlines()[2:]]
    lines += ["", f"- 行情截止 {args.as_of}；股票 {len(stocks)} 只（缓存中至少 250 根已收盘 K 线的样本股）。", "",
              "## 判定", ""]
    for name in PRIMARY_TESTS:
        lines.append(f"- {name}：{'通过' if verdicts[name] else '未通过'}")
    lines.append("")
    for window in WINDOWS:
        lines += [f"## 持有上限 {window} 个交易日{'（主口径）' if window == PRIMARY else '（描述）'}", "",
                  f"| 入场方式 | 时段 | 笔数 | 平均净收益 | 中位数 | 胜率 | 平均盈/平均亏 | 利润因子 | 跌破止损离场 | 中位持有 | 中位风险 | 对照平均 | 配对差（{LEVEL:.1%} 区间） |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in PRIMARY_TESTS:
            for label, _, _ in PERIODS:
                cell = report[name][window][label]
                s, ci = cell["signal"], cell["paired_ci"]
                lines.append(
                    f"| {name} | {label} | {s['n']} | {s['mean']}% | {s['median']}% | {s['win_rate']}% | "
                    f"+{s['avg_win']}% / {s['avg_loss']}% | {s['profit_factor']} | {s['stopped_rate']}% | "
                    f"{s['median_days']} 天 | {s['median_risk_pct']}% | {cell['control_mean']}% | "
                    f"{cell['paired_diff']}%（{ci[0] if ci else '—'}～{ci[1] if ci else '—'}） |")
        lines.append("")
    lines += ["## 描述：三角形识别后 20 日的波动与方向", "",
              "| 时段 | 识别次数 | 20日绝对涨跌幅 | 同日同ATR五分位对照 | 差（95% CI） | 向上突破 | 向下突破 | 40日内未突破/作废 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, item in vol_report.items():
        ci = item["ci95"]
        lines.append(f"| {label} | {item['n']} | {item['own']}% | {item['control']}% | {item['diff']}%"
                     f"（{ci[0] if ci else '—'}～{ci[1] if ci else '—'}） | {item['up']} | {item['down']} | {item['none']} |")
    lines += ["", "## 怎么读", "",
              "- 配对差 > 0：同一天、同样止损距离，按形态买入比随机买入赚得多——这才是形态本身的价值。",
              "- 前复权价格、当前上市股票池（幸存者偏差）、对照只在同一抽样池内抽取；不构成投资建议。", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"verdicts": verdicts, "report": report,
                                                         "triangle_volatility": vol_report},
                                                         ensure_ascii=False, indent=2))
    for name in PRIMARY_TESTS:
        print(f"VERDICT {name}: {'PASS' if verdicts[name] else 'FAIL'}")
    print(f"REPORT {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
