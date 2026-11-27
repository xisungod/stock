"""收盘跌破精准线才离场：用这套方法自己的失效条件回测，并与「止损距离相同的随机买点」配对比较。

口径（先于结果固定，2026-11-27）：
- 信号：validate_logic_fixes.py 产出的 signals.csv（585 只哈希抽样股票，事件去重，截至 as-of 的已收盘行情）。
- 买入：信号次日开盘价；次日一字涨停视为买不到，不计入。
- 止损线：精准线价格 × (1 − 1.5%)，即引擎判定「突破失败」的同一水平（max_intervening_close_break_pct）。
- 离场：某日收盘价低于止损线 → 下一个非一字跌停交易日的开盘价卖出；持有满 N 个交易日（含买入日）仍未跌破，
  按第 N 日收盘价卖出。主口径 N=60；N=20/120 只作描述，不参与判定。
- 收益：扣双边成本 20bps；初始风险 R = 1 − 止损线 / 买入价；R 倍数 = 净收益 / R（仅 R ≥ 0.5% 的交易）。
- 对照：每个信号在同一信号日随机抽 20 只样本内其他股票，次日开盘买入，止损线 = 对照买入价 × (止损线 / 信号买入价)，
  即止损距离与信号完全相同，离场规则完全相同。配对差 = 信号收益 − 该信号 20 个对照的平均收益。
  它回答的是：「把止损放在精准线下」是否比「在同样距离放一条随意的线」更好，同时扣掉了当天的大盘与市场情绪。
- 时段：开发段 2019-01-01～2023-09-30（年末留出 60 个交易日的隔离期，避免持有期跨入留出段），留出段 2024-01-01 起。
- 判定（事先声明）：主口径、当前默认规则（修复后规则，大盘只提示）的配对差均值 > 0，且按月分块自助法 95% 下限 > 0，
  开发段与留出段同时满足才算「跌破线止损下，精准线回踩有超出随机买点的价值」。其余组只作参考。
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

from validate_logic_fixes import block_bootstrap, load_settled_bars

BREAK_PCT = 0.015
COST_PCT = 0.20
WINDOWS = (20, 60, 120)
PRIMARY = 60
CONTROLS = 20
MIN_R = 0.005
SEED = 20261127
PERIODS = (("开发段 2019-01～2023-09", "2019-01-01", "2023-09-30"),
           ("留出段 2024-01～2026-11", "2024-01-01", "2099-12-31"))


class Stock:
    def __init__(self, bars: pd.DataFrame):
        self.dates = bars["trade_date"].to_numpy()
        self.o, self.h, self.l, self.c = (bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close"))
        self.row = {day: i for i, day in enumerate(self.dates)}

    def locked(self, k: int, up: bool) -> bool:
        """一字板：up=True 涨停买不到，up=False 跌停卖不出。"""
        if k <= 0:
            return False
        flat = self.o[k] == self.h[k] == self.l[k] == self.c[k]
        return flat and (self.c[k] > self.c[k - 1] if up else self.c[k] < self.c[k - 1])

    def trade(self, signal_row: int, stop_ratio: float, window: int):
        """返回 (净收益%, 持有天数, 是否止损, 初始风险R)；买不到或行情不足返回 None。"""
        entry = signal_row + 1
        n = len(self.o)
        if entry >= n or self.o[entry] <= 0 or self.locked(entry, up=True):
            return None
        price = self.o[entry]
        stop = price * stop_ratio
        end = entry + window - 1
        last = min(end, n - 1)
        broken = np.flatnonzero(self.c[entry:last + 1] < stop)
        k = entry + int(broken[0]) if broken.size else None
        if k is not None and k < end:
            exit_k = k + 1
            while exit_k < n and self.locked(exit_k, up=False):
                exit_k += 1
            if exit_k >= n:
                return None
            exit_price, days, stopped = self.o[exit_k], exit_k - entry + 1, True
        elif end <= n - 1:
            exit_price, days, stopped = self.c[end], window, k is not None
        else:
            return None
        return (exit_price / price - 1) * 100 - COST_PCT, days, stopped, 1 - stop_ratio


def stats(frame: pd.DataFrame) -> dict:
    ret = frame["ret"]
    wins, losses = ret[ret > 0], ret[ret <= 0]
    risk_ok = frame[frame["risk"] >= MIN_R]
    return {
        "n": int(len(frame)),
        "mean": round(float(ret.mean()), 3) if len(frame) else None,
        "median": round(float(ret.median()), 3) if len(frame) else None,
        "win_rate": round(len(wins) / len(frame) * 100, 1) if len(frame) else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "payoff": round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) and losses.mean() < 0 else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
        "stopped_rate": round(float(frame["stopped"].mean() * 100), 1) if len(frame) else None,
        "median_days": float(frame["days"].median()) if len(frame) else None,
        "median_risk_pct": round(float(frame["risk"].median() * 100), 2) if len(frame) else None,
        "mean_r": round(float((risk_ok["ret"] / (risk_ok["risk"] * 100)).mean()), 3) if len(risk_ok) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", type=Path, default=ROOT / "reports/logic_fixes_20261127/validation/signals.csv")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/logic_fixes_20261127/line_stop")
    parser.add_argument("--limit-stocks", type=int, default=0, help="自检用：只跑前 N 只股票")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.signals, dtype={"stock_code": str}, encoding="utf-8-sig")
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    groups = {
        "当前默认（修复后规则，大盘只提示）": signals[signals.arm == "new"],
        "修复后 + 大盘偏弱不买": signals[(signals.arm == "new") & (signals.market_ok == True)],  # noqa: E712
        "修复前规则": signals[signals.arm == "old"],
    }
    primary_group = next(iter(groups))

    codes = sorted(signals.stock_code.unique())
    if args.limit_stocks:
        codes = codes[:args.limit_stocks]
    stocks: dict[str, Stock] = {}
    for code in codes:
        path = args.cache / f"{code}.json"
        if path.exists():
            stocks[code] = Stock(load_settled_bars(path, args.as_of))
    on_day: dict[str, list[str]] = {}
    for code, stock in stocks.items():
        for day in stock.dates:
            on_day.setdefault(day, []).append(code)

    rng = np.random.default_rng(SEED)
    rows = []
    for name, group in groups.items():
        for item in group[group.stock_code.isin(stocks)].itertuples(index=False):
            stock = stocks[item.stock_code]
            index = int(item.signal_index)
            if stock.dates[index] != item.signal_date:
                raise AssertionError(f"{item.stock_code} {item.signal_date} 行号与日期不一致")
            entry = index + 1
            if entry >= len(stock.o) or stock.o[entry] <= 0:
                continue
            ratio = float(item.line_price) * (1 - BREAK_PCT) / stock.o[entry]
            peers = [code for code in on_day[item.signal_date] if code != item.stock_code]
            picked = rng.choice(peers, size=min(CONTROLS, len(peers)), replace=False) if peers else []
            for window in WINDOWS:
                own = stock.trade(index, ratio, window)
                if own is None:
                    continue
                control = [stocks[code].trade(stocks[code].row[item.signal_date], ratio, window) for code in picked]
                control = [x[0] for x in control if x is not None]
                rows.append({
                    "group": name, "window": window, "stock_code": item.stock_code, "signal_date": item.signal_date,
                    "ret": own[0], "days": own[1], "stopped": own[2], "risk": own[3],
                    "control_mean": float(np.mean(control)) if control else np.nan, "control_n": len(control),
                })
    trades = pd.DataFrame(rows)
    trades["paired_diff"] = trades["ret"] - trades["control_mean"]
    trades.to_csv(args.output / "trades.csv", index=False, encoding="utf-8-sig")

    report: dict = {}
    for name in groups:
        report[name] = {}
        for window in WINDOWS:
            report[name][window] = {}
            for label, start, end in PERIODS:
                part = trades[(trades.group == name) & (trades.window == window)
                              & (trades.signal_date >= start) & (trades.signal_date <= end)]
                paired = part.dropna(subset=["control_mean"])
                report[name][window][label] = {
                    "signal": stats(part),
                    "control_mean": round(float(paired["control_mean"].mean()), 3) if len(paired) else None,
                    "paired_diff": round(float(paired["paired_diff"].mean()), 3) if len(paired) else None,
                    "paired_ci": block_bootstrap(paired, "paired_diff"),
                    "paired_positive_share": round(float((paired["paired_diff"] > 0).mean() * 100), 1) if len(paired) else None,
                }
    verdict = all(
        (item := report[primary_group][PRIMARY][label])["paired_diff"] is not None and item["paired_diff"] > 0
        and item["paired_ci"] is not None and item["paired_ci"][0] > 0
        for label, _, _ in PERIODS
    )

    lines = ["# 收盘跌破精准线才离场：与止损距离相同的随机买点配对比较", "", "## 口径（先于结果固定）", ""]
    lines += [line for line in __doc__.strip().splitlines()[2:] if line.startswith(("-", " "))]
    lines += [f"- 行情截止 {args.as_of}；股票 {len(stocks)} 只。", "",
              f"## 判定：{'通过' if verdict else '未通过'}（{primary_group}，持有上限 {PRIMARY} 日，两段配对差 95% 下限均 > 0）", ""]
    for window in WINDOWS:
        lines += [f"## 持有上限 {window} 个交易日{'（主口径）' if window == PRIMARY else '（描述）'}", "",
                  "| 组 | 时段 | 笔数 | 平均净收益 | 中位数 | 胜率 | 平均盈/平均亏 | 盈亏比 | 利润因子 | 跌破线离场 | 中位持有 | 中位风险 | 平均R | 对照平均 | 配对差（95% CI） | 胜过对照占比 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in groups:
            for label, _, _ in PERIODS:
                item = report[name][window][label]
                s = item["signal"]
                ci = item["paired_ci"]
                lines.append(
                    f"| {name} | {label} | {s['n']} | {s['mean']}% | {s['median']}% | {s['win_rate']}% | "
                    f"+{s['avg_win']}% / {s['avg_loss']}% | {s['payoff']} | {s['profit_factor']} | {s['stopped_rate']}% | "
                    f"{s['median_days']} 天 | {s['median_risk_pct']}% | {s['mean_r']} | {item['control_mean']}% | "
                    f"{item['paired_diff']}%（{ci[0] if ci else '—'}～{ci[1] if ci else '—'}） | {item['paired_positive_share']}% |"
                )
        lines.append("")
    lines += ["## 怎么读", "",
              "- 配对差 > 0：同一天、同样止损距离，把止损放在精准线下比随意放一条线赚得多——这才是「线」本身的价值。",
              "- 盈亏比 = 平均盈利 / 平均亏损；利润因子 = 总盈利 / 总亏损，> 1 才赚钱。",
              "- 平均R：每冒 1 份初始风险换回多少，排除了风险小于 0.5% 的交易（买入价已贴着止损线）。",
              "- 前复权价格、当前上市股票池（幸存者偏差）、对照只在同一抽样池内抽取；不构成投资建议。", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"verdict": verdict, "report": report},
                                                         ensure_ascii=False, indent=2))
    print(f"VERDICT {'PASS' if verdict else 'FAIL'}")
    print(f"REPORT {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
