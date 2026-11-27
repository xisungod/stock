"""买入后先涨 +10% 还是先跌 -3%：对 validate_logic_fixes.py 产出的信号做止盈/止损竞速回测。

口径（先于结果固定）：
- 买入：信号次日开盘价；次日一字涨停视为买不到，不计入。
- 成功：持有期内最高价先触及买入价 × 1.10；失败：最低价先触及买入价 × 0.97。
- 同一根K线同时触及两者时，日线无法分辨先后，按失败计（保守），并单独统计。
- 跳空：开盘已越过止盈/止损价时按开盘价成交（跳空高开成交更好，跳空低开亏得更多）。
- 持有期为 20/60/120 个交易日（含买入日）；到期都未触及记为「未分胜负」，按到期收盘价计收益。
- 行情不足以走完持有期且尚未分出胜负的，记为「未完成」，不进分母。
- 对照：同一批股票在评估区间内「每一天都买入」，用完全相同的规则计算；另给只在大盘不弱日买入的对照。
- 收益扣双边成本 20bps。
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

from app.services.market_regime import regime_on, regime_table
from app.services.tencent_bars import fetch_tencent_bars
from validate_logic_fixes import load_settled_bars

TARGET, STOP, COST_PCT = 0.10, 0.03, 0.20
WINDOWS = (20, 60, 120)
EVAL_START = "2019-01-01"
PERIODS = (("2019–2023（开发段）", "2019-01-01", "2023-12-31"), ("2024–2026（留出段）", "2024-01-01", "2099-12-31"),
           ("全部 2019–2026", "2019-01-01", "2099-12-31"))
OUTCOMES = ("success", "failure", "neither", "incomplete", "blocked")


def race(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    """每个信号日 i（次日买入）的竞速结果；只用 i 之后的行情判定结果，信号本身不看未来。"""
    o, h, l, c = (bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close"))
    n = len(o)
    outcome = np.full(n, "incomplete", dtype=object)
    ret = np.full(n, np.nan)
    ambiguous = np.zeros(n, dtype=bool)
    days = np.full(n, np.nan)
    for i in range(n - 1):
        entry = i + 1
        price = o[entry]
        if price <= 0:
            continue
        if o[entry] == h[entry] == l[entry] == c[entry] and c[entry] > c[i]:
            outcome[i] = "blocked"
            continue
        target, stop = price * (1 + TARGET), price * (1 - STOP)
        end = entry + window - 1
        last = min(end, n - 1)
        hi, lo = h[entry:last + 1], l[entry:last + 1]
        hit_t, hit_s = hi >= target, lo <= stop
        first_t = int(np.argmax(hit_t)) if hit_t.any() else None
        first_s = int(np.argmax(hit_s)) if hit_s.any() else None
        if first_s is not None and (first_t is None or first_s <= first_t):
            k = entry + first_s
            ambiguous[i] = first_t is not None and first_s == first_t
            exit_price = min(o[k], stop) if k > entry else stop
            outcome[i], days[i] = "failure", first_s + 1
        elif first_t is not None:
            k = entry + first_t
            exit_price = max(o[k], target) if k > entry else target
            outcome[i], days[i] = "success", first_t + 1
        elif end <= n - 1:
            exit_price = c[end]
            outcome[i], days[i] = "neither", window
        else:
            continue
        ret[i] = (exit_price / price - 1) * 100 - COST_PCT
    return pd.DataFrame({"outcome": outcome, "ret": ret, "ambiguous": ambiguous, "days": days})


def tally(frame: pd.DataFrame) -> dict:
    counts = frame["outcome"].value_counts()
    done = int(sum(counts.get(name, 0) for name in ("success", "failure", "neither")))
    out = {name: int(counts.get(name, 0)) for name in OUTCOMES}
    out["decided"] = done
    resolved = frame[frame.outcome.isin(["success", "failure", "neither"])]
    out["success_rate"] = round(out["success"] / done * 100, 1) if done else None
    out["failure_rate"] = round(out["failure"] / done * 100, 1) if done else None
    out["neither_rate"] = round(out["neither"] / done * 100, 1) if done else None
    out["ambiguous_failures"] = int(frame["ambiguous"].sum())
    out["mean_return"] = round(float(resolved["ret"].mean()), 3) if done else None
    out["median_days_to_success"] = (float(frame.loc[frame.outcome == "success", "days"].median())
                                     if out["success"] else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", type=Path, default=ROOT / "reports/logic_fixes_20261127/validation/signals.csv")
    parser.add_argument("--cache", type=Path, required=True, help="validate_logic_fixes.py 使用的行情缓存目录")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/logic_fixes_20261127/target_stop")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.signals, dtype={"stock_code": str}, encoding="utf-8-sig")
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    groups = {
        "当前规则（修复后 + 大盘偏弱不买）": signals[(signals.arm == "new") & (signals.market_ok == True)],  # noqa: E712
        "修复后规则（不看大盘）": signals[signals.arm == "new"],
        "修复前规则": signals[signals.arm == "old"],
    }
    index = regime_table(fetch_tencent_bars("sh000300"))
    index = index[index.index <= args.as_of]

    rows = {window: {name: [] for name in [*groups, "对照：每天都买", "对照：大盘不弱日都买"]} for window in WINDOWS}
    codes = sorted(signals.stock_code.unique())
    for code in codes:
        path = args.cache / f"{code}.json"
        if not path.exists():
            continue
        bars = load_settled_bars(path, args.as_of)
        dates = bars["trade_date"].to_numpy()
        weak = np.array([(regime_on(index, day) or {}).get("weak") is True for day in dates])
        in_eval = dates >= EVAL_START
        for window in WINDOWS:
            result = race(bars, window)
            result["signal_date"] = dates
            result["stock_code"] = code
            rows[window]["对照：每天都买"].append(result[in_eval])
            rows[window]["对照：大盘不弱日都买"].append(result[in_eval & ~weak])
            for name, group in groups.items():
                rows_of_code = group[group.stock_code == code]
                picked = rows_of_code["signal_index"].to_numpy(dtype=int)
                if (dates[picked] != rows_of_code["signal_date"].to_numpy()).any():
                    raise AssertionError(f"{code} 信号行号与日期不一致")
                rows[window][name].append(result.iloc[picked])

    report = {}
    for window in WINDOWS:
        report[window] = {}
        for name, parts in rows[window].items():
            frame = pd.concat(parts, ignore_index=True)
            report[window][name] = {
                label: tally(frame[(frame.signal_date >= start) & (frame.signal_date <= end)])
                for label, start, end in PERIODS
            }

    lines = ["# 买入后先涨10%还是先跌3%：竞速回测", "", "## 口径（先于结果固定）", ""]
    lines += [line for line in __doc__.strip().splitlines()[2:] if line.startswith("- ")]
    lines += [f"- 信号来源：`{args.signals.relative_to(ROOT)}`（591只哈希抽样股票）；行情截止 {args.as_of}。", ""]
    for window in WINDOWS:
        lines += [f"## 持有期 {window} 个交易日", "",
                  "| 组 | 时段 | 已分胜负 | 先涨10%（成功） | 先跌3%（失败） | 其中同日难分 | 到期未分胜负 | 每笔平均净收益 | 成功平均用时 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, periods in report[window].items():
            for label, item in periods.items():
                lines.append(
                    f"| {name} | {label} | {item['decided']} | {item['success']}（{item['success_rate']}%） | "
                    f"{item['failure']}（{item['failure_rate']}%） | {item['ambiguous_failures']} | "
                    f"{item['neither']}（{item['neither_rate']}%） | {item['mean_return']}% | "
                    f"{item['median_days_to_success']} 天 |"
                )
        lines.append("")
    lines += ["## 怎么读", "",
              f"- 盈亏比 10:3，不计成本时成功率达到 {STOP / (TARGET + STOP) * 100:.1f}% 才打平（忽略未分胜负与跳空）。",
              "- 与「对照：每天都买」比较才能看出选股本身有没有用：成功率高出对照的部分才是信号的贡献。",
              "- 同日同时触及止盈止损的按失败计，真实结果只会更好或相同。",
              "- 前复权价格、当前上市股票池（幸存者偏差）、未模拟跌停卖不出，结果偏乐观。", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
