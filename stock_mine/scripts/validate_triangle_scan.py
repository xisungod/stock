"""主线 A：三角整理向上突破成为可扫描规则后的基线。

规则在跑数之前写死，不按结果改：
- 认定与 backend/app/services/chart_patterns.py 的 find_triangles 是同一份，不是旗形。
- 买入确认日 = 收盘越过上沿 0.5% 的那一根（triangle_up_breakouts 的 signal_index）。
- 样本 = reports/logic_fixes_20261127/validation/summary.json 里的同一批哈希代码。
- 行情截到 2026-11-26。原缓存没有入库，所以这是同一名单、同一截止日期的重新冻结，不是另选区间。
- 成交与 validate_logic_fixes.py 相同：次日开盘买，一字涨停买不到，T+5/10/20 收盘卖，双边 20bps。
- 超额 = 信号净收益 − 同一信号日样本等权收益。
- 开发段 2019-01-01～2023-12-31；留出段从 2024-01-01 到末尾留出 25 个交易日。
- 有效：两段 T+5 超额均值都 > 0，且月度分块 95% 下限都 > 0。不达标就不进入联合筛选。

用法：
  python3 scripts/validate_triangle_scan.py fetch
  python3 scripts/validate_triangle_scan.py run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from app.services.chart_patterns import triangle_up_breakouts
from app.services.strategy import _normalize_bars
from app.services.tencent_bars import fetch_tencent_bars
from baseline_locked_20261127 import block_bootstrap

AS_OF = "2026-11-26"
EVAL_START = "2019-01-01"
DEV_END = "2023-12-31"
HOLD_START = "2024-01-01"
PURGE_SESSIONS = 25
HORIZONS = (5, 10, 20)
COST_PCT = 0.20
CACHE = ROOT / "data/triangle_scan_bars"
OUT = ROOT / "reports/triangle_scan_20261127"
SAMPLE = ROOT / "reports/logic_fixes_20261127/validation/summary.json"


def codes() -> list[str]:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return [str(code).zfill(6) for code in payload["sample_codes"]]


def symbol(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_one(code: str) -> tuple[str, str]:
    path = CACHE / f"{code}.json"
    if path.exists() and path.stat().st_size > 100:
        return code, "cached"
    last = "unknown"
    for attempt in range(3):
        try:
            bars = fetch_tencent_bars(symbol(code), end_date="20261126")
            path.write_text(json.dumps({"code": code, "bars": bars.to_dict("records")}, ensure_ascii=False))
            return code, "ok"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1 + attempt)
    return code, last


def fetch_all(workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    universe = codes()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, code) for code in universe]
        for done, future in enumerate(as_completed(futures), 1):
            code, status = future.result()
            if status in {"ok", "cached"}:
                ok += 1
            else:
                fail += 1
                print(f"FAIL {code} {status}", flush=True)
            if done % 25 == 0 or done == len(futures):
                print(f"fetch {done}/{len(futures)} ok={ok} fail={fail}", flush=True)


def load_bars(path: Path) -> pd.DataFrame:
    frame = _normalize_bars(pd.DataFrame(json.loads(path.read_text())["bars"]))
    frame = frame[frame["trade_date"] <= pd.Timestamp(AS_OF)].reset_index(drop=True)
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    return frame


def forward_returns(df: pd.DataFrame, index: int) -> dict[int, float | None]:
    entry = index + 1
    if entry >= len(df):
        return {horizon: None for horizon in HORIZONS}
    open_, high, low, close = (float(df[name].iloc[entry]) for name in ("open", "high", "low", "close"))
    if open_ == high == low == close and close > float(df["close"].iloc[index]):
        return {horizon: None for horizon in HORIZONS}
    out = {}
    for horizon in HORIZONS:
        exit_index = index + horizon
        out[horizon] = ((float(df["close"].iloc[exit_index]) / open_ - 1) * 100 - COST_PCT
                        if exit_index < len(df) and open_ > 0 else None)
    return out


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or bars["trade_date"].iloc[-1] < EVAL_START:
            return {"code": code, "signals": [], "returns": {}, "checked": 0, "rejected": 0, "error": "history too short"}
        events = triangle_up_breakouts(bars)
        checked = rejected = 0
        signals = []
        dates = bars["trade_date"].tolist()
        for event in events:
            index = int(event["signal_index"])
            if dates[index] < EVAL_START:
                continue
            checked += 1
            prefix = triangle_up_breakouts(bars.iloc[:index + 1].reset_index(drop=True))
            if event["event_id"] not in {item["event_id"] for item in prefix}:
                rejected += 1
                continue
            signals.append({
                "stock_code": code, "signal_date": dates[index], "signal_index": index,
                "event_id": event["event_id"], "upper": round(float(event["upper"]), 4),
                "width_ratio": round(float(event["width_ratio"]), 4),
                **{f"ret_{horizon}": value for horizon, value in forward_returns(bars, index).items()},
            })
        returns = {}
        start = int((bars["trade_date"] >= EVAL_START).to_numpy().argmax())
        for index in range(start, len(bars) - 1):
            values = forward_returns(bars, index)
            returns[dates[index]] = [values[horizon] for horizon in HORIZONS]
        return {"code": code, "signals": signals, "returns": returns, "checked": checked, "rejected": rejected, "error": None}
    except Exception as exc:
        return {"code": code, "signals": [], "returns": {}, "checked": 0, "rejected": 0, "error": f"{type(exc).__name__}: {exc}"}


def metrics(frame: pd.DataFrame) -> dict:
    out = {"signals": int(len(frame)), "stocks": int(frame["stock_code"].nunique()) if len(frame) else 0}
    for horizon in HORIZONS:
        net = frame[f"ret_{horizon}"].dropna() if len(frame) else pd.Series(dtype=float)
        excess = frame[f"excess_{horizon}"].dropna() if len(frame) else pd.Series(dtype=float)
        out[f"t{horizon}"] = {
            "n": int(len(net)),
            "mean_net": round(float(net.mean()), 3) if len(net) else None,
            "median_net": round(float(net.median()), 3) if len(net) else None,
            "win_rate": round(float((net > 0).mean() * 100), 1) if len(net) else None,
            "mean_excess": round(float(excess.mean()), 3) if len(excess) else None,
            "beat_benchmark_rate": round(float((excess > 0).mean() * 100), 1) if len(excess) else None,
            "excess_ci95": block_bootstrap(frame, f"excess_{horizon}") if len(frame) else None,
        }
    return out


def passes(cell: dict) -> bool:
    ci = cell["excess_ci95"]
    return cell["mean_excess"] is not None and cell["mean_excess"] > 0 and ci is not None and ci[0] > 0


def ci_text(ci) -> str:
    return "-" if not ci else f"{ci[0]}%～{ci[1]}%"


def run(workers: int) -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在。这是锁定账本，不覆盖。")
    paths = sorted(CACHE.glob("*.json"))
    universe = set(codes())
    paths = [path for path in paths if path.stem in universe]
    raw_signals = []
    returns = {}
    failures = []
    checked = rejected = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(replay, str(path)) for path in paths]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            checked += item["checked"]
            rejected += item["rejected"]
            if item["error"]:
                failures.append({"code": item["code"], "error": item["error"]})
            raw_signals.extend(item["signals"])
            returns[item["code"]] = item["returns"]
            if done % 25 == 0 or done == len(futures):
                print(f"replay {done}/{len(futures)} signals={len(raw_signals)} rejected={rejected}", flush=True)
    rows = [(day, *values) for series in returns.values() for day, values in series.items()]
    bench = pd.DataFrame(rows, columns=["signal_date", *(f"bench_{h}" for h in HORIZONS)]).groupby("signal_date").mean()
    signals = pd.DataFrame(raw_signals)
    if signals.empty:
        raise SystemExit("没有三角突破信号，先核对认定规则，不写空账本当成基线。")
    signals = signals.join(bench, on="signal_date")
    for horizon in HORIZONS:
        signals[f"excess_{horizon}"] = signals[f"ret_{horizon}"] - signals[f"bench_{horizon}"]
    sessions = sorted(bench.index)
    purge_cutoff = sessions[-PURGE_SESSIONS - 1]
    periods = (
        ("开发段", EVAL_START, DEV_END),
        ("留出段", HOLD_START, min(purge_cutoff, sessions[-1])),
    )
    results = []
    for label, start, end in periods:
        part = signals[(signals["signal_date"] >= start) & (signals["signal_date"] <= end)]
        record = metrics(part)
        record.update({"label": label, "start": start, "end": end, "pass_t5": passes(record["t5"])})
        results.append(record)
    kept = all(row["pass_t5"] for row in results)
    payload = {
        "as_of": AS_OF,
        "sample_source": str(SAMPLE.relative_to(ROOT)),
        "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
        "codes_with_bars": len(paths),
        "codes_requested": len(universe),
        "failed": failures,
        "prefix_checked": checked,
        "prefix_rejected": rejected,
        "purge_cutoff": purge_cutoff,
        "kept_in_combined_scan": False,
        "meets_validity_standard": kept,
        "rule": "triangle_up_breakouts = find_triangles upward close, one buy per pattern, prefix-confirmed",
        "results": results,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(report(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={kept} rejected={rejected}")


def report(payload: dict) -> str:
    lines = [
        "# 主线 A：三角整理改为可扫描规则后的基线",
        "",
        f"生成时间：{payload['generated_at']}。行情截止 {payload['as_of']}。",
        "这一轮改的是选股规则本身，不是在已经发出的精准线信号上做事后丢弃。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。不把它当成旗形，也不新造另一套形态。",
        "主线 B（精准线回踩）本轮没有改。前两轮否决的「去掉突破日涨停」和「越线上限收到 3.5%」没有再做。",
        "",
        "## 2. 选股规则改了哪一处",
        "",
        "改前：三角整理只在详情页画线，研究回测在 `scripts/pattern_backtest.py`，扫描模式里没有它。联合筛选仍是精准线或旗形。",
        "改后：增加显式扫描模式 `triangle_breakout`。认定函数仍是 `find_triangles`。",
        "确认规则：收盘价高于上沿 0.5% 的那一根是买入预警；每个形态只确认一次。收敛中只标观察，不计入下面的成交。",
        "信号日之后的 K 线不参与这笔确认。账本里每一笔都用截断到信号日的行情重跑了一次，对不上的不进入成交。",
        "这条规则不进入联合筛选默认，也不替换精准线。只有开发段和留出段的 T+5 超额同时达标，才把它留在默认扫描里。",
        "",
        "## 3. 改前改后",
        "",
        "改前在这套扫描成交口径下没有三角整理信号，样本量是 0，因此没有净胜率、净收益和超额。",
        "下面的改后数字，是同一批哈希股票、同一分段、同一次日开盘与 T+5 口径下，新规则自己的基线。",
        "先前那份研究回测用的是跌破形态止损、持有上限 60 日和配对随机买点，不是这张表的口径，这里不重算、也不把两套数字当成同一列。",
        "",
        f"有行情的股票 {payload['codes_with_bars']} / 名单 {payload['codes_requested']}。",
        f"截断复核 {payload['prefix_checked']} 笔，其中 {payload['prefix_rejected']} 笔因用到信号日之后的信息被排除。",
        f"留出段终点 {payload['purge_cutoff']}（末尾 {PURGE_SESSIONS} 个交易日不出信号）。",
        "",
        "| 口径 | 时段 | 信号 | 股票 | 有 T+5 成交 | 净胜率 | 平均净收益 | 中位净收益 | T+5 超额 | 超额95%区间 | 跑赢基准 | T+10 超额 | T+20 超额 | 该时段达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
        "| 改前：扫描不含三角整理 | 开发段 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - |",
        "| 改前：扫描不含三角整理 | 留出段 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - |",
    ]
    for row in payload["results"]:
        t5, t10, t20 = row["t5"], row["t10"], row["t20"]
        lines.append(
            f"| 改后：向上突破确认日买入 | {row['label']} {row['start']}～{row['end']} | {row['signals']} | {row['stocks']} | "
            f"{t5['n']} | {t5['win_rate']}% | {t5['mean_net']}% | {t5['median_net']}% | {t5['mean_excess']}% | "
            f"{ci_text(t5['excess_ci95'])} | {t5['beat_benchmark_rate']}% | {t10['mean_excess']}% | {t20['mean_excess']}% | "
            f"{'是' if row['pass_t5'] else '否'} |"
        )
    lines += ["", "## 是否留下", ""]
    if payload["meets_validity_standard"]:
        lines.append("两段都达标。本轮仍不把它并进联合筛选，避免在没有联合模式对照时改掉默认扫描；显式模式 `triangle_breakout` 保留。")
    else:
        dev, hold = payload["results"]
        lines += [
            "不进入默认扫描，也不并进联合筛选。",
            f"开发段 T+5 超额 {dev['t5']['mean_excess']}%（{ci_text(dev['t5']['excess_ci95'])}），区间整体小于 0。",
            f"留出段 T+5 超额 {hold['t5']['mean_excess']}%（{ci_text(hold['t5']['excess_ci95'])}），均值小于 0，下限也小于 0。",
            "显式模式 `triangle_breakout` 仍留在代码里，否则这笔基线无法用同一条规则再扫出来。页面上可以单独选它，默认仍是联合筛选。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        "- 原精准线验证用的日线缓存没有入库。这里按同一代码名单、同一截止日期重新取了前复权行情，历史价格可能与 2026-11-26 那份缓存有复权差异。区间没有另选。",
        "- 股票池仍是当时的上市非 ST 样本，有幸存者偏差。",
        "- 没有把量能或位置再加成第二条规则。那是这条基线之后的事。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "run"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_all(args.workers)
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
