"""主线 A：三角形上沿不许上升、下沿不许下降。

现行认定允许后峰比前峰高 1%、后谷比前谷低 1%（TRIANGLE_EDGE_SLACK=0.01）。
本轮把这份容差收成 0，在同一冻结行情上整段重放。轨道仍由这两点决定，
只是略微张开的两边不再算三角形。不是确认日的量能或位置过滤，也不删旧表的行。

改前必须与 reports/triangle_scan_20261127 一致，否则退出。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把默认容差改成 0。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from app.services.chart_patterns import TRIANGLE_EDGE_SLACK, triangle_up_breakouts
from validate_logic_fixes import HORIZONS, forward_returns
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

BASELINE = ROOT / "reports/triangle_scan_20261127"
OUT = ROOT / "reports/triangle_edge_slack_20261127"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or str(bars["trade_date"].iloc[-1]) < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        signals = []
        for slack, arm in ((TRIANGLE_EDGE_SLACK, "before"), (0.0, "after")):
            for event in triangle_up_breakouts(bars, edge_slack=slack):
                index = int(event["signal_index"])
                day = dates[index]
                if day < "2019-01-01" or day > "2026-10-21":
                    continue
                signals.append({
                    "arm": arm, "stock_code": code, "signal_date": day, "signal_index": index,
                    "event_id": event["event_id"],
                    **{f"ret_{horizon}": value for horizon, value in forward_returns(bars, index).items()},
                })
        returns = {}
        start = int((bars["trade_date"] >= "2019-01-01").to_numpy().argmax())
        for index in range(start, len(bars) - 1):
            if dates[index] > "2026-10-21":
                break
            values = forward_returns(bars, index)
            returns[dates[index]] = [values[horizon] for horizon in HORIZONS]
        return {"code": code, "signals": signals, "returns": returns, "error": None}
    except Exception as exc:
        return {"code": code, "signals": [], "returns": {}, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    if abs(TRIANGLE_EDGE_SLACK - 0.01) > 1e-12:
        raise SystemExit("默认边界容差已不是 1%。先停下来。")
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    published = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    paths = sorted(CACHE.glob("*.json"))
    raw, returns, failures = [], {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(replay, str(path)) for path in paths]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if item["error"]:
                failures.append({"code": item["code"], "error": item["error"]})
            else:
                raw.extend(item["signals"])
                returns[item["code"]] = item["returns"]
            if done % 100 == 0 or done == len(futures):
                print(f"replay {done}/{len(futures)} signals={len(raw)} fail={len(failures)}", flush=True)
    bench = pd.DataFrame(
        [(day, *values) for series in returns.values() for day, values in series.items()],
        columns=["signal_date", *(f"bench_{horizon}" for horizon in HORIZONS)],
    ).groupby("signal_date").mean()
    signals = pd.DataFrame(raw)
    if signals.empty:
        raise SystemExit("没有信号。")
    signals = signals.join(bench, on="signal_date")
    for horizon in HORIZONS:
        signals[f"excess_{horizon}"] = signals[f"ret_{horizon}"] - signals[f"bench_{horizon}"]
    rows = []
    for arm, variant in (("before", "改前：后峰可高 1%、后谷可低 1%"), ("after", "改后：上沿不许上升、下沿不许下降")):
        part = signals[signals.arm == arm]
        for label, start, end in PERIODS:
            window = part[(part.signal_date >= start) & (part.signal_date <= end)]
            record = metrics(window)
            record.update({"label": label, "variant": variant, "arm": arm})
            rows.append(record)
    for row in rows:
        if row["arm"] != "before":
            continue
        expected = next(item for item in published["results"] if item["label"] == row["label"])
        for field in ("n", "mean_net", "win_rate", "mean_excess", "excess_ci95"):
            if row["t5"][field] != expected["t5"][field]:
                raise SystemExit(f"改前对不上三角基线 {row['label']} {field}: {row['t5'][field]} != {expected['t5'][field]}")
        if row["signals"] != expected["signals"]:
            raise SystemExit(f"改前信号数对不上 {row['label']}")
    valid = all(passes(row["t5"]) for row in rows if row["arm"] == "after")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meets_validity_standard": valid, "retained_as_default": False,
        "failures": failures, "rows": rows,
        "elapsed_min": round((time.monotonic() - started) / 60, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 A：三角形上沿不许上升、下沿不许下降",
        "",
        f"生成时间：{payload['generated_at']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。改的是两边是否允许略微张开，不是确认日再加过滤，也不是每边 3 个触点。",
        "精准线默认和联合筛选不动。已否决的放量、越线和触点试验都不重做。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：后一个峰可以比前一个峰高 1%，后一个谷可以比前一个谷低 1%。轨道仍穿过这两个点。",
        "改后：后峰不得高于前峰，后谷不得低于前谷。容差为 0。略微张开的两边不再被认成三角形，因此也不会产生那次收盘突破。",
        "突破确认仍是收盘越过上沿 0.5%。触点个数仍是每边 2 个。",
        "默认容差仍是 1%。只有两段都达标，才把默认改成 0。",
        "",
        "## 3. 改前改后",
        "",
        "改前在同一冻结行情上重放，必须与 `reports/triangle_scan_20261127` 的笔数、胜率、超额和区间一致。",
        "股票池、分段（开发段到 2023-12-31，留出段到 2026-10-21）、次日开盘和同日样本等权超额都不变。",
        "",
        "| 口径 | 时段 | 信号 | 股票 | 有 T+5 成交 | 净胜率 | 平均净收益 | T+5 超额 | 超额95%区间 | 该时段达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        t5 = row["t5"]
        lines.append(
            f"| {row['variant']} | {row['label']} | {row['signals']} | {row['stocks']} | {t5['n']} | "
            f"{t5['win_rate']}% | {t5['mean_net']}% | {t5['mean_excess']}% | {ci_text(t5['excess_ci95'])} | "
            f"{'是' if passes(t5) else '否'} |"
        )
    lines += ["", "## 是否留下", ""]
    if payload["meets_validity_standard"]:
        lines.append("两段都达标。默认边界容差可以改成 0。联合筛选和精准线仍不改。")
    else:
        before = [row for row in payload["rows"] if row["arm"] == "before"]
        after = [row for row in payload["rows"] if row["arm"] == "after"]
        lines += [
            "不把零容差留成默认认定。联合筛选和精准线也不改。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段信号从 {before[0]['signals']} 笔变成 {after[0]['signals']} 笔，超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%；"
                f"留出段从 {before[1]['signals']} 笔变成 {after[1]['signals']} 笔，超额从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "对照是 `triangle_up_breakouts(..., edge_slack=0)`。默认 `TRIANGLE_EDGE_SLACK` 仍是 0.01。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 只比较了 1% 和 0，没有再试中间的容差。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有锁定成交，不报胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
