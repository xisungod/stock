"""主线 A：三角形在突破前上下沿各至少 3 个已确认触点。

这是认定改动，不是成交日的量能或位置过滤。
默认仍是两峰两谷。本轮把每边最少触点改成 3，在同一冻结行情上整段重放。
触点只计突破检查之前已经被收进边界的峰谷，不看突破日之后。

改前必须与 reports/triangle_scan_20261127 的开发段 766/764、超额 -0.383%，
留出段 485/483、超额 -0.395% 一致，否则退出。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把 3 触点留成扫描默认。
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

from app.services.chart_patterns import TRIANGLE_MIN_TOUCHES_PER_SIDE, triangle_up_breakouts
from validate_logic_fixes import HORIZONS, forward_returns
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

BASELINE = ROOT / "reports/triangle_scan_20261127"
OUT = ROOT / "reports/triangle_touches_20261127"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or str(bars["trade_date"].iloc[-1]) < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        signals = []
        for touches, arm in ((2, "before"), (3, "after")):
            for event in triangle_up_breakouts(bars, min_touches_per_side=touches):
                index = int(event["signal_index"])
                day = dates[index]
                if day < "2019-01-01" or day > "2026-10-21":
                    continue
                signals.append({
                    "arm": arm, "stock_code": code, "signal_date": day, "signal_index": index,
                    "event_id": event["event_id"],
                    "peak_touches": event["peak_touches"], "valley_touches": event["valley_touches"],
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
    if TRIANGLE_MIN_TOUCHES_PER_SIDE != 2:
        raise SystemExit("默认触点已不是每边 2 个。先停下来，不要把对照写成默认。")
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
    for arm, variant in (("before", "改前：两峰两谷即可突破"), ("after", "改后：突破前上下沿各至少 3 个触点")):
        part = signals[signals.arm == arm]
        for label, start, end in PERIODS:
            window = part[(part.signal_date >= start) & (part.signal_date <= end)]
            record = metrics(window)
            record.update({"label": label, "variant": variant, "arm": arm})
            rows.append(record)
    before_rows = [row for row in rows if row["arm"] == "before"]
    for row in before_rows:
        expected = next(item for item in published["results"] if item["label"] == row["label"])
        for field in ("n", "mean_net", "win_rate", "mean_excess", "excess_ci95"):
            if row["t5"][field] != expected["t5"][field]:
                raise SystemExit(f"改前对不上三角基线 {row['label']} {field}: {row['t5'][field]} != {expected['t5'][field]}")
        if row["signals"] != expected["signals"]:
            raise SystemExit(f"改前信号数对不上 {row['label']}")
    after_rows = [row for row in rows if row["arm"] == "after"]
    valid = all(passes(row["t5"]) for row in after_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meets_validity_standard": valid,
        "retained_as_default": False,
        "failures": failures,
        "rows": rows,
        "elapsed_min": round((time.monotonic() - started) / 60, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 A：三角形突破前上下沿各至少 3 个触点",
        "",
        f"生成时间：{payload['generated_at']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。改的是形态怎么认定，不是确认日的放量，也不是收盘离上沿 3.5%。",
        "精准线默认和联合筛选不动。已否决的 1.5 倍放量、0.75 倍常态量、精准线越线 3.5% 都不重做。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：两峰两谷确立上下沿并收敛后，收盘越过上沿 0.5% 就是买入。之后的触点可以验证边界，但不是必需的。",
        "改后：在收盘突破被接受之前，上沿至少还要有一个峰、下沿至少还要有一个谷，在突破检查之前已经被确认贴在轨道上。",
        "也就是上下沿各至少 3 个触点。只有两下的收盘突破不再算作这个形态的买入。",
        "触点的确认日必须早于突破日的收盘检查，不使用突破日之后新出现的峰谷。",
        "默认扫描仍是两峰两谷。3 个触点只有在本轮两段都达标时才改成默认。",
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
        lines.append("两段都达标。上下沿各 3 个触点可以留成三角扫描的默认识定。联合筛选和精准线仍不改。")
    else:
        before = [row for row in payload["rows"] if row["arm"] == "before"]
        after = [row for row in payload["rows"] if row["arm"] == "after"]
        lines += [
            "不把「每边至少 3 个触点」留成默认认定。联合筛选和精准线也不改。",
            f"改后开发段只有 {after[0]['signals']} 笔信号，留出段只有 {after[1]['signals']} 笔。",
            "不同信号日少于 10 天，月度分块区间算不出来，不能把开发段那点正超额当成达标。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "对照函数是 `triangle_up_breakouts(..., min_touches_per_side=3)`。默认参数仍是 2。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 这是两峰两谷基线出来之后的一次认定对照。没有再改触点个数。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有单独的锁定成交，这里不报旗形胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
