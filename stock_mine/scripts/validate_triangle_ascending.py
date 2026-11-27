"""主线 A：只扫描上升三角。上沿跌幅不超过 3%，下沿必须抬高。

现行认定不分类。后峰可以比前峰低很多，只要宽度在收敛。
本轮不认上沿跌幅超过 3% 的，也不认下沿没有抬高的。
不够的四点组合在认定时不成立，后面的峰谷可以另组三角形。
不是确认日过滤，也不删旧表的行。

改前必须与 reports/triangle_scan_20261127 一致，否则退出。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把上升三角留成默认。
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

from app.services.chart_patterns import triangle_up_breakouts
from validate_logic_fixes import HORIZONS, forward_returns
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

BASELINE = ROOT / "reports/triangle_scan_20261127"
OUT = ROOT / "reports/triangle_ascending_20261127"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))
FLAT = 0.03


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or str(bars["trade_date"].iloc[-1]) < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        signals = []
        for flat, arm in ((None, "before"), (FLAT, "after")):
            for event in triangle_up_breakouts(bars, ascending_flat_pct=flat):
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
    for arm, variant in (
        ("before", "改前：不区分对称、上升、下降"),
        ("after", "改后：只认上升三角，上沿跌幅不超过 3%"),
    ):
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
    before_ids = set(zip(
        signals.loc[signals.arm == "before", "stock_code"],
        signals.loc[signals.arm == "before", "event_id"],
    ))
    after_ids = set(zip(
        signals.loc[signals.arm == "after", "stock_code"],
        signals.loc[signals.arm == "after", "event_id"],
    ))
    shift = {
        "same_event": len(before_ids & after_ids),
        "new_event": len(after_ids - before_ids),
        "dropped_event": len(before_ids - after_ids),
    }
    valid = all(passes(row["t5"]) for row in rows if row["arm"] == "after")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meets_validity_standard": valid, "retained_as_default": False,
        "ascending_flat_pct": FLAT, "shift": shift,
        "failures": failures, "rows": rows,
        "elapsed_min": round((time.monotonic() - started) / 60, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid} shift={shift}")


def render(payload: dict) -> str:
    shift = payload["shift"]
    lines = [
        "# 主线 A：只扫描上升三角",
        "",
        f"生成时间：{payload['generated_at']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。改的是只认上升三角，不是确认日再加过滤。",
        "不重做对称约束、到顶点 10 根、20～60 根跨度、高低价偏离、零容差和每边 3 个触点。精准线默认和联合筛选不动。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：不区分对称、上升、下降。上沿可以比前峰低很多，只要两边在收敛。",
        "改后：只认上升三角。后一个谷必须高于前一个谷。后一个峰不得低于前一个峰 3%，也不得高出前一个峰 1%。",
        "上沿跌得更多，或下沿没有抬高，这组边界不成立。引擎会用后面的峰谷另组三角形。",
        f"同一股票上，{shift['same_event']} 个突破事件没变，{shift['new_event']} 个是新事件，{shift['dropped_event']} 个旧事件不再出现。",
        "突破确认仍是收盘越过上沿 0.5%。每边仍是 2 个触点。形成期仍只限制收盘。跨度仍不超过 100 根。",
        "默认仍不分类。只有两段都达标，才把上升三角留成默认。",
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
        lines.append("两段都达标。三角扫描默认可以只认上沿跌幅不超过 3%、下沿抬高的上升三角。联合筛选和精准线默认不改。")
    else:
        before = [row for row in payload["rows"] if row["arm"] == "before"]
        after = [row for row in payload["rows"] if row["arm"] == "after"]
        lines += [
            "不把「只认上升三角」留成默认。当前默认仍不分类。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "对照是 `triangle_up_breakouts(..., ascending_flat_pct=0.03)`。默认参数仍是关闭。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 只比较了「不分类」和「上沿跌幅不超过 3% 的上升三角」，没有再试 2% 或下降三角。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有锁定成交，不报胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
