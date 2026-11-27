"""主线 A：只把对称三角认成可扫描形态。

现行认定不区分对称、上升、下降。本轮要求较陡一边的斜率绝对值
不得超过较缓一边的 2 倍。一边几乎走平的上升或下降三角不再成立，
后面的峰谷可以另组三角形。不是确认日过滤，也不删旧表的行。

改前必须与 reports/triangle_scan_20261127 一致，否则退出。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把对称约束留成默认。
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
OUT = ROOT / "reports/triangle_symmetric_20261127"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))
BALANCE = 2.0


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or str(bars["trade_date"].iloc[-1]) < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        signals = []
        for balance, arm in ((None, "before"), (BALANCE, "after")):
            for event in triangle_up_breakouts(bars, max_slope_balance=balance):
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
        ("after", "改后：较陡一边不超过较缓一边的 2 倍"),
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
        "max_slope_balance": BALANCE, "shift": shift,
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
        "# 主线 A：只扫描对称三角",
        "",
        f"生成时间：{payload['generated_at']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。改的是什么样的两边才算对称三角，不是确认日再加过滤。",
        "不重做每边 3 个触点、零边界容差、形成期高低价偏离和 20～60 根跨度。精准线默认和联合筛选不动。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：两峰两谷只要收敛，上升三角、下降三角和对称三角都认。代码里原先没有这三类的开关。",
        "改后：较陡一边的斜率绝对值不得超过较缓一边的 2 倍。一边几乎走平，就不算对称，这组边界不成立。",
        "引擎会用后面的峰谷另组三角形，因此突破日也可以换成另一天。",
        f"同一股票上，{shift['same_event']} 个突破事件没变，{shift['new_event']} 个是新事件，{shift['dropped_event']} 个旧事件不再出现。",
        "突破确认仍是收盘越过上沿 0.5%。每边仍是 2 个触点。边界容差仍是 1%。形成期仍只限制收盘。跨度仍不超过 100 根。",
        "默认仍不区分三类。只有两段都达标，才把 2 倍对称约束留成默认。",
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
        lines.append("两段都达标。三角扫描默认可以只认两边斜率绝对值相差不超过 2 倍的对称三角。联合筛选和精准线默认不改。")
    else:
        before = [row for row in payload["rows"] if row["arm"] == "before"]
        after = [row for row in payload["rows"] if row["arm"] == "after"]
        lines += [
            "不把「只认对称三角」留成默认。当前默认仍不区分对称、上升、下降。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "对照是 `triangle_up_breakouts(..., max_slope_balance=2)`。默认参数仍是关闭。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 只比较了「不分类」和「较陡一边不超过较缓一边的 2 倍」，没有再试 1.5 倍或 3 倍。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有锁定成交，不报胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
