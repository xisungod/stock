"""主线 B：把回踩量不超过突破前 10 日均量的 0.75 倍写进引擎，整段重放。

当前默认是 1.0 倍。0.75 曾在旧信号表上事后过滤过，留出段失败；那次不能代替引擎重放。
这一轮把 pullback_max_normal_volume_ratio 设为 0.75，platform_events 各自重放。
回踩日量高于 0.75 倍时，这一天的回踩不确认，引擎可以换一条线或换一个触发日。
均量是突破前 vol_lookback=10 根，不含突破日和回踩日之后。不要求突破日 1.5 倍放量。

改前用 reports/precision_breakout_volume_20261127 里「不要求突破放量」的重放，
也就是现行默认。改后在同一 600 个代码、同一份截到 2026-11-26 的行情上重放。
分段仍是开发段到 2023-12-31、留出段到 2026-10-21。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把默认从 1.0 改成 0.75。
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

from app.schemas import production_strategy_config
from app.services.backtest import evaluate_bars
from validate_logic_fixes import COST_PCT, HORIZONS, forward_returns
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

OUT = ROOT / "reports/precision_pullback_volume_20261127"
PRIOR = ROOT / "reports/precision_breakout_volume_20261127"
AS_OF = "2026-11-26"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))
RATIO = 0.75


def after_config():
    return production_strategy_config(
        strategy_mode="precision_pullback", include_chart=False, pullback_max_normal_volume_ratio=RATIO,
    )


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or bars["trade_date"].iloc[-1] < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        report = evaluate_bars(
            code, code, bars, lookback_days=0, config=after_config(), as_of=AS_OF, calendar=dates,
        )
        signals = []
        for record in report["signals"]:
            index = int(record["signal_index"])
            if dates[index] != record["signal_date"]:
                raise AssertionError(f"{code} {index} {dates[index]} != {record['signal_date']}")
            if record["signal_date"] < "2019-01-01" or record["signal_date"] > "2026-10-21":
                continue
            signals.append({
                "stock_code": code, "signal_date": record["signal_date"], "signal_index": index,
                "event_id": record.get("signal_event_id"),
                "normal_volume_ratio": (record.get("pullback_check") or {}).get("pullback_normal_volume_ratio")
                if isinstance(record.get("pullback_check"), dict) else None,
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


def prior_rows() -> list[dict]:
    payload = json.loads((PRIOR / "summary.json").read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["arm"] == "before"]
    if [row["label"] for row in rows] != ["开发段", "留出段"]:
        raise SystemExit("上一轮改前对照不完整。")
    return rows


def main() -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    default = production_strategy_config(strategy_mode="precision_pullback", include_chart=False)
    after = after_config()
    if default.pullback_max_normal_volume_ratio != 1.0 or after.pullback_max_normal_volume_ratio != RATIO:
        raise SystemExit("回踩常态量门槛不是 1.0 对 0.75。先停下来。")
    if default.require_breakout_volume or after.require_breakout_volume:
        raise SystemExit("这一轮不能同时打开突破日 1.5 倍放量。")
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
            if done % 25 == 0 or done == len(futures):
                print(f"replay {done}/{len(futures)} signals={len(raw)} fail={len(failures)} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    bench = pd.DataFrame(
        [(day, *values) for series in returns.values() for day, values in series.items()],
        columns=["signal_date", *(f"bench_{horizon}" for horizon in HORIZONS)],
    ).groupby("signal_date").mean()
    prior = pd.read_csv(PRIOR / "signals.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    prior = prior[prior.arm == "before"]
    check = prior.merge(bench.reset_index(), on="signal_date", suffixes=("_old", ""))
    gap = (check["bench_5_old"] - check["bench_5"]).abs()
    if gap.dropna().max() > 1e-6:
        raise SystemExit(f"同日基准与上一轮不一致，最大差 {gap.max()}")
    signals = pd.DataFrame(raw)
    if signals.empty:
        raise SystemExit("没有信号。")
    signals = signals.join(bench, on="signal_date")
    for horizon in HORIZONS:
        signals[f"excess_{horizon}"] = signals[f"ret_{horizon}"] - signals[f"bench_{horizon}"]
    rows = []
    for row in prior_rows():
        rows.append({**row, "variant": "改前：回踩量 ≤ 突破前 10 日均量的 1.0 倍"})
    for label, start, end in PERIODS:
        window = signals[(signals.signal_date >= start) & (signals.signal_date <= end)]
        record = metrics(window)
        record.update({"label": label, "variant": "改后：回踩量 ≤ 突破前 10 日均量的 0.75 倍", "arm": "after"})
        rows.append(record)
    after_rows = [row for row in rows if row.get("arm") == "after"]
    valid = all(passes(row["t5"]) for row in after_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": AS_OF, "cost_pct": COST_PCT, "ratio": RATIO,
        "codes": len(paths), "replayed": len(returns), "failures": failures,
        "meets_validity_standard": valid, "retained_as_default": False, "rows": rows,
        "elapsed_min": round((time.monotonic() - started) / 60, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 B：精准线回踩量改为不超过突破前均量的 0.75 倍",
        "",
        f"生成时间：{payload['generated_at']}；行情截止 {payload['as_of']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 B：精准线回踩。三角整理默认不改。不重做突破日 1.5 倍放量，也不重做三角的触点、放量和离上沿试验。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：回踩日成交量不超过突破前 10 日均量的 1.0 倍。这是当前新扫描的默认。",
        "改后：同一条规则收紧到 0.75 倍。回踩日量更高时，这一天不算回踩确认。",
        "引擎会继续找别的线，或等到后面另一个回踩日，所以入选日和入选的线都可以变。",
        "均量在突破日之前截断，不含突破日，也不含回踩日之后。突破日放量开关保持关闭。",
        "0.75 是引擎里相对突破日的那条缩量比例，这一轮把它用到突破前均量上，并且放进确认条件里重放。",
        "以前在已经发出的信号上删掉高于 0.75 的行，不算这次试验。",
        "",
        "## 3. 改前改后",
        "",
        "改前是上一轮同一行情、同一默认配置的重放。同日等权基准与那次保存的基准一致，否则脚本退出。",
        "分段仍是开发段 2019-01-01～2023-12-31、留出段 2024-01-01～2026-10-21。",
        "成交是次日开盘买入，一字涨停买不到，第 5 个交易日收盘卖出，双边 20bps。",
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
        lines.append("两段都达标。新扫描的回踩常态量上限可以从 1.0 改为 0.75。旧批次已保存的 1.0 不改。三角默认不改。")
    else:
        before = [row for row in payload["rows"] if "1.0" in row["variant"]]
        after = [row for row in payload["rows"] if row.get("arm") == "after"]
        lines += [
            "不把 0.75 留成默认。当前默认仍是回踩量不超过突破前均量的 1.0 倍。三角扫描默认也不改。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "0.75 仍可作为显式参数。没有把它写成新扫描的默认值。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 只试了 0.75 这一档，没有再试别的倍数。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有锁定成交，不报胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
