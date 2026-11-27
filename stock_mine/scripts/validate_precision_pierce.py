"""主线 B：回踩最低价跌破精准线不得超过 1%。引擎整段重放。

现行定义允许最低价低于精准线最多 3%（max_intraday_undercut_pct=0.03）。
本轮把这个上限收到 1%。跌破更深的当天不算踩到，引擎可以改等后面跌破更浅的一天，
或让这次突破过期后再等下一次。不是收盘必须站回线，也不是最低价必须碰到线。

改前用现行默认重放。两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把默认跌破上限改成 1%。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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

OUT = ROOT / "reports/precision_pierce_20261127"
PRIOR = ROOT / "reports/precision_breakout_volume_20261127"
AS_OF = "2026-11-26"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))


def configs():
    base = production_strategy_config(strategy_mode="precision_pullback", include_chart=False)
    return base, replace(base, max_intraday_undercut_pct=0.01)


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or bars["trade_date"].iloc[-1] < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        _before, after = configs()
        dates = bars["trade_date"].tolist()
        report = evaluate_bars(code, code, bars, lookback_days=0, config=after, as_of=AS_OF, calendar=dates)
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
        raise SystemExit("现行默认对照不完整。")
    return rows


def main() -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    before, after = configs()
    if abs(before.max_intraday_undercut_pct - 0.03) > 1e-12 or abs(after.max_intraday_undercut_pct - 0.01) > 1e-12:
        raise SystemExit("跌破上限不是 3% 对 1%。先停下来。")
    if before.pullback_touch_pct != 0.012 or after.pullback_touch_pct != 0.012:
        raise SystemExit("不能同时改触线容差。")
    if abs(before.allow_undercut_pct - 0.003) > 1e-12 or abs(after.allow_undercut_pct - 0.003) > 1e-12:
        raise SystemExit("不能同时改收盘站回容差。")
    if before.max_pullback_days != 10 or after.max_pullback_days != 10:
        raise SystemExit("回踩窗口必须仍是 10 日。")
    if before.require_breakout_volume or after.require_breakout_volume:
        raise SystemExit("不能同时打开突破日放量。")
    if before.pullback_max_normal_volume_ratio != 1.0 or after.pullback_max_normal_volume_ratio != 1.0:
        raise SystemExit("回踩常态量必须保持 1.0 倍。")
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
        raise SystemExit(f"同日基准与现行默认重放不一致，最大差 {gap.max()}")
    signals = pd.DataFrame(raw)
    if signals.empty:
        raise SystemExit("没有信号。")
    signals = signals.join(bench, on="signal_date")
    for horizon in HORIZONS:
        signals[f"excess_{horizon}"] = signals[f"ret_{horizon}"] - signals[f"bench_{horizon}"]
    before_ids = set(zip(prior.stock_code, prior.event_id))
    after_ids = set(zip(signals.stock_code, signals.event_id))
    matched = prior.merge(signals, on=["stock_code", "event_id"], suffixes=("_b", "_a"))
    shift = {
        "same_event": len(before_ids & after_ids),
        "new_event": len(after_ids - before_ids),
        "dropped_event": len(before_ids - after_ids),
        "date_shifted": int((matched.signal_date_b != matched.signal_date_a).sum()) if len(matched) else 0,
    }
    rows = []
    for row in prior_rows():
        rows.append({**row, "variant": "改前：最低价可低于线 3%"})
    for label, start, end in PERIODS:
        window = signals[(signals.signal_date >= start) & (signals.signal_date <= end)]
        record = metrics(window)
        record.update({"label": label, "variant": "改后：最低价跌破不得超过 1%", "arm": "after"})
        rows.append(record)
    valid = all(passes(row["t5"]) for row in rows if row.get("arm") == "after")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": AS_OF, "cost_pct": COST_PCT,
        "codes": len(paths), "replayed": len(returns), "failures": failures,
        "meets_validity_standard": valid, "retained_as_default": False,
        "shift": shift, "rows": rows,
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
        "# 主线 B：回踩最低价跌破精准线不得超过 1%",
        "",
        f"生成时间：{payload['generated_at']}；行情截止 {payload['as_of']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 B：精准线回踩。三角整理默认不改。不重做收盘站稳，也不重做必须触线。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：回踩日最低价可以低于精准线最多 3%，仍算踩到。也可以停在线上方 1.2% 内。",
        "改后：最低价低于线超过 1%，这一天不算回踩。跌破 1% 以内、或停在线上方 1.2% 内，仍算踩到。",
        "确认日收盘仍可低于线 0.3%。回踩仍可在突破后 1～10 个交易日内完成。",
        "跌破过深的当天被拒绝后，引擎可以改等后面跌破更浅的一天，或让这次突破过期后再等下一次。",
        (
            f"同一事件里，{shift['date_shifted']} 笔确认日被换成了另一天；"
            f"{shift['new_event']} 个事件是新的，{shift['dropped_event']} 个旧事件不再给出买入，"
            f"{shift['same_event']} 个事件仍在。"
        ),
        "回踩量仍是不超过突破前均量的 1.0 倍。突破日放量开关仍关闭。",
        "默认仍允许最低价低于线 3%。只有两段都达标，才把默认跌破上限改成 1%。",
        "",
        "## 3. 改前改后",
        "",
        "改前是现行默认在同一冻结行情上的重放。同日等权基准与那次保存的基准一致，否则脚本退出。",
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
        lines.append("两段都达标。新扫描可以把回踩跌破上限从 3% 改成 1%。旧批次不追溯。三角默认不改。")
    else:
        before = [row for row in payload["rows"] if "3%" in row["variant"]]
        after = [row for row in payload["rows"] if row.get("arm") == "after"]
        lines += [
            "不把「最低价跌破不得超过 1%」留成默认。当前默认仍允许最低价低于精准线 3%。三角扫描默认也不改。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "对照配置是 `max_intraday_undercut_pct=0.01`。默认仍是 0.03。收盘容差仍是 0.003，触线容差仍是 0.012。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        f"- 重放失败 {len(payload['failures'])} 只，没有换股票。",
        "- 只比较了 3% 和 1%，没有再试 1.5% 或 2%。",
        "- 前复权与幸存者偏差同锁定行情。旗形没有锁定成交，不报胜率。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
