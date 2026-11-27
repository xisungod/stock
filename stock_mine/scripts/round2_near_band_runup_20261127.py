"""第二轮只做一步：在当前默认信号发出之后，用近线带把冲高过高的回踩丢掉。

假设（先写死，再算数字）：
买入预警已经要求信号日收盘离精准线不超过 max_extension_pct（0.035，即 3.5%）。
冲高上限却是这条近线带的大约 3 倍（10%）。本轮只问：如果突破后的最高价也没有离开
同一条近线带，这批回踩的 T+5 超额是否达到项目既有的有效标准。

只使用已提交的 signals.csv 里、信号日就已经写下的 runup_pct。
不换样本、不换区间、不重抓行情、不扫其他阈值。
这是信号发出后的丢弃，不改引擎当天选哪一条线，因此和「大盘偏弱不买」一样，
可以在同一张锁定表上对照。达不到两段标准就不改选股规则。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_locked_20261127 import (  # noqa: E402
    arm_record,
    ci_text,
    load_signals,
    period_slice,
    select_arm,
)

OUT = ROOT / "reports/iteration_20261127_round2"


def near_band_pct() -> float:
    source = (ROOT / "backend/app/services/strategy.py").read_text(encoding="utf-8")
    match = re.search(r"max_extension_pct: float = ([0-9.]+)", source)
    if match is None:
        raise SystemExit("strategy.py 里找不到 max_extension_pct")
    return float(match.group(1)) * 100


def main() -> None:
    band = near_band_pct()
    if abs(band - 3.5) > 1e-9:
        raise SystemExit(f"近线带不是 3.5%：{band}。先停下来，不要换一个更好看的阈值。")
    signals, summary, dev_end, hold_end = load_signals()
    published = summary["results"]
    dev_label = next(label for label in published if label.startswith("development"))
    hold_label = next(label for label in published if label.startswith("holdout"))
    current = select_arm(signals, "new")
    kept = current[current["runup_pct"] <= band]
    dropped = current[current["runup_pct"] > band]
    periods = (
        ("开发段", "2019-01-01", dev_end, dev_label),
        ("留出段", "2024-01-01", hold_end, hold_label),
    )
    rows = []
    for label, start, end, published_label in periods:
        before = arm_record(period_slice(current, start, end))
        after = arm_record(period_slice(kept, start, end))
        removed = arm_record(period_slice(dropped, start, end))
        expected = published[published_label]["new"]["t5"]
        got = before["t5"]
        for field in ("n", "mean_net", "win_rate", "mean_excess", "excess_ci95"):
            if got[field] != expected[field]:
                raise SystemExit(f"改前对不上已发表的当前默认 {label} {field}: {got[field]} != {expected[field]}")
        before.update({"label": label, "variant": "改前：当前默认（冲高上限 10%）"})
        after.update({"label": label, "variant": f"改后：越线不超过近线带 {band:g}%"})
        removed.update({"label": label, "variant": f"被丢掉：越线高于 {band:g}%"})
        rows.extend((before, after, removed))

    after_rows = [row for row in rows if row["variant"].startswith("改后")]
    kept_rule = all(
        cell["mean_excess"] is not None and cell["mean_excess"] > 0 and cell["excess_ci95"] and cell["excess_ci95"][0] > 0
        for cell in (row["t5"] for row in after_rows)
    )
    payload = {
        "band_pct": band,
        "kept_rule": kept_rule,
        "logic_change_retained": False,
        "rows": rows,
        "as_of": summary["meta"]["as_of"],
        "dev_end": dev_end,
        "hold_end": hold_end,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(markdown(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} kept={kept_rule}")


def _row(payload: dict, label: str, prefix: str) -> dict:
    return next(row for row in payload["rows"] if row["label"] == label and row["variant"].startswith(prefix))


def attribution(payload: dict) -> str:
    before_dev, after_dev = _row(payload, "开发段", "改前"), _row(payload, "开发段", "改后")
    before_hold, after_hold = _row(payload, "留出段", "改前"), _row(payload, "留出段", "改后")
    drop_dev, drop_hold = _row(payload, "开发段", "被丢掉"), _row(payload, "留出段", "被丢掉")
    return (
        f"改后开发段 T+5 超额从 {before_dev['t5']['mean_excess']}% 变成 {after_dev['t5']['mean_excess']}%"
        f"（{ci_text(after_dev['t5']['excess_ci95'])}），均值不再大于 0。"
        f"留出段从 {before_hold['t5']['mean_excess']}% 变成 {after_hold['t5']['mean_excess']}%"
        f"（{ci_text(after_hold['t5']['excess_ci95'])}），下限仍小于 0。"
        f"被丢掉的部分两段超额点估计都更高（开发段 {drop_dev['t5']['mean_excess']}%，留出段 {drop_hold['t5']['mean_excess']}%）。"
        "近线带留下的是超额更低的那一组。"
        f"改后开发段的 T+20 超额是 {after_dev['t20']['mean_excess']}%，留出段的 T+10 / T+20 是 "
        f"{after_hold['t10']['mean_excess']}% / {after_hold['t20']['mean_excess']}%。判定仍只看事先声明的 T+5。"
    )


def markdown(payload: dict) -> str:
    band = payload["band_pct"]
    lines = [
        "# 第二轮：近线带内的冲高（一步位置过滤）",
        "",
        "生成方式：`python3 scripts/round2_near_band_runup_20261127.py`。",
        "第一轮的基线、三角整理结论和「去掉突破日涨停」没有重做。本轮只加这一步。",
        "",
        "## 结论",
        "",
    ]
    if payload["kept_rule"]:
        lines.append(
            f"越线不超过 {band:g}% 的子集在开发段和留出段都达到了 T+5 有效标准。"
            "按本轮事先写好的规则，这一条可以进入选股逻辑。"
        )
    else:
        lines += [
            f"不改选股规则。把已发出信号里「突破后最高越线不超过 {band:g}%」的子集留下来，"
            "开发段和留出段没有同时达到既有有效标准（T+5 超额均值大于 0，且月度分块 95% 下限大于 0）。",
            "没有改阈值再试，也没有把旗形或三角整理补成新的选股策略。",
        ]
    lines += [
        "",
        "## 这一步问什么",
        "",
        "当前默认已经要求两件位置上的事：信号日收盘离精准线不超过 `max_extension_pct`（3.5%），",
        "突破后最高价越线不超过 10%。`schemas.py` 把 10% 写成近线带的大约 3 倍。",
        "第一轮没有把「最高价也留在 3.5% 近线带里」当成一条通过/不通过的规则，只描述过按收益切出来的五档。",
        "那些分档边界（大约 2.87%、4.05%、5.34%、7.17%）不是本轮的阈值。本轮只用策略里已经写死的 3.5%，用一次。",
        "",
        "它只用信号日已经知道的 `runup_pct`（突破日到信号日的最高价相对线价）。",
        "信号发出之后再丢掉越线过高的预警，不让引擎改选另一条线。",
        "因此对照就是同一张 `validation/signals.csv` 里 `arm=new` 的当前默认，时段仍是",
        f"开发段 2019-01-01～{payload['dev_end']}、留出段 2024-01-01～{payload['hold_end']}。",
        "改前的 T+5 样本数、净收益、胜率、超额和区间必须与已发表的当前默认一致，否则脚本退出。",
        "",
        "## 改前改后",
        "",
        "| 时段 | 口径 | 信号 | 有 T+5 成交 | 净胜率 | 平均净收益 | 中位净收益 | T+5 超额 | 超额95%区间 | T+10 超额 | T+20 超额 | 该时段达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        t5, t10, t20 = row["t5"], row["t10"], row["t20"]
        ok = t5["mean_excess"] is not None and t5["mean_excess"] > 0 and t5["excess_ci95"] and t5["excess_ci95"][0] > 0
        lines.append(
            f"| {row['label']} | {row['variant']} | {row['signals']} | {t5['n']} | {t5['win_rate']}% | "
            f"{t5['mean_net']}% | {t5['median_net']}% | {t5['mean_excess']}% | {ci_text(t5['excess_ci95'])} | "
            f"{t10['mean_excess']}% | {t20['mean_excess']}% | {'是' if ok else '否'} |"
        )
    lines += [
        "",
        "「被丢掉」一行是对照，不是第三条候选规则。它说明去掉的那部分自己什么样，不参与是否保留的判定。",
        "",
        attribution(payload),
        "",
        "## 为什么这样处理",
        "",
    ]
    if payload["kept_rule"]:
        lines.append("两段都达标，见仓库里随后的规则实现。旧批次没有这个字段时不追溯启用。")
    else:
        lines += [
            "有效标准沿用 `validate_logic_fixes.py`：必须开发段和留出段同时达标。只看点估计更高不算。",
            "旗形仍然没有锁定成交，本轮不编造胜率。三角整理仍只作标注，不新造选股规则。",
            "行情缓存不在仓库里。把 10% 上限直接改成 3.5% 会让引擎有机会改选另一条线，那一步无法在这份锁定表上复现，所以没有改引擎。",
            "",
            "## 限制",
            "",
            "- 这是第一轮之后追加的一次比较。标准没有放宽，也没有因为没通过就改用别的持有期或别的切点。",
            "- 近线带约束的是信号日收盘；拿它去卡突破后的最高价，是同一数字的新用途，不是引擎里已经生效的条件。",
            "- 前复权、幸存者偏差、以及留出段已被先前研究看过，限制与第一轮相同。",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
