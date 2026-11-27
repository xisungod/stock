"""第三轮只做一步：在当前默认信号发出之后，用已有的 0.75 缩量比例再卡常态量。

假设（先写死，再算数字）：
当前默认已经要求回踩量不超过突破前 10 日均量（pullback_max_normal_volume_ratio = 1），
并且回踩量不超过突破日的 75%（pullback_vol_max_ratio = 0.75）。
0.75 目前只用来和突破日比。本轮只问：把同一个 0.75 用到突破前均量上，
留下的信号是否在开发段和留出段都达到 T+5 有效标准。

不重做第一轮基线，不重做涨停过滤，不重做 3.5% 近线带。
不换样本、不换区间、不扫其他倍数。缺失的常态量比值不算通过。
这是信号发出后的丢弃，不改引擎当天选哪一条线。达不到两段标准就不改选股规则。
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

OUT = ROOT / "reports/iteration_20261127_round3"


def shrink_ratio() -> float:
    source = (ROOT / "backend/app/services/strategy.py").read_text(encoding="utf-8")
    match = re.search(r"pullback_vol_max_ratio: float = ([0-9.]+)", source)
    if match is None:
        raise SystemExit("strategy.py 里找不到 pullback_vol_max_ratio")
    return float(match.group(1))


def _row(payload: dict, label: str, prefix: str) -> dict:
    return next(row for row in payload["rows"] if row["label"] == label and row["variant"].startswith(prefix))


def attribution(payload: dict) -> str:
    ratio = payload["ratio"]
    before_dev, after_dev = _row(payload, "开发段", "改前"), _row(payload, "开发段", "改后")
    before_hold, after_hold = _row(payload, "留出段", "改前"), _row(payload, "留出段", "改后")
    drop_dev, drop_hold = _row(payload, "开发段", "被丢掉"), _row(payload, "留出段", "被丢掉")
    return (
        f"改后开发段 T+5 超额从 {before_dev['t5']['mean_excess']}% 变成 {after_dev['t5']['mean_excess']}%"
        f"（{ci_text(after_dev['t5']['excess_ci95'])}）。"
        f"留出段从 {before_hold['t5']['mean_excess']}% 变成 {after_hold['t5']['mean_excess']}%"
        f"（{ci_text(after_hold['t5']['excess_ci95'])}）。"
        f"被丢掉的部分（常态量比高于 {ratio:g}，或比值缺失）超额是开发段 {drop_dev['t5']['mean_excess']}%"
        f"、留出段 {drop_hold['t5']['mean_excess']}%。"
        "判定只看事先声明的 T+5：两段都要均值大于 0 且 95% 下限大于 0。"
    )


def markdown(payload: dict) -> str:
    ratio = payload["ratio"]
    lines = [
        "# 第三轮：把 0.75 缩量比例用到常态量上",
        "",
        "生成方式：`python3 scripts/round3_normal_volume_20261127.py`。",
        "第一轮基线、去掉突破日涨停、第二轮 3.5% 近线带都没有重做。本轮只加这一步量能过滤。",
        "",
        "## 结论",
        "",
    ]
    if payload["kept_rule"]:
        lines.append(
            f"回踩量不超过突破前均量的 {ratio:g} 倍，在开发段和留出段都达到了 T+5 有效标准。"
            "按本轮事先写好的规则，这一条可以进入选股逻辑。旧批次没有这个字段时不追溯启用。"
        )
    else:
        dev, hold = _row(payload, "开发段", "改后"), _row(payload, "留出段", "改后")
        lines += [
            f"不改选股规则。在已发出的当前默认信号里，只留下回踩量不超过突破前 10 日均量 {ratio:g} 倍的那些，"
            "开发段和留出段没有同时达到既有有效标准（T+5 超额均值大于 0，且月度分块 95% 下限大于 0）。",
            f"开发段单独看是达标的：超额 {dev['t5']['mean_excess']}%（{ci_text(dev['t5']['excess_ci95'])}）。"
            f"留出段超额变成 {hold['t5']['mean_excess']}%（{ci_text(hold['t5']['excess_ci95'])}），方向反过来了。",
            "没有改用别的倍数，也没有再试 3.5% 近线带、涨停过滤、旗形胜率或新的三角选股。",
        ]
    lines += [
        "",
        "## 这一步问什么",
        "",
        "当前默认已经有两条量能条件：回踩量不超过突破日的 75%（`pullback_vol_max_ratio`），",
        "并且不超过突破前 10 日均量的 1 倍（`pullback_max_normal_volume_ratio`）。",
        "1 倍这条已经作为修复项测过，没有单独通过有效标准。75% 只和突破日比较，突破日本身通常放量，所以这条很容易满足。",
        "本轮只问：同一个 0.75，如果改成相对突破前均量，留下的信号还达不达标。",
        "第一轮按收益切过常态量的五档，那些边界不是本轮的阈值。这里只用策略源码里的 `pullback_vol_max_ratio`，用一次。",
        "",
        "它只用信号日已经写下的 `normal_volume_ratio`。比值缺失的信号不放进保留组。",
        "信号发出之后再丢弃，不让引擎改选另一条线。",
        "对照仍是 `validation/signals.csv` 里 `arm=new` 的当前默认，时段仍是",
        f"开发段 2019-01-01～{payload['dev_end']}、留出段 2024-01-01～{payload['hold_end']}。",
        "改前的 T+5 数字必须与已发表的当前默认一致，否则脚本退出。",
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
        "「被丢掉」一行是对照，不是另一条候选。它不参与是否保留的判定。",
        "",
        attribution(payload),
        "",
        "## 为什么这样处理",
        "",
    ]
    if payload["kept_rule"]:
        lines.append("两段都达标。实现时只在信号发出后丢弃，不把引擎内部的 1 倍常态量上限直接改成 0.75，避免改选另一条线。旧批次缺字段则关闭。")
    else:
        lines += [
            "有效标准沿用 `validate_logic_fixes.py`：必须开发段和留出段同时达标。点估计更高不算通过。",
            "旗形仍然没有锁定成交，本轮不编造胜率。三角整理仍只作标注。",
            "行情缓存不在仓库里。若把引擎内部的常态量上限从 1 改成 0.75，当天可能会改选另一条线，那一步无法在这份锁定表上复现，所以没有改引擎。",
            "",
            "## 限制",
            "",
            "- 0.75 在引擎里的原意是相对突破日，不是相对突破前均量。本轮是把这个已有比例换一个参照，不是拟合出来的新阈值。",
            "- 这是前两轮之后追加的一次比较。没有因为没通过就改持有期或改倍数。",
            "- 前复权、幸存者偏差、以及留出段已被先前研究看过，限制与第一轮相同。",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ratio = shrink_ratio()
    if abs(ratio - 0.75) > 1e-9:
        raise SystemExit(f"缩量比例不是 0.75：{ratio}。先停下来，不要改成别的阈值。")
    signals, summary, dev_end, hold_end = load_signals()
    published = summary["results"]
    dev_label = next(label for label in published if label.startswith("development"))
    hold_label = next(label for label in published if label.startswith("holdout"))
    current = select_arm(signals, "new")
    known = current["normal_volume_ratio"].notna()
    kept = current[known & (current["normal_volume_ratio"] <= ratio)]
    dropped = current[~(known & (current["normal_volume_ratio"] <= ratio))]
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
        before.update({"label": label, "variant": "改前：当前默认（常态量 ≤ 1 倍）"})
        after.update({"label": label, "variant": f"改后：常态量 ≤ {ratio:g} 倍"})
        removed.update({"label": label, "variant": f"被丢掉：常态量 > {ratio:g} 或缺失"})
        rows.extend((before, after, removed))

    after_rows = [row for row in rows if row["variant"].startswith("改后")]
    kept_rule = all(
        cell["mean_excess"] is not None and cell["mean_excess"] > 0 and cell["excess_ci95"] and cell["excess_ci95"][0] > 0
        for cell in (row["t5"] for row in after_rows)
    )
    payload = {
        "ratio": ratio,
        "missing_ratio": int((~known).sum()),
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
    print(f"REPORT {OUT / 'report.md'} kept={kept_rule} missing_ratio={payload['missing_ratio']}")


if __name__ == "__main__":
    sys.exit(main())
