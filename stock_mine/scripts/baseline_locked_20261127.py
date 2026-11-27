"""从已提交的锁定账本复现基线，并比较信号日已知字段上的赢单与亏单。

只读这些文件，不抓行情、不重写旧报告、不改选股规则：

- reports/logic_fixes_20261127/validation/signals.csv 与 summary.json
- reports/logic_fixes_20261127/line_stop/trades.csv
- reports/logic_fixes_20261127/patterns/trades.csv 与 report.md
- backend/app/services/chart_patterns.py（只用来核对三角整理的代码是否仍是初版回测）

月度分块自助法与 scripts/validate_logic_fixes.py 的 block_bootstrap 相同：
种子 20261127，4000 次，按自然月重抽样，取 2.5% 与 97.5% 分位。
脚本会对照 summary.json 里已经发表的 T+5 数字，对不上就退出。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "reports/logic_fixes_20261127"
OUT = ROOT / "reports/iteration_20261127_baseline"
SEED = 20261127
ITERATIONS = 4000
HORIZONS = (5, 10, 20)
ARMS = (
    ("old", "修复前规则"),
    ("only_normal_volume", "仅加：回踩量≤常态量"),
    ("only_runup", "仅加：突破后冲高≤10%"),
    ("only_event_age", "仅加：证据时效750根"),
    ("new", "修复后规则（当前默认，大盘只标注）"),
    ("new_downgrade", "修复后 + 大盘偏弱不买（已有对照，不是默认）"),
)
CONTINUOUS = (
    ("normal_volume_ratio", "回踩日量 / 突破前10日均量"),
    ("runup_pct", "突破后最高越线 %"),
    ("line_points", "精准线独立验证事件数"),
    ("calendar_gap", "突破日到信号日的自然日间隔"),
)
# 与 success_feature_analysis.py 相同的稳定门槛。本轮连续字段只有 4 个，
# 仍用 |z|≥2.6，不因为检验更少就把门槛放宽。
STABLE_Z = 2.6
STABLE_AUC_GAP = 0.03
LINE_GROUP = "当前默认（修复后规则，大盘只提示）"
TRIANGLE = "三角形向上突破"
# 已发表、本脚本必须重算到同一位的单元格。来源是锁定报告，不是新跑出来的目标。
LINE_EXPECT = {
    ("20", "2019-01-01", "2023-09-30"): {"n": 3096, "mean": 0.358, "win_rate": 24.4, "paired": 0.42},
    ("60", "2019-01-01", "2023-09-30"): {
        "n": 3095, "mean": 0.075, "win_rate": 14.1, "paired": 0.276, "paired_ci95": [-0.354, 0.915],
    },
    ("60", "2024-01-01", "2099-12-31"): {
        "n": 2070, "mean": 0.64, "win_rate": 16.5, "paired": 0.423, "paired_ci95": [-0.312, 1.092],
    },
}
TRIANGLE_LEVEL = 1 - 0.05 / 3
TRIANGLE_EXPECT = {
    ("60", "2019-01-01", "2023-09-30"): {
        "n": 2878, "mean": 0.493, "median": -7.477, "win_rate": 34.0, "paired": -0.614,
        "paired_ci95": [-1.637, 0.417],
    },
    ("60", "2024-01-01", "2099-12-31"): {
        "n": 1917, "mean": 0.188, "median": -8.221, "win_rate": 32.0, "paired": -0.946,
        "paired_ci95": [-2.199, 0.276],
    },
}


def block_bootstrap(frame: pd.DataFrame, column: str, iterations: int = ITERATIONS,
                    level: float = 0.95) -> list[float] | None:
    valid = frame[["signal_date", column]].dropna()
    if valid["signal_date"].nunique() < 10:
        return None
    months = pd.to_datetime(valid["signal_date"]).dt.to_period("M")
    blocks = valid.groupby(months, observed=True)[column].agg(["sum", "count"])
    values, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    means = values[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return [round(float(x), 3) for x in np.percentile(means, [tail, 100 - tail])]


def auc(success: pd.Series, failure: pd.Series) -> tuple[float | None, float | None]:
    a, b = success.dropna().to_numpy(), failure.dropna().to_numpy()
    if len(a) < 20 or len(b) < 20:
        return None, None
    ranks = pd.Series(np.r_[a, b]).rank().to_numpy()
    value = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    z = (value - 0.5) / math.sqrt((len(a) + len(b) + 1) / (12 * len(a) * len(b)))
    return round(float(value), 3), round(float(z), 2)


def board_of(code: str) -> str:
    return "科创板" if str(code).startswith("68") else "创业板" if str(code).startswith("30") else "主板"


def load_signals() -> tuple[pd.DataFrame, dict, str, str]:
    summary = json.loads((LOCKED / "validation/summary.json").read_text())
    labels = list(summary["results"])
    dev_label = next(label for label in labels if label.startswith("development"))
    hold_label = next(label for label in labels if label.startswith("holdout"))
    dev_end = dev_label.split("～")[1].rstrip("）")
    hold_end = hold_label.split("～")[1].rstrip("）")
    frame = pd.read_csv(LOCKED / "validation/signals.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    frame["stock_code"] = frame["stock_code"].str.zfill(6)
    frame["market_ok"] = frame["market_weak"] != True  # noqa: E712  缺数据时与原脚本一样不拦截
    frame["calendar_gap"] = (pd.to_datetime(frame["signal_date"]) - pd.to_datetime(frame["breakout_date"])).dt.days
    frame["board"] = frame["stock_code"].map(board_of)
    return frame, summary, dev_end, hold_end


def select_arm(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "new_downgrade":
        return frame[(frame["arm"] == "new") & frame["market_ok"]]
    return frame[frame["arm"] == key]


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame[(frame["signal_date"] >= start) & (frame["signal_date"] <= end)].copy()


def horizon_metrics(frame: pd.DataFrame, horizon: int) -> dict:
    net = frame[f"ret_{horizon}"].dropna()
    excess = frame[f"excess_{horizon}"].dropna()
    gains, losses = net[net > 0], net[net < 0]
    return {
        "n": int(len(net)),
        "mean_net": round(float(net.mean()), 3) if len(net) else None,
        "median_net": round(float(net.median()), 3) if len(net) else None,
        "win_rate": round(float((net > 0).mean() * 100), 1) if len(net) else None,
        "mean_excess": round(float(excess.mean()), 3) if len(excess) else None,
        "beat_benchmark_rate": round(float((excess > 0).mean() * 100), 1) if len(excess) else None,
        "excess_ci95": block_bootstrap(frame, f"excess_{horizon}") if len(frame) else None,
        "profit_factor": round(float(gains.sum() / -losses.sum()), 3) if len(losses) else None,
        "payoff": round(float(gains.mean() / -losses.mean()), 3) if len(gains) and len(losses) else None,
        "avg_win": round(float(gains.mean()), 3) if len(gains) else None,
        "avg_loss": round(float(losses.mean()), 3) if len(losses) else None,
    }


def arm_record(frame: pd.DataFrame) -> dict:
    record = {
        "signals": int(len(frame)),
        "stocks": int(frame["stock_code"].nunique()) if len(frame) else 0,
        "signal_days": int(frame["signal_date"].nunique()) if len(frame) else 0,
    }
    for horizon in HORIZONS:
        record[f"t{horizon}"] = horizon_metrics(frame, horizon)
    return record


def assert_matches_published(got: dict, published: dict, period: str, key: str) -> None:
    for field in ("n", "mean_net", "median_net", "win_rate", "mean_excess", "beat_benchmark_rate", "excess_ci95"):
        if got[field] != published[field]:
            raise SystemExit(f"复现失败 {period} {key} T+5 {field}: {got[field]} != {published[field]}")


def trade_stats(frame: pd.DataFrame, level: float = 0.95) -> dict:
    ret = frame["ret"]
    wins, losses = ret[ret > 0], ret[ret <= 0]
    paired = frame.dropna(subset=["paired_diff"])
    return {
        "n": int(len(frame)),
        "mean": round(float(ret.mean()), 3) if len(frame) else None,
        "median": round(float(ret.median()), 3) if len(frame) else None,
        "win_rate": round(float((ret > 0).mean() * 100), 1) if len(frame) else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
        "stopped_rate": round(float(frame["stopped"].mean() * 100), 1) if "stopped" in frame and len(frame) else None,
        "median_days": float(frame["days"].median()) if "days" in frame and len(frame) else None,
        "paired": round(float(paired["paired_diff"].mean()), 3) if len(paired) else None,
        "paired_ci95": block_bootstrap(paired, "paired_diff", level=level) if len(paired) else None,
    }


def assert_trade(got: dict, expected: dict, label: str) -> None:
    for field, value in expected.items():
        if got[field] != value:
            raise SystemExit(f"复现失败 {label} {field}: {got[field]} != {value}")


def feature_rows(frame: pd.DataFrame, dev_end: str) -> list[dict]:
    usable = frame.dropna(subset=["ret_5"]).copy()
    win, lose = usable["ret_5"] > 0, usable["ret_5"] <= 0
    dev, hold = usable["signal_date"] <= dev_end, usable["signal_date"] >= "2024-01-01"
    rows = []
    for feature, label in CONTINUOUS:
        overall, z = auc(usable.loc[win, feature], usable.loc[lose, feature])
        dev_auc, _ = auc(usable.loc[win & dev, feature], usable.loc[lose & dev, feature])
        hold_auc, _ = auc(usable.loc[win & hold, feature], usable.loc[lose & hold, feature])
        stable = (
            overall is not None and dev_auc is not None and hold_auc is not None and z is not None
            and abs(z) >= STABLE_Z and (dev_auc - 0.5) * (hold_auc - 0.5) > 0
            and min(abs(dev_auc - 0.5), abs(hold_auc - 0.5)) >= STABLE_AUC_GAP
        )
        rows.append({
            "feature": feature, "label": label, "auc": overall, "z": z,
            "auc_dev": dev_auc, "auc_holdout": hold_auc, "stable": stable,
            "median_win": round(float(usable.loc[win, feature].median()), 3),
            "median_loss": round(float(usable.loc[lose, feature].median()), 3),
            "wins": int(win.sum()), "losses": int(lose.sum()),
        })
    return rows


def quintiles(frame: pd.DataFrame, feature: str, dev_end: str, hold_end: str) -> list[dict]:
    usable = frame.dropna(subset=["ret_5", feature]).copy()
    dev = usable[usable["signal_date"] <= dev_end]
    edges = np.unique(np.quantile(dev[feature].to_numpy(), [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 3:
        return []
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    usable["bin"] = pd.cut(usable[feature], edges, include_lowest=True)
    rows = []
    for period, start, end in (("development", "2019-01-01", dev_end), ("holdout", "2024-01-01", hold_end)):
        part = usable[(usable["signal_date"] >= start) & (usable["signal_date"] <= end)]
        for interval, group in part.groupby("bin", observed=True):
            net = group["ret_5"].dropna()
            rows.append({
                "feature": feature, "period": period, "bin": str(interval), "n": int(len(group)),
                "win_rate": round(float((net > 0).mean() * 100), 1) if len(net) else None,
                "mean_net": round(float(net.mean()), 3) if len(net) else None,
                "mean_excess": round(float(group["excess_5"].mean()), 3) if group["excess_5"].notna().any() else None,
            })
    return rows


def fmt(value, suffix="") -> str:
    if value is None or (isinstance(value, float) and abs(value) < 5e-4):
        return f"0{suffix}" if isinstance(value, float) else "-"
    return f"{value}{suffix}"


def ci_text(ci) -> str:
    return "-" if not ci else f"{ci[0]}%～{ci[1]}%"


def passes(cell: dict) -> bool:
    ci = cell["excess_ci95"]
    return cell["mean_excess"] is not None and cell["mean_excess"] > 0 and ci is not None and ci[0] > 0


def _cat(payload: dict, field: str, value: str, period: str) -> float:
    return next(row["mean_excess"] for row in payload["categories"]
                if row["field"] == field and row["value"] == value and row["period"] == period)


def markdown(payload: dict) -> str:
    meta = payload["meta"]
    default = payload["precision_default"]
    lines = [
        "# 迭代第一步：锁定样本基线与正向案例特征",
        "",
        f"生成方式：`python3 scripts/baseline_locked_20261127.py`。数字全部来自已提交账本，评估截止 {meta['as_of']}。",
        "本轮没有修改选股规则，也没有重写 `reports/logic_fixes_20261127/` 里的旧结果。",
        "",
        "## 结论",
        "",
        "1. 三角整理已经写在代码里，但是 K 线标注和一份独立研究回测，不是扫描策略，也不等于旗形。向上突破在锁定回测里没有通过事先声明的判定。本轮不新造一套三角整理选股。",
        "2. 能复现的收益基线是精准线回调。当前默认（三条修复都打开、大盘只标注）在开发段和留出段的 T+5 超额都是正的，月度分块 95% 下限都小于 0，按项目既有标准不算有效。净胜率约 46%～48%，中位数为负。",
        "3. 旗形突破是工作台可选策略，但这份仓库里没有它的锁定成交账本，不能报净胜率或净收益。",
        "4. 在信号日已经知道的字段上，T+5 赢单和亏单分不开。没有一个连续字段达到既有的稳定标准。两条现成标注（涨停突破、大盘偏弱）做成过滤之后，两段区间下限仍然小于 0，所以不写进规则。",
        "",
        "## 三角整理在代码和文档里的实际情况",
        "",
        "扫描模式只有三种：`combined`（工作台默认，精准线或旗形）、`precision_pullback`、`flag_breakout`。没有三角整理这个 `strategy_mode`。",
        "",
        "- 识别函数是 `backend/app/services/chart_patterns.py` 的 `find_triangles`。详情页用它画上下沿，`scripts/pattern_backtest.py` 用同一个函数做研究回测。",
        "- 旗形在 `backend/app/services/flag_pattern.py`。它要求旗杆之后的通道近似平行，并明确排除明显楔形；通道、ATR 和基准量截止信号日前，信号当天不参与重画边界。三角整理要求两峰两谷把宽度收到原来的 60% 以内，形状是收敛，不是平行旗面。两者不是同一个形态。",
        "- README 把「精准线回调」和「旗形突破」写成可选筛选；「K线形态标注」一节写明三角形只作结构标注。这和代码一致。",
        f"- 锁定回测目录 `reports/logic_fixes_20261127/patterns/` 的说明文字仍是生成当时的简版（报告里{'有' if meta['pattern_report_uses_short_prose'] else '没有'}「最近四个峰谷点」这段旧表述）。当前 `find_triangles` {'包含' if meta['code_has_shared_span_rule'] else '不包含'}共同区间长度、首次波幅和形成期内收盘不越界这些约束。`chart_patterns.py` 写明那份历史回测用的是初版口径。",
        "- 行情缓存没有入库。本轮只把锁定的 `trades.csv` 重新汇总，确认笔数和收益与已发表的表一致，没有用今天的函数重放形态，也没有改旧成交。",
        "",
        "主口径是持有上限 60 个交易日、收盘跌破形态止损才离场、扣双边 20bps，对照是同一天、同样止损距离的随机买点。判定要开发段和留出段的配对差都大于 0，且 Bonferroni 月度分块区间下限都大于 0。三角形向上突破未通过。",
        "",
        "| 时段 | 笔数 | 平均净收益 | 中位数 | 净胜率 | 利润因子 | 配对差（98.3% 区间） | 已发表判定 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["triangle_primary"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['mean']}% | {row['median']}% | {row['win_rate']}% | "
            f"{row['profit_factor']} | {row['paired']}%（{ci_text(row['paired_ci95'])}） | 未通过 |"
        )
    lines += [
        "",
        "同一文件里的 W 底 A、W 底 B 也未通过。它们不是本轮要补的形态，这里只说明那份账本不只有三角形。",
        "该目录 `report.md` 的描述统计写的是：识别后 20 日绝对涨跌幅与同日、同 ATR 分位的随机股票几乎相同，向上和向下突破的次数也接近。那份锁定结论是，初版三角形识别既不指示方向，也不带来额外波动。",
        "",
        "## 验证管线，以及本轮用哪一套锁定样本",
        "",
        "`scripts/validate_platform_v2.py` 是更早的三引擎协议：先冻结源码哈希、分段、参数、抽样和行情快照，再跑 `legacy` / `pivot_fixed` / `platform_events`。",
        "默认分段是 2024 开发段、2025 留出段、2026-01-01～2026-09-09；每段末尾清掉 20 个共同交易日；成本 20bps；平台事件按冻结突破去重。",
        "README 和 `REVIEW_FIX_WORKLOG.md` 把 v4 的对照放在 `reports/platform_v2_validation/locked_20260910_r3/`。这个目录没有进入 git，仓库里也没有那一轮的收益率。r3 的完成记录是 24/24 只股票跑完。更早的平台事件工作日志写过「盈利能力没有确立」，那不是 r3 的收益表，本轮不把它改写成 r3 的数字，也不另抽一组股票代替缺失的报告。",
        "",
        "当前规则的锁定样本是之后修正了行号错误的这一套，已经提交：",
        "",
        f"- 账本：`reports/logic_fixes_20261127/validation/signals.csv`。哈希抽样，种子 `logic-fix-20261127`，成功 {meta['succeeded']} / {meta['sampled']} 只，截止 {meta['as_of']}。",
        "- 信号：`platform_events` 精准线，`strategy_mode=precision_pullback`。这不是工作台默认的联合筛选。联合模式里的旗形没有落在这张表里。",
        "- 成交：信号次日开盘买入，次日一字涨停则整笔没有收益；第 5/10/20 个交易日收盘卖出；双边成本 20bps。",
        "- 超额：信号净收益减去同一信号日、同一期限、样本内等权收益。",
        f"- 时段：开发段 2019-01-01～{meta['dev_end']}；留出段 2024-01-01～{meta['hold_end']}（末尾 {meta['purge_sessions']} 个交易日不出信号，终点取自已发表的 summary.json）。",
        "- 判定：开发段和留出段的 T+5 超额均值都大于 0，且月度分块 95% 下限都大于 0。",
        "",
        "本脚本对六个已发表组的 T+5 样本数、净均值、胜率、超额和区间做了逐项对照，全部一致。跌破精准线的主口径和三角形主口径的笔数、平均净收益、胜率、配对差也与已发表的表一致。",
        "",
        "## 按策略拆开的基线",
        "",
        "### 精准线回调：当前默认",
        "",
        "当前默认等于验证账本里的 `new`：回踩量不超过突破前 10 日均量、突破后冲高不超过线价 10%、构线证据在突破前 750 根以内，大盘状态只标注、不降级。旧批次没有这些字段时规则保持关闭。",
        "",
        "| 时段 | 信号 | 有 T+5 成交 | 股票 | 净胜率 | 平均净收益 | 中位净收益 | 平均盈/平均亏 | 利润因子 | T+5 超额 | 超额95%区间 | 跑赢基准 | T+10 超额 | T+20 超额 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in default:
        t5, t10, t20 = row["t5"], row["t10"], row["t20"]
        lines.append(
            f"| {row['label']} | {row['signals']} | {t5['n']} | {row['stocks']} | {fmt(t5['win_rate'], '%')} | "
            f"{fmt(t5['mean_net'], '%')} | {fmt(t5['median_net'], '%')} | "
            f"+{t5['avg_win']}% / {t5['avg_loss']}% | {t5['profit_factor']} | {fmt(t5['mean_excess'], '%')} | "
            f"{ci_text(t5['excess_ci95'])} | {fmt(t5['beat_benchmark_rate'], '%')} | "
            f"{fmt(t10['mean_excess'], '%')} | {fmt(t20['mean_excess'], '%')} |"
        )
    dev_t20, hold_t20 = default[0]["t20"], default[1]["t20"]
    lines += [
        "",
        f"留出段有 {payload['unfilled']['n']} 笔信号三个期限的收益都为空（{payload['unfilled']['stock_code']}，{payload['unfilled']['signal_date']}）。按原脚本，这是次日无法按开盘成交，或其后没有可用 K 线，不进入胜率分母。",
        f"T+20 超额在开发段的区间是 {ci_text(dev_t20['excess_ci95'])}，留出段是 {ci_text(hold_t20['excess_ci95'])}。事先声明的判定看 T+5；不能改用 T+20 把这一组说成有效。",
        "两段中位数都是负的，平均盈亏比大约 1.24～1.26，利润因子只略高于 1。正的平均值来自右侧尾部，不是多数交易赚钱。",
        "",
        "同一批当前默认信号、改用「收盘跌破精准线×(1−1.5%) 才离场」的主口径（持有上限 60 日，开发段截止 2023-09-30，避免持有期跨进留出段）：",
        "",
        "| 时段 | 笔数 | 净胜率 | 平均净收益 | 中位数 | 利润因子 | 跌破线离场 | 中位持有 | 配对差 | 配对差95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["line_stop_primary"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['win_rate']}% | {row['mean']}% | {row['median']}% | "
            f"{row['profit_factor']} | {row['stopped_rate']}% | {row['median_days']} 天 | {row['paired']}% | {ci_text(row['paired_ci95'])} |"
        )
    held = "、".join(f"{row['stopped_rate']}%" for row in payload["line_stop_primary"])
    days = "、".join(f"{row['median_days']:g}" for row in payload["line_stop_primary"])
    lines += [
        "",
        "配对差的点估计为正，两段下限都小于 0。已发表判定是未通过：把止损放在精准线下，并没有被证明好于同样距离的随机买点。"
        f"两段跌破线离场的比例是 {held}，中位持有 {days} 天。",
        "",
        "### 精准线回调：规则变体（同一锁定样本，都未达标）",
        "",
        "这些组是 2026-11-27 事先写好的对照，不是本轮新切出来的。最后一列只表示这一段自己的下限是否大于 0；一组要两段都为「是」才算通过。六组都没有。",
        "",
        "| 组 | 时段 | 信号 | 净胜率 | 平均净收益 | T+5 超额 | 超额95%区间 | 该时段下限>0 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["precision_arms"]:
        t5 = row["t5"]
        lines.append(
            f"| {row['arm']} | {row['label']} | {row['signals']} | {fmt(t5['win_rate'], '%')} | "
            f"{fmt(t5['mean_net'], '%')} | {fmt(t5['mean_excess'], '%')} | {ci_text(t5['excess_ci95'])} | "
            f"{'是' if row['pass_t5'] else '否'} |"
        )
    lines += [
        "",
        "### 旗形突破",
        "",
        "没有可复现的锁定样本。README 记录 2026-09-07 在本地缓存 5012 只、最近 250 根上只有 18 次确认突破，原始统计文件 `reports/flag_backtest_20260907_100351.md` 不在仓库里。本轮不估计旗形的胜率，也不把三角形的回测借给旗形。",
        "",
        "### 三角整理向上突破",
        "",
        "见上面的表。它不是扫描产出的买入预警。主口径配对差在两段都是负的。",
        "",
        "## 赢单和亏单：只用信号日已经知道的信息",
        "",
        "样本是当前默认精准线、并且有 T+5 成交的全部信号。赢单定义为 T+5 净收益大于 0，亏单为小于或等于 0。",
        "特征只取 `signals.csv` 在信号生成时就写下的列，加上由突破日和信号日算出的自然日间隔。自然日间隔不是引擎里的「距突破交易日数」，表里用它自己的名字。",
        "不用信号日之后的价格给特征赋值。T+5 收益只做标签。",
        "先前 `success_features` 用的是「20 日内先碰 +10% 还是先碰 −3%」，而且只保留分出胜负的、大盘不弱的交易。那份样本的入选本身用到了信号日之后的走势，不能当成这次 T+5 的样本。那次结论仍然有效：28 个信号日特征没有一个过稳定标准，较大的两个在随机买点里同样存在。行情缓存不在仓库里，本轮没有重算那 28 列。",
        "",
        f"有 T+5 成交的当前默认信号共 {payload['feature_sample']['n']} 笔，赢单 {payload['feature_sample']['wins']} 笔，亏单 {payload['feature_sample']['losses']} 笔。",
        "",
        "区分力沿用原定义：AUC 是随机抽一笔赢单和一笔亏单时、赢单该字段更大的概率，0.5 表示分不开。",
        f"稳定的要求也沿用原定义：开发段和留出段的 AUC 在 0.5 的同一侧，各自偏离至少 {STABLE_AUC_GAP}，且全样本 |z| ≥ {STABLE_Z}。",
        "",
        "| 特征 | 赢单中位数 | 亏单中位数 | AUC | z | 开发段AUC | 留出段AUC | 稳定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["features"]:
        lines.append(
            f"| {row['label']} | {row['median_win']} | {row['median_loss']} | {row['auc']} | {row['z']} | "
            f"{row['auc_dev']} | {row['auc_holdout']} | {'是' if row['stable'] else ''} |"
        )
    lines += [
        "",
        "四个连续字段的中位数几乎重合，AUC 都在 0.49～0.52。没有一个达到稳定标准。",
        "分档边界只用开发段算出，再原样套到留出段，避免用留出段决定怎么切。",
        "",
        "| 特征 | 开发段分档 | 开发段笔数 | 开发段超额 | 留出段笔数 | 留出段超额 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["quintile_pairs"]:
        lines.append(
            f"| {row['label']} | {row['bin']} | {row['dev_n']} | {fmt(row['dev_excess'], '%')} | "
            f"{row['hold_n']} | {fmt(row['hold_excess'], '%')} |"
        )
    lines += [
        "",
        "没有任何一档在两段都稳定地高于其他档。回踩量、越线幅度、事件数、自然日间隔都不能单独指出赢单。",
        "",
        "分类字段是同一批 T+5 成交，不是又一次搜索出来的阈值。",
        "",
        "| 字段 | 取值 | 时段 | 笔数 | 净胜率 | 平均净收益 | T+5 超额 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["categories"]:
        lines.append(
            f"| {row['field']} | {row['value']} | {row['period']} | {row['n']} | {row['win_rate']}% | "
            f"{row['mean_net']}% | {row['mean_excess']}% |"
        )
    lines += [
        "",
        (
            "大盘不弱的交易，两段超额都更高（开发段 "
            f"{_cat(payload, '信号日大盘偏弱', '否', '开发段')}% 对 {_cat(payload, '信号日大盘偏弱', '是', '开发段')}%，留出段 "
            f"{_cat(payload, '信号日大盘偏弱', '否', '留出段')}% 对 {_cat(payload, '信号日大盘偏弱', '是', '留出段')}%）。"
            "方向一致，但这就是已经单独成组、并且留出段区间下限小于 0 的那条规则，不能因为方向一致就重新打开。"
        ),
        "突破日涨停的信号两段超额都更差，留出段只有 46 笔。它是标注，不是默认拦截。",
        "板块对不上：创业板开发段超额为负、留出段为正；科创板两段都为负，开发段只有 133 笔。先前竞速分析里科创板、创业板成功率更高，换到 T+5 超额上并不成立，本轮不采用板块过滤。",
        "",
        "## 检查过的过滤，以及为什么不留下",
        "",
        "候选只有两条，都是引擎已经算好、默认不拦截的标注。没有扫阈值，也没有改构线、通道或回踩窗口。",
        "它们都是信号发出之后的丢弃，不改变引擎当天选哪一条线。项目对「大盘偏弱不买」用的就是这种过滤，所以在同一张 `signals.csv` 上重算，就是该锁定样本上的对照。",
        "没有重跑引擎：行情缓存不在仓库里，重新向腾讯拉前复权数据不会是同一份锁定快照。",
        "",
        "事先沿用的保留标准：过滤之后，开发段和留出段的 T+5 超额都大于 0，且月度分块 95% 下限都大于 0。做不到这一点，就不为了更高的点估计改代码。",
        "",
        "### 去掉突破日涨停",
        "",
        "| 时段 | 口径 | 信号 | 有 T+5 成交 | 净胜率 | 平均净收益 | T+5 超额 | 超额95%区间 | T+10 超额 | T+20 超额 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["limit_up_compare"]:
        t5, t10, t20 = row["t5"], row["t10"], row["t20"]
        lines.append(
            f"| {row['label']} | {row['variant']} | {row['signals']} | {t5['n']} | {fmt(t5['win_rate'], '%')} | "
            f"{fmt(t5['mean_net'], '%')} | {fmt(t5['mean_excess'], '%')} | {ci_text(t5['excess_ci95'])} | "
            f"{fmt(t10['mean_excess'], '%')} | {fmt(t20['mean_excess'], '%')} |"
        )
    lines += [
        "",
        (
            f"点估计略高（开发段超额 {payload['limit_up_deltas']['dev_before']}% → {payload['limit_up_deltas']['dev_after']}%，"
            f"留出段 {payload['limit_up_deltas']['hold_before']}% → {payload['limit_up_deltas']['hold_after']}%），两段下限仍小于 0。"
            f"开发段去掉 {payload['limit_up_deltas']['dev_dropped']} 笔，留出段去掉 {payload['limit_up_deltas']['hold_dropped']} 笔。"
        ),
        (
            f"换到跌破精准线的 60 日口径，留出段这些涨停突破共 {payload['limit_up_line']['n']} 笔，"
            f"配对差平均 {payload['limit_up_line']['paired']}%。"
            "点估计为正，和「涨停突破在这种离场下一定更差」对不上。这条过滤不留下。"
        ),
        "",
        "### 大盘偏弱不买",
        "",
        "这不是新规则。上表里它的 T+5 超额是开发段 0.37%（0.025%～0.72%）、留出段 0.401%（-0.049%～0.788%）。开发段下限大于 0，留出段没有。默认因此保持「只标注」。本轮重算得到同一组数字，不改回去。",
        "",
        "## 为什么还没有一条稳妥、可归因的修正值得保留",
        "",
        "当前默认相对修复前，留出段的超额点估计从 −0.015% 变成 +0.222%，但区间仍跨过 0。再往下切，信号日字段没有给出一段在两段都站得住的子集。",
        "大盘状态是唯一方向稳定、幅度也看得出来的差异，项目已经按事先标准否决过把它当成拦截。涨停突破的差异样本太小，而且另一种离场口径并不支持。",
        "在这个证据下改阈值、换持有期或新写三角整理，只会得到一个事后更好看的数字。本轮的代码改动停在这份可复现报告。",
        "",
        "## 限制",
        "",
        "- 股票池是评估日仍上市、非 ST 的公司，退市股的失败不在样本里。",
        "- 前复权价格不是当年的成交价。没有模拟跌停卖不出、停牌和随成交额变化的滑点。",
        "- 留出段只是事先划开的时间，更早的研究已经看过同期行情，不是真正的前向样本。",
        "- `locked_20260910_r3` 的收益率不在仓库里，本报告没有那些数字。",
        "- 旗形和联合筛选没有这张锁定成交表，不能从精准线的数字推过去。",
        "- 多个字段同时查看。即便用了原来的 |z| 门槛，这仍是看过账本之后的描述；没有一项被写成新规则。",
        "",
    ]
    return "\n".join(lines)


def pair_quintiles(rows: list[dict], labels: dict[str, str]) -> list[dict]:
    paired = []
    grouped: dict[tuple[str, str], dict] = {}
    for row in rows:
        grouped.setdefault((row["feature"], row["bin"]), {})[row["period"]] = row
    for (feature, bin_label), sides in grouped.items():
        dev, hold = sides.get("development"), sides.get("holdout")
        if dev is None or hold is None:
            continue
        paired.append({
            "label": labels[feature], "bin": bin_label,
            "dev_n": dev["n"], "dev_excess": dev["mean_excess"],
            "hold_n": hold["n"], "hold_excess": hold["mean_excess"],
        })
    return paired


def category_table(frame: pd.DataFrame, dev_end: str, hold_end: str) -> list[dict]:
    usable = frame.dropna(subset=["ret_5"])
    rows = []
    specs = (
        ("突破日涨停", "limit_up_breakout", {True: "是", False: "否"}),
        ("信号日大盘偏弱", "market_weak", {True: "是", False: "否"}),
        ("板块", "board", {"主板": "主板", "创业板": "创业板", "科创板": "科创板"}),
    )
    for period, start, end in (("开发段", "2019-01-01", dev_end), ("留出段", "2024-01-01", hold_end)):
        part = usable[(usable["signal_date"] >= start) & (usable["signal_date"] <= end)]
        for field, column, mapping in specs:
            for raw, label in mapping.items():
                group = part[part[column] == raw]
                net = group["ret_5"]
                rows.append({
                    "field": field, "value": label, "period": period, "n": int(len(group)),
                    "win_rate": round(float((net > 0).mean() * 100), 1) if len(group) else None,
                    "mean_net": round(float(net.mean()), 3) if len(group) else None,
                    "mean_excess": round(float(group["excess_5"].mean()), 3) if len(group) else None,
                })
    return rows


def main() -> None:
    signals, summary, dev_end, hold_end = load_signals()
    published = summary["results"]
    dev_label = next(label for label in published if label.startswith("development"))
    hold_label = next(label for label in published if label.startswith("holdout"))
    periods = (("开发段", "2019-01-01", dev_end, dev_label), ("留出段", "2024-01-01", hold_end, hold_label))

    precision_rows = []
    for key, arm_name in ARMS:
        for label, start, end, published_label in periods:
            part = period_slice(select_arm(signals, key), start, end)
            record = arm_record(part)
            record.update({"arm": arm_name, "key": key, "label": label, "pass_t5": passes(record["t5"])})
            assert_matches_published(record["t5"], published[published_label][key]["t5"], label, key)
            if record["signals"] != published[published_label][key]["signals"]:
                raise SystemExit(f"信号数不一致 {label} {key}")
            precision_rows.append(record)

    current = select_arm(signals, "new")
    current.attrs["dev_end"] = dev_end
    current.attrs["hold_end"] = hold_end
    no_limit = current[current["limit_up_breakout"] == False]  # noqa: E712
    comparisons = []
    for label, start, end in (("开发段", "2019-01-01", dev_end), ("留出段", "2024-01-01", hold_end)):
        for variant, frame in (("当前默认", current), ("去掉突破日涨停", no_limit)):
            record = arm_record(period_slice(frame, start, end))
            record.update({"label": label, "variant": variant})
            comparisons.append(record)

    line = pd.read_csv(LOCKED / "line_stop/trades.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    line["stock_code"] = line["stock_code"].str.zfill(6)
    line_primary = []
    line_base = line[line["group"] == LINE_GROUP]
    for window, start, end, label in (
        ("60", "2019-01-01", "2023-09-30", "开发段 2019-01～2023-09"),
        ("60", "2024-01-01", "2099-12-31", "留出段 2024-01～"),
    ):
        part = line_base[(line_base["window"] == int(window)) & (line_base["signal_date"] >= start) & (line_base["signal_date"] <= end)]
        stats = trade_stats(part)
        assert_trade(stats, LINE_EXPECT[(window, start, end)], f"line {label}")
        stats["label"] = label
        line_primary.append(stats)
    # 20 日开发段只用于确认账本没被换过，不写进主表。
    check = line_base[(line_base["window"] == 20) & (line_base["signal_date"] >= "2019-01-01") & (line_base["signal_date"] <= "2023-09-30")]
    assert_trade(trade_stats(check), LINE_EXPECT[("20", "2019-01-01", "2023-09-30")], "line window 20")

    pattern = pd.read_csv(LOCKED / "patterns/trades.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    triangle_primary = []
    tri = pattern[pattern["type"] == TRIANGLE]
    for window, start, end, label in (
        ("60", "2019-01-01", "2023-09-30", "开发段 2019-01～2023-09"),
        ("60", "2024-01-01", "2099-12-31", "留出段 2024-01～"),
    ):
        part = tri[(tri["window"] == int(window)) & (tri["signal_date"] >= start) & (tri["signal_date"] <= end)]
        stats = trade_stats(part, level=TRIANGLE_LEVEL)
        assert_trade(stats, TRIANGLE_EXPECT[(window, start, end)], f"triangle {label}")
        stats["label"] = label
        triangle_primary.append(stats)

    features = feature_rows(period_slice(current, "2019-01-01", hold_end), dev_end)
    if any(row["stable"] for row in features):
        raise SystemExit("出现了达到稳定标准的字段；先停下来核对，不要让脚本按『没有稳定字段』继续写结论")
    labels = dict(CONTINUOUS)
    quintile_rows = []
    for feature, _ in CONTINUOUS:
        quintile_rows.extend(quintiles(period_slice(current, "2019-01-01", hold_end), feature, dev_end, hold_end))
    pattern_report = (LOCKED / "patterns/report.md").read_text(encoding="utf-8")
    code = (ROOT / "backend/app/services/chart_patterns.py").read_text(encoding="utf-8")
    filled = period_slice(current, "2019-01-01", hold_end).dropna(subset=["ret_5"])
    missing = period_slice(current, "2024-01-01", hold_end)
    missing = missing[missing["ret_5"].isna() & missing["ret_10"].isna() & missing["ret_20"].isna()]
    if len(missing) != 1:
        raise SystemExit(f"留出段无成交信号不是 1 笔：{len(missing)}")
    by_variant = {(row["label"], row["variant"]): row for row in comparisons}
    limit_flags = current.loc[current["limit_up_breakout"] == True, ["stock_code", "signal_date"]]  # noqa: E712
    limit_line = line_base[line_base["window"] == 60].merge(limit_flags, on=["stock_code", "signal_date"], how="inner")
    limit_line = limit_line[(limit_line["signal_date"] >= "2024-01-01") & (limit_line["signal_date"] <= hold_end)]
    payload = {
        "meta": {
            "as_of": summary["meta"]["as_of"],
            "sampled": summary["meta"]["sampled"],
            "succeeded": summary["meta"]["succeeded"],
            "dev_end": dev_end,
            "hold_end": hold_end,
            "purge_sessions": 25,
            "pattern_report_uses_short_prose": "最近四个峰谷点" in pattern_report,
            "code_has_shared_span_rule": "TRIANGLE_MIN_SHARED_BARS" in code,
            "logic_change_retained": False,
        },
        "precision_default": [row for row in precision_rows if row["key"] == "new"],
        "precision_arms": precision_rows,
        "line_stop_primary": line_primary,
        "triangle_primary": triangle_primary,
        "features": features,
        "feature_sample": {"n": int(len(filled)), "wins": int((filled["ret_5"] > 0).sum()), "losses": int((filled["ret_5"] <= 0).sum())},
        "quintile_pairs": pair_quintiles(quintile_rows, labels),
        "categories": category_table(period_slice(current, "2019-01-01", hold_end), dev_end, hold_end),
        "limit_up_compare": comparisons,
        "limit_up_deltas": {
            "dev_before": by_variant[("开发段", "当前默认")]["t5"]["mean_excess"],
            "dev_after": by_variant[("开发段", "去掉突破日涨停")]["t5"]["mean_excess"],
            "hold_before": by_variant[("留出段", "当前默认")]["t5"]["mean_excess"],
            "hold_after": by_variant[("留出段", "去掉突破日涨停")]["t5"]["mean_excess"],
            "dev_dropped": by_variant[("开发段", "当前默认")]["signals"] - by_variant[("开发段", "去掉突破日涨停")]["signals"],
            "hold_dropped": by_variant[("留出段", "当前默认")]["signals"] - by_variant[("留出段", "去掉突破日涨停")]["signals"],
        },
        "limit_up_line": {"n": int(len(limit_line)), "paired": round(float(limit_line["paired_diff"].mean()), 3)},
        "unfilled": {
            "n": int(len(missing)),
            "stock_code": str(missing.iloc[0]["stock_code"]),
            "signal_date": str(missing.iloc[0]["signal_date"]),
        },
        "published_verdicts": summary["verdicts"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(features).to_csv(OUT / "features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quintile_rows).to_csv(OUT / "quintiles.csv", index=False, encoding="utf-8-sig")
    report = markdown(payload)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
