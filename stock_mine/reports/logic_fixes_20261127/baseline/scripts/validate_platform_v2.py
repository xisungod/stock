"""Freeze inputs first, then evaluate without tuning or touching live scan records."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pandas as pd

from app.services.platform_events import PlatformEngine, iter_platform_results
from app.services.strategy import (
    StrategyConfig, SignalCategory, _analyze_normalized_bars, _normalize_bars,
    _swings, analyze_bars, version_for_engine,
)
from app.services.validation import (
    block_mean_difference, portfolio_curve, trade_on_calendar, trade_statistics,
)
from app.services.market_calendar import calendar_info, session_dates


ENGINES = ("legacy", "pivot_fixed", "platform_events")
PARTITIONS = (
    ("development", "2024-01-01", "2024-12-31"),
    ("holdout_2025", "2025-01-01", "2025-12-31"),
    ("holdout_2026", "2026-01-01", "2026-09-09"),
)
SOURCES = (
    "backend/app/services/strategy.py", "backend/app/services/platform_events.py",
    "backend/app/services/price_bands.py", "backend/app/services/zigzag.py",
    "backend/app/services/backtest.py", "backend/app/services/validation.py",
    "backend/app/services/market_calendar.py", "backend/app/services/bars_cache.py",
    "backend/app/services/data_provider.py", "backend/app/services/scanner.py",
    "backend/app/schemas.py", "backend/requirements.txt",
    "scripts/validate_platform_v2.py",
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def segment(code: str) -> str:
    if code.startswith(("30",)):
        return "growth"
    if code.startswith("68"):
        return "star"
    return "sh_main" if code.startswith("6") else "sz_main"


def select_sample(rows: list[dict], cache: Path, per_group: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        code = str(row["stock_code"]).zfill(6)
        if code.startswith(("0", "3", "6")) and (cache / f"{code}.json").is_file():
            groups[segment(code)].append({**row, "stock_code": code})
    selected = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda row: digest(f"platform-v2-20260910:{row['stock_code']}".encode()))
        selected.extend(ordered[:per_group])
    return sorted(selected, key=lambda row: row["stock_code"])


def normalized_records(payload: dict) -> list[dict]:
    frame = _normalize_bars(pd.DataFrame(payload["bars"]))
    frame["trade_date"] = frame.trade_date.dt.strftime("%Y-%m-%d")
    return frame.to_dict("records")


def freeze_revision(folder: Path, source_protocol: Path) -> None:
    old = json.loads(source_protocol.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    inputs, previous_signals = folder / "inputs", folder / "previous_signals"
    inputs.mkdir(exist_ok=True)
    previous_signals.mkdir(exist_ok=True)
    baseline_hashes = {}
    for member in old["members"]:
        code = member["code"]
        source = source_protocol.parent / "inputs" / f"{code}.json"
        if digest(source.read_bytes()) != member["snapshot_sha256"]:
            raise ValueError(f"Original frozen input changed: {code}")
        shutil.copy2(source, inputs / source.name)
        if member["included_in_evaluation"]:
            replay = source_protocol.parent / "replays" / f"{code}.json"
            shutil.copy2(replay, previous_signals / replay.name)
            baseline_hashes[code] = digest(replay.read_bytes())
    dates = session_dates(min(member["first_date"] for member in old["members"]),
                          old["partitions"][-1]["end"])
    protocol = {
        **old, "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in SOURCES},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "calendar": dates, "calendar_source": calendar_info()["source"], "calendar_details": calendar_info(),
        "engineering_revision": "v4: complete trial ownership; bounded candidate inclusion; shared exchange-calendar execution; quote readiness and strict providers",
        "baseline_protocol": str(source_protocol), "baseline_protocol_sha256": digest(source_protocol.read_bytes()),
        "previous_signal_sha256": baseline_hashes,
        "calendar_change": {
            "added": sorted(set(dates) - set(old["calendar"])),
            "removed": sorted(set(old["calendar"]) - set(dates)),
        },
    }
    StrategyConfig(**protocol["config"])
    write_json(folder / "protocol.json", protocol)
    print(f"FROZEN {folder / 'protocol.json'} unchanged_inputs={len(old['members'])} baseline_stocks={len(baseline_hashes)}", flush=True)


def freeze(folder: Path, per_group: int, source_protocol: Path | None = None) -> None:
    if (folder / "protocol.json").exists():
        raise ValueError("Protocol exists; choose a new output directory, never overwrite a frozen run.")
    if source_protocol is not None:
        freeze_revision(folder, source_protocol)
        return
    folder.mkdir(parents=True, exist_ok=True)
    snapshots = folder / "inputs"
    snapshots.mkdir(exist_ok=True)
    cache = ROOT / "data/bars_cache"
    stock_list = json.loads((ROOT / "data/stock_list_cache.json").read_text())
    selected = select_sample(stock_list, cache, per_group)
    sources = {name: digest((ROOT / name).read_bytes()) for name in SOURCES}
    names = {str(row["stock_code"]).zfill(6): row["short_name"] for row in stock_list}
    members = []
    calendar_dates = set()
    # The stock-based calendar is an explicit proxy; it does not impute bars for suspended stocks.
    all_codes = sorted({row["stock_code"] for row in selected} | {"000001", "600000", "002120"})
    for code in all_codes:
        original = (cache / f"{code}.json").read_bytes()
        payload = json.loads(original)
        records = normalized_records(payload)
        records = [row for row in records if row["trade_date"] <= PARTITIONS[-1][2]]
        snapshot = snapshots / f"{code}.json"
        write_json(snapshot, {"code": code, "name": names.get(code, code), "bars": records})
        calendar_dates.update(row["trade_date"] for row in records)
        members.append({
            "code": code, "name": names.get(code, code), "group": segment(code),
            "included_in_evaluation": code in {row["stock_code"] for row in selected},
            "original_sha256": digest(original), "snapshot_sha256": digest(snapshot.read_bytes()),
            "bar_count": len(records), "raw_bar_count": len(payload["bars"]),
            "first_date": records[0]["trade_date"] if records else None,
            "last_date": records[-1]["trade_date"] if records else None,
            "checked_at": payload.get("checked_at"),
        })
    cfg = StrategyConfig(signal_offset=0, include_chart=False, precision_engine="platform_events")
    protocol = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"), "source_sha256": sources,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "config": asdict(cfg), "engines": list(ENGINES),
        "partitions": [{"name": name, "start": start, "end": end} for name, start, end in PARTITIONS],
        "purge_sessions_at_partition_end": 20, "horizons": [5, 10, 20],
        "roundtrip_cost_bps": 20, "cost_sensitivity_bps": [40, 60],
        "max_positions": 10, "members": members,
        "calendar": session_dates(min(calendar_dates), PARTITIONS[-1][2]),
        "sampling": f"SHA256 ranking within SZ main/SH main/growth/STAR, {per_group} stocks each; no performance selection",
        "deduplication": {"legacy": "first alert per stock then 20 stock bars",
                          "pivot_fixed": "first alert per stock then 20 stock bars",
                          "platform_events": "first alert per frozen breakout event"},
        "control": "One unique non-alert stock per V2 alert on the same date, deterministic hash; no replacement if unavailable",
        "portfolio": "10 positions, same-stock exclusion, cash constrained, fractional shares, next open entry, close exit, no leverage",
        "calendar_source": calendar_info()["source"], "calendar_details": calendar_info(),
        "qualification": "Historical holdouts on a previously examined survivor cache, not genuinely untouched or forward data",
        "engineering_revision": "v4: complete trial ownership, strict quote readiness/providers, and shared calendar execution",
        "release_gate": "No claim of improvement unless both holdouts have adequate samples and positive net deltas with uncertainty assessed; no tuning from this run",
    }
    write_json(folder / "protocol.json", protocol)
    print(f"FROZEN {folder / 'protocol.json'} stocks={len(selected)}", flush=True)


def load_frame(folder: Path, code: str) -> pd.DataFrame:
    payload = json.loads((folder / "inputs" / f"{code}.json").read_text())
    return _normalize_bars(pd.DataFrame(payload["bars"]))


def summarize_result(result, code: str, index: int, stamp: str) -> dict:
    return {
        "stock_code": code, "signal_date": stamp, "signal_index": index,
        "structure_score": float(result.structure_score if result.structure_score is not None else result.score),
        "signal_event_id": result.signal_event_id, "line_price": result.key_line_price,
        "line_snapshot": result.line_snapshot, "resistance_anchor": result.resistance_anchor,
        "prebreakout_check": result.prebreakout_check, "pullback_check": result.pullback_check,
    }


def replay_stock(task: tuple[str, dict, dict]) -> dict:
    directory, member, protocol = task
    folder, code = Path(directory), member["code"]
    df = load_frame(folder, code)
    dates = df.trade_date.dt.strftime("%Y-%m-%d").tolist()
    start_index = max(119, int(np.searchsorted(dates, protocol["partitions"][0]["start"])))
    cfg = StrategyConfig(**protocol["config"])
    output = {"code": code, "name": member["name"], "engines": {}, "bar_count": len(df), "error": None}
    for engine in ENGINES:
        selected = replace(cfg, precision_engine=engine)
        if engine == "platform_events":
            iterator = iter_platform_results(code, member["name"], df, selected, start_index)
        else:
            iterator = ((index, _analyze_normalized_bars(code, member["name"], df.iloc[:index + 1], selected))
                        for index in range(start_index, len(df)))
        rows, raw_dates, states, seen = [], [], Counter(), set()
        last_index = -10**9
        for index, result in iterator:
            state = result.platform_state or result.category.value if result else "no_line"
            states[state] += 1
            if result is None or result.category != SignalCategory.BUY_ALERT:
                continue
            raw_dates.append(dates[index])
            key = result.signal_event_id or f"{code}:{dates[index]}"
            if (engine == "platform_events" and key in seen) or (engine != "platform_events" and index - last_index < 20):
                continue
            seen.add(key)
            last_index = index
            rows.append(summarize_result(result, code, index, dates[index]))
        output["engines"][engine] = {"signals": rows, "raw_alert_dates": raw_dates, "state_counts": dict(states)}
    if protocol.get("previous_signal_sha256"):
        previous = json.loads((folder / "previous_signals" / f"{code}.json").read_text())
        output["engines"]["previous_platform"] = previous["engines"]["platform_events"]
    return output


def partition_rows(rows: list[dict], part: dict, calendar: list[str], purge: int = 20) -> list[dict]:
    days = [day for day in calendar if part["start"] <= day <= part["end"]]
    if len(days) <= purge:
        return []
    cutoff = days[-purge - 1]
    return [row for row in rows if part["start"] <= row["signal_date"] <= cutoff]


def match_controls(new_rows: list[dict], results: list[dict], frames: dict) -> tuple[list[dict], list[dict], int]:
    alerts = defaultdict(set)
    for result in results:
        for stamp in result["engines"]["platform_events"]["raw_alert_dates"]:
            alerts[stamp].add(result["code"])
    used = set()
    matched, controls, missing = [], [], 0
    eligible_dates = {
        code: set(list(prices)[119:]) for code, prices in frames.items()
    }
    for row in sorted(new_rows, key=lambda item: (item["signal_date"], item["stock_code"])):
        stamp, code = row["signal_date"], row["stock_code"]
        options = [other for other, prices in frames.items()
                   if other != code and other not in alerts[stamp] and stamp in eligible_dates[other]
                   and prices[stamp]["volume"] > 0 and (stamp, other) not in used]
        if not options:
            missing += 1
            continue
        other = min(options, key=lambda candidate: digest(f"control-20260910:{code}:{stamp}:{candidate}".encode()))
        used.add((stamp, other))
        pair = f"{stamp}:{code}:{other}"
        matched.append({**row, "control_pair": pair})
        controls.append({"stock_code": other, "signal_date": stamp, "structure_score": 0,
                         "signal_event_id": f"control:{pair}", "control_pair": pair})
    return matched, controls, missing


def evaluate(protocol: dict, results: list[dict], frames: dict) -> dict:
    calendar = protocol["calendar"]
    arms = {engine: [row for result in results for row in result["engines"][engine]["signals"]]
            for engine in ENGINES}
    if protocol.get("previous_signal_sha256"):
        arms["previous_platform"] = [
            row for result in results for row in result["engines"]["previous_platform"]["signals"]
        ]
    matched, control, missing = match_controls(arms["platform_events"], results, frames)
    arms.update(platform_matched=matched, non_alert_control=control)
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_at": protocol["frozen_at"], "successful_stocks": len(results),
        "matched_control_pairs": len(matched), "unmatched_new_signals": missing,
        "partitions": {}, "decision": "not_established",
    }
    if "previous_platform" in arms:
        new_keys = {(row["stock_code"], row["signal_date"]) for row in arms["platform_events"]}
        old_keys = {(row["stock_code"], row["signal_date"]) for row in arms["previous_platform"]}
        out["signal_changes"] = {
            "same_stock_date": len(new_keys & old_keys), "new_only": len(new_keys - old_keys),
            "previous_only": len(old_keys - new_keys), "new_count": len(new_keys), "previous_count": len(old_keys),
        }
    for part in protocol["partitions"]:
        part_calendar = [day for day in calendar if day <= part["end"]]
        section = {"start": part["start"], "end": part["end"], "horizons": {}}
        for horizon in protocol["horizons"]:
            reports, records = {}, {}
            for arm, rows in arms.items():
                selected = partition_rows(rows, part, calendar, protocol["purge_sessions_at_partition_end"])
                trades = [trade_on_calendar(row["stock_code"], row, frames[row["stock_code"]], part_calendar,
                                            horizon, protocol["roundtrip_cost_bps"]) for row in selected]
                records[arm] = trades
                performance = portfolio_curve(trades, frames, part_calendar, part["start"], part["end"],
                                              protocol["max_positions"], protocol["roundtrip_cost_bps"])
                reports[arm] = {"trades": trade_statistics(trades), "portfolio": performance}
                if arm == "platform_events":
                    reports[arm]["cost_sensitivity"] = {
                        str(cost): trade_statistics([
                            {**trade, "net_return_pct": trade["net_return_pct"] - (cost - protocol["roundtrip_cost_bps"]) / 100
                             if trade["net_return_pct"] is not None else None} for trade in trades
                        ]) for cost in protocol["cost_sensitivity_bps"]
                    }
            date_range = [day for day in calendar if part["start"] <= day <= part["end"]]
            comparisons = {
                "new_minus_old": block_mean_difference(records["platform_events"], records["legacy"], date_range),
                "new_minus_fixed": block_mean_difference(records["platform_events"], records["pivot_fixed"], date_range),
                "matched_new_minus_non_alert": block_mean_difference(records["platform_matched"], records["non_alert_control"], date_range),
            }
            if "previous_platform" in records:
                comparisons["new_minus_previous_platform"] = block_mean_difference(
                    records["platform_events"], records["previous_platform"], date_range,
                )
            section["horizons"][str(horizon)] = {"arms": reports, "mean_difference_95pct": comparisons}
        out["partitions"][part["name"]] = section
    return out


def point_audit(folder: Path, protocol: dict) -> dict:
    frame = load_frame(folder, "002120")
    frame = frame[frame.trade_date <= pd.Timestamp("2016-10-10")].reset_index(drop=True)
    cfg = StrategyConfig(**protocol["config"])
    engine = PlatformEngine("002120", "韵达股份", frame, cfg)
    checkpoints = {}
    for index in range(len(frame)):
        engine.feed(index)
        if engine.dates[index] in {"2016-03-23", "2016-03-24", "2016-03-29", "2016-10-10"}:
            entries = {}
            for mode in ENGINES:
                result = analyze_bars("002120", "韵达股份", frame.iloc[:index + 1],
                                      replace(cfg, precision_engine=mode, include_chart=True))
                entries[mode] = result.to_dict() if result else None
            checkpoints[engine.dates[index]] = entries
    highs, lows, pending, meta = _swings(frame, replace(cfg, precision_engine="legacy"), with_confirmation=True)
    peak_confirms = dict(zip(highs, meta["peak_confirmed_at"]))
    low_confirms = dict(zip(lows, meta["valley_confirmed_at"]))
    targets = {}
    for stamp in ["2007-06-20", "2015-03-27", "2015-10-30", "2016-02-22"]:
        index = engine.dates.index(stamp)
        events = [engine.event_dict(event) for event in engine.events.values() if event.pivot == index]
        memberships = []
        for level in engine.platforms:
            for event in events:
                if any(engine.event_dict(item)["event_id"] == event["event_id"] for item in level.events.values()):
                    memberships.append({
                        "line_id": level.id, "line_price": level.band.price,
                        "current_count": len(level.events), "snapshot": level.snapshot,
                    })
        surrounding = [engine.event_dict(event) for event in engine.events.values()
                       if abs(event.pivot - index) <= 8]
        encompassing = [event for event in surrounding if event["date"] < stamp <= event["confirmed_date"]]
        explanation = (
            "已确认独立事件；具体所属线、容差与角色见明细" if events and memberships else
            "局部事件已记录，但未通过同价位带的独立性验证" if events else
            f"属于同次试探；代表极值发生于{encompassing[-1]['date']}，价格{encompassing[-1]['price']}"
            if encompassing else "未满足独立局部转折或价位带验证；附近事件见明细"
        )
        targets[stamp] = {
            "ohlcv": {key: float(frame.iloc[index][key]) for key in ["open", "high", "low", "close", "volume"]},
            "legacy_peak": index in highs, "legacy_valley": index in lows,
            "legacy_confirmation": engine.dates[peak_confirms.get(index, low_confirms.get(index))]
            if index in peak_confirms or index in low_confirms else None,
            "new_events": events, "memberships": memberships, "nearby_events": surrounding,
            "explanation": explanation,
            "reason": ("confirmed_local_or_band_event" if events and memberships else
                       "event_detected_but_no_independent_band_membership" if events else
                       "not_a_separately_confirmed_local_or_band_event"),
        }
    result = {"code": "002120", "as_of": "2016-10-10", "targets": targets, "checkpoints": checkpoints}
    write_json(folder / "002120_point_audit.json", result)
    return result


def fmt(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(report: dict, protocol: dict, audit: dict) -> str:
    labels = {"legacy": "旧版", "pivot_fixed": "修容差峰谷版", "platform_events": "两层事件版",
              "platform_matched": "事件版已配对", "non_alert_control": "同日未预警对照",
              "previous_platform": "上轮事件版v3（同日历计价）"}
    lines = [
        "# 两层平台事件：规则锁定与历史留出验证", "",
        f"规则冻结：{protocol['frozen_at']}；报告生成：{report['generated_at']}。",
        f"实际完成 {report['successful_stocks']} 只股票，失败 {len(report['failures'])} 只。不是全A股回测。", "",
        f"本轮工程修订：{protocol['engineering_revision']}。", "",
        "## 结论边界", "",
        "**本轮只提供工程验收和锁定规则后的历史对照，不认定新规则已提高收益。**",
        "2025和2026为预先指定的时间留出段，但行情缓存及个别案例已被研究过，不能称为从未见过的真正样本外数据。"
        "后续需要保持同一规则哈希开展前向验证；本脚本不会按本轮收益自动调参。", "",
        "## 固定方法", "",
        f"- 按股票代码哈希分组抽样：{protocol['sampling']}。未按收益、是否入选或002120形态挑样本。",
        "- 三引擎读取同一冻结行情快照；每个信号只使用截至信号日的数据。旗形本轮关闭，单独验证精准线逻辑。",
        "- 阻力最高价、支撑最低价；局部反应阈值为极值日ATR20的1倍与价位1%中较大值。贡献价必须落入最终线的±2%价位带。",
        "- 高低价记录候选与离带幅度，后续收盘确认反应；新事件需在前次事件之后、再次触线之前明确离开对应价位带，离带幅度不低于反应阈值，不以固定K线根数去重。",
        "- 已有价位带的整轮试探负责构线计数：局部转折保留，但不能抢在整轮确认前加点；已拒绝试探中的迟到确认也不能绕过。",
        "- 2个独立事件为候选，3个为已验证。收盘上穿先观察，超过线0.6%确认突破，随后1至10个交易日缩量回踩、趋势通过才预警。",
        "- 突破前路径继续限定最高不超过线1.50倍、不能连续30个交易日收盘高于线2%；突破量比门槛保持关闭。回踩缩量条件仍保留。",
        "- 突破使用此前已可用的冻结线版本，本次回踩不补入突破前证据；旧版按20根同股K线去重，新版按冻结突破事件去重。",
        "- 每段末尾去掉20个共同交易日的信号以隔离收益标签；退出数据只使用本分段终点以前，卖出受阻未完成交易单列。",
        "- 次一共同交易日开盘近似入场，T+5/10/20收盘近似退出，双边费滑合计假设20bps，另给40/60bps压力检查。",
        f"- 页面回测与本报告共用成交函数；日历为 {protocol['calendar_source']}，不按下一根可见股票K线顺延入场。",
        "- 缺失、一字、零量日不买入，卖出受阻顺延；无法重建开盘队列、历史涨跌停和ST状态，表中的成交比例仅是日线可成交代理，不是真实成交率。",
        "- 组合按日盯市，最多10只、同股不叠加、现金约束、不开杠杆；当日收盘卖出的现金不能用于当日开盘买入。",
        "- 组合采用碎股份额，未模拟100股整数手、最小佣金、市场冲击、股息现金和现金利息，资金容量结论不成立。",
        "- 对照股票与事件版同日、未触发预警，按固定哈希选取、不重复用同日同股；未匹配历史行业、市值、估值等特征，不能归因于形态本身。",
        "- 同步以20个共同交易日为块重采样1000次，给均值差95%区间；样本小、信号相关、多指标比较都限制推断。",
        "- 当前上市且有缓存股票池存在幸存者偏差；当前前复权历史不等于当年实盘价格，缺少历史复权因子及行业归属，因此不回填当前行业或股东评分。", "",
        "## 分段收益与风险", "",
    ]
    if protocol.get("previous_signal_sha256"):
        changes = report["signal_changes"]
        lines[lines.index("## 分段收益与风险"):lines.index("## 分段收益与风险")] = [
            "## 与修复前对照", "",
            "- 股票名单、行情文件、筛选参数和时间分段沿用上一轮冻结清单，未重新挑样本。",
            "- v3预警来自上一轮已冻结逐股文件；所有收益用本轮同一交易日历和成本重新计算，不直接混用旧报告净值。",
            f"- 未清除边界前：v3有{changes['previous_count']}个、新版有{changes['new_count']}个；"
            f"相同股票和日期{changes['same_stock_date']}个、仅旧版{changes['previous_only']}个、仅新版{changes['new_only']}个。",
            "- 信号日期变化可能来自确认延后、事件合并或线价变化；不能把所有消失日期都称为错误预警。", "",
        ]
    for name, part in report["partitions"].items():
        lines += [f"### {name}：{part['start']} 至 {part['end']}", "",
                  "| 引擎 | 持有期 | 信号/闭仓 | 日线入场代理% | 单笔均值净% | 盈亏比 | 利润因子 | 组合净% | 最大回撤% | 净胜率% |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for horizon, section in part["horizons"].items():
            for arm, item in section["arms"].items():
                if arm == "platform_matched":
                    continue
                stats, nav = item["trades"], item["portfolio"]
                lines.append(f"| {labels[arm]} | T+{horizon} | {stats['signal_count']}/{stats['closed_trade_count']} | "
                             f"{fmt(stats['entry_feasibility_pct'])} | {fmt(stats['mean_net_return_pct'])} | "
                             f"{fmt(stats['payoff_ratio'])} | {fmt(stats['profit_factor'])} | {fmt(nav['net_return_pct'])} | "
                             f"{fmt(nav['max_drawdown_pct'])} | {fmt(stats['net_win_rate_pct'])} |")
        lines += ["", "T+5平均净收益差95%区间（百分点）："]
        for comparison, values in part["horizons"]["5"]["mean_difference_95pct"].items():
            lines.append(f"- {comparison}：{[round(v, 3) for v in values] if values else '样本不足'}。")
        new = part["horizons"]["5"]["arms"]["platform_events"]
        lines.append("- T+5事件版费用压力："
                     + "；".join(f"{cost}bps时均值{fmt(values['mean_net_return_pct'])}%" for cost, values in new["cost_sensitivity"].items()) + "。")
        lines.append("")
    lines += ["## 002120指定点验收", "", "以下属于规则解释案例，不进入抽样的统计推断，除非被预先哈希选中。", "",
              "| 发生日 | 旧版大波峰/波谷 | 新版确认事件 | 新版确认日 | 审计结论 |",
              "|---|---|---|---|---|"]
    for stamp, row in audit["targets"].items():
        old = "波峰" if row["legacy_peak"] else "波谷" if row["legacy_valley"] else "无"
        events = row["new_events"]
        lines.append(f"| {stamp} | {old} | {', '.join(event['kind'] + ' ' + str(event['price']) for event in events) or '无'} | "
                     f"{', '.join(event['confirmed_date'] for event in events) or '-'} | {row['explanation']} |")
    lines += ["", "各历史检查日的当时判定：", "",
              "| 日期 | 引擎 | 状态 | 当时主线 | 说明 |", "|---|---|---|---:|---|"]
    for stamp, modes in audit["checkpoints"].items():
        for mode, row in modes.items():
            lines.append(f"| {stamp} | {labels[mode]} | {row['category'] if row else 'none'} | "
                         f"{fmt(row['key_line_price'], 6) if row else '-'} | {row['suggestion'] if row else '无合格线'} |")
    lines += ["", "## 文件与复现", "",
              "- `protocol.json`：先于计算收益生成的规则、样本、行情和源码哈希。",
              "- `inputs/`：输入快照；`replays/`：逐股三引擎预警及被冻结的事件证据。",
              "- `summary.json`：所有样本数、未完成交易、费用压力、资金约束与逐日净值。",
              "- `002120_point_audit.json`：指定点及附近事件、真实极值、确认日期、线版本和历史检查日详情。",
              "- 结果不同只能说明定义或行为不同，不能因指定股票现在入选就宣称策略更好。", ""]
    return "\n".join(lines)


def run(folder: Path, workers: int) -> None:
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (folder / "summary.json").exists():
        raise ValueError("Completed report exists; keep it immutable and create a new run.")
    for name, expected in protocol["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"Rules changed since freeze: {name}")
    for member in protocol["members"]:
        if digest((folder / "inputs" / f"{member['code']}.json").read_bytes()) != member["snapshot_sha256"]:
            raise ValueError(f"Input changed since freeze: {member['code']}")
    for code, expected in protocol.get("previous_signal_sha256", {}).items():
        if digest((folder / "previous_signals" / f"{code}.json").read_bytes()) != expected:
            raise ValueError(f"Baseline signal file changed since freeze: {code}")
    members = [member for member in protocol["members"] if member["included_in_evaluation"]]
    replays = folder / "replays"
    replays.mkdir(exist_ok=True)
    results, failures = [], []
    tasks = []
    for member in members:
        checkpoint = replays / f"{member['code']}.json"
        if checkpoint.exists():
            results.append(json.loads(checkpoint.read_text()))
        else:
            tasks.append((str(folder), member, protocol))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=max(1, min(4, workers))) as pool:
        futures = {pool.submit(replay_stock, task): task[1]["code"] for task in tasks}
        for future in as_completed(futures):
            code = futures[future]
            try:
                result = future.result()
                write_json(replays / f"{code}.json", result)
                results.append(result)
                counts = {mode: len(result["engines"][mode]["signals"]) for mode in ENGINES}
                print(f"{len(results) + len(failures)}/{len(members)} {code} {counts} elapsed={time.monotonic()-started:.0f}s", flush=True)
            except Exception as exc:
                failures.append({"code": code, "error": str(exc)})
                print(f"FAILED {code}: {exc}", flush=True)
    frames = {}
    for result in results:
        frame = load_frame(folder, result["code"])
        frame["trade_date"] = frame.trade_date.dt.strftime("%Y-%m-%d")
        frames[result["code"]] = {row["trade_date"]: row for row in frame.to_dict("records")}
    report = evaluate(protocol, results, frames)
    report["failures"] = failures
    report["protocol_sha256"] = digest(protocol_path.read_bytes())
    audit = point_audit(folder, protocol)
    write_json(folder / "summary.json", report)
    (folder / "report.md").write_text(markdown(report, protocol, audit), encoding="utf-8")
    print(f"REPORT {folder / 'report.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/platform_v2_validation/locked_20260910_r3")
    parser.add_argument("--from-protocol", type=Path, help="Reuse identical frozen input files and previous event signals")
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if options.action == "freeze":
        if options.per_group < 1:
            parser.error("--per-group must be positive")
        freeze(options.output.resolve(), options.per_group,
               options.from_protocol.resolve() if options.from_protocol else None)
    else:
        run(options.output.resolve(), options.workers)
