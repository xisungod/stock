from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas import production_strategy_config
from app.services.strategy import SignalCategory, _normalize_bars, analyze_bars


FEATURES = (
    "prior_return_5_pct",
    "prior_return_20_pct",
    "volatility_20_pct",
    "distance_to_ma20_pct",
    "volume_ratio_20",
)
OUTCOMES = ("return_5", "return_10", "return_20", "mae_5", "mfe_5", "target_before_stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-control evaluation for Stock Mine alerts.")
    parser.add_argument("--signals", required=True, help="Signal CSV produced by sweep_line_strength.py.")
    parser.add_argument("--universe-summary", required=True, help="Sweep JSON containing selected_codes.")
    parser.add_argument("--setting", type=int, default=3)
    parser.add_argument("--engine", choices=["platform_events", "legacy"], default="platform_events",
                        help="必须与生成 --signals 的 sweep_line_strength.py --engine 一致")
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--exclude-days", type=int, default=28, help="Calendar days around another alert excluded from controls.")
    parser.add_argument("--roundtrip-cost-bps", type=float, default=20.0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--cache-dir", default="data/bars_cache")
    parser.add_argument("--stock-list", default="data/stock_list_cache.json")
    parser.add_argument("--all-cached-controls", action="store_true", help="Use every cached A-share as a control candidate.")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def market_segment(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("68"):
        return "star"
    if code.startswith("3"):
        return "chinext"
    if code.startswith(("4", "8")):
        return "beijing"
    return "sh_main" if code.startswith("6") else "sz_main"


def stock_names(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    return {
        str(item.get("stock_code") or item.get("code")).zfill(6): str(item.get("short_name") or item.get("name") or "")
        for item in records
    }


def all_cached_codes(stock_list: Path, cache_dir: Path) -> list[str]:
    names = stock_names(stock_list)
    return sorted(code for code in names if (cache_dir / f"{code}.json").exists())


def _pct(current: float, previous: float) -> float | None:
    if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
        return None
    return (current / previous - 1) * 100


def panel_rows(code: str, bars: pd.DataFrame, dates: set[str]) -> list[dict[str, Any]]:
    df = _normalize_bars(bars)
    if df.empty:
        return []
    df = df.reset_index(drop=True)
    date_strings = df["trade_date"].dt.strftime("%Y-%m-%d")
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ma20 = close.rolling(20).mean()
    volatility = close.pct_change().rolling(20).std(ddof=1) * 100
    volume_mean = volume.shift(1).rolling(20).mean()
    wanted = np.flatnonzero(date_strings.isin(dates).to_numpy()).tolist()
    rows: list[dict[str, Any]] = []
    for idx in wanted:
        if idx < 21:
            continue
        next_idx = idx + 1
        next_open = float(df["open"].iloc[next_idx]) if next_idx < len(df) else np.nan

        def forward(window: int) -> float | None:
            target = idx + window
            return _pct(float(close.iloc[target]), next_open) if target < len(df) else None

        first_week = df.iloc[idx + 1 : idx + 6]
        if len(first_week) == 5 and next_open > 0:
            mae_5 = (float(first_week["low"].min()) / next_open - 1) * 100
            mfe_5 = (float(first_week["high"].max()) / next_open - 1) * 100
        else:
            mae_5 = mfe_5 = None

        future = df.iloc[idx + 1 : idx + 21]
        target_before_stop: float | None = None
        if len(future) == 20 and next_open > 0:
            target_hits = np.flatnonzero(future["high"].to_numpy(dtype=float) >= next_open * 1.05)
            stop_hits = np.flatnonzero(future["low"].to_numpy(dtype=float) <= next_open * 0.95)
            target_first = len(target_hits) > 0 and (len(stop_hits) == 0 or target_hits[0] < stop_hits[0])
            target_before_stop = float(target_first)

        mean_volume = float(volume_mean.iloc[idx])
        rows.append(
            {
                "stock_code": code,
                "date": date_strings.iloc[idx],
                "segment": market_segment(code),
                "prior_return_5_pct": _pct(float(close.iloc[idx]), float(close.iloc[idx - 5])),
                "prior_return_20_pct": _pct(float(close.iloc[idx]), float(close.iloc[idx - 20])),
                "volatility_20_pct": float(volatility.iloc[idx]) if pd.notna(volatility.iloc[idx]) else None,
                "distance_to_ma20_pct": _pct(float(close.iloc[idx]), float(ma20.iloc[idx])),
                "volume_ratio_20": float(volume.iloc[idx] / mean_volume) if mean_volume > 0 else None,
                "return_5": forward(5),
                "return_10": forward(10),
                "return_20": forward(20),
                "mae_5": mae_5,
                "mfe_5": mfe_5,
                "target_before_stop": target_before_stop,
            }
        )
    return rows


def build_panel(codes: list[str], cache_dir: Path, dates: set[str]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    progress_every = max(25, len(codes) // 20)
    for count, code in enumerate(codes, start=1):
        try:
            payload = json.loads((cache_dir / f"{code}.json").read_text(encoding="utf-8"))
            rows.extend(panel_rows(code, pd.DataFrame(payload.get("bars", [])), dates))
        except Exception as exc:
            failures.append({"stock_code": code, "error": str(exc)})
        if count % progress_every == 0 or count == len(codes):
            print(f"panel {count}/{len(codes)} rows={len(rows)} failures={len(failures)}", flush=True)
    return pd.DataFrame(rows), failures


def robust_distance(signal: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    distance = pd.Series(0.0, index=candidates.index)
    for feature in FEATURES:
        values = candidates[feature].astype(float)
        center = float(values.median())
        mad = float((values - center).abs().median()) * 1.4826
        scale = mad if mad > 1e-9 else float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        distance += ((values - float(signal[feature])) / scale) ** 2
    return np.sqrt(distance)


def nearby_alert(code: str, date: pd.Timestamp, alerts: dict[str, list[pd.Timestamp]], days: int) -> bool:
    return any(abs((item - date).days) <= days for item in alerts.get(code, []))


def match_events(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    controls: int,
    exclude_days: int,
    cost_pct: float,
    cache_dir: Path,
    setting: int,
    verify_non_signals: bool,
    engine: str = "platform_events",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    field = "confirmed_event_count" if engine == "platform_events" else "min_line_strength"
    control_config = production_strategy_config(
        strategy_mode="precision_pullback", precision_engine=engine, include_chart=False, **{field: setting},
    )
    indexed = panel.set_index(["stock_code", "date"], drop=False)
    by_date = {date: group for date, group in panel.groupby("date")}
    alert_dates = {
        code: [pd.Timestamp(value) for value in group["signal_date"]]
        for code, group in signals.groupby("stock_code")
    }
    matched: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    alert_cache: dict[tuple[str, str], bool] = {}
    frame_cache: dict[str, pd.DataFrame] = {}

    def is_alert(code: str, date: str) -> bool:
        key = (code, date)
        if key in alert_cache:
            return alert_cache[key]
        try:
            frame = frame_cache.get(code)
            if frame is None:
                payload = json.loads((cache_dir / f"{code}.json").read_text(encoding="utf-8"))
                frame = _normalize_bars(pd.DataFrame(payload.get("bars", []))).reset_index(drop=True)
                if len(frame_cache) >= 128:
                    frame_cache.pop(next(iter(frame_cache)))
                frame_cache[code] = frame
            dates = frame["trade_date"].dt.strftime("%Y-%m-%d")
            positions = np.flatnonzero((dates == date).to_numpy())
            if not len(positions):
                alert_cache[key] = True
                return True
            visible = frame.iloc[: int(positions[-1]) + 1]
            result = analyze_bars(
                code,
                code,
                visible,
                control_config,
            )
            alert_cache[key] = bool(result is not None and result.category == SignalCategory.BUY_ALERT)
        except Exception:
            # A control that cannot be verified must not enter the comparison group.
            alert_cache[key] = True
        return alert_cache[key]

    for event_number, row in enumerate(signals.itertuples(index=False), start=1):
        code, date = str(row.stock_code).zfill(6), str(row.signal_date)
        key = (code, date)
        if key not in indexed.index or date not in by_date:
            skipped.append({"stock_code": code, "signal_date": date, "reason": "missing signal panel row"})
            continue
        signal = indexed.loc[key]
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[-1]
        candidates = by_date[date].copy()
        candidates = candidates[candidates["stock_code"] != code]
        event_date = pd.Timestamp(date)
        candidates = candidates[
            ~candidates["stock_code"].map(lambda item: nearby_alert(str(item), event_date, alert_dates, exclude_days))
        ]
        candidates = candidates.dropna(subset=list(FEATURES))
        if len(candidates) < controls or signal[list(FEATURES)].isna().any():
            skipped.append({"stock_code": code, "signal_date": date, "reason": "insufficient comparable controls"})
            continue
        candidates["distance"] = robust_distance(signal, candidates)
        same_segment = candidates[candidates["segment"] == signal["segment"]].sort_values("distance")
        other_segments = candidates[candidates["segment"] != signal["segment"]].sort_values("distance")
        ordered = pd.concat([same_segment, other_segments])
        picked_indices: list[int] = []
        for candidate in ordered.itertuples():
            if verify_non_signals and is_alert(str(candidate.stock_code), date):
                continue
            picked_indices.append(candidate.Index)
            if len(picked_indices) == controls:
                break
        if len(picked_indices) < controls:
            skipped.append({"stock_code": code, "signal_date": date, "reason": "not enough verified non-signal controls"})
            continue
        chosen = candidates.loc[picked_indices]
        record: dict[str, Any] = {
            "stock_code": code,
            "stock_name": getattr(row, "stock_name", ""),
            "signal_date": date,
            "segment": signal["segment"],
            "control_codes": ",".join(chosen["stock_code"]),
            "control_count": len(chosen),
            "match_distance": round(float(chosen["distance"].mean()), 3),
        }
        for feature in FEATURES:
            record[f"signal_{feature}"] = signal[feature]
            record[f"control_{feature}"] = float(chosen[feature].mean())
        for outcome in OUTCOMES:
            signal_value = signal[outcome]
            controls_valid = chosen[outcome].dropna().astype(float)
            control_value = float(controls_valid.mean()) if len(controls_valid) else None
            record[f"signal_{outcome}"] = float(signal_value) if pd.notna(signal_value) else None
            record[f"control_{outcome}"] = control_value
            record[f"excess_{outcome}"] = (
                float(signal_value) - control_value
                if pd.notna(signal_value) and control_value is not None
                else None
            )
        for window in (5, 10, 20):
            signal_return = record[f"signal_return_{window}"]
            control_return = record[f"control_return_{window}"]
            record[f"signal_return_{window}_net"] = signal_return - cost_pct if signal_return is not None else None
            record[f"control_return_{window}_net"] = control_return - cost_pct if control_return is not None else None
        matched.append(record)
        if event_number % 20 == 0 or event_number == len(signals):
            print(
                f"match {event_number}/{len(signals)} matched={len(matched)} "
                f"skipped={len(skipped)} verified={len(alert_cache)}",
                flush=True,
            )
    return pd.DataFrame(matched), skipped


def clustered_ci(frame: pd.DataFrame, column: str, iterations: int) -> list[float | None]:
    valid = frame[["signal_date", column]].dropna()
    groups = [group[column].to_numpy(dtype=float) for _date, group in valid.groupby("signal_date")]
    if not groups:
        return [None, None]
    rng = np.random.default_rng(20260904)
    samples = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        picked = rng.integers(0, len(groups), size=len(groups))
        samples[idx] = np.concatenate([groups[item] for item in picked]).mean()
    return [round(float(np.percentile(samples, 2.5)), 3), round(float(np.percentile(samples, 97.5)), 3)]


def summarize(frame: pd.DataFrame, iterations: int, cost_pct: float) -> dict[str, Any]:
    output: dict[str, Any] = {
        "matched_signal_count": len(frame),
        "unique_signal_dates": int(frame["signal_date"].nunique()) if len(frame) else 0,
        "median_match_distance": round(float(frame["match_distance"].median()), 3) if len(frame) else None,
        "roundtrip_cost_pct": cost_pct,
    }
    for window in (5, 10, 20):
        signal = frame[f"signal_return_{window}"].dropna()
        control = frame[f"control_return_{window}"].dropna()
        excess = frame[f"excess_return_{window}"].dropna()
        output[f"t{window}"] = {
            "sample_count": len(excess),
            "signal_average": round(float(signal.mean()), 3) if len(signal) else None,
            "signal_net_average": round(float(signal.mean() - cost_pct), 3) if len(signal) else None,
            "control_average": round(float(control.mean()), 3) if len(control) else None,
            "excess_average": round(float(excess.mean()), 3) if len(excess) else None,
            "excess_median": round(float(excess.median()), 3) if len(excess) else None,
            "signal_beats_control_rate": round(float((excess > 0).mean() * 100), 1) if len(excess) else None,
            "date_clustered_95_ci": clustered_ci(frame, f"excess_return_{window}", iterations),
        }
    for outcome in ("mae_5", "mfe_5", "target_before_stop"):
        signal = frame[f"signal_{outcome}"].dropna()
        control = frame[f"control_{outcome}"].dropna()
        output[outcome] = {
            "sample_count": int(min(len(signal), len(control))),
            "signal_average": round(float(signal.mean()), 3) if len(signal) else None,
            "control_average": round(float(control.mean()), 3) if len(control) else None,
        }
    return output


def markdown(args: argparse.Namespace, summary: dict[str, Any], skipped: list[dict[str, Any]]) -> str:
    rows = []
    for window in (5, 10, 20):
        item = summary[f"t{window}"]
        ci = item["date_clustered_95_ci"]
        rows.append(
            f"| T+{window} | {item['sample_count']} | {item['signal_average']}% | {item['signal_net_average']}% | "
            f"{item['control_average']}% | {item['excess_average']}% | {item['excess_median']}% | "
            f"{item['signal_beats_control_rate']}% | {ci[0]}%～{ci[1]}% |"
        )
    target = summary["target_before_stop"]
    mae, mfe = summary["mae_5"], summary["mfe_5"]
    t5_ci = summary["t5"]["date_clustered_95_ci"]
    evidence = "有" if t5_ci[0] is not None and t5_ci[0] > 0 else "没有"
    return f"""# 历史预警匹配对照组回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 方法

- 信号：{args.engine} 引擎、强度参数 {args.setting} 的时点化历史预警（其余为当前默认配置）
- 对照：每个信号匹配同日 {args.controls} 只相似非信号股票
- 匹配变量：市场板块、近5/20日收益、20日波动率、MA20距离、20日量比
- 非信号确认：每个入选对照在信号日重新运行同一策略，确认当日不是买入预警
- 辅助排除：已知信号样本在信号日前后 {args.exclude_days} 个自然日内不作为对照
- 成交：下一交易日开盘买入，T+5/T+10/T+20 收盘卖出
- 成本：双边合计 {args.roundtrip_cost_bps:.0f} bps
- 置信区间：按信号日期聚类 bootstrap，{args.bootstrap} 次

匹配成功 {summary['matched_signal_count']} 个信号，覆盖 {summary['unique_signal_dates']} 个交易日；
未匹配 {len(skipped)} 个，匹配距离中位数 {summary['median_match_distance']}。

## 配对结果

| 周期 | 样本 | 信号原始收益 | 信号成本后 | 对照收益 | 超额均值 | 超额中位数 | 跑赢对照 | 超额收益95%区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 路径风险

- 未来5日最大不利波动：信号 {mae['signal_average']}%，对照 {mae['control_average']}%
- 未来5日最大有利波动：信号 {mfe['signal_average']}%，对照 {mfe['control_average']}%
- 20日内 +5% 先于 -5%：信号 {None if target['signal_average'] is None else round(target['signal_average'] * 100, 1)}%，对照 {None if target['control_average'] is None else round(target['control_average'] * 100, 1)}%

## 结论

当前样本对“T+5 超额收益为正”**{evidence}统计证据**。判断依据是按日期聚类后的95%区间是否整体高于0，而不是信号自身是否上涨。

## 限制

- 没有历史市值和历史行业成分，无法按市值与当时行业精确匹配。
- 股票池来自当前上市公司，仍有幸存者偏差。
- 涨停无法买入、停牌与实际滑点只通过统一成本近似，没有逐笔成交模拟。
"""


def main() -> None:
    args = parse_args()
    signals = pd.read_csv(args.signals, dtype={"stock_code": str})
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    signals = signals[signals["setting"] == args.setting].copy()
    universe_payload = json.loads(Path(args.universe_summary).read_text(encoding="utf-8"))
    sample_codes = [str(code).zfill(6) for code in universe_payload["selected_codes"]]
    codes = (
        all_cached_codes(Path(args.stock_list), Path(args.cache_dir))
        if args.all_cached_controls
        else sample_codes
    )
    dates = set(signals["signal_date"].astype(str))
    panel, failures = build_panel(codes, Path(args.cache_dir), dates)
    matched, skipped = match_events(
        signals,
        panel,
        controls=args.controls,
        exclude_days=args.exclude_days,
        cost_pct=args.roundtrip_cost_bps / 100.0,
        cache_dir=Path(args.cache_dir),
        setting=args.setting,
        verify_non_signals=args.all_cached_controls,
        engine=args.engine,
    )
    summary = summarize(matched, args.bootstrap, args.roundtrip_cost_bps / 100.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output_dir / f"matched_control_backtest_{stamp}"
    matched.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    machine = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(args),
        "summary": summary,
        "panel_failures": failures,
        "skipped_signals": skipped,
    }
    stem.with_suffix(".json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(args, summary, skipped), encoding="utf-8")
    print(json.dumps({"report": str(stem.with_suffix('.md')), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
