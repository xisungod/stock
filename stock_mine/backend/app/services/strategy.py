from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from app.services.zigzag import confirmed_swings
from app.services.flag_pattern import FLAG_DIRECTIONS, detect_flag
from app.services.price_bands import bounded_band, bounded_clusters

LEGACY_STRATEGY_VERSION = "2026-09-09-price-breakout-v1"
PIVOT_FIXED_VERSION = "2026-09-10-bounded-pivots-v1"
STRATEGY_VERSION = "2026-09-10-platform-events-v4"
MAX_PULLBACK_DAYS = 60
MARKET_FILTERS = ("off", "risk", "downgrade")


def version_for_engine(engine: str = "platform_events") -> str:
    return {
        "legacy": LEGACY_STRATEGY_VERSION,
        "pivot_fixed": PIVOT_FIXED_VERSION,
        "platform_events": STRATEGY_VERSION,
    }[engine]


def validate_pullback_days(
    minimum: int, maximum: int | None, legacy_minimum: int = 0, legacy_maximum: int = 0,
) -> None:
    for value in (minimum, maximum):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not 1 <= value <= MAX_PULLBACK_DAYS
        ):
            raise ValueError(f"回调天数必须为1～{MAX_PULLBACK_DAYS}的整数")
    if minimum is None:
        raise ValueError("回调天数下限不能为空")
    if maximum is None:
        if minimum != 1:
            raise ValueError("设置回调天数下限时必须同时设置上限")
        return
    if minimum > maximum:
        raise ValueError("回调天数上限不能小于下限")
    if max(minimum, legacy_minimum) > min(maximum, legacy_maximum or maximum):
        raise ValueError("回调天数与旧版距突破天数窗口没有交集")


class SignalCategory(StrEnum):
    BUY_ALERT = "buy_alert"
    WATCH = "watch"
    NONE = "none"


@dataclass(frozen=True)
class StrategyConfig:
    # None 表示使用数据源可提供的全部历史。精准线是跨周期结构，不能先裁成固定窗口。
    lookback_days: int | None = None
    min_history_bars: int = 120
    signal_offset: int = 1
    # 同一价格区域的高点应归为一条线。1.2% 过窄会把相邻 2% 内的高点拆成两条
    # （实测格力 41.36 与 40.41 相差仅 2.3% 却成了两条独立的线）。
    error_margin_pct: float = 0.02
    min_line_strength: int = 3
    breakout_window: int = 10
    # 距突破日的交易日数窗口，0 表示不限制。用来只挑"突破后过了几天、
    # 正处在回踩阶段"的股票：太新还没来得及回踩，太旧已经走远。
    # 旧批次兼容参数；新筛选使用 min/max_pullback_days 同步控制搜索和回踩窗口。
    min_days_since_breakout: int = 0
    max_days_since_breakout: int = 0
    # 只保留符合当前突破规则的股票；关闭时近线但未突破的观察机会照常入选。
    require_breakout: bool = False
    pullback_window: int = 10
    require_pullback: bool = True
    min_pullback_days: int = 1
    # None preserves the original breakout_window / pullback_window of saved batches.
    max_pullback_days: int | None = 10
    breakout_eps_pct: float = 0.006
    pullback_touch_pct: float = 0.012
    allow_undercut_pct: float = 0.003
    max_intraday_undercut_pct: float = 0.03
    max_intervening_close_break_pct: float = 0.015
    max_extension_pct: float = 0.035
    # None preserves batches made before this filter existed; 0 is strict, not disabled.
    max_prebreakout_extension_pct: float | None = 0.50
    # Reject at this run length, not one day after it. None disables the duration filter.
    prebreakout_reject_above_days: int | None = 30
    prebreakout_above_buffer_pct: float = 0.02
    vol_lookback: int = 10
    breakout_vol_mult: float = 1.5
    # Applies only to precision-line breakouts; flags retain their volume confirmation.
    require_breakout_volume: bool = False
    pullback_vol_max_ratio: float = 0.75
    # 「缩量」还必须相对常态：回踩量不高于突破前 vol_lookback 日均量的该倍数。
    # 只和突破日比几乎必然成立（突破日本身就是放量日）。None 保留此规则之前的批次。
    pullback_max_normal_volume_ratio: float | None = None
    # 突破日至信号日的最高价越线超过该比例后又回到线附近，属冲高回落而不是回踩。
    # None 保留此规则之前的批次。
    max_post_breakout_runup_pct: float | None = None
    # 买入预警要求突破前该根数之内有足够的独立验证（platform_events 为 confirmed_event_count，
    # legacy 为 min_line_strength）；更早的证据只保留展示。None 保留全历史计数的旧批次。
    max_event_age_bars: int | None = None
    # 沪深300环境：off 不评估；risk 仅标注；downgrade 偏弱时买入预警降为观察。
    market_filter: str = "off"
    ma_period: int = 20
    ma_slope_days: int = 5
    watch_near_pct: float = 0.05
    watch_below_pct: float = 0.015
    above_line_wave_max_count: int = 1
    above_line_wave_min_gain_pct: float = 0.05
    min_long_term_peak_gap: int = 60
    # 拓扑显著度门槛 = ratio × ATR20%，再夹在 [min, max] 之间。
    # 用 ATR 归一化：中国核建 ATR 4.4%，与平安银行 ATR 1.6%，
    # 同一个绝对百分比门槛在两者上必然一严一松。
    # 0.7 而非 1.0：292 只 × 回看 400 日的扫参显示，这个系数只决定信号数量、
    # 不决定信号质量——降门槛多出来的 571 个信号触及+10% 率 44.1%，比共有信号的
    # 41.4% 还高（z=+1.01，不显著即"没变差"）；升到 1.5 丢掉的信号也不比留下的差。
    # 而覆盖度是实的：1.0 时有 3.8% 的股票一个信号都选不出，0.7 降到 0.7%。
    # 摆动识别（ZigZag）：候选极值取最高/最低价，确认反转取收盘价。
    # 反转阈值 = clip(ratio × ATR20%, min, max)，让银行股和小盘股各按自己的尺度。
    swing_reversal_atr_ratio: float = 1.5
    swing_reversal_min_pct: float = 3.0
    swing_reversal_max_pct: float = 8.0
    recent_touch_window: int = 0
    recent_touch_min: int = 2
    max_display_lines: int = 6
    # 精准线与现价的最大偏离。超出该范围的线对当前交易没有参考意义
    # （实测茅台现价 1272 时仍保留 428.22 的线，偏离 -197%）。0 表示不限制。
    max_line_distance_pct: float = 0.35
    include_chart: bool = True
    strategy_mode: str = "precision_pullback"
    flag_direction: str = "all"
    # Explicit construction remains backward compatible; new API requests opt into V2.
    precision_engine: str = "legacy"
    event_reversal_atr: float = 1.0
    event_min_reversal_pct: float = 0.01
    candidate_event_count: int = 2
    confirmed_event_count: int = 3

    def __post_init__(self) -> None:
        validate_pullback_days(
            self.min_pullback_days, self.max_pullback_days,
            self.min_days_since_breakout, self.max_days_since_breakout,
        )
        if self.precision_engine not in {"legacy", "pivot_fixed", "platform_events"}:
            raise ValueError("unknown precision_engine")
        if not 0 <= self.error_margin_pct < 1:
            raise ValueError("error_margin_pct must be in [0, 1)")
        if not np.isfinite(self.event_reversal_atr) or self.event_reversal_atr <= 0:
            raise ValueError("event_reversal_atr must be positive")
        if not 0 < self.event_min_reversal_pct < 1:
            raise ValueError("event_min_reversal_pct must be in (0, 1)")
        if not 2 <= self.candidate_event_count <= self.confirmed_event_count:
            raise ValueError("event counts must satisfy 2 <= candidate <= confirmed")
        if self.precision_engine == "platform_events" and self.lookback_days is not None:
            raise ValueError("platform_events uses full history; limit replay output instead")
        for name in ("pullback_max_normal_volume_ratio", "max_post_breakout_runup_pct"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or None")
        age = self.max_event_age_bars
        if age is not None and (isinstance(age, bool) or not isinstance(age, (int, np.integer)) or age < 1):
            raise ValueError("max_event_age_bars must be a positive integer or None")
        if self.market_filter not in MARKET_FILTERS:
            raise ValueError("unknown market_filter")


@dataclass
class KeyLine:
    price: float
    strength: int
    touch_indices: list[int]
    last_touch_index: int
    formation_index: int
    above_wave_count: int = 0
    support_indices: list[int] = field(default_factory=list)
    line_type: str = "resistance"
    resistance_confirmed_at: dict[int, int] = field(default_factory=dict)
    support_confirmed_at: dict[int, int] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return len(self.touch_indices) + len(self.support_indices)

    @property
    def last_pivot_index(self) -> int:
        """最后一次触及的**发生日**下标。

        注意不是"确认日"：波峰波谷在发生日成形，要等回撤够了才可识别
        （实测中位滞后 2 根）。用于按新近程度排序时这点差别可以忽略，
        但对外展示必须叫对名字。
        """
        return max([*self.touch_indices, *self.support_indices])


@dataclass
class SignalResult:
    stock_code: str
    stock_name: str
    category: SignalCategory
    score: float
    suggestion: str
    key_line_price: float | None
    line_strength: int
    signal_date: str | None
    line_formation_date: str | None = None
    breakout_date: str | None = None
    pullback_date: str | None = None
    current_price: float | None = None
    distance_to_line_pct: float | None = None
    breakout_volume_ratio: float | None = None
    pullback_volume_ratio: float | None = None
    pullback_normal_volume_ratio: float | None = None
    breakout_status: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    chart: dict[str, Any] = field(default_factory=dict)
    key_lines: list[dict[str, Any]] = field(default_factory=list)
    structure_score: float | None = None
    industry: str | None = None
    industry_score: float | None = None
    industry_adjustment: float = 0.0
    industry_risk_level: str = "uncovered"
    industry_risk_message: str = "行业数据暂未覆盖"
    industry_source: str | None = None
    industry_data_as_of: str | None = None
    industry_stale: bool = False
    pattern_type: str = "precision_pullback"
    flag_pattern: dict[str, Any] | None = None
    pattern_matches: list[dict[str, Any]] = field(default_factory=list)
    flag_observation: dict[str, Any] | None = field(default=None, repr=False)
    prebreakout_check: dict[str, Any] | None = None
    pullback_check: dict[str, Any] | None = None
    require_breakout_volume: bool | None = None
    breakout_volume_threshold: float | None = None
    resistance_anchor: dict[str, Any] | None = None
    precision_engine: str = "legacy"
    platform_state: str | None = None
    signal_event_id: str | None = None
    line_snapshot: dict[str, Any] | None = None
    platform_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    breakout_limit_up: bool | None = None
    market_regime: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        structure_score = self.score if self.structure_score is None else self.structure_score
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "category": self.category.value,
            "score": round(float(self.score), 2),
            # 行业快照无法回填历史、未经验证，不再计入排序分；只作为环境提示展示。
            "score_type": "structure",
            "strategy_version": version_for_engine(self.precision_engine),
            "precision_engine": self.precision_engine,
            "platform_state": self.platform_state,
            "signal_event_id": self.signal_event_id,
            "line_snapshot": self.line_snapshot,
            "platform_diagnostics": self.platform_diagnostics,
            "structure_score": round(float(structure_score), 2),
            "industry": self.industry,
            "industry_score": self.industry_score,
            "industry_adjustment": round(float(self.industry_adjustment), 2),
            "industry_risk_level": self.industry_risk_level,
            "industry_risk_message": self.industry_risk_message,
            "industry_source": self.industry_source,
            "industry_data_as_of": self.industry_data_as_of,
            "industry_stale": self.industry_stale,
            "suggestion": self.suggestion,
            "key_line_price": self.key_line_price,
            "line_strength": self.line_strength,
            "signal_date": self.signal_date,
            "line_formation_date": self.line_formation_date,
            "breakout_date": self.breakout_date,
            "pullback_date": self.pullback_date,
            "current_price": self.current_price,
            "distance_to_line_pct": self.distance_to_line_pct,
            "breakout_volume_ratio": self.breakout_volume_ratio,
            "require_breakout_volume": self.require_breakout_volume,
            "breakout_volume_threshold": self.breakout_volume_threshold,
            "resistance_anchor": self.resistance_anchor,
            "pullback_volume_ratio": self.pullback_volume_ratio,
            "pullback_normal_volume_ratio": self.pullback_normal_volume_ratio,
            "breakout_status": self.breakout_status,
            "reasons": self.reasons,
            "risks": self.risks,
            "chart": self.chart,
            "key_lines": self.key_lines,
            "pattern_type": self.pattern_type,
            "flag_pattern": self.flag_pattern,
            "pattern_matches": self.pattern_matches or ([_pattern_snapshot(self)] if self.category != SignalCategory.NONE else []),
            "prebreakout_check": self.prebreakout_check,
            "pullback_check": self.pullback_check,
            "breakout_limit_up": self.breakout_limit_up,
            "market_regime": self.market_regime,
        }


def build_strategy_config(params: dict[str, Any]) -> StrategyConfig:
    """把扫描/回测请求参数翻译成策略配置——扫描、详情、回测与研究脚本共用这一份。

    参数缺失表示该批次创建时规则尚不存在，一律按「未启用」解释，旧批次的结论不会被静默改写。
    新请求的默认值由 `app.schemas.ScanRequest` 给出。
    """
    precision_enabled = params.get("strategy_mode") != "flag_breakout"

    def optional(name: str, cast):
        value = params.get(name)
        return None if value is None else cast(value)

    return StrategyConfig(
        precision_engine=params.get("precision_engine", "legacy"),
        signal_offset=int(params.get("signal_offset", 1)),
        min_days_since_breakout=int(params.get("min_days_since_breakout", 0) or 0),
        max_days_since_breakout=int(params.get("max_days_since_breakout", 0) or 0),
        require_breakout=bool(params.get("require_breakout", False)),
        require_pullback=precision_enabled and bool(params.get("require_pullback", False)),
        min_pullback_days=int(params.get("min_pullback_days", 1)) if precision_enabled else 1,
        max_pullback_days=(
            int(params["max_pullback_days"])
            if precision_enabled and params.get("max_pullback_days") is not None else None
        ),
        breakout_vol_mult=float(params.get("breakout_vol_mult") or 1.5),
        # Older batches always required volume; do not silently change their signals.
        require_breakout_volume=not precision_enabled or bool(params.get("require_breakout_volume", True)),
        strategy_mode=params.get("strategy_mode", "precision_pullback"),
        flag_direction=params.get("flag_direction", "all"),
        # An absent field in an old saved batch means the filter was not enabled.
        max_prebreakout_extension_pct=optional("max_prebreakout_extension_pct", float),
        prebreakout_reject_above_days=optional("prebreakout_reject_above_days", int),
        prebreakout_above_buffer_pct=float(params.get("prebreakout_above_buffer_pct", 0.02)),
        pullback_max_normal_volume_ratio=optional("pullback_max_normal_volume_ratio", float),
        max_post_breakout_runup_pct=optional("max_post_breakout_runup_pct", float),
        max_event_age_bars=optional("max_event_age_bars", int),
        market_filter=params.get("market_filter") or "off",
        **{name: params[name] for name in ("include_chart", "confirmed_event_count", "min_line_strength")
           if name in params},
    )


def daily_price_limit(stock_code: str, stock_name: str = "", trade_date: Any = None) -> float:
    """A股日涨跌幅限制（近似）：ST 5%，创业板 2020-08-24 起与科创板 20%，北交所 30%，其余 10%。"""
    code = str(stock_code).zfill(6)
    if "ST" in str(stock_name or "").upper():
        return 0.05
    if code.startswith("68"):
        return 0.20
    if code.startswith("30"):
        when = pd.to_datetime(trade_date) if trade_date is not None else None
        return 0.20 if when is None or when >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def is_limit_up_close(df: pd.DataFrame, index: int, stock_code: str, stock_name: str = "") -> bool:
    """该根以涨停价收盘：涨幅达到限制（留 0.3% 取整/复权误差）且收在最高价。"""
    if index < 1 or index >= len(df):
        return False
    previous = float(df["close"].iloc[index - 1])
    close = float(df["close"].iloc[index])
    high = float(df["high"].iloc[index])
    if previous <= 0:
        return False
    limit = daily_price_limit(stock_code, stock_name, df["trade_date"].iloc[index])
    return close / previous - 1 >= limit - 0.003 and bool(np.isclose(close, high, rtol=1e-4))


LIMIT_UP_BREAKOUT_RISK = "突破日以涨停收盘：多为消息驱动，当日通常无法买入，随后回落更可能是兑现而非健康回踩"


def _run_pipeline(
    stock_code: str,
    stock_name: str,
    bars: pd.DataFrame,
    config: StrategyConfig | None = None,
    bars_are_normalized: bool = False,
) -> tuple[str, str, SignalResult | None]:
    """返回 (stage, message, result)。stage 见 diagnose_bars 文档。"""
    cfg = config or StrategyConfig()
    df = bars if bars_are_normalized else _normalize_bars(bars)
    df = _completed_session_bars(df, bars.attrs.get("data_quality"), cfg)
    minimum_bars = max(cfg.min_history_bars, cfg.ma_period + cfg.ma_slope_days)
    if df.empty or len(df) < minimum_bars:
        return (
            "insufficient_history",
            f"历史数据不足 {minimum_bars} 根（当前 {len(df)} 根），可能是新股或长期停牌",
            None,
        )

    signal_idx = len(df) - 1 - max(0, cfg.signal_offset)
    if signal_idx < 2:
        return ("signal_offset_out_of_range", "可用数据不足以回看到所选的信号 K 线", None)

    df_upto = df.iloc[: signal_idx + 1].copy().reset_index(drop=True)
    if cfg.lookback_days is not None and cfg.lookback_days > 0:
        df_upto = df_upto.tail(cfg.lookback_days).reset_index(drop=True)
    if len(df_upto) < minimum_bars:
        return ("insufficient_history", f"截取到信号日后仅剩 {len(df_upto)} 根 K 线，不足以分析", None)

    if cfg.strategy_mode == "combined":
        return _run_combined_pipeline(stock_code, stock_name, df_upto, cfg)
    if cfg.strategy_mode == "flag_breakout":
        outcome = _run_flag_pipeline(stock_code, stock_name, df_upto, cfg)
    elif cfg.strategy_mode == "precision_pullback":
        outcome = _run_precision_pipeline(stock_code, stock_name, df_upto, cfg)
    else:
        raise ValueError("unknown strategy_mode")
    result = outcome[2]
    if result is not None and result.category != SignalCategory.NONE:
        result.pattern_matches = [_pattern_snapshot(result)]
    return outcome


def _run_precision_pipeline(
    stock_code: str, stock_name: str, df_upto: pd.DataFrame, cfg: StrategyConfig,
) -> tuple[str, str, SignalResult | None]:
    if cfg.precision_engine == "platform_events":
        from app.services.platform_events import analyze_platform
        result = analyze_platform(stock_code, stock_name, df_upto, cfg)
        if result is None:
            return "no_key_line", "尚未形成具备独立验证事件的价位带", None
        return ("ok" if result.category != SignalCategory.NONE else "no_signal"), result.suggestion, result
    swing_highs, swing_lows, pending_swing, swing_meta = _swings(df_upto, cfg, with_confirmation=True)
    lines = _cluster_key_lines(df_upto, swing_highs, cfg, swing_lows, swing_meta, swing_highs)
    if not lines:
        history_label = "全部历史" if cfg.lookback_days is None else f"近 {cfg.lookback_days} 日"
        return (
            "no_key_line",
            f"{history_label}未找到有效精准线",
            None,
        )

    ma_min_periods = min(cfg.ma_period, max(5, cfg.ma_period // 2))
    df_upto["ma"] = df_upto["close"].rolling(cfg.ma_period, min_periods=ma_min_periods).mean()
    best = _select_best_signal(stock_code, stock_name, df_upto, lines, cfg)
    if best is None:
        return ("no_key_line", "未能在候选精准线上形成可评估的结构", None)
    if best.category == SignalCategory.NONE:
        return ("no_signal", best.suggestion, best)
    return ("ok", best.suggestion, best)


def _pattern_snapshot(result: SignalResult) -> dict[str, Any]:
    fields = (
        "pattern_type", "suggestion", "signal_date", "key_line_price", "line_strength",
        "line_formation_date", "breakout_date", "pullback_date", "current_price",
        "distance_to_line_pct", "breakout_volume_ratio", "require_breakout_volume",
        "breakout_volume_threshold", "pullback_volume_ratio",
        "pullback_normal_volume_ratio", "breakout_status", "reasons", "risks",
        "prebreakout_check", "pullback_check", "resistance_anchor",
        "precision_engine", "platform_state", "signal_event_id", "line_snapshot",
    )
    return {
        **{name: getattr(result, name) for name in fields},
        "category": result.category.value,
        "structure_score": round(float(result.structure_score if result.structure_score is not None else result.score), 2),
        "flag_direction": result.flag_pattern.get("direction") if result.flag_pattern else None,
    }


def _run_combined_pipeline(
    stock_code: str, stock_name: str, df: pd.DataFrame, cfg: StrategyConfig,
) -> tuple[str, str, SignalResult | None]:
    precision = _run_precision_pipeline(stock_code, stock_name, df, replace(cfg, strategy_mode="precision_pullback"))
    # Breakout age belongs only to precision-line pullbacks, not to the flag branch.
    flag = _run_flag_pipeline(stock_code, stock_name, df, replace(
        cfg, strategy_mode="flag_breakout", min_days_since_breakout=0, max_days_since_breakout=0,
        max_prebreakout_extension_pct=None, prebreakout_reject_above_days=None,
        require_pullback=False, min_pullback_days=1, max_pullback_days=None,
        require_breakout_volume=True,
    ))
    return _merge_pattern_results(stock_code, stock_name, df, cfg, precision, flag)


def _merge_pattern_results(stock_code, stock_name, df, cfg, precision, flag):
    evaluated = [outcome[2] for outcome in (precision, flag) if outcome[2] is not None]
    matched = [result for result in evaluated if result.category != SignalCategory.NONE]
    observation = flag[2].flag_pattern if flag[2] is not None else None
    if not matched:
        message = f"精准线：{precision[1]}；旗形：{flag[1]}"
        diagnostic = replace(max(evaluated, key=_rank), pattern_matches=[], flag_observation=observation) if evaluated else None
        return "no_signal", message, diagnostic
    primary = max(matched, key=_rank)
    precision_hit = next((r for r in matched if r.pattern_type == "precision_pullback"), None)
    flag_hit = next((r for r in matched if r.pattern_type == "flag_breakout"), None)
    snapshots = [_pattern_snapshot(result) for result in matched]
    labels = [
        ("精准线回调" if item["pattern_type"] == "precision_pullback" else FLAG_DIRECTIONS.get(item["flag_direction"], "旗形"))
        + ("（已确认）" if item["category"] == "buy_alert" else "（待确认）")
        for item in snapshots
    ]
    merged = replace(
        primary,
        pattern_matches=snapshots,
        flag_pattern=flag_hit.flag_pattern if flag_hit else None,
        flag_observation=observation,
        key_lines=precision_hit.key_lines if precision_hit else [],
        suggestion=primary.suggestion if len(matched) == 1 else f"联合筛选：{'、'.join(labels)}",
        risks=list(dict.fromkeys(risk for result in matched for risk in result.risks)),
    )
    if cfg.include_chart:
        merged.chart = _build_chart_payload(df, merged)
        markers = {}
        for result in matched:
            for marker in result.chart.get("markers", []):
                markers[(marker["date"], marker["type"])] = marker
        merged.chart["markers"] = list(markers.values())
    return "ok", merged.suggestion, merged


def _run_flag_pipeline(
    stock_code: str, stock_name: str, df: pd.DataFrame, cfg: StrategyConfig,
) -> tuple[str, str, SignalResult | None]:
    if cfg.flag_direction not in {"all", *FLAG_DIRECTIONS}:
        raise ValueError("unknown flag_direction")
    if cfg.min_days_since_breakout or cfg.max_days_since_breakout:
        raise ValueError("旗形突破不使用精准线的距突破天数窗口")
    pattern = detect_flag(df, breakout_vol_mult=cfg.breakout_vol_mult)
    if pattern is None:
        return "no_flag", "未找到满足旗杆、通道、缩量条件的旗形", None
    if cfg.flag_direction != "all" and pattern["direction"] != cfg.flag_direction:
        return "flag_direction_filtered", f"当前为{pattern['direction_label']}，不属于所选方向", None
    status = pattern["status"]
    label = pattern["direction_label"]
    state_labels = {
        "forming": "旗面形成中，等待放量突破",
        "breakout": "收盘放量突破旗面上轨",
        "unconfirmed_breakout": "越过上轨但量能未确认",
        "failed": "收盘跌破通道或回撤过深，形态失效",
        "untradable": "信号日无成交，暂不预警",
    }
    category = SignalCategory.BUY_ALERT if status == "breakout" else (
        SignalCategory.WATCH if status in {"forming", "unconfirmed_breakout"} else SignalCategory.NONE
    )
    if cfg.require_breakout and status != "breakout":
        category = SignalCategory.NONE
    score = 45 + min(15, pattern["pole_gain_pct"] / 2) + min(10, (1 - pattern["volume_contraction_ratio"]) * 20)
    score += min(10, pattern["upper_touch_count"] + pattern["lower_touch_count"])
    score += 15 if status == "breakout" else 0
    score = min(95, score)
    current = float(df.close.iloc[-1])
    result = SignalResult(
        stock_code, stock_name, category, score, f"{label}：{state_labels[status]}",
        None, 0, pattern["signal_date"],
        breakout_date=pattern["signal_date"] if status == "breakout" else None,
        current_price=round(current, 3),
        distance_to_line_pct=round((current / pattern["upper_price"] - 1) * 100, 3),
        breakout_volume_ratio=pattern["breakout_volume_ratio"],
        require_breakout_volume=True, breakout_volume_threshold=cfg.breakout_vol_mult,
        structure_score=score, pattern_type="flag_breakout", flag_pattern=pattern,
        flag_observation=pattern,
        precision_engine=cfg.precision_engine,
        signal_event_id=f"{stock_code}:flag:{pattern['id']}" if status == "breakout" else None,
        breakout_status={"tags": [state_labels[status]]},
        reasons=[
            f"旗杆 {pattern['pole_bars']} 根，涨幅 {pattern['pole_gain_pct']}%",
            f"旗面 {pattern['flag_bars']} 根，独立上轨触碰 {pattern['upper_touch_count']} 次、下轨 {pattern['lower_touch_count']} 次",
            f"通道宽度/旗杆高度 {pattern['width_to_pole_pct']}%，回撤/旗杆高度 {pattern['retracement_pct']}%",
            f"整理后半段/前半段成交量中位数 {pattern['volume_contraction_ratio']:.2f}",
        ],
        risks=["旗形为实验规则，得分不是胜率；未通过独立前向验证"],
    )
    if status in {"failed", "untradable", "unconfirmed_breakout"}:
        result.risks.append(state_labels[status])
    if cfg.include_chart:
        result.chart = _build_chart_payload(df, result)
    return ("ok" if category != SignalCategory.NONE else "no_signal"), result.suggestion, result


def analyze_bars(stock_code: str, stock_name: str, bars: pd.DataFrame, config: StrategyConfig | None = None) -> SignalResult | None:
    _stage, _message, result = _run_pipeline(stock_code, stock_name, bars, config)
    return result


def _analyze_normalized_bars(
    stock_code: str,
    stock_name: str,
    bars: pd.DataFrame,
    config: StrategyConfig,
) -> SignalResult | None:
    _stage, _message, result = _run_pipeline(
        stock_code, stock_name, bars, config, bars_are_normalized=True
    )
    return result


def diagnose_bars(
    stock_code: str, stock_name: str, bars: pd.DataFrame, config: StrategyConfig | None = None
) -> dict[str, Any]:
    """单股诊断：即便未入选也说明卡在哪一步。

    stage 取值：insufficient_history / signal_offset_out_of_range / no_key_line / no_signal / ok
    """
    stage, message, result = _run_pipeline(stock_code, stock_name, bars, config)
    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "found": stage == "ok",
        "stage": stage,
        "message": message,
        "result": result.to_dict() if result is not None else None,
        "chart": result.chart if result is not None else {},
    }


def _completed_session_bars(df: pd.DataFrame, quality: dict[str, Any] | None, cfg: StrategyConfig) -> pd.DataFrame:
    """signal_offset=0 表示「最近一个已收盘交易日」。

    行情核验会给出 expected_bar_date（最近已收盘交易日）；晚于它的 K 线是盘中未定型数据，
    在这里去掉。回测同样只用已收盘 K 线并按 offset=0 重放，这样扫描与回测是同一根信号 K 线。
    signal_offset>=1 是旧批次的字面含义「去掉最后 N 根」，原样保留不做裁剪。
    """
    expected = (quality or {}).get("expected_bar_date")
    if cfg.signal_offset != 0 or not expected or df.empty:
        return df
    cutoff = pd.Timestamp(expected)
    if pd.Timestamp(df["trade_date"].iloc[-1]) <= cutoff:
        return df
    return df[pd.to_datetime(df["trade_date"]) <= cutoff].reset_index(drop=True)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["trade_date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in bars.columns]
    if missing:
        return pd.DataFrame(columns=required)
    df = bars[required].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).sort_values("trade_date")
    df = df.drop_duplicates(subset=["trade_date"], keep="last")
    df = df[np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)).all(axis=1)]
    valid_prices = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = df["volume"] >= 0
    valid_ohlc = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    )
    df = df[valid_prices & valid_volume & valid_ohlc].reset_index(drop=True)
    return df


def _atr_pct(df: pd.DataFrame, period: int = 20) -> float | None:
    """最近 period 日的 ATR，以当前收盘价的百分比表示。"""
    if len(df) < period + 1:
        return None
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr = float(pd.Series(tr).rolling(period).mean().iloc[-1])
    last = float(close[-1])
    if not np.isfinite(atr) or last <= 0:
        return None
    return atr / last * 100


def _atr_pct_series(df: pd.DataFrame, period: int = 20) -> np.ndarray:
    """逐根 K 线的 ATR，以该根收盘价的百分比表示。

    每个值只依赖截至该根为止的数据，因此一根 K 线进入历史后就永久固定。
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=float)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    if n > 1:
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    out = np.full(n, np.nan, dtype=float)
    ok = close > 0
    out[ok] = atr[ok] / close[ok] * 100
    return out


def _swing_thresholds(df: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    """逐根返回反转阈值（百分数）。"""
    atr = _atr_pct_series(df)
    fallback = float(np.nanmedian(atr)) if np.isfinite(np.nanmedian(atr)) else 3.0
    atr = np.nan_to_num(atr, nan=fallback, posinf=fallback, neginf=fallback)
    reversal = np.clip(
        cfg.swing_reversal_atr_ratio * atr,
        cfg.swing_reversal_min_pct,
        cfg.swing_reversal_max_pct,
    )
    return reversal


def _swings(df: pd.DataFrame, cfg: StrategyConfig, with_confirmation: bool = False):
    """生产路径的波峰波谷：交替出现，并把尚未回撤确认的候选点单独给出。

    替代了原来"各自独立找局部极值 + min_peak_gap"的做法——那样会连续标出多个
    波峰（实测 18% 的摆动点是连续同类），本质上只是候选局部极值，不是波段拐点。
    """
    reversal = _swing_thresholds(df, cfg)
    return confirmed_swings(
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        reversal,
        with_pending=True,
        with_confirmation=with_confirmation,
        # 传开盘价 → 确认反转看 K 线实体而非收盘：阴线的收盘已是实体最低处，
        # 只看收盘会让波峰确认得偏松。
        opens=df["open"].to_numpy(dtype=float),
    )


def _support_touch_indices(
    df: pd.DataFrame,
    line_price: float,
    formation_index: int,
    cfg: StrategyConfig,
    low_indices: list[int] | None = None,
    high_indices: list[int] | None = None,
) -> list[int]:
    """统计有效上破后的独立回踩点；连续回踩只保留最贴近精准线的一天。"""
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    if low_indices is None or high_indices is None:
        computed = _swings(df, cfg)
        low_indices = computed[1] if low_indices is None else low_indices
        high_indices = computed[0] if high_indices is None else high_indices
    swing_lows, swing_highs = low_indices, high_indices
    breakout_level = line_price * (1 + cfg.breakout_eps_pct)
    failed_level = line_price * (1 - cfg.max_intervening_close_break_pct)
    indices = np.arange(len(df), dtype=int)
    crossed = np.zeros(len(df), dtype=bool)
    crossed[1:] = (closes[:-1] <= breakout_level) & (closes[1:] >= breakout_level)
    crossed[: formation_index + 1] = False
    failed = closes < failed_level
    last_cross = np.maximum.accumulate(np.where(crossed, indices, -1))
    last_failure = np.maximum.accumulate(np.where(failed, indices, -1))
    active_above = last_cross > last_failure
    is_swing_low = np.zeros(len(df), dtype=bool)
    is_swing_low[swing_lows] = True
    touched = (
        (lows >= line_price * (1 - cfg.max_intraday_undercut_pct))
        & (lows <= line_price * (1 + cfg.pullback_touch_pct))
    )
    recovered = closes >= line_price * (1 - cfg.allow_undercut_pct)
    candidates = np.flatnonzero(active_above & is_swing_low & touched & recovered).tolist()

    # 两次支撑触碰之间必须走出一个已确认波峰，才算两段独立的回踩。
    # 原来用"间隔至少 N 根"去重，那是主观的根数门槛，横盘里连碰十次仍会被
    # 拆成好几个精准点，把线的强度撑虚。
    independent: list[int] = []
    for idx in candidates:
        if independent and not any(independent[-1] < peak < idx for peak in swing_highs):
            previous = independent[-1]
            if abs(lows[idx] - line_price) < abs(lows[previous] - line_price):
                independent[-1] = idx
        else:
            independent.append(idx)
    return independent


def _cluster_key_lines(df: pd.DataFrame, high_indices: list[int], cfg: StrategyConfig,
                       low_indices: list[int] | None = None,
                       swing_meta: dict[str, list[int]] | None = None,
                       swing_highs: list[int] | None = None) -> list[KeyLine]:
    high_values = df["high"].to_numpy(dtype=float)
    if low_indices is None or swing_meta is None:
        high_indices, low_indices, _pending, swing_meta = _swings(df, cfg, with_confirmation=True)
        swing_highs = high_indices
    swing_highs = swing_highs if swing_highs is not None else high_indices
    # 波峰"发生"在 pivot 那天，却要到确认日才可识别（实测中位滞后 2 根、最长 19）。
    # 线的形成日必须用确认日，否则会把一段当时还画不出这条线的上穿记成"突破本线"。
    confirmed_at = dict(zip(high_indices, (swing_meta or {}).get("peak_confirmed_at", [])))
    clusters: list[list[int]] = []
    if cfg.precision_engine == "pivot_fixed":
        clusters = [
            [high_indices[i] for i in group]
            for group in bounded_clusters([float(high_values[i]) for i in high_indices], cfg.error_margin_pct)
        ]
    else:
        for idx in sorted(high_indices, key=lambda i: high_values[i]):
            price = float(high_values[idx])
            placed = False
            for cluster in clusters:
                center = float(np.mean(high_values[cluster]))
                if abs(price - center) / center <= cfg.error_margin_pct:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

    lines: list[KeyLine] = []
    for cluster in clusters:
        touch_indices = sorted(cluster)
        standard_line = len(touch_indices) >= cfg.min_line_strength
        long_term_pair = (
            len(touch_indices) == 2
            and touch_indices[-1] - touch_indices[0] >= cfg.min_long_term_peak_gap
        )
        if not standard_line and not long_term_pair:
            continue
        if cfg.recent_touch_window > 0:
            cutoff = len(df) - cfg.recent_touch_window
            if sum(1 for i in touch_indices if i >= cutoff) < cfg.recent_touch_min:
                continue
        pivot_idx = touch_indices[cfg.min_line_strength - 1] if standard_line else touch_indices[-1]
        formation_idx = confirmed_at.get(pivot_idx, pivot_idx)
        price = (
            bounded_band(high_values[touch_indices], cfg.error_margin_pct).price
            if cfg.precision_engine == "pivot_fixed" else float(np.median(high_values[touch_indices]))
        )
        support_indices = _support_touch_indices(
            df, price, formation_idx, cfg, low_indices, swing_highs
        )
        if cfg.precision_engine == "pivot_fixed":
            support_indices = [
                idx for idx in support_indices
                if abs(float(df["low"].iloc[idx]) / price - 1) <= cfg.error_margin_pct + 1e-10
            ]
        if long_term_pair and not support_indices:
            continue
        above_count = 0
        for idx in high_indices:
            if idx <= formation_idx:
                continue
            if idx in touch_indices:
                continue
            high = float(high_values[idx])
            if high > price * (1 + cfg.above_line_wave_min_gain_pct):
                above_count += 1
        line_type = "role_reversal" if long_term_pair else (
            "resistance_support" if support_indices else "resistance"
        )
        lines.append(
            KeyLine(
                price=price,
                strength=len(touch_indices),
                touch_indices=touch_indices,
                last_touch_index=max(touch_indices),
                formation_index=formation_idx,
                above_wave_count=above_count,
                support_indices=support_indices,
                line_type=line_type,
                resistance_confirmed_at={idx: confirmed_at[idx] for idx in touch_indices if idx in confirmed_at},
                support_confirmed_at={
                    idx: confirmation for idx, confirmation in zip(low_indices, (swing_meta or {}).get("valley_confirmed_at", []))
                    if idx in support_indices
                },
            )
        )
    # 丢弃离现价过远的线：它们对当前交易没有参考意义，只会占据展示位
    if cfg.max_line_distance_pct > 0 and len(df):
        current = float(df["close"].iloc[-1])
        if current > 0:
            lines = [
                line
                for line in lines
                if abs(line.price - current) / current <= cfg.max_line_distance_pct
            ]

    return sorted(
        lines,
        key=lambda line: (line.point_count, line.last_pivot_index),
        reverse=True,
    )


def _last_confirmed_resistance_index(line: KeyLine, end: int) -> int | None:
    return max((
        idx for idx in line.touch_indices
        if 0 <= idx < end and idx < line.resistance_confirmed_at.get(idx, -1) < end
    ), default=None)


def _serialize_resistance_point(df: pd.DataFrame, line: KeyLine, index: int) -> dict[str, Any]:
    confirmed = line.resistance_confirmed_at.get(index)
    return {
        "date": _date_str(df["trade_date"].iloc[index]),
        "price": round(float(df["high"].iloc[index]), 6),
        "price_basis": "high",
        "confirmed_date": (
            _date_str(df["trade_date"].iloc[confirmed])
            if confirmed is not None and 0 <= confirmed < len(df) else None
        ),
    }


def _check_prebreakout_extension(
    df: pd.DataFrame, line: KeyLine, breakout: dict[str, Any] | None, cfg: StrategyConfig,
) -> dict[str, Any]:
    """Check excursion and consecutive effective closes over the same pre-breakout path."""
    limit = cfg.max_prebreakout_extension_pct
    reject_days = cfg.prebreakout_reject_above_days
    buffer = cfg.prebreakout_above_buffer_pct
    enabled = limit is not None or reject_days is not None
    check: dict[str, Any] = {
        "status": "no_breakout" if enabled else "disabled",
        "passed": None, "limit_pct": None if limit is None else float(limit) * 100,
        "anchor_date": None, "anchor_confirmed_date": None, "anchor_price": None,
        "anchor_price_basis": "high", "breakout_date": None,
        "checked_bars": 0, "max_high": None, "max_high_date": None,
        "max_above_line_pct": None, "threshold_price": None,
        "extension_passed": None, "consecutive_passed": None,
        "reject_above_days": reject_days,
        "above_buffer_pct": float(buffer) * 100 if reject_days is not None else None,
        "effective_close_threshold": None, "max_consecutive_above_days": None,
        "longest_run_start_date": None, "longest_run_end_date": None,
    }
    if not enabled:
        return check
    if limit is not None and (not np.isfinite(limit) or not 0 <= limit <= 1):
        raise ValueError("max_prebreakout_extension_pct 必须为0～1或None")
    if reject_days is not None:
        if isinstance(reject_days, bool) or not isinstance(reject_days, (int, np.integer)) or not 1 <= reject_days <= 250:
            raise ValueError("prebreakout_reject_above_days 必须为1～250的整数或None")
        if not np.isfinite(buffer) or not 0 <= buffer <= 1:
            raise ValueError("prebreakout_above_buffer_pct 必须为0～1")
    if breakout is None:
        return check
    end = int(breakout["index"])
    check["breakout_date"] = _date_str(df["trade_date"].iloc[end])
    anchor = _last_confirmed_resistance_index(line, end)
    if anchor is None:
        check.update(status="no_prior_point", passed=False)
        return check
    point = _serialize_resistance_point(df, line, anchor)
    check["anchor_date"] = point["date"]
    check["anchor_confirmed_date"] = point["confirmed_date"]
    check["anchor_price"] = point["price"]
    # Neither the anchor candle nor the breakout candle is part of the intervening path.
    window = df.iloc[anchor + 1:end]
    check["checked_bars"] = len(window)
    if not np.isfinite(line.price) or line.price <= 0:
        check.update(status="invalid_data", passed=False)
        return check
    threshold = line.price * (1 + limit) if limit is not None else None
    close_threshold = line.price * (1 + buffer) if reject_days is not None else None
    check["threshold_price"] = round(float(threshold), 6) if threshold is not None else None
    check["effective_close_threshold"] = round(float(close_threshold), 6) if close_threshold is not None else None
    highs = window["high"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    if not np.isfinite(highs).all() or (reject_days is not None and not np.isfinite(closes).all()):
        check.update(status="invalid_data", passed=False)
        return check
    tolerance = max(1, abs(line.price)) * 1e-10
    if window.empty:
        check["max_above_line_pct"] = 0.0
        if limit is not None:
            check["extension_passed"] = True
    else:
        position = int(np.argmax(highs))
        highest = float(highs[position])
        check.update(
            max_high=round(highest, 6),
            max_high_date=_date_str(window["trade_date"].iloc[position]),
            max_above_line_pct=round(max(0.0, (highest / line.price - 1) * 100), 6),
        )
        if limit is not None:
            check["extension_passed"] = bool(
                highest <= threshold or np.isclose(highest, threshold, rtol=0, atol=tolerance)
            )
    if reject_days is not None:
        above = (closes > close_threshold) & ~np.isclose(closes, close_threshold, rtol=0, atol=tolerance)
        # Sentinel edges find every run, including one ending immediately before breakout.
        edges = np.diff(np.r_[False, above, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        longest = 0
        if len(starts):
            position = int(np.argmax(ends - starts))
            start, end = int(starts[position]), int(ends[position])
            longest = end - start
            check["longest_run_start_date"] = _date_str(window["trade_date"].iloc[start])
            check["longest_run_end_date"] = _date_str(window["trade_date"].iloc[end - 1])
        check["max_consecutive_above_days"] = longest
        check["consecutive_passed"] = longest < reject_days
    failed_extension = check["extension_passed"] is False
    failed_duration = check["consecutive_passed"] is False
    check["passed"] = not (failed_extension or failed_duration)
    check["status"] = (
        "extension_and_duration_exceeded" if failed_extension and failed_duration
        else "exceeded" if failed_extension
        else "duration_exceeded" if failed_duration
        else "passed"
    )
    return check


def _select_best_signal(
    stock_code: str,
    stock_name: str,
    df: pd.DataFrame,
    lines: list[KeyLine],
    cfg: StrategyConfig,
) -> SignalResult | None:
    best: SignalResult | None = None
    best_line: KeyLine | None = None
    breakout_label = "放量突破" if cfg.require_breakout_volume else "收盘突破"
    for line in lines:
        current_close = float(df["close"].iloc[-1])
        distance_pct = (current_close - line.price) / line.price * 100
        breakout = _find_breakout(df, line, cfg)
        anchor_index = _last_confirmed_resistance_index(line, int(breakout["index"]) if breakout else len(df))
        resistance_anchor = (
            {**_serialize_resistance_point(df, line, anchor_index),
             "role": "before_breakout" if breakout else "latest_confirmed"}
            if anchor_index is not None else None
        )
        prebreakout_check = _check_prebreakout_extension(df, line, breakout, cfg)
        pullback = _confirm_pullback(df, line.price, breakout, cfg) if breakout else None
        pullback_check = _pullback_check(df, breakout, pullback, cfg)
        breakout_age_ok = _breakout_age_ok(df, breakout, cfg)
        status = _breakout_status(df, line.price, breakout, cfg) if breakout else None
        trend_ok = _is_trend_positive(df, cfg)
        signal_strength = len([idx for idx in line.touch_indices if not breakout or idx < breakout["index"]])
        avg_volume = breakout.get("avg_volume") if breakout else None
        normal_volume_ratio = float(df["volume"].iloc[-1]) / avg_volume if avg_volume else None
        recent_strength: int | None = None
        recent_ok = True
        if cfg.max_event_age_bars is not None and breakout:
            cutoff = int(breakout["index"]) - cfg.max_event_age_bars
            recent_strength = len([idx for idx in line.touch_indices if cutoff <= idx < breakout["index"]])
            recent_ok = recent_strength >= min(cfg.min_line_strength, signal_strength)
        limit_up = bool(breakout) and is_limit_up_close(df, int(breakout["index"]), stock_code, stock_name)

        reasons: list[str] = [
            f"全历史累计 {line.point_count} 个精准点（阻力 {len(line.touch_indices)}、支撑 {len(line.support_indices)}）",
            f"精准线在本次突破前被独立波峰确认 {signal_strength} 次",
        ]
        risks: list[str] = []
        score = 20.0 + min(15.0, max(0, signal_strength - cfg.min_line_strength + 1) * 5.0)

        if breakout:
            score += 15.0
            if cfg.require_breakout_volume:
                score += min(10.0, max(0.0, breakout["volume_ratio"] - cfg.breakout_vol_mult) * 10.0)
                reasons.append(f"最近出现放量突破，量比约 {breakout['volume_ratio']:.2f} 倍")
            else:
                reasons.append("最近出现收盘突破，突破量比过滤已关闭")
        else:
            risks.append(f"尚未出现明确{breakout_label}")

        if limit_up:
            risks.append(LIMIT_UP_BREAKOUT_RISK)
        if pullback:
            score += 20.0
            reasons.append(_pullback_reason(pullback, cfg))
            reasons.append("突破后收盘未出现明显失守")
            if cfg.max_pullback_days is not None:
                reasons.append(
                    f"突破后第 {pullback_check['days_since_breakout']} 个交易日回踩确认"
                    f"（设定 {cfg.min_pullback_days}～{cfg.max_pullback_days} 个交易日）"
                )
            if cfg.pullback_max_normal_volume_ratio is None and normal_volume_ratio is not None and normal_volume_ratio > 1:
                risks.append(f"回踩量仍为突破前 {cfg.vol_lookback} 日均量的 {normal_volume_ratio:.2f} 倍，并非低于常态量")
        elif breakout:
            risks.append("突破后尚未完成缩量回踩确认")
            risks.extend(_pullback_rule_risks(df, line.price, breakout, cfg))
            if float(df["low"].iloc[-1]) < line.price * (1 - cfg.max_intraday_undercut_pct):
                risks.append("信号日盘中跌破精准线过深")
        if not recent_ok:
            risks.append(
                f"突破前 {cfg.max_event_age_bars} 个交易日内阻力确认仅 {recent_strength} 次"
                f"（需 {min(cfg.min_line_strength, signal_strength)} 次），更早的证据不升级买入预警"
            )

        if trend_ok:
            score += 15.0
            reasons.append(f"收盘站上 MA{cfg.ma_period}，且均线保持向上")
        else:
            risks.append("短期趋势尚未走强")

        if prebreakout_check["passed"] is True:
            if prebreakout_check["extension_passed"] is True:
                reasons.append(
                    f"阻力点 {prebreakout_check['anchor_date']} 之后至突破前，"
                    f"最高越线 {prebreakout_check['max_above_line_pct']:.2f}%"
                    f"（上限 {prebreakout_check['limit_pct']:g}%）"
                )
            if prebreakout_check["consecutive_passed"] is True:
                reasons.append(
                    f"同期最长连续 {prebreakout_check['max_consecutive_above_days']} 个交易日有效收在线上"
                    f"（收盘高于线价 {prebreakout_check['above_buffer_pct']:g}%，"
                    f"未达 {prebreakout_check['reject_above_days']} 日排除阈值）"
                )

        if distance_pct < -cfg.allow_undercut_pct * 100:
            risks.append("收盘价跌回关键线下方")
        elif distance_pct > cfg.max_extension_pct * 100:
            risks.append("当前价格距离关键线偏远")
        else:
            score += 10.0

        buy_alert = (
            breakout
            and pullback
            and trend_ok
            and recent_ok
            and distance_pct >= -cfg.allow_undercut_pct * 100
            and distance_pct <= cfg.max_extension_pct * 100
        )
        near_line = -cfg.watch_below_pct * 100 <= distance_pct <= cfg.watch_near_pct * 100
        breakout_not_failed = not breakout or distance_pct >= -cfg.max_intervening_close_break_pct * 100

        if prebreakout_check["passed"] is False:
            category = SignalCategory.NONE
            violations = []
            if prebreakout_check["extension_passed"] is False:
                violations.append(
                    f"上次阻力点至本次突破前最高越线 "
                    f"{prebreakout_check['max_above_line_pct']:.2f}%，"
                    f"超过 {prebreakout_check['limit_pct']:g}% 上限"
                )
            if prebreakout_check["consecutive_passed"] is False:
                violations.append(
                    f"同期曾连续 {prebreakout_check['max_consecutive_above_days']} 个交易日"
                    f"收盘高于线价 {prebreakout_check['above_buffer_pct']:g}%，"
                    f"达到 {prebreakout_check['reject_above_days']} 日排除阈值"
                )
            suggestion = "暂不行动：" + ("；".join(violations) if violations else
                "缺少突破前已确认的阻力触点或有效价格，无法验证突破前路径")
            risks.append(suggestion)
        elif cfg.require_breakout and not breakout:
            category = SignalCategory.NONE
            suggestion = f"暂不行动：尚未出现{breakout_label}（已开启「只要已突破」）"
        elif not breakout_age_ok:
            # 人工设定的"距突破天数"窗口没满足，整只排除（不降级为观察机会）
            category = SignalCategory.NONE
            age = len(df) - 1 - int(breakout["index"])
            suggestion = f"暂不行动：距突破已 {age} 个交易日，不在设定的 {cfg.min_days_since_breakout}~{cfg.max_days_since_breakout} 天窗口内"
            risks.append(suggestion)
        elif cfg.require_pullback and not pullback:
            category = SignalCategory.NONE
            if breakout is None:
                suggestion = f"暂不行动：尚未出现{breakout_label}，不能确认突破后回踩"
            elif pullback_check["status"] == "outside_window":
                suggestion = (
                    f"暂不行动：距突破 {pullback_check['days_since_breakout']} 个交易日，"
                    f"不在回调天数 {pullback_check['min_days']}～{pullback_check['max_days']} 范围内"
                )
            else:
                suggestion = "暂不行动：信号日未完成缩量回踩确认（已开启「只选突破后回踩」）"
            risks.append(suggestion)
        elif buy_alert:
            category = SignalCategory.BUY_ALERT
            suggestion = f"买入预警：{breakout_label}后缩量回踩确认，仅作为技术提醒"
        elif near_line and breakout_not_failed:
            category = SignalCategory.WATCH
            suggestion = "观察机会：接近或站上关键线，等待突破回踩确认"
            if not breakout:
                reasons.append("价格接近关键精准线，适合加入观察")
        else:
            category = SignalCategory.NONE
            suggestion = "暂不行动：未形成可跟踪结构"

        result = SignalResult(
            stock_code=stock_code,
            stock_name=stock_name,
            category=category,
            score=max(0.0, min(95.0, score)),
            structure_score=max(0.0, min(95.0, score)),
            suggestion=suggestion,
            key_line_price=round(line.price, 6 if cfg.precision_engine == "pivot_fixed" else 3),
            line_strength=line.point_count,
            signal_date=_date_str(df["trade_date"].iloc[-1]),
            line_formation_date=_date_str(df["trade_date"].iloc[line.formation_index]),
            breakout_date=_date_str(df["trade_date"].iloc[breakout["index"]]) if breakout else None,
            pullback_date=_date_str(df["trade_date"].iloc[pullback["index"]]) if pullback else None,
            current_price=round(current_close, 3),
            distance_to_line_pct=round(distance_pct, 2),
            breakout_status=status,
            breakout_volume_ratio=round(breakout["volume_ratio"], 2) if breakout and breakout["volume_ratio"] is not None else None,
            require_breakout_volume=cfg.require_breakout_volume,
            breakout_volume_threshold=cfg.breakout_vol_mult if cfg.require_breakout_volume else None,
            pullback_volume_ratio=round(pullback["volume_ratio"], 2) if pullback else None,
            pullback_normal_volume_ratio=round(normal_volume_ratio, 3) if normal_volume_ratio is not None else None,
            reasons=reasons,
            risks=risks,
            chart={},
            prebreakout_check=prebreakout_check,
            pullback_check=pullback_check,
            resistance_anchor=resistance_anchor,
            precision_engine=cfg.precision_engine,
            breakout_limit_up=limit_up if breakout else None,
        )
        if best is None or _rank(result) > _rank(best):
            best = result
            best_line = line

    if best is None or best_line is None:
        return None
    if cfg.include_chart:
        display_lines = _select_display_lines(lines, best_line, float(df["close"].iloc[-1]), cfg)
        best.key_lines = [
            _serialize_key_line(df, line, line is best_line, float(df["close"].iloc[-1]),
                                precision=6 if cfg.precision_engine == "pivot_fixed" else 3)
            for line in display_lines
        ]
        best.chart = _build_chart_payload(df, best)
    return best


def _breakout_age_ok(df: pd.DataFrame, breakout: dict[str, Any] | None, cfg: StrategyConfig) -> bool:
    """突破距今的交易日数是否落在人工设定的窗口内。

    没有突破的股票不受这个条件约束——它说的是"距突破多少天"，
    对还没突破的股票无从谈起；否则一开启就会把观察机会整类清空。
    """
    if breakout is None:
        return True
    age = len(df) - 1 - int(breakout["index"])
    if cfg.min_days_since_breakout > 0 and age < cfg.min_days_since_breakout:
        return False
    if cfg.max_days_since_breakout > 0 and age > cfg.max_days_since_breakout:
        return False
    return True


def _find_breakout(df: pd.DataFrame, line: KeyLine, cfg: StrategyConfig) -> dict[str, Any] | None:
    signal_idx = len(df) - 1
    # Include age == maximum: the breakout candle itself occupies one extra row.
    window = cfg.max_pullback_days + 1 if cfg.max_pullback_days is not None else cfg.breakout_window
    start = max(line.formation_index + 1, signal_idx - window + 1, 1)
    volumes = df["volume"].astype(float)
    for idx in range(signal_idx, start - 1, -1):
        prev_close = float(df["close"].iloc[idx - 1])
        close = float(df["close"].iloc[idx])
        if prev_close > line.price * (1 + cfg.breakout_eps_pct):
            continue
        if close < line.price * (1 + cfg.breakout_eps_pct):
            continue
        vol_start = max(0, idx - cfg.vol_lookback)
        history = volumes.iloc[vol_start:idx]
        volume = float(volumes.iloc[idx])
        if not np.isfinite(volume) or volume < 0:
            continue
        avg_volume = float(history.mean()) if len(history) >= min(5, cfg.vol_lookback) else None
        if avg_volume is not None and (not np.isfinite(avg_volume) or avg_volume <= 0):
            avg_volume = None
        volume_ratio = volume / avg_volume if avg_volume is not None else None
        if volume_ratio is not None and not np.isfinite(volume_ratio):
            volume_ratio = None
        if cfg.require_breakout_volume and (volume_ratio is None or volume_ratio < cfg.breakout_vol_mult):
            continue
        return {"index": idx, "volume_ratio": volume_ratio, "volume": volume, "avg_volume": avg_volume}
    return None


def _breakout_status(
    df: pd.DataFrame,
    line_price: float,
    breakout: dict[str, Any],
    cfg: StrategyConfig,
) -> dict[str, Any]:
    """突破之后发生了什么：回踩了没、缩量还是放量、有没有跌破。

    与 `_confirm_pullback` 的分工：那个只判定"信号日当天是否正在回踩"，
    用于给出买入预警；这里扫描突破之后的整段行情，产出页面上的状态标记，
    让"已突破但还没回踩""回踩过又跌破了"这些中间状态也能看见。
    """
    breakout_idx = int(breakout["index"])
    breakout_volume = float(breakout["volume"])
    last = len(df) - 1
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float)

    touch_lo = line_price * (1 - cfg.max_intraday_undercut_pct)
    touch_hi = line_price * (1 + cfg.pullback_touch_pct)
    hold_level = line_price * (1 - cfg.allow_undercut_pct)
    break_level = line_price * (1 - cfg.max_intervening_close_break_pct)

    average = breakout.get("avg_volume")
    pullback_day: int | None = None
    pullback_ratio: float | None = None
    pullback_normal_ratio: float | None = None
    breakdown_day: int | None = None
    for i in range(breakout_idx + 1, last + 1):
        day = i - breakout_idx
        if breakdown_day is None and closes[i] < break_level:
            breakdown_day = day
        if (
            pullback_day is None
            and touch_lo <= lows[i] <= touch_hi
            and closes[i] >= hold_level
        ):
            pullback_day = day
            # 转成原生 float/bool：numpy 标量会一路漏进 JSON 序列化
            pullback_ratio = float(volumes[i] / breakout_volume) if breakout_volume > 0 else None
            pullback_normal_ratio = float(volumes[i] / average) if average else None

    shrinking: bool | None = None
    if pullback_ratio is not None:
        shrinking = bool(pullback_ratio <= cfg.pullback_vol_max_ratio)
        # 与买入判定同一口径：启用常态量规则后，高于常态量的回踩不再标成「缩量」。
        if cfg.pullback_max_normal_volume_ratio is not None:
            shrinking = shrinking and pullback_normal_ratio is not None and (
                pullback_normal_ratio <= cfg.pullback_max_normal_volume_ratio + 1e-12
            )

    marks: list[tuple[int, str]] = []
    if pullback_day is not None:
        kind = "缩量回踩" if shrinking else "放量回踩"
        marks.append((pullback_day, f"{kind}（第{pullback_day}天）"))
    if breakdown_day is not None:
        marks.append((breakdown_day, f"跌破突破位（第{breakdown_day}天）"))
    marks.sort(key=lambda item: item[0])
    tags = [text for _day, text in marks] or ["未回踩"]

    after = highs[breakout_idx + 1 : last + 1]
    max_gain = float(after.max() / line_price - 1) * 100 if len(after) else 0.0
    return {
        "days_since_breakout": last - breakout_idx,
        "pullback_day": pullback_day,
        "pullback_volume_ratio": round(pullback_ratio, 2) if pullback_ratio is not None else None,
        "pullback_normal_volume_ratio": round(pullback_normal_ratio, 2) if pullback_normal_ratio is not None else None,
        "pullback_is_shrinking": shrinking,
        "breakdown_day": breakdown_day,
        "max_gain_since_breakout_pct": round(max_gain, 2),
        "tags": tags,
    }


def _pullback_check(
    df: pd.DataFrame, breakout: dict[str, Any] | None, pullback: dict[str, Any] | None, cfg: StrategyConfig,
) -> dict[str, Any]:
    maximum = cfg.max_pullback_days if cfg.max_pullback_days is not None else cfg.pullback_window
    age = len(df) - 1 - int(breakout["index"]) if breakout is not None else None
    status = (
        "no_breakout" if age is None
        else "outside_window" if not cfg.min_pullback_days <= age <= maximum
        else "confirmed" if pullback is not None
        else "awaiting_pullback"
    )
    return {
        "required": cfg.require_pullback,
        "range_mode": "legacy" if cfg.max_pullback_days is None else "configured",
        "min_days": cfg.min_pullback_days, "max_days": maximum,
        "days_since_breakout": age,
        "status": status, "passed": pullback is not None,
    }


PULLBACK_CONDITIONS = (
    "in_window", "touched", "recovered", "not_extended", "shrink_vs_breakout",
    "held_line", "normal_volume_ok", "runup_ok",
)


def _pullback_conditions(df: pd.DataFrame, line_price: float, breakout: dict[str, Any], cfg: StrategyConfig) -> dict[str, Any]:
    """信号日是否构成回踩的每一项条件；只用截至信号日的数据。"""
    idx = len(df) - 1
    breakout_idx = int(breakout["index"])
    maximum = cfg.max_pullback_days if cfg.max_pullback_days is not None else cfg.pullback_window
    low = float(df["low"].iloc[idx])
    close = float(df["close"].iloc[idx])
    volume = float(df["volume"].iloc[idx])
    breakout_volume = float(breakout["volume"])
    average = breakout.get("avg_volume")
    average = float(average) if average is not None and np.isfinite(average) and average > 0 else None
    intermediate_closes = df["close"].iloc[breakout_idx + 1 : idx].astype(float)
    # 突破日本身也计入：涨停突破当天就可能远离线价。
    peak = float(df["high"].iloc[breakout_idx : idx + 1].astype(float).max())
    normal_ratio = volume / average if average is not None else None
    limit = cfg.pullback_max_normal_volume_ratio
    runup_limit = cfg.max_post_breakout_runup_pct
    return {
        "in_window": cfg.min_pullback_days <= idx - breakout_idx <= maximum,
        "touched": line_price * (1 - cfg.max_intraday_undercut_pct) <= low <= line_price * (1 + cfg.pullback_touch_pct),
        "recovered": close >= line_price * (1 - cfg.allow_undercut_pct),
        "not_extended": close <= line_price * (1 + cfg.max_extension_pct),
        "shrink_vs_breakout": breakout_volume > 0 and volume <= breakout_volume * cfg.pullback_vol_max_ratio,
        "held_line": intermediate_closes.empty or bool(
            (intermediate_closes >= line_price * (1 - cfg.max_intervening_close_break_pct)).all()
        ),
        # 无法核验常态量（历史太短）时不放行：规则启用后宁缺毋滥。
        "normal_volume_ok": limit is None or (normal_ratio is not None and normal_ratio <= limit + 1e-12),
        "runup_ok": runup_limit is None or peak <= line_price * (1 + runup_limit) * (1 + 1e-12),
        "volume_ratio": volume / breakout_volume if breakout_volume > 0 else None,
        "normal_volume_ratio": normal_ratio,
        "max_runup_pct": (peak / line_price - 1) * 100 if line_price > 0 else None,
    }


def _confirm_pullback(df: pd.DataFrame, line_price: float, breakout: dict[str, Any], cfg: StrategyConfig) -> dict[str, Any] | None:
    conditions = _pullback_conditions(df, line_price, breakout, cfg)
    if not all(conditions[name] for name in PULLBACK_CONDITIONS):
        return None
    return {
        "index": len(df) - 1,
        "volume_ratio": conditions["volume_ratio"],
        "normal_volume_ratio": conditions["normal_volume_ratio"],
        "max_runup_pct": conditions["max_runup_pct"],
    }


def _pullback_reason(pullback: dict[str, Any], cfg: StrategyConfig) -> str:
    text = f"缩量回踩确认（相对突破日），回踩量约为突破量的 {pullback['volume_ratio']:.2f}"
    if cfg.pullback_max_normal_volume_ratio is not None and pullback.get("normal_volume_ratio") is not None:
        text += f"、突破前 {cfg.vol_lookback} 日均量的 {pullback['normal_volume_ratio']:.2f} 倍"
    return text


def _pullback_rule_risks(df: pd.DataFrame, line_price: float, breakout: dict[str, Any] | None, cfg: StrategyConfig) -> list[str]:
    """价格已回到线附近、只因量能或冲高回落规则未通过时，说清楚原因。"""
    if breakout is None:
        return []
    conditions = _pullback_conditions(df, line_price, breakout, cfg)
    positional = ("in_window", "touched", "recovered", "not_extended", "held_line")
    if not all(conditions[name] for name in positional):
        return []
    risks = []
    if not conditions["normal_volume_ok"]:
        ratio = conditions["normal_volume_ratio"]
        risks.append(
            f"回踩到位但成交量为突破前 {cfg.vol_lookback} 日均量的 {ratio:.2f} 倍"
            f"（上限 {cfg.pullback_max_normal_volume_ratio:g} 倍），不算缩量回踩"
            if ratio is not None else "突破前均量不足以核验回踩是否缩量，不发买入预警"
        )
    if not conditions["runup_ok"]:
        risks.append(
            f"突破后最高越线 {conditions['max_runup_pct']:.1f}%（上限 {cfg.max_post_breakout_runup_pct * 100:g}%）"
            "后回到线附近，属冲高回落，不按回踩预警"
        )
    return risks


def _is_trend_positive(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    if len(df) < cfg.ma_slope_days + 1:
        return False
    close = float(df["close"].iloc[-1])
    ma_now = float(df["ma"].iloc[-1])
    ma_before = float(df["ma"].iloc[-1 - cfg.ma_slope_days])
    if np.isnan(ma_now) or np.isnan(ma_before):
        return False
    return close >= ma_now and ma_now >= ma_before


def _select_display_lines(
    lines: list[KeyLine],
    primary: KeyLine,
    current_close: float,
    cfg: StrategyConfig,
) -> list[KeyLine]:
    remaining = [line for line in lines if line is not primary]
    nearest = sorted(
        remaining,
        key=lambda line: (
            abs(line.price - current_close) / current_close,
            -line.point_count,
            -line.last_pivot_index,
        )
    )
    slots = max(0, cfg.max_display_lines - 1)
    nearest_slots = min(3, slots)
    selected = nearest[:nearest_slots]
    selected_ids = {id(line) for line in selected}
    nearby = [
        line
        for line in remaining
        if id(line) not in selected_ids and abs(line.price - current_close) / current_close <= 0.15
    ]
    strength_ranked = sorted(
        nearby,
        key=lambda line: (
            -line.point_count,
            abs(line.price - current_close) / current_close,
            -line.last_pivot_index,
        ),
    )
    selected.extend(strength_ranked[: slots - len(selected)])
    if len(selected) < slots:
        selected_ids = {id(line) for line in selected}
        selected.extend(
            line for line in nearest if id(line) not in selected_ids
        )
        selected = selected[:slots]
    return [primary, *selected]


def _serialize_key_line(
    df: pd.DataFrame,
    line: KeyLine,
    is_primary: bool,
    current_close: float,
    precision: int = 3,
) -> dict[str, Any]:
    resistance_dates = [_date_str(df["trade_date"].iloc[idx]) for idx in line.touch_indices]
    support_dates = [_date_str(df["trade_date"].iloc[idx]) for idx in line.support_indices]
    confirmations = [(idx, "resistance") for idx in line.touch_indices]
    confirmations.extend((idx, "support") for idx in line.support_indices)
    confirmations.sort(key=lambda item: item[0])
    return {
        "price": round(line.price, precision),
        "point_count": line.point_count,
        "resistance_point_count": len(line.touch_indices),
        "support_point_count": len(line.support_indices),
        "resistance_dates": resistance_dates,
        "resistance_price_basis": "high",
        "resistance_points": [_serialize_resistance_point(df, line, idx) for idx in line.touch_indices],
        "support_dates": support_dates,
        "support_price_basis": "low",
        "support_points": [{
            "date": _date_str(df["trade_date"].iloc[idx]), "price": round(float(df["low"].iloc[idx]), 6),
            "price_basis": "low",
            "confirmed_date": _date_str(df["trade_date"].iloc[line.support_confirmed_at[idx]]) if idx in line.support_confirmed_at else None,
        } for idx in line.support_indices],
        "touch_dates": [_date_str(df["trade_date"].iloc[idx]) for idx, _kind in confirmations],
        "formation_date": _date_str(df["trade_date"].iloc[line.formation_index]),
        "last_pivot_date": _date_str(df["trade_date"].iloc[line.last_pivot_index]),
        "line_type": line.line_type,
        "is_primary": is_primary,
        "history_bars": len(df),
        "distance_to_current_pct": round((current_close - line.price) / line.price * 100, 2),
    }


def _build_chart_payload(df: pd.DataFrame, result: SignalResult) -> dict[str, Any]:
    compact = df.tail(140).reset_index(drop=True)
    bars = [
        {
            "date": _date_str(row.trade_date),
            "open": round(float(row.open), 3),
            "close": round(float(row.close), 3),
            "low": round(float(row.low), 3),
            "high": round(float(row.high), 3),
            "volume": float(row.volume),
        }
        for row in compact.itertuples(index=False)
    ]
    markers: list[dict[str, Any]] = []
    compact_dates = set(compact["trade_date"].map(_date_str))
    if result.breakout_date and result.breakout_date in compact_dates:
        markers.append({"date": result.breakout_date, "type": "breakout", "label": "突破"})
    if result.pullback_date and result.pullback_date in compact_dates:
        markers.append({"date": result.pullback_date, "type": "pullback", "label": "回踩"})
    return {
        "bars": bars,
        "key_line": result.key_line_price,
        "key_lines": result.key_lines,
        "markers": markers,
        "flag_pattern": result.flag_pattern,
    }


def _rank(result: SignalResult) -> tuple[int, float]:
    category_rank = {
        SignalCategory.BUY_ALERT: 2,
        SignalCategory.WATCH: 1,
        SignalCategory.NONE: 0,
    }[result.category]
    return category_rank, result.score


def _date_str(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")
