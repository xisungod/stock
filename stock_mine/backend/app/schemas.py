from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.strategy import MAX_PULLBACK_DAYS, validate_pullback_days

MAX_BREAKOUT_AGE = 9
MIN_BACKTEST_LOOKBACK = 120
MAX_BACKTEST_LOOKBACK = 1500
StrategyMode = Literal["combined", "precision_pullback", "flag_breakout", "triangle_breakout"]
FlagDirection = Literal["all", "descending", "horizontal", "ascending"]
PrecisionEngine = Literal["legacy", "pivot_fixed", "platform_events"]
ProviderName = Literal["auto", "adata", "akshare", "sample", "demo", "fake"]
MarketFilter = Literal["off", "risk", "downgrade"]


class RuleDefaults(BaseModel):
    """2026-11-27 逻辑修复引入的规则；扫描与回测共用同一组默认值。

    这些阈值是事前按规则含义选定的，没有按收益调参，也尚未经过冻结后的前向验证。
    """
    # 回踩量 ≤ 突破前10日均量：「缩量」的字面含义，而不只是低于突破日的放量。
    pullback_max_normal_volume_ratio: float | None = Field(default=1.0, gt=0, le=5)
    # 突破后最高价越线超过 10%（主板一个涨停幅度，约为回踩区间上沿 3.5% 的 3 倍）再回到线附近，视为冲高回落。
    max_post_breakout_runup_pct: float | None = Field(default=0.10, gt=0, le=1)
    # 突破前约 3 年（750 个交易日）内须有足够独立验证；更早的前复权价位只展示不升级预警。
    max_event_age_bars: int | None = Field(default=750, ge=20, le=5000)
    # 大盘环境默认只标注（risk）。修正行号 bug 后的真实行情验证（reports/logic_fixes_20261127/validation）中，
    # 「修复后规则 + 偏弱不买」留出段 95% 下限 < 0，没有通过预设标准，因此不默认降级；
    # downgrade 偏弱时买入预警降为观察，off 不评估。
    market_filter: MarketFilter = "risk"


def validate_backtest_lookback(value: int) -> int:
    if value == 0 or MIN_BACKTEST_LOOKBACK <= value <= MAX_BACKTEST_LOOKBACK:
        return value
    raise ValueError(
        f"lookback_days 必须为 0（全部历史）或 {MIN_BACKTEST_LOOKBACK}～{MAX_BACKTEST_LOOKBACK}"
    )


class ScanRequest(RuleDefaults):
    provider: ProviderName = "auto"
    # 0 = 扫描全部股票，与 /api/config/defaults 广告的默认值保持一致
    limit: int = Field(default=0, ge=0, le=6000)
    # 0 = 最近一个已收盘交易日（盘中未收盘的K线自动去掉），与回测同一根信号K线；
    # N>=1 = 旧版字面含义「去掉最后 N 根」。
    signal_offset: int = Field(default=0, ge=0, le=5)
    # 旧版距突破窗口保留兼容；新页面使用 min/max_pullback_days。
    min_days_since_breakout: int = Field(default=0, ge=0, le=MAX_BREAKOUT_AGE)
    max_days_since_breakout: int = Field(default=0, ge=0, le=MAX_BREAKOUT_AGE)
    require_breakout: bool = Field(default=False)
    require_pullback: bool = True
    min_pullback_days: int = Field(default=1, ge=1, le=MAX_PULLBACK_DAYS, strict=True)
    max_pullback_days: int | None = Field(default=10, ge=1, le=MAX_PULLBACK_DAYS, strict=True)
    # 突破日量比门槛：当日成交量 / 前 10 日均量
    breakout_vol_mult: float = Field(default=1.5, ge=1.0, le=5.0)
    require_breakout_volume: bool = False
    stock_codes: list[str] | None = None
    strategy_mode: StrategyMode = "combined"
    precision_engine: PrecisionEngine = "platform_events"
    flag_direction: FlagDirection = "all"
    max_prebreakout_extension_pct: float | None = Field(default=0.50, ge=0, le=1)
    prebreakout_reject_above_days: int | None = Field(default=30, ge=1, le=250)
    prebreakout_above_buffer_pct: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def validate_breakout_window(self):
        if self.strategy_mode == "flag_breakout" and (self.min_days_since_breakout or self.max_days_since_breakout):
            raise ValueError("旗形突破不使用精准线的距突破天数窗口，请将上下限设为0")
        if self.max_days_since_breakout > 0 and self.min_days_since_breakout > self.max_days_since_breakout:
            raise ValueError("max_days_since_breakout 必须为 0，或不小于 min_days_since_breakout")
        validate_pullback_days(
            self.min_pullback_days, self.max_pullback_days,
            self.min_days_since_breakout, self.max_days_since_breakout,
        )
        return self


class BacktestRequest(RuleDefaults):
    provider: ProviderName = "auto"
    limit: int = Field(default=0, ge=0, le=6000)
    lookback_days: int = Field(default=500, ge=0, le=MAX_BACKTEST_LOOKBACK)
    stock_codes: list[str] | None = None
    scan_id: int | None = Field(default=None, ge=1)
    strategy_mode: StrategyMode = "combined"
    precision_engine: PrecisionEngine = "platform_events"
    flag_direction: FlagDirection = "all"
    breakout_vol_mult: float = Field(default=1.5, ge=1.0, le=5.0)
    require_breakout_volume: bool = False
    max_prebreakout_extension_pct: float | None = Field(default=0.50, ge=0, le=1)
    prebreakout_reject_above_days: int | None = Field(default=30, ge=1, le=250)
    prebreakout_above_buffer_pct: float = Field(default=0.02, ge=0, le=1)
    require_pullback: bool = True
    min_pullback_days: int = Field(default=1, ge=1, le=MAX_PULLBACK_DAYS, strict=True)
    max_pullback_days: int | None = Field(default=10, ge=1, le=MAX_PULLBACK_DAYS, strict=True)

    @model_validator(mode="after")
    def validate_pullback_window(self):
        validate_pullback_days(self.min_pullback_days, self.max_pullback_days)
        return self

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback_days(cls, value: int) -> int:
        return validate_backtest_lookback(value)


class WatchCreate(BaseModel):
    stock_code: str
    stock_name: str = ""
    note: str = ""
    status: str = "active"
    entry_price: float | None = Field(default=None, gt=0)
    entry_date: str | None = None


class WatchBulkCreate(BaseModel):
    items: list[WatchCreate] = Field(min_length=1)


class WatchBulkDelete(BaseModel):
    stock_codes: list[str] = Field(min_length=1)


class WatchUpdate(BaseModel):
    note: str | None = None
    status: str | None = None


def production_strategy_config(**overrides):
    """与新建扫描完全相同的策略配置；研究脚本必须用它，避免验证的不是线上逻辑。

    overrides 可改 strategy_mode / include_chart / confirmed_event_count / min_line_strength
    以及任何 ScanRequest 字段；回测内部总会把 signal_offset 置 0。
    """
    from app.services.strategy import build_strategy_config

    return build_strategy_config({**ScanRequest().model_dump(), **overrides})
