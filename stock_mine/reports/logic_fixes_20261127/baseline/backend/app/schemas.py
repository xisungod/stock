from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.strategy import MAX_PULLBACK_DAYS, validate_pullback_days

MAX_BREAKOUT_AGE = 9
MIN_BACKTEST_LOOKBACK = 120
MAX_BACKTEST_LOOKBACK = 1500
StrategyMode = Literal["combined", "precision_pullback", "flag_breakout"]
FlagDirection = Literal["all", "descending", "horizontal", "ascending"]
PrecisionEngine = Literal["legacy", "pivot_fixed", "platform_events"]
ProviderName = Literal["auto", "adata", "akshare", "sample", "demo", "fake"]


def validate_backtest_lookback(value: int) -> int:
    if value == 0 or MIN_BACKTEST_LOOKBACK <= value <= MAX_BACKTEST_LOOKBACK:
        return value
    raise ValueError(
        f"lookback_days 必须为 0（全部历史）或 {MIN_BACKTEST_LOOKBACK}～{MAX_BACKTEST_LOOKBACK}"
    )


class ScanRequest(BaseModel):
    provider: ProviderName = "auto"
    # 0 = 扫描全部股票，与 /api/config/defaults 广告的默认值保持一致
    limit: int = Field(default=0, ge=0, le=6000)
    signal_offset: int = Field(default=1, ge=0, le=5)
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


class BacktestRequest(BaseModel):
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
