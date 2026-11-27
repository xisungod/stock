"""Causal platform events, bounded line versions, and breakout lifecycles.

This is separate from the large-scale ZigZag. A bar is processed against yesterday's
line versions before today's newly confirmed evidence may create another version.
"""
from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import numpy as np
import pandas as pd

from app.services.price_bands import PriceBand, bounded_band


@dataclass(frozen=True)
class ValidationEvent:
    kind: str
    pivot: int
    confirmed: int
    start: int
    price: float
    atr: float
    max_high: float
    min_low: float
    origins: tuple[str, ...] = ("local_turn",)
    episode_id: str | None = None
    local_confirmed: int | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.pivot}"


class LocalTurnDetector:
    """Independent directional tests; rearming requires a real price excursion."""

    def __init__(self, kind: str):
        self.kind = kind
        self.floor = float("inf")
        self.floor_index = 0
        self.floor_atr = 0.0
        self.peak = -float("inf")
        self.peak_index = 0
        self.peak_atr = 0.0
        self.start = 0
        self.max_high = -float("inf")
        self.min_low = float("inf")
        self.armed = False

    def feed(self, index: int, high: float, low: float, close: float, atr: float, cfg) -> ValidationEvent | None:
        upper, lower, value = (high, low, close) if self.kind == "resistance" else (-low, -high, -close)
        if not self.armed:
            if lower <= self.floor:
                self.floor, self.floor_index, self.floor_atr = lower, index, atr
                self.peak, self.peak_index, self.peak_atr = upper, index, atr
                self.max_high, self.min_low = high, low
            elif upper >= self.peak:
                self.peak, self.peak_index, self.peak_atr = upper, index, atr
            self.max_high, self.min_low = max(self.max_high, high), min(self.min_low, low)
            distance = max(self.floor_atr * cfg.event_reversal_atr, abs(self.floor) * cfg.event_min_reversal_pct)
            if self.peak - self.floor >= distance:
                self.armed, self.start = True, self.floor_index
            return None

        self.max_high, self.min_low = max(self.max_high, high), min(self.min_low, low)
        if upper > self.peak:
            self.peak, self.peak_index, self.peak_atr = upper, index, atr
        elif upper == self.peak:
            self.peak_index = index
        distance = max(self.peak_atr * cfg.event_reversal_atr, abs(self.peak) * cfg.event_min_reversal_pct)
        if index <= self.peak_index or self.peak - value < distance:
            return None
        event = ValidationEvent(
            self.kind, self.peak_index, index, self.start, abs(self.peak),
            self.peak_atr, self.max_high, self.min_low,
        )
        self.armed = False
        self.floor, self.floor_index, self.floor_atr = lower, index, atr
        self.peak, self.peak_index, self.peak_atr = upper, index, atr
        self.max_high, self.min_low = high, low
        return event


@dataclass
class Probe:
    kind: str
    start: int
    pivot: int
    price: float
    distance: float
    atr: float
    high: float
    low: float
    band: PriceBand
    id: str = ""
    end: int | None = None
    status: str = "pending"
    local_event_ids: set[str] = field(default_factory=set)


@dataclass
class Observation:
    snapshot: dict[str, Any]
    started: int
    breakout: dict[str, Any] | None = None
    precheck: dict[str, Any] | None = None


@dataclass
class Platform:
    id: str
    events: dict[str, ValidationEvent]
    band: PriceBand
    version: int = 0
    epoch: int = 0
    born: int | None = None
    confirmed: int | None = None
    invalidated: int | None = None
    observation: Observation | None = None
    probe: Probe | None = None
    episodes: list[Probe] = field(default_factory=list)
    episode_starts: list[int] = field(default_factory=list)
    armed_side: str | None = None
    can_start: bool = True
    snapshot: dict[str, Any] | None = None
    versions: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    lifetime_events: set[str] = field(default_factory=set)
    last_state: str = "candidate"


class PlatformEngine:
    def __init__(self, code: str, name: str, frame: pd.DataFrame, cfg):
        self.code, self.name, self.frame, self.cfg = code, name, frame, cfg
        self.dates = pd.to_datetime(frame.trade_date).dt.strftime("%Y-%m-%d").tolist()
        self.high = frame.high.to_numpy(dtype=float)
        self.low = frame.low.to_numpy(dtype=float)
        self.close = frame.close.to_numpy(dtype=float)
        self.volume = frame.volume.to_numpy(dtype=float)
        previous = np.r_[self.close[0], self.close[:-1]]
        tr = np.maximum(self.high - self.low, np.maximum(abs(self.high - previous), abs(self.low - previous)))
        self.atr = pd.Series(tr).rolling(20, min_periods=1).mean().to_numpy()
        self.ma = frame.close.rolling(cfg.ma_period, min_periods=min(cfg.ma_period, max(5, cfg.ma_period // 2))).mean().to_numpy()
        self.detectors = [LocalTurnDetector("resistance"), LocalTurnDetector("support")]
        self.events: dict[str, ValidationEvent] = {}
        self.platforms: list[Platform] = []
        self.index = -1

    def event_dict(self, event: ValidationEvent) -> dict[str, Any]:
        return {
            "event_id": f"{self.code}:{event.key}", "source_candle_id": f"{self.code}:{self.dates[event.pivot]}",
            "date": self.dates[event.pivot], "price": round(event.price, 6),
            "kind": event.kind, "price_basis": "high" if event.kind == "resistance" else "low",
            "confirmed_date": self.dates[event.confirmed], "event_start_date": self.dates[event.start],
            "max_high": round(event.max_high, 6), "min_low": round(event.min_low, 6),
            "atr_at_pivot": round(event.atr, 6), "origins": list(event.origins),
            "pivot_index": event.pivot, "confirmed_index": event.confirmed, "start_index": event.start,
            "episode_id": event.episode_id,
            "local_confirmed_date": self.dates[event.local_confirmed] if event.local_confirmed is not None else None,
        }

    def _snapshot(self, platform: Platform, index: int):
        events = sorted(platform.events.values(), key=lambda item: (item.pivot, item.kind))
        if len(events) < self.cfg.candidate_event_count:
            platform.snapshot = None
            return
        if platform.born is None:
            platform.born = index
        if platform.confirmed is None and len(events) >= self.cfg.confirmed_event_count:
            platform.confirmed = index
        platform.version += 1
        resistance = [self.event_dict(item) for item in events if item.kind == "resistance"]
        support = [self.event_dict(item) for item in events if item.kind == "support"]
        payload = {
            "line_id": platform.id, "version_id": f"{platform.id}:e{platform.epoch}:v{platform.version}",
            "version": platform.version, "epoch": platform.epoch,
            "price": platform.band.price, "band_lower": platform.band.lower, "band_upper": platform.band.upper,
            "center_min": platform.band.center_min, "center_max": platform.band.center_max,
            "tolerance_pct": self.cfg.error_margin_pct * 100,
            "point_count": len(events), "resistance_point_count": len(resistance), "support_point_count": len(support),
            "lifetime_event_count": len(platform.lifetime_events),
            "resistance_dates": [item["date"] for item in resistance], "support_dates": [item["date"] for item in support],
            "resistance_points": resistance, "support_points": support,
            "resistance_price_basis": "high", "support_price_basis": "low",
            "touch_dates": [self.dates[item.pivot] for item in events],
            "formation_date": self.dates[platform.born],
            "confirmation_date": self.dates[platform.confirmed] if platform.confirmed is not None else None,
            "version_created_at": self.dates[index], "usable_after_index": index,
            "formation_index": platform.born,
            "last_pivot_date": self.dates[max(item.pivot for item in events)],
            "last_confirmation_date": self.dates[max(item.confirmed for item in events)],
            "line_type": "resistance_support" if support and resistance else "support" if support else "resistance",
            "platform_state": "confirmed" if platform.confirmed is not None else "candidate",
            "confirmed_before_breakout": False,
            "history_bars": index + 1, "is_primary": False,
        }
        assert all(platform.band.accepts(item.price) for item in events)
        assert all(item.confirmed <= index for item in events)
        platform.snapshot = payload
        platform.versions.append(copy.deepcopy(payload))
        if platform.probe is None and platform.armed_side is None:
            distance = max(self.atr[index] * self.cfg.event_reversal_atr, platform.band.price * self.cfg.event_min_reversal_pct)
            if self.close[index] < platform.band.lower - distance:
                platform.armed_side = "below"
            elif self.close[index] > platform.band.upper + distance:
                platform.armed_side = "above"

    def _remember_event(self, event: ValidationEvent) -> ValidationEvent:
        local = event.local_confirmed
        if local is None and event.origins == ("local_turn",):
            local = event.confirmed
        event = replace(event, local_confirmed=local)
        previous = self.events.get(event.key)
        if previous:
            local_dates = [stamp for stamp in (previous.local_confirmed, local) if stamp is not None]
            event = replace(
                previous, origins=tuple(sorted(set(previous.origins + event.origins))),
                confirmed=min(previous.confirmed, event.confirmed), start=min(previous.start, event.start),
                max_high=max(previous.max_high, event.max_high), min_low=min(previous.min_low, event.min_low),
                local_confirmed=min(local_dates) if local_dates else None,
            )
        self.events[event.key] = event
        return event

    @staticmethod
    def _covering_trial(platform: Platform, event: ValidationEvent) -> Probe | None:
        position = bisect_right(platform.episode_starts, event.pivot) - 1
        if position < 0:
            return None
        trial = platform.episodes[position]
        if trial.end is None or event.pivot <= trial.end:
            return trial
        return None

    def _register(self, event: ValidationEvent, index: int, owner: Platform | None = None):
        raw = self._remember_event(event)
        if owner is None:
            # Raw turns remain immutable evidence, but a managed price-band trial
            # supplies the contribution only after its entire reaction is known.
            governed = False
            for platform in self.platforms:
                trial = self._covering_trial(platform, event)
                if trial is not None and platform.band.can_expand(event.price):
                    trial.local_event_ids.add(event.key)
                    governed = True
                if platform.snapshot is not None and platform.band.can_expand(event.price):
                    governed = True
                if event.key in platform.events:
                    previous = platform.events[event.key]
                    updated = replace(previous, origins=raw.origins, local_confirmed=raw.local_confirmed)
                    if updated != previous:
                        platform.events[event.key] = updated
                        self._snapshot(platform, index)
                    governed = True
            if governed:
                return
        else:
            if event.key in owner.events or any(
                item.episode_id == event.episode_id for item in owner.events.values()
                if event.episode_id is not None
            ):
                return
            event = replace(event, origins=raw.origins, local_confirmed=raw.local_confirmed)
        options = []
        duplicate = False
        retired_nearby = False
        same_trial = False
        for position, platform in enumerate(self.platforms):
            if owner is not None and platform is not owner:
                continue
            if not platform.band.can_expand(event.price):
                continue
            if platform.invalidated is not None and event.pivot <= platform.invalidated:
                retired_nearby = True
                continue
            expanded = bounded_band([*(item.price for item in platform.events.values()), event.price], self.cfg.error_margin_pct)
            if expanded is None:
                continue
            if any(item.pivot == event.pivot for item in platform.events.values()):
                duplicate = True
                continue
            prior = max((item for item in platform.events.values() if item.pivot < event.pivot),
                        key=lambda item: item.pivot, default=None)
            if prior is not None and owner is None:
                distance = max(event.atr * self.cfg.event_reversal_atr,
                               expanded.price * self.cfg.event_min_reversal_pct)
                path = (self.low if event.kind == "resistance" else self.high)[prior.pivot + 1:event.pivot]
                departed = (
                    np.any(path < expanded.lower - distance) if event.kind == "resistance"
                    else np.any(path > expanded.upper + distance)
                )
                if not departed:
                    same_trial = True
                    platform.rejected.append({
                        "date": self.dates[index], "reason": "same_trial_without_price_departure",
                        "event_id": f"{self.code}:{event.key}", "event_date": self.dates[event.pivot],
                        "previous_event_date": self.dates[prior.pivot],
                        "max_high": event.max_high, "min_low": event.min_low,
                    })
                    platform.rejected = platform.rejected[-16:]
                    continue
            options.append((abs(event.price / platform.band.price - 1), -len(platform.events), position, expanded))
        if not options:
            if owner is not None:
                owner.rejected.append({
                    "date": self.dates[index], "reason": "completed_trial_does_not_fit_current_band",
                    "episode_id": event.episode_id, "max_high": event.max_high, "min_low": event.min_low,
                })
                owner.rejected = owner.rejected[-16:]
                return
            if duplicate or retired_nearby or same_trial:
                return
            platform = Platform(f"{self.code}:band:{event.kind}:{event.pivot}", {}, bounded_band([event.price], self.cfg.error_margin_pct))
            self.platforms.append(platform)
        else:
            _, _, position, expanded = min(options, key=lambda item: item[:3])
            platform = self.platforms[position]
            platform.band = expanded
        platform.events[event.key] = event
        platform.lifetime_events.add(event.key)
        platform.last_state = "reforming" if platform.invalidated is not None else "candidate"
        self._snapshot(platform, index)
        if platform.snapshot:
            platform.last_state = platform.snapshot["platform_state"]

    def _end_trial(self, platform: Platform, index: int, status: str):
        trial = platform.probe
        if trial is not None:
            trial.high = max(trial.high, self.high[index])
            trial.low = min(trial.low, self.low[index])
            trial.end, trial.status = index, status
        platform.probe = None

    def _invalidate(self, platform: Platform, index: int, reason: str):
        platform.last_state = "invalid"
        platform.rejected.append({"date": self.dates[index], "reason": reason,
                                  "version_id": platform.observation.snapshot["version_id"] if platform.observation else None})
        platform.rejected = platform.rejected[-16:]
        platform.observation = None
        self._end_trial(platform, index, "invalidated")
        platform.armed_side = None
        platform.invalidated = index
        # A failed trade attempt does not erase the historical evidence at this price.
        platform.epoch += 1
        platform.can_start = False

    def _probe(self, platform: Platform, index: int) -> ValidationEvent | None:
        snapshot = platform.snapshot
        if snapshot is None or snapshot["usable_after_index"] >= index:
            return None
        band = platform.band
        close, high, low = self.close[index], self.high[index], self.low[index]
        distance = max(self.atr[index] * self.cfg.event_reversal_atr, band.price * self.cfg.event_min_reversal_pct)
        if platform.probe is None:
            touched = high >= band.lower and low <= band.upper
            if platform.armed_side and touched:
                kind = "resistance" if platform.armed_side == "below" else "support"
                platform.probe = Probe(kind, index, index, high if kind == "resistance" else low,
                                       distance, self.atr[index], high, low, band,
                                       id=f"{platform.id}:trial:{index}")
                platform.episode_starts.append(index)
                platform.episodes.append(platform.probe)
                platform.armed_side = None
            elif close < band.lower - distance:
                platform.armed_side = "below"
                platform.can_start = True
            elif close > band.upper + distance:
                platform.armed_side = "above"
            return None

        probe = platform.probe
        probe.high, probe.low = max(probe.high, high), min(probe.low, low)
        price = high if probe.kind == "resistance" else low
        if (probe.kind == "resistance" and price >= probe.price) or (probe.kind == "support" and price <= probe.price):
            probe.price, probe.pivot = price, index
        crossed = close > probe.band.upper + probe.distance if probe.kind == "resistance" else close < probe.band.lower - probe.distance
        reacted = close < probe.band.lower - probe.distance if probe.kind == "resistance" else close > probe.band.upper + probe.distance
        if crossed:
            platform.rejected.append({"date": self.dates[index], "reason": "price_traversal_not_validation",
                                      "event_start": self.dates[probe.start], "max_high": probe.high, "min_low": probe.low})
            platform.rejected = platform.rejected[-16:]
            self._end_trial(platform, index, "traversed")
            platform.armed_side = "above" if probe.kind == "resistance" else "below"
            return None
        if not reacted or index <= probe.pivot:
            return None
        platform.armed_side = "below" if probe.kind == "resistance" else "above"
        if not probe.band.accepts(probe.price):
            platform.rejected.append({"date": self.dates[index], "reason": "event_extreme_outside_band",
                                      "event_start": self.dates[probe.start], "max_high": probe.high, "min_low": probe.low,
                                      "episode_id": probe.id, "local_event_ids": sorted(probe.local_event_ids)})
            platform.rejected = platform.rejected[-16:]
            self._end_trial(platform, index, "rejected")
            return None
        self._end_trial(platform, index, "confirmed")
        return ValidationEvent(probe.kind, probe.pivot, index, probe.start, probe.price,
                               probe.atr, probe.high, probe.low, (f"{probe.kind}_test",), episode_id=probe.id)

    def _precheck(self, snapshot: dict, breakout: dict, index: int) -> dict:
        from app.services.strategy import KeyLine, _check_prebreakout_extension
        resistance = snapshot["resistance_points"]
        line = KeyLine(
            snapshot["price"], len(resistance), [point["pivot_index"] for point in resistance],
            max((point["pivot_index"] for point in resistance), default=0), snapshot["formation_index"],
            resistance_confirmed_at={point["pivot_index"]: point["confirmed_index"] for point in resistance},
        )
        return _check_prebreakout_extension(self.frame.iloc[:index + 1], line, breakout, self.cfg)

    def _observe(self, platform: Platform, index: int):
        snapshot = platform.snapshot
        observation = platform.observation
        close, previous = self.close[index], self.close[index - 1]
        maximum = self.cfg.max_pullback_days or self.cfg.pullback_window
        if observation:
            price = observation.snapshot["price"]
            if close < price * (1 - self.cfg.max_intervening_close_break_pct):
                self._invalidate(platform, index, "close_broke_frozen_line")
                return
            origin = observation.breakout["index"] if observation.breakout else observation.started
            if index - origin > maximum:
                platform.observation = observation = None
                platform.can_start = False
                platform.last_state = "expired"
        if snapshot is None or snapshot["usable_after_index"] >= index:
            return
        if observation is None and platform.can_start and previous <= snapshot["price"] < close:
            observation = platform.observation = Observation(copy.deepcopy(snapshot), index)
            platform.last_state = "breakout_observation"
            platform.can_start = False
        if observation is None or observation.breakout is not None:
            return
        threshold = snapshot["price"] * (1 + self.cfg.breakout_eps_pct)
        if not previous <= threshold <= close:
            return
        history = self.volume[max(0, index - self.cfg.vol_lookback):index]
        average = float(np.mean(history)) if len(history) >= min(5, self.cfg.vol_lookback) else None
        ratio = float(self.volume[index] / average) if average and average > 0 else None
        if self.cfg.require_breakout_volume and (ratio is None or ratio < self.cfg.breakout_vol_mult):
            return
        frozen = copy.deepcopy(snapshot)
        frozen["breakout_date"] = self.dates[index]
        frozen["confirmed_before_breakout"] = bool(frozen["confirmation_date"] and frozen["confirmation_date"] < self.dates[index])
        assert all(point["confirmed_index"] < index for point in frozen["resistance_points"] + frozen["support_points"])
        age = self.cfg.max_event_age_bars
        if age is not None:
            # 证据时效按突破当时冻结：之后再久也不回头改写这次突破的判定。
            frozen["recent_event_window_bars"] = age
            frozen["recent_event_count"] = sum(
                point["pivot_index"] >= index - age
                for point in frozen["resistance_points"] + frozen["support_points"]
            )
        observation.snapshot = frozen
        observation.breakout = {"index": index, "volume": float(self.volume[index]), "avg_volume": average, "volume_ratio": ratio}
        observation.precheck = self._precheck(frozen, observation.breakout, index)
        platform.last_state = "awaiting_pullback"

    def feed(self, index: int):
        if index != self.index + 1:
            raise ValueError("PlatformEngine must be advanced one bar at a time")
        self.index = index
        if self.volume[index] <= 0:
            return
        pending = []
        qualified = []
        if index:
            for platform in self.platforms:
                self._observe(platform, index)
                event = self._probe(platform, index)
                if event:
                    qualified.append((platform, event))
        for detector in self.detectors:
            event = detector.feed(index, self.high[index], self.low[index], self.close[index], self.atr[index], self.cfg)
            if event:
                self._remember_event(event)
                pending.append(event)
        # All events confirmed on this bar become available for future bars, never this bar's breakout.
        for owner, event in qualified:
            self._register(event, index, owner)
        for event in sorted(pending, key=lambda item: (item.pivot, item.kind, item.origins)):
            self._register(event, index)

    def _result(self, platform: Platform, index: int):
        from app.services.strategy import (
            LIMIT_UP_BREAKOUT_RISK, SignalCategory as C, SignalResult, _confirm_pullback, _pullback_check,
            _pullback_rule_risks, is_limit_up_close,
        )
        observation = platform.observation
        snapshot = observation.snapshot if observation else platform.snapshot or (
            platform.versions[-1] if platform.last_state == "invalid" and platform.versions else None
        )
        if snapshot is None:
            return None
        price, close = snapshot["price"], self.close[index]
        distance = (close / price - 1) * 100
        breakout = observation.breakout if observation else None
        view = self.frame.iloc[:index + 1]
        pullback = _confirm_pullback(view, price, breakout, self.cfg) if breakout else None
        check = copy.deepcopy(observation.precheck) if breakout else {
            "status": "awaiting_confirmed_breakout", "passed": None,
            "limit_pct": None if self.cfg.max_prebreakout_extension_pct is None else self.cfg.max_prebreakout_extension_pct * 100,
            "reject_above_days": self.cfg.prebreakout_reject_above_days,
            "above_buffer_pct": self.cfg.prebreakout_above_buffer_pct * 100,
        }
        trend = bool(index >= self.cfg.ma_slope_days and np.isfinite(self.ma[index])
                     and close >= self.ma[index] >= self.ma[index - self.cfg.ma_slope_days])
        category, state = C.NONE, platform.last_state
        reasons = [
            f"独立验证事件 {snapshot['point_count']} 个（阻力 {snapshot['resistance_point_count']}、支撑 {snapshot['support_point_count']}）",
            f"价位带 {snapshot['band_lower']:.3f}～{snapshot['band_upper']:.3f}，所有贡献点均满足最终容差",
            f"线版本 {snapshot['version_id']}，可用证据截止 {snapshot['version_created_at']}",
        ]
        risks = []
        if state == "invalid":
            risks.append("本次突破已失效；保留价位验证历史，需离开价位带后重新试探")
        score = 20 + min(20, snapshot["point_count"] * 5)
        if observation:
            category = C.WATCH
            state = "awaiting_pullback" if breakout else "breakout_observation"
            reasons.append(f"{self.dates[observation.started]} 收盘上穿代表线，进入观察")
            score += 10
        limit_up = None
        if breakout:
            score += 15
            reasons.append(f"{self.dates[breakout['index']]} 完成收盘突破确认")
            limit_up = is_limit_up_close(self.frame, breakout["index"], self.code, self.name)
            if limit_up:
                risks.append(LIMIT_UP_BREAKOUT_RISK)
            age = self.cfg.max_event_age_bars
            recent = snapshot.get("recent_event_count")
            recent_ok = age is None or (recent is not None and recent >= self.cfg.confirmed_event_count)
            if check.get("passed") is False and check.get("status") != "no_prior_point":
                category, state = C.NONE, "rejected"
                risks.append(f"突破前路径未通过：{check['status']}")
            elif check.get("status") == "no_prior_point":
                risks.append("缺少突破前已确认阻力验证，保留观察，不发买入预警")
            elif pullback and trend and snapshot["confirmed_before_breakout"] and recent_ok:
                category, state = C.BUY_ALERT, "pullback_alert"
                score += 20
                text = f"突破后第 {index - breakout['index']} 个交易日回踩确认，回踩量为突破日的 {pullback['volume_ratio']:.2f}"
                if self.cfg.pullback_max_normal_volume_ratio is not None and pullback.get("normal_volume_ratio") is not None:
                    text += f"、突破前 {self.cfg.vol_lookback} 日均量的 {pullback['normal_volume_ratio']:.2f} 倍"
                reasons.append(text)
            if not snapshot["confirmed_before_breakout"]:
                risks.append("突破前仅为候选价位带，验证事件不足，不升级买入预警")
            if not recent_ok:
                risks.append(
                    f"突破前 {age} 个交易日内独立验证事件仅 {recent or 0} 个"
                    f"（需 {self.cfg.confirmed_event_count} 个），更早的证据只作展示，不升级买入预警"
                )
            if not pullback:
                risks.append("尚未完成当前交易日的缩量回踩确认")
                risks.extend(_pullback_rule_risks(view, price, breakout, self.cfg))
        if trend:
            score += 10
            reasons.append(f"收盘站上 MA{self.cfg.ma_period}，均线未走低")
        else:
            risks.append("趋势确认尚未通过")
        if self.cfg.require_breakout and not breakout:
            category = C.NONE
        if breakout and category == C.BUY_ALERT and self.cfg.min_days_since_breakout:
            if index - breakout["index"] < self.cfg.min_days_since_breakout:
                category, state = C.WATCH, "awaiting_pullback"
        if breakout and self.cfg.max_days_since_breakout:
            if index - breakout["index"] > self.cfg.max_days_since_breakout:
                category, state = C.NONE, "expired"
        if self.volume[index] <= 0:
            category, state = C.NONE, "no_trading"
            risks.append("当前日线没有有效成交量，不确认事件或触发预警")
        labels = {
            "candidate": "候选价位带", "confirmed": "价位带已验证，等待上穿",
            "breakout_observation": "突破观察：已上穿代表线，等待有效突破与回踩",
            "awaiting_pullback": "突破观察：有效突破，等待回踩确认",
            "pullback_alert": "买入预警：已验证价位带突破后缩量回踩",
            "rejected": "暂不行动：突破前路径条件未通过",
            "invalid": "本次突破已失效", "reforming": "价位带重新验证中",
            "expired": "观察窗口已结束，等待新的独立机会",
            "no_trading": "当前交易日无有效成交",
        }
        anchor = None
        if breakout:
            eligible = [point for point in snapshot["resistance_points"] if point["confirmed_index"] < breakout["index"]]
            if eligible:
                anchor = {**max(eligible, key=lambda point: point["pivot_index"]), "role": "before_breakout"}
        event_id = (
            f"{self.code}:{snapshot['version_id']}:breakout:{self.dates[breakout['index']]}"
            if breakout else f"{self.code}:{snapshot['version_id']}:observation:{self.dates[observation.started]}"
            if observation else None
        )
        pullback_check = _pullback_check(view, breakout, pullback, self.cfg)
        pullback_check["required"] = True
        status_tags = [labels.get(state, state)]
        average = breakout.get("avg_volume") if breakout else None
        result = SignalResult(
            self.code, self.name, category, min(95, score), labels.get(state, state),
            price, snapshot["point_count"], self.dates[index],
            line_formation_date=snapshot["formation_date"],
            breakout_date=self.dates[breakout["index"]] if breakout else None,
            pullback_date=self.dates[index] if pullback else None,
            current_price=round(float(close), 6), distance_to_line_pct=round(distance, 3),
            breakout_volume_ratio=round(breakout["volume_ratio"], 3) if breakout and breakout["volume_ratio"] is not None else None,
            pullback_volume_ratio=round(pullback["volume_ratio"], 3) if pullback else None,
            pullback_normal_volume_ratio=round(float(self.volume[index] / average), 3) if average else None,
            require_breakout_volume=self.cfg.require_breakout_volume,
            breakout_volume_threshold=self.cfg.breakout_vol_mult if self.cfg.require_breakout_volume else None,
            breakout_status={"tags": status_tags, "days_since_breakout": index - breakout["index"] if breakout else None},
            reasons=reasons, risks=risks, prebreakout_check=check, pullback_check=pullback_check,
            resistance_anchor=anchor, precision_engine="platform_events", platform_state=state,
            signal_event_id=event_id, line_snapshot=copy.deepcopy(snapshot),
            breakout_limit_up=limit_up,
        )
        return result

    def result(self, index: int):
        from app.services.strategy import _rank, _build_chart_payload
        eligible = [
            platform for platform in self.platforms
            if platform.snapshot or platform.observation or platform.last_state == "invalid" and platform.versions
            if platform.observation or self.cfg.max_line_distance_pct <= 0
            or abs(platform.band.price / self.close[index] - 1) <= self.cfg.max_line_distance_pct
        ]
        results = [(self._result(platform, index), platform) for platform in eligible]
        results = [(result, platform) for result, platform in results if result is not None]
        if not results:
            return None
        best, owner = max(results, key=lambda pair: (*_rank(pair[0]), -abs(pair[0].distance_to_line_pct)))
        if self.cfg.include_chart:
            primary = copy.deepcopy(best.line_snapshot)
            primary["is_primary"] = True
            other = sorted(
                [platform for platform in eligible if platform.id != owner.id and platform.snapshot],
                key=lambda platform: abs(platform.band.price / self.close[index] - 1),
            )[:max(0, self.cfg.max_display_lines - 1)]
            best.key_lines = [primary, *(copy.deepcopy(platform.snapshot) for platform in other)]
            for line in best.key_lines:
                line["distance_to_current_pct"] = round((self.close[index] / line["price"] - 1) * 100, 3)
            best.chart = _build_chart_payload(self.frame.iloc[:index + 1], best)
            best.platform_diagnostics = [
                {"line_id": platform.id, "state": platform.last_state,
                 "latest_version": platform.snapshot["version_id"] if platform.snapshot else None,
                 "version_count": len(platform.versions), "rejections": platform.rejected[-5:],
                 "recent_trials": [{
                     "episode_id": trial.id, "kind": trial.kind, "start_date": self.dates[trial.start],
                     "end_date": self.dates[trial.end] if trial.end is not None else None,
                     "status": trial.status, "max_high": float(trial.high), "min_low": float(trial.low),
                     "local_event_ids": sorted(trial.local_event_ids),
                 } for trial in platform.episodes[-5:]]}
                for platform in [owner, *other]
            ]
        return best


def iter_platform_results(code: str, name: str, frame: pd.DataFrame, cfg, start_index: int = 0) -> Iterator[tuple[int, Any]]:
    from app.services.strategy import _merge_pattern_results, _pattern_snapshot, _run_flag_pipeline, SignalCategory
    if frame.empty:
        return
    engine = PlatformEngine(code, name, frame, cfg)
    minimum = max(cfg.min_history_bars, cfg.ma_period + cfg.ma_slope_days)
    for index in range(len(frame)):
        engine.feed(index)
        if index < max(start_index, minimum - 1):
            continue
        result = engine.result(index)
        if cfg.strategy_mode == "combined":
            view = frame.iloc[:index + 1]
            flag = _run_flag_pipeline(code, name, view, replace(
                cfg, strategy_mode="flag_breakout", min_days_since_breakout=0, max_days_since_breakout=0,
                max_prebreakout_extension_pct=None, prebreakout_reject_above_days=None,
                require_pullback=False, min_pullback_days=1, max_pullback_days=None,
                require_breakout_volume=True,
            ))
            result = _merge_pattern_results(code, name, view, cfg,
                ("ok" if result and result.category != SignalCategory.NONE else "no_signal", "", result), flag)[2]
        elif result and result.category != SignalCategory.NONE:
            result.pattern_matches = [_pattern_snapshot(result)]
        yield index, result


def analyze_platform(code: str, name: str, frame: pd.DataFrame, cfg):
    precision = replace(cfg, strategy_mode="precision_pullback")
    return next((result for _, result in iter_platform_results(code, name, frame, precision, len(frame) - 1)), None)
