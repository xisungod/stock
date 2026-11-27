"""日线行情的本地缓存。

为什么值得做：实测单只股票取数 8.95 秒、分析只要 0.12 秒——取数占 99%。
腾讯接口每次都返回全历史（33 年、约 8500 根），而昨天以前的 K 线永远不会变。
缓存后隔天再扫只需取新增的几根。

必须处理的陷阱：**前复权价格会在除权日被整体重算**。把新 K 线直接拼到旧缓存
后面，旧段还是老的复权基准，接缝处价格会突变，后面所有精准线和涨跌幅都算错。
这里靠重叠段比对来发现基准变化，一旦对不上就重新全量取数。
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.services.market_calendar import can_reuse_quotes, china_now, quote_quality

class FetchFailed(RuntimeError):
    """取数失败（与"确实没有新K线"区分开）。

    provider 内部对网络异常做了重试和降级，最终会返回空表；缓存层若只看
    "是否抛异常"，就分不清失败和无新数据，会把失败记成当天已定稿。
    """


COLUMNS = ["trade_date", "open", "high", "low", "close", "volume"]
EPOCH_START = "19900101"
# 增量取数时向前多要几根，用来比对复权基准有没有变
OVERLAP_BARS = 5
# 重叠段收盘价的相对容差；超过即认为复权基准已变
RESCALE_TOLERANCE = 0.005
# A 股 15:00 收盘，留几分钟给数据源落定
MARKET_CLOSE = time(15, 5)
# 盘中数据是临时的，必须重查；但短时间内连问多次没有意义
INTRADAY_RECHECK = timedelta(minutes=20)


def with_data_quality(
    frame: pd.DataFrame, checked_at: str = "", error: str | None = None,
    *, verify: bool = False, now: datetime | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if verify:
        result.attrs["data_quality"] = quote_quality(result, checked_at, now, error)
        return result
    dates = pd.to_datetime(result.get("trade_date", pd.Series(dtype=str)), errors="coerce").dropna()
    result.attrs["data_quality"] = {
        "status": "unavailable" if result.empty else "stale" if error else "fresh",
        "checked_at": checked_at or None,
        "last_bar_date": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
        "error": error,
    }
    return result


class BarsCache:
    def __init__(
        self,
        cache_dir: str | Path,
        fetch: Callable[[str, str], pd.DataFrame],
        now: Callable[[], datetime] = china_now,
    ):
        self.cache_dir = Path(cache_dir)
        self.fetch = fetch
        self.now = now

    def today(self) -> date:
        return self.now().date()

    def _path(self, code: str) -> Path:
        return self.cache_dir / f"{code}.json"

    def _load(self, code: str) -> tuple[pd.DataFrame, str] | None:
        """返回 (K线, 上次向数据源确认的日期)。"""
        try:
            payload = json.loads(self._path(code).read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload["bars"])
            if frame.empty or any(col not in frame.columns for col in COLUMNS):
                return None
            return frame[COLUMNS], str(payload.get("checked_at") or "")
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save(self, code: str, frame: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(code).write_text(
                json.dumps({"checked_at": self.now().isoformat(timespec="seconds"),
                            "bars": frame[COLUMNS].to_dict("records")}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def peek(self, code: str) -> pd.DataFrame:
        """Read a snapshot without refreshing or claiming it is current."""
        loaded = self._load(code)
        if loaded is None:
            return pd.DataFrame(columns=COLUMNS)
        frame = with_data_quality(loaded[0], loaded[1])
        frame.attrs["data_quality"]["status"] = "cached"
        return frame

    def _full_fetch(self, code: str) -> pd.DataFrame:
        frame = _normalize(self.fetch(code, EPOCH_START))
        if frame.empty:
            raise FetchFailed(f"{code}: 全历史行情为空")
        self._save(code, frame)
        return self._quality(frame, self.now().isoformat(timespec="seconds"))

    def _quality(self, frame, checked_at="", error=None):
        return with_data_quality(frame, checked_at, error, verify=True, now=self.now())

    def get(self, code: str) -> pd.DataFrame:
        loaded = self._load(code)
        if loaded is None or loaded[0].empty:
            try:
                return self._full_fetch(code)
            except Exception as exc:
                return self._quality(pd.DataFrame(columns=COLUMNS), error=str(exc))
        cached, checked_at = loaded

        now = self.now()
        if can_reuse_quotes(cached, checked_at, now, INTRADAY_RECHECK):
            return self._quality(cached, checked_at)

        overlap_from = str(cached["trade_date"].iloc[max(0, len(cached) - OVERLAP_BARS)])
        try:
            fresh = _normalize(self.fetch(code, overlap_from.replace("-", "")))
        except Exception as exc:
            # 取数失败 ≠ 今天没有新数据。旧数据比没有数据强，但**不能**盖上定稿
            # 时间戳——否则当天就再也不会重试，晚上的筛选会一直用着昨天的行情。
            return self._quality(cached, checked_at, str(exc))
        if fresh.empty:
            # An empty successful response is a check, not proof of a completed daily bar.
            self._save(code, cached)
            return self._quality(cached, now.isoformat(timespec="seconds"))

        if _adjustment_changed(cached, fresh):
            try:
                return self._full_fetch(code)
            except Exception as exc:
                return self._quality(cached, checked_at, f"复权基准更新失败：{exc}")

        merged = pd.concat([cached[cached["trade_date"] < fresh["trade_date"].iloc[0]], fresh])
        merged = merged.drop_duplicates(subset=["trade_date"], keep="last")
        merged = merged.sort_values("trade_date").reset_index(drop=True)
        self._save(code, merged)
        return self._quality(merged, now.isoformat(timespec="seconds"))


def _normalize(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out[out["trade_date"].notna()]
    return out[COLUMNS].sort_values("trade_date").reset_index(drop=True)


def _adjustment_changed(cached: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """重叠交易日的收盘价对不上，说明前复权基准变了（除权/分红）。"""
    shared = cached.merge(fresh, on="trade_date", suffixes=("_old", "_new"))
    if shared.empty:
        return False
    old = shared["close_old"].astype(float)
    new = shared["close_new"].astype(float)
    denom = old.abs().where(old.abs() > 0, 1.0)
    return bool(((old - new).abs() / denom > RESCALE_TOLERANCE).any())
