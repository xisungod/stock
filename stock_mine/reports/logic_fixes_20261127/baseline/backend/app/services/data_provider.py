from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from app.services.akshare_guard import call_akshare, call_akshare_parallel, get_akshare
from app.services.bars_cache import BarsCache, FetchFailed, with_data_quality
from app.services.market_calendar import china_now
from app.services.tencent_bars import fetch_tencent_bars

DEFAULT_BARS_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "bars_cache"


class Provider(Protocol):
    name: str

    def get_stock_list(self) -> pd.DataFrame:
        ...

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        ...

    def get_full_daily_bars(self, stock_code: str, years: int = 0) -> pd.DataFrame:
        ...


FETCH_MAX_ATTEMPTS = 3
FETCH_RETRY_DELAY = 1.0


def _fetch_with_retry(call):
    """重试取数：空结果与异常都算失败。

    本机网络/代理抖动时数据源会间歇性返回空或抛错，不重试的话整批扫描会大面积失败。
    最后一次尝试后不再等待。返回最后一次的结果（可能为 None 或空表）。
    """
    df = None
    for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):
        try:
            df = call()
            if df is not None and not df.empty:
                return df
        except Exception:
            df = None
        if attempt < FETCH_MAX_ATTEMPTS:
            time.sleep(FETCH_RETRY_DELAY * attempt)
    return df


def create_provider(provider: str = "auto") -> Provider:
    if not isinstance(provider, str):
        raise ValueError("数据源必须为 auto、adata、akshare 或显式示例数据源")
    provider = provider.strip().lower()
    if provider in {"sample", "demo", "fake"}:
        return SampleProvider()
    if provider == "auto":
        return AutoProvider()
    if provider == "adata":
        return ADataProvider()
    if provider == "akshare":
        return AkshareProvider()
    raise ValueError(f"未知数据源：{provider!r}，不会回退到示例行情")


@dataclass
class ADataProvider:
    name: str = "adata"

    def __post_init__(self) -> None:
        import adata  # type: ignore

        self._adata = adata

    def get_stock_list(self) -> pd.DataFrame:
        df = self._adata.stock.info.all_code()
        if df is None or df.empty:
            return pd.DataFrame(columns=["stock_code", "short_name", "exchange"])
        df = df[df["exchange"].isin(["SZ", "SH"])].copy()
        df = df[~df["short_name"].astype(str).str.contains("ST|退", na=False, regex=True)]
        return df[["stock_code", "short_name", "exchange"]].reset_index(drop=True)

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        df = _fetch_with_retry(
            lambda: self._adata.stock.market.get_market(
                stock_code=stock_code, k_type=1, adjust_type=1
            )
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values("trade_date").reset_index(drop=True)
        bars = df[["trade_date", "open", "high", "low", "close", "volume"]].copy()
        now = china_now()
        return with_data_quality(bars, now.isoformat(timespec="seconds"), verify=True, now=now)

    def get_full_daily_bars(self, stock_code: str, years: int = 0) -> pd.DataFrame:
        return self.get_daily_bars(stock_code)


@dataclass
class AkshareProvider:
    name: str = "akshare"
    cache_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "stock_list_cache.json"
    )
    legacy_cache_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[4] / "cache" / "stock_list_20250728.json"
    )
    # 传 None 关闭缓存（测试用）；生产默认落盘，隔天只取增量
    bars_cache_dir: Path | None = field(default_factory=lambda: DEFAULT_BARS_CACHE_DIR)
    tencent_fetch: Any = fetch_tencent_bars
    _history_source: str = field(default="eastmoney", init=False, repr=False)
    _history_source_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ak = get_akshare()
        self._bars_cache = (
            BarsCache(self.bars_cache_dir, self._fetch_range)
            if self.bars_cache_dir is not None
            else None
        )

    def _fetch_range(self, code: str, start_date: str) -> pd.DataFrame:
        """取 [start_date, 今天] 的前复权日线：东财优先，连续失败后本轮改走腾讯。

        两条路都走 `call_akshare_parallel`（不加全局锁）——锁是为了保护
        libmini_racer 的初始化，行情接口碰不到它，加了只会把并发压成 1。
        """
        end_date = china_now().strftime("%Y%m%d")
        with self._history_source_lock:
            source = self._history_source
        if source == "eastmoney":
            df = _fetch_with_retry(
                lambda: call_akshare_parallel(
                    lambda: self._ak.stock_zh_a_hist(
                        symbol=code, period="daily",
                        start_date=start_date, end_date=end_date, adjust="qfq",
                    )
                )
            )
            normalized = self._normalize_daily_bars(df)
            if not normalized.empty:
                return normalized
            # 东方财富会在高频访问后直接断开连接。本轮一旦连续重试仍失败，
            # 后续股票直接走腾讯，避免每只股票继续撞同一个限流接口。
            with self._history_source_lock:
                self._history_source = "tencent"

        market_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
        df = _fetch_with_retry(
            lambda: call_akshare_parallel(
                lambda: self.tencent_fetch(
                    market_code, start_date=start_date, end_date=end_date, adjust="qfq"
                )
            )
        )
        normalized = self._normalize_daily_bars(df)
        if normalized.empty:
            # 两个源都没给出数据。这跟"今天没有新K线"不是一回事——必须让缓存层
            # 知道这是失败，否则它会盖上定稿时间戳，当天不再重试。
            raise FetchFailed(f"{code}: 东财与腾讯均未返回数据")
        return normalized

    def get_stock_list(self) -> pd.DataFrame:
        errors: list[str] = []
        loaders = (
            lambda: self._ak.stock_info_a_code_name(),
            lambda: self._ak.stock_zh_a_spot_em(),
        )
        for loader in loaders:
            try:
                stock_list = self._normalize_stock_list(call_akshare(loader))
                if not stock_list.empty:
                    self._save_stock_list_cache(stock_list)
                    return stock_list
            except Exception as exc:
                errors.append(str(exc))

        cached = self.get_cached_stock_list()
        if not cached.empty:
            return cached

        detail = "; ".join(errors[:2]) or "接口返回空数据"
        raise RuntimeError(f"无法获取股票列表，且没有可用缓存: {detail}")

    def get_cached_stock_list(self) -> pd.DataFrame:
        for path in (self.cache_path, self.legacy_cache_path):
            cached = self._load_stock_list_cache(path)
            if not cached.empty:
                return cached
        return pd.DataFrame(columns=["stock_code", "short_name", "exchange"])

    def get_cached_daily_bars(self, stock_code: str) -> pd.DataFrame:
        if self._bars_cache is None:
            return pd.DataFrame()
        frame = self._bars_cache.peek(str(stock_code).zfill(6))
        if not frame.empty:
            frame.attrs["data_quality"]["source"] = self.name
        return frame

    def _normalize_stock_list(self, df: pd.DataFrame | None) -> pd.DataFrame:
        columns = ["stock_code", "short_name", "exchange"]
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)
        normalized = df.rename(
            columns={
                "code": "stock_code",
                "name": "short_name",
                "代码": "stock_code",
                "名称": "short_name",
            }
        ).copy()
        if "stock_code" not in normalized.columns or "short_name" not in normalized.columns:
            return pd.DataFrame(columns=columns)
        normalized["stock_code"] = normalized["stock_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        normalized = normalized[normalized["stock_code"].str.match(r"^[036]\d{5}$", na=False)]
        normalized["exchange"] = normalized["stock_code"].apply(lambda code: "SH" if code.startswith("6") else "SZ")
        normalized = normalized[
            ~normalized["short_name"].astype(str).str.contains("ST|退|PT", na=False, case=False, regex=True)
        ]
        return normalized[columns].drop_duplicates(subset=["stock_code"], keep="last").sort_values("stock_code").reset_index(drop=True)

    def _save_stock_list_cache(self, stock_list: pd.DataFrame) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(stock_list.to_dict("records"), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load_stock_list_cache(self, path: Path) -> pd.DataFrame:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("records", []) if isinstance(payload, dict) else payload
            return self._normalize_stock_list(pd.DataFrame(records))
        except (OSError, ValueError, TypeError):
            return pd.DataFrame(columns=["stock_code", "short_name", "exchange"])

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        """扫描循环的契约：取不到就返回空表，不要抛异常打断整轮扫描。

        缓存层走的是 `_fetch_range`，它会把失败以 FetchFailed 抛出——那条路径必须
        看得见失败，否则会把"数据源全挂"当成"今天没有新K线"记为定稿。
        """
        code = str(stock_code).zfill(6)
        try:
            if self._bars_cache is not None:
                return self._bars_cache.get(code)
            bars = self._fetch_range(code, "19900101")
            now = china_now()
            return with_data_quality(bars, now.isoformat(timespec="seconds"), verify=True, now=now)
        except FetchFailed:
            return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume"])

    def get_full_daily_bars(self, stock_code: str, years: int = 0) -> pd.DataFrame:
        # 缓存里存的就是全历史，按年数切片即可，不必再取一次数
        bars = self.get_daily_bars(stock_code)
        if years <= 0 or bars.empty:
            return bars
        cutoff = (date.today() - timedelta(days=max(1, years) * 366)).strftime("%Y-%m-%d")
        return bars[bars["trade_date"].astype(str) >= cutoff].reset_index(drop=True)

    @staticmethod
    def _normalize_daily_bars(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "日期": "trade_date",
                "date": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "amount": "volume",
            }
        )
        needed = ["trade_date", "open", "high", "low", "close", "volume"]
        if any(col not in df.columns for col in needed):
            return pd.DataFrame()
        return df[needed].copy()


@dataclass
class AutoProvider:
    """按单次请求在 AData 与 AKShare 之间回退。

    仅在初始化时选择数据源不够可靠：AData 能正常 import，并不代表随后的行情
    请求一定成功。自动模式必须把空表也视为失败，才能在网络或上游接口抖动时
    继续尝试备用源。显式选择 adata/akshare 时仍保持严格，不做跨源回退。
    """

    name: str = "auto"
    adata: Provider | None = None
    akshare: Provider | None = None

    def __post_init__(self) -> None:
        initialization_errors: list[str] = []
        if self.adata is None:
            try:
                self.adata = ADataProvider()
            except Exception as exc:
                initialization_errors.append(f"AData: {exc}")
        if self.akshare is None:
            try:
                self.akshare = AkshareProvider()
            except Exception as exc:
                initialization_errors.append(f"AKShare: {exc}")
        if self.adata is None and self.akshare is None:
            detail = "; ".join(initialization_errors) or "没有可用数据源"
            raise RuntimeError(f"自动数据源初始化失败: {detail}")

    def _providers(self) -> list[Provider]:
        return [item for item in (self.adata, self.akshare) if item is not None]

    def _fetch(self, method: str, *args, raise_on_empty: bool = False, **kwargs) -> pd.DataFrame:
        errors: list[str] = []
        stale = None
        for provider in self._providers():
            try:
                result = getattr(provider, method)(*args, **kwargs)
                if result is not None and not result.empty:
                    if result.attrs.get("data_quality", {}).get("status") == "stale":
                        stale = result
                        continue
                    return result
                errors.append(f"{provider.name} 返回空数据")
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
        if raise_on_empty:
            raise RuntimeError("；".join(errors) or "数据源均不可用")
        return stale if stale is not None else pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        return self._fetch("get_stock_list", raise_on_empty=True)

    def get_cached_stock_list(self) -> pd.DataFrame:
        return self._fetch("get_cached_stock_list")

    def get_cached_daily_bars(self, stock_code: str) -> pd.DataFrame:
        return self._fetch("get_cached_daily_bars", stock_code)

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        return self._fetch("get_daily_bars", stock_code)

    def get_full_daily_bars(self, stock_code: str, years: int = 0) -> pd.DataFrame:
        return self._fetch("get_full_daily_bars", stock_code, years=years)


@dataclass
class SampleProvider:
    name: str = "sample"

    def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stock_code": "000001", "short_name": "平安银行", "exchange": "SZ"},
                {"stock_code": "000002", "short_name": "万科A", "exchange": "SZ"},
                {"stock_code": "600000", "short_name": "浦发银行", "exchange": "SH"},
            ]
        )

    def get_daily_bars(self, stock_code: str) -> pd.DataFrame:
        prefix = [6.8 + idx * 0.009 for idx in range(130)]
        if stock_code == "000002":
            pattern = [8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4, 9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.7, 9.9, 10.03, 10.07, 10.11]
            closes = prefix + pattern
            volumes = [800] * len(prefix) + [1000] * len(pattern)
        else:
            pattern = [8.0, 8.3, 8.8, 9.4, 10.0, 9.2, 8.9, 9.5, 10.05, 9.4, 9.1, 9.7, 10.02, 9.5, 9.6, 9.8, 10.08, 9.6, 9.9, 10.35, 10.18, 10.12]
            closes = prefix + pattern
            volumes = [800] * len(prefix) + [1000] * len(pattern)
            volumes[len(prefix) + 19] = 2600
            volumes[len(prefix) + 21] = 1100
        highs = [c * 1.01 for c in closes]
        lows = [c * 0.99 for c in closes]
        if len(lows) > len(prefix) + 21:
            lows[len(prefix) + 21] = min(lows[len(prefix) + 21], 10.01)
        start = date.today() - timedelta(days=len(closes) - 1)
        return pd.DataFrame(
            {
                "trade_date": [start + timedelta(days=i) for i in range(len(closes))],
                "open": [close * 0.995 for close in closes],
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

    def get_full_daily_bars(self, stock_code: str, years: int = 0) -> pd.DataFrame:
        return self.get_daily_bars(stock_code)
