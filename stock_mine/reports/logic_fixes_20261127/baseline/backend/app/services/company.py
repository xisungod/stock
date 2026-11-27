from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.akshare_guard import call_akshare, get_akshare


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "company_cache"
COMPANY_SCHEMA_VERSION = 2


class CompanyService:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, client: Any | None = None, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.client = client
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, stock_code: str) -> dict[str, Any]:
        code = str(stock_code).zfill(6)
        cached = self._load(code)
        if cached is not None:
            return cached
        try:
            payload = self._fetch(code)
            self._save(code, payload)
            return payload
        except Exception as exc:
            stale = self._load(code, ignore_age=True)
            if stale is not None:
                stale["stale"] = True
                stale["message"] = "实时股东接口不可用，当前展示最近一次缓存"
                stale["shareholders"]["sentiment"] = "unknown"
                stale["shareholders"]["sentiment_label"] = "数据过期"
                return stale
            return empty_company_payload(f"公司及股东数据暂不可用：{exc}")

    def _ak(self) -> Any:
        if self.client is None:
            self.client = get_akshare()
        return self.client

    def _fetch(self, code: str) -> dict[str, Any]:
        client = self._ak()
        profile = pd.DataFrame()
        holders = pd.DataFrame()
        counts = pd.DataFrame()
        errors: list[str] = []
        for name, call in (
            ("公司资料", lambda: client.stock_profile_cninfo(symbol=code)),
            ("前十大股东", lambda: client.stock_main_stock_holder(stock=code)),
            ("股东户数", lambda: client.stock_zh_a_gdhs_detail_em(symbol=code)),
        ):
            try:
                value = call_akshare(call)
                if name == "公司资料":
                    profile = value
                elif name == "前十大股东":
                    holders = value
                else:
                    counts = value
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        company = _profile_payload(profile)
        shareholder = _shareholder_payload(holders, counts)
        return {
            "schema_version": COMPANY_SCHEMA_VERSION,
            "company": company,
            "shareholders": shareholder,
            "source": "巨潮资讯 / 新浪财经 / 东方财富（经 AKShare）",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "stale": False,
            "message": "；".join(errors) if errors else "",
        }

    def _path(self, code: str) -> Path:
        return self.cache_dir / f"{code}.json"

    def _load(self, code: str, ignore_age: bool = False) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._path(code).read_text(encoding="utf-8"))
            updated = datetime.fromisoformat(payload["updated_at"])
            if not ignore_age and datetime.now() - updated > self.ttl:
                return None
            if payload.get("schema_version") != COMPANY_SCHEMA_VERSION:
                if not ignore_age:
                    return None
                payload["shareholders"] = sanitize_legacy_shareholders(payload.get("shareholders", {}))
            return payload
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _save(self, code: str, payload: dict[str, Any]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(code).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def empty_company_payload(message: str = "暂无公司与股东数据") -> dict[str, Any]:
    return {
        "schema_version": COMPANY_SCHEMA_VERSION,
        "company": {},
        "shareholders": {
            "as_of": None,
            "holder_count": None,
            "holder_count_change_pct": None,
            "top10_concentration_pct": None,
            "top10_concentration_change_pct": None,
            "top_holders": [],
            "sentiment": "unknown",
            "sentiment_label": "数据不足",
            "signals": ["股东数据不足，暂不作利好或利空判断"],
            "data_quality": {"status": "unavailable", "concentration_status": "unavailable"},
        },
        "source": None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stale": False,
        "message": message,
    }


def _profile_payload(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[0]
    mapping = {
        "full_name": "公司名称",
        "industry": "所属行业",
        "market": "所属市场",
        "chairman": "法人代表",
        "established_date": "成立日期",
        "listing_date": "上市日期",
        "website": "官方网站",
        "registered_address": "注册地址",
        "main_business": "主营业务",
        "introduction": "机构简介",
    }
    return {key: _json_value(row.get(column)) for key, column in mapping.items() if pd.notna(row.get(column))}


def _shareholder_payload(holders: pd.DataFrame | None, counts: pd.DataFrame | None) -> dict[str, Any]:
    top_holders: list[dict[str, Any]] = []
    as_of = None
    concentration = previous_concentration = None
    quality = {"status": "incomplete", "concentration_status": "unavailable",
               "previous_concentration_status": "unavailable", "valid_ratio_count": 0, "row_count": 0}
    if holders is not None and not holders.empty and "截至日期" in holders:
        work = holders.copy()
        work["截至日期"] = pd.to_datetime(work["截至日期"], errors="coerce")
        dates = sorted(work["截至日期"].dropna().unique(), reverse=True)
        if dates:
            latest = work[work["截至日期"] == dates[0]].head(10)
            as_of = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
            concentration, latest_quality = _complete_concentration(latest)
            quality.update(latest_quality)
            if len(dates) > 1:
                previous = work[work["截至日期"] == dates[1]].head(10)
                previous_concentration, previous_quality = _complete_concentration(previous)
                quality["previous_concentration_status"] = previous_quality["concentration_status"]
            for _, row in latest.iterrows():
                top_holders.append(
                    {
                        "name": _json_value(row.get("股东名称")),
                        "shares": _json_value(row.get("持股数量")),
                        "ratio_pct": _ratio(row.get("持股比例")),
                        "nature": _json_value(row.get("股本性质")),
                    }
                )

    holder_count = holder_change = None
    holder_count_as_of = None
    if counts is not None and not counts.empty and "股东户数统计截止日" in counts:
        work = counts.copy()
        work["股东户数统计截止日"] = pd.to_datetime(work["股东户数统计截止日"], errors="coerce")
        work = work.sort_values("股东户数统计截止日", ascending=False)
        work = work[work["股东户数统计截止日"].notna()]
        if not work.empty:
            latest_count = work.iloc[0]
            holder_count_as_of = _date_value(latest_count.get("股东户数统计截止日"))
            holder_count = _number(latest_count.get("股东户数-本次"))
            holder_change = _number(latest_count.get("股东户数-增减比例"))
            if holder_count is None or holder_count <= 0 or holder_change is not None and holder_change < -100:
                holder_count = holder_change = None

    concentration_change = (
        round(concentration - previous_concentration, 2)
        if concentration is not None and previous_concentration is not None
        else None
    )
    signals: list[str] = []
    sentiment_points = 0
    if holder_change is not None:
        if holder_change <= -3:
            sentiment_points += 1
            signals.append(f"股东户数较上期减少 {abs(holder_change):.2f}%，筹码趋于集中（偏利好）")
        elif holder_change >= 3:
            sentiment_points -= 1
            signals.append(f"股东户数较上期增加 {holder_change:.2f}%，筹码趋于分散（偏利空）")
        else:
            signals.append(f"股东户数较上期变化 {holder_change:+.2f}%，整体平稳")
    if concentration_change is not None:
        if concentration_change >= 0.5:
            sentiment_points += 1
            signals.append(f"前十大股东持股比例较上期增加 {concentration_change:.2f} 个百分点（偏利好）")
        elif concentration_change <= -0.5:
            sentiment_points -= 1
            signals.append(f"前十大股东持股比例较上期减少 {abs(concentration_change):.2f} 个百分点（偏利空）")
        else:
            signals.append("前十大股东持股集中度变化不大")
    if concentration is None:
        signals.append("前十大股东比例不完整或格式无效，集中度及其变化不作判断")
    elif previous_concentration is None:
        signals.append("上期前十大股东比例不完整，集中度变化不作判断")
    if not signals:
        signals.append("股东数据不足，暂不作利好或利空判断")
    complete = holder_change is not None and concentration_change is not None
    quality["status"] = "complete" if complete else "incomplete"
    sentiment = ("positive" if sentiment_points > 0 else "negative" if sentiment_points < 0 else "neutral") if complete else "unknown"
    labels = {"positive": "偏利好", "negative": "偏利空", "neutral": "中性", "unknown": "数据不足"}
    return {
        "as_of": as_of or holder_count_as_of,
        "holder_count_as_of": holder_count_as_of,
        "holder_count": holder_count,
        "holder_count_change_pct": round(holder_change, 2) if holder_change is not None else None,
        "top10_concentration_pct": concentration,
        "top10_concentration_change_pct": concentration_change,
        "top_holders": top_holders,
        "sentiment": sentiment,
        "sentiment_label": labels[sentiment],
        "signals": signals,
        "data_quality": quality,
        "disclaimer": "股东标记仅反映股东户数和集中度变化，不代表公司基本面或未来涨跌。",
    }


def _number(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "").removesuffix("%").removesuffix("％")
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _ratio(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and 0 <= number <= 100 else None


def _complete_concentration(frame: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    ratios = [_ratio(value) for value in frame.get("持股比例", pd.Series(index=frame.index, dtype=object))]
    names = [str(value).strip() for value in frame.get("股东名称", pd.Series(index=frame.index, dtype=object))]
    valid = sum(value is not None for value in ratios)
    total = sum(value for value in ratios if value is not None)
    complete = (len(frame) == 10 and valid == 10 and len(set(names)) == 10
                and all(name not in {"", "nan", "None"} for name in names) and total <= 100 + 1e-6)
    return (round(total, 2) if complete else None), {
        "concentration_status": "complete" if complete else "incomplete",
        "valid_ratio_count": valid, "row_count": len(frame),
    }


def sanitize_legacy_shareholders(payload: dict[str, Any]) -> dict[str, Any]:
    """Old caches do not preserve the previous period's rows, so changes are unverifiable."""
    result = dict(payload)
    holders = result.get("top_holders") or []
    frame = pd.DataFrame([
        {"股东名称": row.get("name"), "持股比例": row.get("ratio_pct")}
        for row in holders if isinstance(row, dict)
    ])
    total, quality = _complete_concentration(frame)
    result.update(
        top10_concentration_pct=total, top10_concentration_change_pct=None,
        sentiment="unknown", sentiment_label="数据不足",
        signals=["旧缓存缺少完整的两期股东明细，暂不判断集中度变化"],
        data_quality={**quality, "status": "legacy_unverified"},
    )
    return result


def _date_value(value: Any) -> str | None:
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value
