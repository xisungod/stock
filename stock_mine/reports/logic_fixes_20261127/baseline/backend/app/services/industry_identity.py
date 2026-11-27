"""Conservative cross-check of vendor classifications against cached company profiles."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from app.services.company import DEFAULT_CACHE_DIR


FAMILIES = (
    ("logistics", ("邮政", "快递", "物流")),
    ("home", ("家具", "家居")),
    ("power", ("电力", "热力")),
    ("bank", ("银行", "货币金融")),
    ("securities", ("证券", "资本市场服务")),
    ("insurance", ("保险",)),
    ("real_estate", ("房地产",)),
    ("coal", ("煤炭",)),
    ("steel", ("钢铁", "黑色金属冶炼")),
    ("nonferrous", ("有色金属",)),
    ("paper", ("造纸",)),
    ("software", ("软件", "信息技术服务")),
    ("telecom", ("电信", "通信服务")),
    ("pharma", ("医药", "制药",)),
    ("chemical", ("化工", "化学原料", "化学制品")),
    ("textile", ("纺织",)),
    ("clothing", ("服装", "服饰")),
    ("food", ("食品",)),
    ("beverage", ("酒", "饮料")),
    ("retail", ("零售", "商业百货")),
    ("environment", ("环境治理", "环保")),
    ("gas", ("燃气",)),
)


def industry_family(name: str | None) -> str | None:
    text = str(name or "").strip()
    matches = {family for family, fragments in FAMILIES if any(part in text for part in fragments)}
    return next(iter(matches)) if len(matches) == 1 else None


def cached_company_industry(code: str, cache_dir: Path = DEFAULT_CACHE_DIR,
                            now: datetime | None = None) -> str | None:
    try:
        payload = json.loads((cache_dir / f"{str(code).zfill(6)}.json").read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(payload["updated_at"])
        age = (now or datetime.now()) - stamp
        if not timedelta(0) <= age <= timedelta(days=30):
            return None
        value = payload.get("company", {}).get("industry")
        return value.strip() if isinstance(value, str) and value.strip() else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
