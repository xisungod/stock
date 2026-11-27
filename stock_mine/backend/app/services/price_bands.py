"""Shared bounded one-dimensional price grouping.

All contributing prices must be within the same final representative's tolerance.
The feasible representative interval prevents both chaining and a shifted median.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class PriceBand:
    center_min: float
    center_max: float
    tolerance: float
    representative: float | None = None

    @property
    def price(self) -> float:
        return self.representative if self.representative is not None else (self.center_min + self.center_max) / 2

    @property
    def lower(self) -> float:
        return self.price * (1 - self.tolerance)

    @property
    def upper(self) -> float:
        return self.price * (1 + self.tolerance)

    def accepts(self, price: float) -> bool:
        return isfinite(price) and abs(price / self.price - 1) <= self.tolerance + 1e-10

    def can_expand(self, price: float) -> bool:
        slack = max(1.0, self.price) * 1e-10
        return (isfinite(price) and price > 0
                and self.center_min * (1 - self.tolerance) - slack <= price
                <= self.center_max * (1 + self.tolerance) + slack)


def bounded_band(prices: Iterable[float], tolerance: float = .02) -> PriceBand | None:
    if not isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("price tolerance must be finite and in [0, 1)")
    values = [float(value) for value in prices]
    if not values or any(not isfinite(value) or value <= 0 for value in values):
        return None
    lower = max(value / (1 + tolerance) for value in values)
    upper = min(value / (1 - tolerance) for value in values)
    if lower > upper + max(1.0, abs(lower), abs(upper)) * 1e-12:
        return None
    if lower > upper:
        lower = upper = (lower + upper) / 2
    representative = min(upper, max(lower, float(median(values))))
    band = PriceBand(lower, upper, tolerance, representative)
    assert all(band.accepts(value) for value in values)
    return band


def bounded_clusters(prices: list[float], tolerance: float = .02) -> list[list[int]]:
    """Deterministic grouping without a transitive-neighbor price chain."""
    groups: list[list[int]] = []
    for index in sorted(range(len(prices)), key=lambda i: (prices[i], i)):
        options = []
        for position, group in enumerate(groups):
            current = bounded_band((prices[i] for i in group), tolerance)
            enlarged = bounded_band((prices[i] for i in [*group, index]), tolerance)
            if enlarged is not None:
                options.append((abs(prices[index] / current.price - 1), position))
        if options:
            groups[min(options)[1]].append(index)
        else:
            groups.append([index])
    return groups
