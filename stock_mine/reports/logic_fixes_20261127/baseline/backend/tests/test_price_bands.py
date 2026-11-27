import random

import pytest

from app.services.price_bands import bounded_band, bounded_clusters


def test_symmetric_prices_fit_the_same_final_band():
    band = bounded_band([6.67, 6.81], .02)
    assert band is not None
    assert band.accepts(6.67) and band.accepts(6.81)
    assert len(bounded_clusters([6.67, 6.81], .02)) == 1


def test_repeated_low_prices_do_not_pull_the_line_outside_the_high_point():
    prices = [2.69] * 30 + [2.7, 2.74, 2.76, 2.77, 2.78]
    band = bounded_band(prices, .02)
    assert band is not None
    assert all(abs(price / band.price - 1) <= .02 + 1e-10 for price in prices)


def test_chained_neighbors_do_not_form_an_unbounded_group():
    prices = [100 * 1.019**i for i in range(30)]
    groups = bounded_clusters(prices)
    assert len(groups) > 1
    for group in groups:
        assert bounded_band(prices[i] for i in group) is not None


def test_invalid_values_and_boundary():
    assert bounded_band([0, 1]) is None
    assert bounded_band([float("nan")]) is None
    assert bounded_band([98, 102], .02).price == pytest.approx(100)
    assert bounded_band([98, 102.001], .02) is None
    with pytest.raises(ValueError):
        bounded_band([1], 1)
    assert bounded_band([98], .02).can_expand(102)
    assert not bounded_band([98], .02).can_expand(102.001)


def test_all_random_contributions_obey_the_final_bound():
    rng = random.Random(7)
    prices = [rng.uniform(1, 50) for _ in range(150)]
    groups = bounded_clusters(prices)
    assert sorted(i for group in groups for i in group) == list(range(len(prices)))
    for group in groups:
        band = bounded_band(prices[i] for i in group)
        assert all(band.accepts(prices[i]) for i in group)


def test_fixed_pivot_mode_applies_final_tolerance_to_support_too(monkeypatch):
    import pandas as pd
    from app.services import strategy

    frame = pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-01", periods=8),
        "open": [10] * 8, "close": [10] * 8, "high": [10.1] * 8,
        "low": [9.7, 9.7, 9.7, 9.7, 9.85, 9.7, 9.7, 9.7], "volume": [100] * 8,
    })
    monkeypatch.setattr(strategy, "_support_touch_indices", lambda *args: [4, 6])
    lines = strategy._cluster_key_lines(
        frame, [0, 2, 4], strategy.StrategyConfig(precision_engine="pivot_fixed"), [4, 6],
        {"peak_confirmed_at": [1, 3, 5], "valley_confirmed_at": [5, 7]}, [0, 2, 4],
    )
    assert len(lines) == 1
    assert lines[0].support_indices == []
