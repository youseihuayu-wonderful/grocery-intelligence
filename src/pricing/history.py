"""Deterministic 90-day price-history simulator.

For every product we want to draw a believable price chart: a mostly
flat-ish series with occasional sale-event dips and a few percent of
day-over-day jitter. The series is seeded on ``product_id`` so the
same product always shows the same chart across runs.

Public API:
    - :func:`generate_price_history`
    - :func:`detect_price_drop`
"""

from __future__ import annotations

import datetime as _dt
import random
import statistics
from typing import Iterable


# --------------------------------------------------------------------
# Tuning knobs. Pulled out so they're easy to find / tweak.
# --------------------------------------------------------------------

# Max daily random-walk step (as a fraction of the current price).
_MAX_DAILY_STEP_PCT = 0.02

# Mean-reversion factor toward the current price each day (0..1).
# Small value = slow drift back; larger = sticks closer to current.
_MEAN_REVERSION = 0.10

# Starting offset from current_price (the walk starts near here).
# Picked deterministically per product in [-12%, +12%].
_START_OFFSET_PCT_RANGE = (-0.12, 0.12)

# Hard floor on any simulated price (so we never produce $0 or negative).
_MIN_PRICE = 0.50

# Sale events: 0 to 2 per series, each lasting 3-7 days, 15-30% drop.
_MAX_SALES_PER_SERIES = 2
_SALE_LEN_RANGE = (3, 7)
_SALE_DROP_PCT_RANGE = (0.15, 0.30)

# Significance threshold for detect_price_drop.
_SIGNIFICANT_DROP_PCT = 10.0


def generate_price_history(
    product_id: int,
    current_price: float,
    days: int = 90,
) -> list[dict]:
    """Simulate a 90-day price series ending at ``current_price``.

    The series is deterministic per ``product_id``: same id always
    returns the same sequence. Daily step is a bounded random walk
    around ``current_price`` with mild mean reversion, plus up to two
    "sale event" windows that knock 15-30% off for 3-7 days.

    The final entry (today) is forced to equal ``current_price``
    exactly so the chart joins seamlessly with the live cart price.

    Parameters
    ----------
    product_id:
        Used as the RNG seed. Same id → identical history.
    current_price:
        The "today" price; the last row of the returned series.
    days:
        How many entries to return (default 90).

    Returns
    -------
    list[dict]
        ``[{date: "YYYY-MM-DD", price: float}, ...]``, oldest first.
        Length ``== days``. Final entry's price is exactly
        ``current_price``.
    """
    if days <= 0:
        return []
    # Guard against a zero / negative seed price.
    if current_price <= 0:
        current_price = _MIN_PRICE

    rng = random.Random(int(product_id))

    # Walk forward day by day from a slightly-offset starting price.
    start_offset = rng.uniform(*_START_OFFSET_PCT_RANGE)
    price = max(_MIN_PRICE, current_price * (1.0 + start_offset))

    prices: list[float] = []
    for _ in range(days):
        prices.append(price)
        # Random step in [-step, +step], plus pull toward current_price.
        step = rng.uniform(-_MAX_DAILY_STEP_PCT, _MAX_DAILY_STEP_PCT)
        revert = (current_price - price) * _MEAN_REVERSION
        price = price * (1.0 + step) + revert
        if price < _MIN_PRICE:
            price = _MIN_PRICE

    # --- Sale events --------------------------------------------------
    n_sales = rng.randint(0, _MAX_SALES_PER_SERIES)
    for _ in range(n_sales):
        sale_len = rng.randint(*_SALE_LEN_RANGE)
        # Window must fit entirely BEFORE the last day so the final
        # row remains == current_price.
        if sale_len >= days - 1:
            continue
        # Latest allowed start is days - sale_len - 1 (leave at least
        # one non-sale day at the very end so we can pin to current).
        start = rng.randint(0, days - sale_len - 1)
        drop_pct = rng.uniform(*_SALE_DROP_PCT_RANGE)
        for i in range(start, start + sale_len):
            prices[i] = max(_MIN_PRICE, prices[i] * (1.0 - drop_pct))

    # Pin the final entry to the exact current price.
    prices[-1] = float(current_price)

    # --- Stamp on calendar dates -------------------------------------
    today = _dt.date.today()
    out: list[dict] = []
    for i, p in enumerate(prices):
        # Index 0 is the oldest day; index days-1 is today.
        day = today - _dt.timedelta(days=days - 1 - i)
        out.append({"date": day.isoformat(), "price": round(float(p), 2)})
    # Make sure the rounded final price is also exact in case of FP slop.
    out[-1]["price"] = round(float(current_price), 2)
    return out


def detect_price_drop(
    history: list[dict], lookback_days: int = 30
) -> dict | None:
    """Compare today's price to the median of the prior ``lookback_days``.

    Parameters
    ----------
    history:
        Output of :func:`generate_price_history` (or any list of
        ``{date, price}`` dicts sorted oldest-first).
    lookback_days:
        How many prior days to median over. We need at least
        ``lookback_days + 1`` entries; otherwise the function returns
        ``None``.

    Returns
    -------
    dict | None
        ``{"current": ..., "previous_median": ..., "drop_pct": ...,
        "is_significant": ...}`` with ``drop_pct`` positive on a drop,
        or ``None`` if we don't have enough data.
    """
    if not history or lookback_days <= 0:
        return None
    if len(history) < lookback_days + 1:
        return None

    current = float(history[-1]["price"])
    # Median of the lookback_days entries immediately preceding today.
    window = history[-(lookback_days + 1) : -1]
    prior_prices = [float(row["price"]) for row in window]
    previous_median = float(statistics.median(prior_prices))

    if previous_median <= 0:
        # Pathological; treat as "no signal" rather than divide-by-zero.
        return {
            "current": round(current, 2),
            "previous_median": round(previous_median, 2),
            "drop_pct": 0.0,
            "is_significant": False,
        }

    # Positive number = price has dropped vs. the median.
    drop_pct = (previous_median - current) / previous_median * 100.0
    return {
        "current": round(current, 2),
        "previous_median": round(previous_median, 2),
        "drop_pct": round(drop_pct, 2),
        "is_significant": drop_pct >= _SIGNIFICANT_DROP_PCT,
    }


__all__ = ["generate_price_history", "detect_price_drop"]
