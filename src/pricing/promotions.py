"""Cart-level promotion engine.

Given a list of cart items (each with ``price``, ``qty``, ``category``,
optionally ``department`` and ``product_name``), evaluate a handful of
built-in promotion rules. Each rule returns a :class:`Promotion` that
indicates whether the cart already qualifies (``is_applied``) and how
close it is to qualifying (``progress``) — the frontend can render
either an "applied" badge or a "you're 60% there" incentive bar.

Public API:
    - :class:`Promotion`
    - :func:`evaluate_promotions`
    - :func:`cart_pricing_summary`

The functions are pure: no I/O, no globals beyond constants.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable


# --------------------------------------------------------------------
# Promotion data class
# --------------------------------------------------------------------


@dataclass
class Promotion:
    """A single promotion evaluated against a specific cart.

    Attributes
    ----------
    code:
        Stable machine identifier (e.g. ``"SPEND_50_SAVE_5"``).
    title:
        Short, human-friendly headline.
    description:
        One-line explanation of how to qualify.
    discount_amount:
        Absolute dollars off this cart. Computed against the actual
        cart contents. ``0.0`` when the promotion isn't yet applied,
        but the field is also used by callers to preview the savings
        once qualified.
    is_applied:
        ``True`` if this cart already qualifies for the promo.
    progress:
        Float in ``[0, 1]``. ``1.0`` when ``is_applied`` is true.
        Otherwise it indicates how close the cart is to qualifying —
        e.g. ``0.6`` if you need $50 and you're at $30.
    """

    code: str
    title: str
    description: str
    discount_amount: float
    is_applied: bool
    progress: float

    def to_dict(self) -> dict:
        """Convert to a plain dict (handy for JSON serialization)."""
        return asdict(self)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _line_price(item: dict) -> float:
    """Effective $ contribution of a cart line: ``price * qty``."""
    price = float(item.get("price", 0.0) or 0.0)
    qty = int(item.get("qty", 1) or 1)
    if qty < 0:
        qty = 0
    if price < 0:
        price = 0.0
    return price * qty


def _subtotal(cart_items: Iterable[dict]) -> float:
    """Sum of ``price * qty`` over every cart line."""
    return sum(_line_price(i) for i in cart_items)


def _department_of(item: dict) -> str:
    """Lower-cased department name; falls back to ``category``."""
    dept = item.get("department") or item.get("category") or ""
    if dept is None:
        return ""
    return str(dept).strip().lower()


def _is_organic(item: dict) -> bool:
    """True if the item has any signal of being organic.

    Checks ``attributes`` (list[str]), ``product_name``, and the legacy
    ``is_organic`` flag if present.
    """
    if item.get("is_organic"):
        return True
    attrs = item.get("attributes") or []
    if isinstance(attrs, (list, tuple)):
        if any(str(a).strip().lower() == "organic" for a in attrs):
            return True
    name = (item.get("product_name") or "").lower()
    return "organic" in name


def _qty_in_department(cart_items: Iterable[dict], dept: str) -> int:
    """Total qty across all lines in the given department."""
    target = dept.lower()
    total = 0
    for item in cart_items:
        if _department_of(item) == target:
            total += max(0, int(item.get("qty", 1) or 1))
    return total


def _subtotal_in_department(cart_items: Iterable[dict], dept: str) -> float:
    target = dept.lower()
    return sum(_line_price(i) for i in cart_items if _department_of(i) == target)


# --------------------------------------------------------------------
# Promotion rules
# --------------------------------------------------------------------
# Each rule is a small function that returns a Promotion. They share a
# common contract: never raise on weird input — return progress=0 and
# is_applied=False instead.


def _promo_spend_threshold(
    cart_items: list[dict],
    *,
    code: str,
    title: str,
    description: str,
    threshold: float,
    discount: float,
) -> Promotion:
    """Generic "spend at least $X → $Y off" promotion."""
    subtotal = _subtotal(cart_items)
    qualified = subtotal >= threshold
    if qualified:
        return Promotion(
            code=code,
            title=title,
            description=description,
            discount_amount=round(discount, 2),
            is_applied=True,
            progress=1.0,
        )
    # Not yet — show progress.
    progress = 0.0 if threshold <= 0 else min(1.0, subtotal / threshold)
    return Promotion(
        code=code,
        title=title,
        description=description,
        discount_amount=0.0,
        is_applied=False,
        progress=round(progress, 4),
    )


def _promo_dept_qty(
    cart_items: list[dict],
    *,
    code: str,
    title: str,
    description: str,
    department: str,
    min_qty: int,
    percent_off: float,
) -> Promotion:
    """``min_qty`` items in ``department`` → ``percent_off`` off those items."""
    qty = _qty_in_department(cart_items, department)
    dept_subtotal = _subtotal_in_department(cart_items, department)
    if qty >= min_qty:
        return Promotion(
            code=code,
            title=title,
            description=description,
            discount_amount=round(dept_subtotal * percent_off, 2),
            is_applied=True,
            progress=1.0,
        )
    progress = 0.0 if min_qty <= 0 else min(1.0, qty / float(min_qty))
    return Promotion(
        code=code,
        title=title,
        description=description,
        discount_amount=0.0,
        is_applied=False,
        progress=round(progress, 4),
    )


def _promo_first_organic(cart_items: list[dict]) -> Promotion:
    """$2 off if the cart contains at least one organic item."""
    has_organic = any(_is_organic(i) for i in cart_items)
    if has_organic:
        return Promotion(
            code="FIRST_ORGANIC",
            title="Try Organic",
            description="Save $2 when your cart includes an organic item.",
            discount_amount=2.00,
            is_applied=True,
            progress=1.0,
        )
    return Promotion(
        code="FIRST_ORGANIC",
        title="Try Organic",
        description="Add any organic item to save $2.",
        discount_amount=0.0,
        is_applied=False,
        progress=0.0,
    )


# --------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------


def evaluate_promotions(cart_items: list[dict]) -> list[Promotion]:
    """Run every built-in promotion rule against ``cart_items``.

    Parameters
    ----------
    cart_items:
        Each item is a dict with at least ``price`` and ``qty``.
        Optional but used: ``department`` (or ``category``),
        ``product_name``, ``attributes`` (list of strings),
        ``is_organic``.

    Returns
    -------
    list[Promotion]
        Every rule, regardless of whether it was applied. Callers can
        split into "applied" and "incentive" sets via :attr:`Promotion.is_applied`.
    """
    items = list(cart_items or [])
    promos = [
        _promo_spend_threshold(
            items,
            code="SPEND_50_SAVE_5",
            title="$5 off $50",
            description="Spend $50 or more, get $5 off your order.",
            threshold=50.0,
            discount=5.0,
        ),
        _promo_spend_threshold(
            items,
            code="SPEND_100_SAVE_15",
            title="$15 off $100",
            description="Spend $100 or more, get $15 off your order.",
            threshold=100.0,
            discount=15.0,
        ),
        _promo_dept_qty(
            items,
            code="BUY_2_DAIRY",
            title="10% off dairy",
            description="Add 2 or more dairy & eggs items, save 10% on dairy.",
            department="dairy eggs",
            min_qty=2,
            percent_off=0.10,
        ),
        _promo_dept_qty(
            items,
            code="BUY_3_PRODUCE",
            title="15% off produce",
            description="Add 3 or more produce items, save 15% on produce.",
            department="produce",
            min_qty=3,
            percent_off=0.15,
        ),
        _promo_first_organic(items),
    ]
    return promos


def cart_pricing_summary(cart_items: list[dict]) -> dict:
    """Compute subtotal, applied discounts, and the final cart total.

    Returns a dict in the shape::

        {
          "subtotal": float,
          "promotions_applied":   [Promotion, ...],
          "promotions_available": [Promotion, ...],  # not yet qualified
          "total_discount":       float,
          "total":                float,    # subtotal - total_discount
          "n_items":              int,
        }

    The total discount is clamped so it never exceeds the subtotal —
    a free-cart edge case from stacking promos against a tiny cart.
    """
    items = list(cart_items or [])
    subtotal = round(_subtotal(items), 2)
    promos = evaluate_promotions(items)

    applied = [p for p in promos if p.is_applied]
    available = [p for p in promos if not p.is_applied]
    total_discount = round(sum(p.discount_amount for p in applied), 2)

    # Never give a negative total.
    if total_discount > subtotal:
        total_discount = subtotal

    total = round(subtotal - total_discount, 2)
    n_items = sum(max(0, int(i.get("qty", 1) or 1)) for i in items)

    return {
        "subtotal": subtotal,
        "promotions_applied": applied,
        "promotions_available": available,
        "total_discount": round(total_discount, 2),
        "total": total,
        "n_items": int(n_items),
    }


__all__ = [
    "Promotion",
    "evaluate_promotions",
    "cart_pricing_summary",
]
