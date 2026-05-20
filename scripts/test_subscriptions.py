"""Sanity script for the Subscribe & Save feature.

Builds an isolated temp store + temp cart, subscribes a demo user to a
few products that are already due, runs the fulfillment cron, and
prints the resulting state plus an estimated monthly recurring revenue
figure.

Run from project root::

    python -m scripts.test_subscriptions
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.shopping.cart import CartStore
from src.shopping.subscriptions import (
    SubscriptionStore,
    fulfill_due_subscriptions,
)


def _pretty(obj) -> str:
    """Compact, deterministic-ish JSON for stdout."""
    return json.dumps(obj, indent=2, default=float, sort_keys=True)


def main() -> None:
    user_id = 42

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        subs_db = tmp_dir / "subscriptions.db"
        cart_db = tmp_dir / "shopping.db"

        # 1. Create temp store + temp cart store.
        sub_store = SubscriptionStore(subs_db)
        cart_store = CartStore(cart_db)

        # 2. Subscribe user_id=42 to 3 products on 3 different cadences.
        #    All set to "already due" via first_delivery_offset_days=0.
        sub_milk = sub_store.subscribe(
            user_id=user_id, product_id=1001,
            frequency="weekly", qty=2,
            first_delivery_offset_days=0,
        )
        sub_eggs = sub_store.subscribe(
            user_id=user_id, product_id=2002,
            frequency="biweekly", qty=1,
            first_delivery_offset_days=0,
        )
        sub_oil = sub_store.subscribe(
            user_id=user_id, product_id=3003,
            frequency="monthly", qty=1,
            first_delivery_offset_days=0,
        )

        # 3. Print initial subscriptions list.
        print("=" * 64)
        print(f"Initial subscriptions for user_id={user_id}")
        print("=" * 64)
        initial = sub_store.get_user_subscriptions(user_id=user_id)
        print(_pretty(initial))
        print(f"\nactive count: {sub_store.count_active(user_id)}")
        print(
            "subscription ids: "
            f"milk={sub_milk}, eggs={sub_eggs}, oil={sub_oil}"
        )

        # 4. Run the cron and print result + new cart contents.
        print("\n" + "=" * 64)
        print("fulfill_due_subscriptions(...)")
        print("=" * 64)
        fired = fulfill_due_subscriptions(sub_store, cart_store)
        print(f"fired {len(fired)} subscription(s):")
        print(_pretty(fired))

        print("\nCart contents after fulfillment:")
        print(_pretty(cart_store.get_cart(user_id)))

        print("\nSubscriptions after advance:")
        print(_pretty(sub_store.get_user_subscriptions(user_id=user_id)))

        # 5. Estimated monthly value with a fake price_map.
        print("\n" + "=" * 64)
        print("estimated_monthly_value(...)")
        print("=" * 64)
        price_map = {
            1001: 4.99,   # milk
            2002: 6.49,   # eggs
            3003: 12.99,  # oil
        }
        emv = sub_store.estimated_monthly_value(
            user_id=user_id, price_map=price_map
        )
        # Show the per-sub math for the reader.
        # weekly milk qty=2 @ 4.99 -> 30/7 * 2 * 4.99
        # biweekly eggs qty=1 @ 6.49 -> 30/14 * 1 * 6.49
        # monthly oil qty=1 @ 12.99 -> 30/30 * 1 * 12.99
        print(f"price_map: {price_map}")
        print(f"estimated_monthly_value: ${emv:.2f}")

        # Cleanup.
        sub_store.close()
        cart_store.close()


if __name__ == "__main__":
    main()
