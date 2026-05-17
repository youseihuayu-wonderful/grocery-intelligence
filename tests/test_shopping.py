"""Tests for the shopping module: cart, wishlist, and order history."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.shopping.cart import CartStore
from src.shopping.orders import OrderHistoryStore


# ======================================================================
# CartStore tests
# ======================================================================


@pytest.fixture
def cart(tmp_path: Path) -> CartStore:
    """A fresh CartStore backed by a per-test temp SQLite file."""
    db_path = tmp_path / "test_shopping.db"
    store = CartStore(db_path)
    yield store
    store.close()


def test_cart_add_then_get(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    items = cart.get_cart(user_id=1)
    assert len(items) == 1
    assert items[0]["product_id"] == 100
    assert items[0]["qty"] == 2
    assert items[0]["added_at"] > 0


def test_cart_increment_on_duplicate_add(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=1)
    cart.add_to_cart(user_id=1, product_id=100, qty=3)
    items = cart.get_cart(user_id=1)
    assert len(items) == 1
    assert items[0]["qty"] == 4  # 1 + 3 = 4


def test_cart_update_qty(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=1)
    cart.update_cart_qty(user_id=1, product_id=100, qty=7)
    items = cart.get_cart(user_id=1)
    assert items[0]["qty"] == 7  # overwritten, not incremented


def test_cart_update_qty_zero_removes(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    cart.update_cart_qty(user_id=1, product_id=100, qty=0)
    assert cart.get_cart(user_id=1) == []


def test_cart_update_qty_can_create(cart: CartStore) -> None:
    """update_cart_qty on a non-existent row should insert it."""
    cart.update_cart_qty(user_id=1, product_id=100, qty=5)
    items = cart.get_cart(user_id=1)
    assert len(items) == 1
    assert items[0]["qty"] == 5


def test_cart_remove(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    cart.add_to_cart(user_id=1, product_id=200, qty=1)
    cart.remove_from_cart(user_id=1, product_id=100)
    items = cart.get_cart(user_id=1)
    assert len(items) == 1
    assert items[0]["product_id"] == 200


def test_cart_clear(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    cart.add_to_cart(user_id=1, product_id=200, qty=1)
    cart.clear_cart(user_id=1)
    assert cart.get_cart(user_id=1) == []


def test_cart_count(cart: CartStore) -> None:
    assert cart.cart_count(user_id=1) == 0
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    assert cart.cart_count(user_id=1) == 1
    cart.add_to_cart(user_id=1, product_id=200, qty=1)
    assert cart.cart_count(user_id=1) == 2
    # Incrementing an existing product does not add a distinct row.
    cart.add_to_cart(user_id=1, product_id=200, qty=5)
    assert cart.cart_count(user_id=1) == 2


def test_cart_isolated_per_user(cart: CartStore) -> None:
    cart.add_to_cart(user_id=1, product_id=100, qty=2)
    cart.add_to_cart(user_id=2, product_id=999, qty=1)
    assert cart.cart_count(user_id=1) == 1
    assert cart.cart_count(user_id=2) == 1
    assert cart.get_cart(user_id=1)[0]["product_id"] == 100
    assert cart.get_cart(user_id=2)[0]["product_id"] == 999


def test_cart_sorted_most_recent_first(cart: CartStore) -> None:
    import time as _t

    cart.add_to_cart(user_id=1, product_id=100, qty=1)
    _t.sleep(0.01)
    cart.add_to_cart(user_id=1, product_id=200, qty=1)
    _t.sleep(0.01)
    cart.add_to_cart(user_id=1, product_id=300, qty=1)
    items = cart.get_cart(user_id=1)
    pids = [it["product_id"] for it in items]
    # Most-recent first.
    assert pids == [300, 200, 100]


def test_cart_invalid_qty_raises(cart: CartStore) -> None:
    with pytest.raises(ValueError):
        cart.add_to_cart(user_id=1, product_id=100, qty=0)
    with pytest.raises(ValueError):
        cart.add_to_cart(user_id=1, product_id=100, qty=-5)


# ======================================================================
# Wishlist tests
# ======================================================================


def test_wishlist_add_and_get(cart: CartStore) -> None:
    cart.add_to_wishlist(user_id=1, product_id=100)
    items = cart.get_wishlist(user_id=1)
    assert len(items) == 1
    assert items[0]["product_id"] == 100
    assert items[0]["added_at"] > 0


def test_wishlist_no_duplicates(cart: CartStore) -> None:
    cart.add_to_wishlist(user_id=1, product_id=100)
    cart.add_to_wishlist(user_id=1, product_id=100)
    cart.add_to_wishlist(user_id=1, product_id=100)
    items = cart.get_wishlist(user_id=1)
    assert len(items) == 1


def test_wishlist_remove(cart: CartStore) -> None:
    cart.add_to_wishlist(user_id=1, product_id=100)
    cart.add_to_wishlist(user_id=1, product_id=200)
    cart.remove_from_wishlist(user_id=1, product_id=100)
    items = cart.get_wishlist(user_id=1)
    assert len(items) == 1
    assert items[0]["product_id"] == 200


def test_wishlist_is_in_wishlist(cart: CartStore) -> None:
    assert cart.is_in_wishlist(user_id=1, product_id=100) is False
    cart.add_to_wishlist(user_id=1, product_id=100)
    assert cart.is_in_wishlist(user_id=1, product_id=100) is True
    cart.remove_from_wishlist(user_id=1, product_id=100)
    assert cart.is_in_wishlist(user_id=1, product_id=100) is False


def test_wishlist_sorted_most_recent_first(cart: CartStore) -> None:
    import time as _t

    cart.add_to_wishlist(user_id=1, product_id=100)
    _t.sleep(0.01)
    cart.add_to_wishlist(user_id=1, product_id=200)
    _t.sleep(0.01)
    cart.add_to_wishlist(user_id=1, product_id=300)
    items = cart.get_wishlist(user_id=1)
    pids = [it["product_id"] for it in items]
    assert pids == [300, 200, 100]


def test_context_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "ctx.db"
    with CartStore(db_path) as store:
        store.add_to_cart(user_id=1, product_id=100, qty=2)
        assert store.cart_count(user_id=1) == 1
    # After context exit, opening again should still see the data.
    with CartStore(db_path) as store2:
        items = store2.get_cart(user_id=1)
        assert len(items) == 1
        assert items[0]["product_id"] == 100


# ======================================================================
# OrderHistoryStore tests (tiny in-memory CSV fixture)
# ======================================================================


@pytest.fixture
def tiny_orders(tmp_path: Path) -> OrderHistoryStore:
    """Build an OrderHistoryStore backed by tiny synthetic CSVs.

    Data shape:
      user 1: 2 prior orders (order_id=10 with [A,B], 11 with [A,C])
      user 2: 1 prior order  (order_id=20 with [A,A] — A bought 2x)
                              Wait, can't buy same product twice in
                              one order (PK is (order_id, product_id)).
                              Use (order_id=20 with [B,C]) and another
                              order (order_id=21 with [B]).
      user 3: 1 'train' order (filtered out by default).
    """
    orders_csv = tmp_path / "orders.csv"
    op_csv = tmp_path / "order_products__prior.csv"

    orders_csv.write_text(
        textwrap.dedent(
            """\
            order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
            10,1,prior,1,0,8,
            11,1,prior,2,3,9,15.0
            20,2,prior,1,1,10,
            21,2,prior,2,4,11,7.0
            30,3,train,1,2,12,
            """
        )
    )
    op_csv.write_text(
        textwrap.dedent(
            """\
            order_id,product_id,add_to_cart_order,reordered
            10,1,1,0
            10,2,2,0
            11,1,1,1
            11,3,2,0
            20,2,1,0
            20,3,2,0
            21,2,1,1
            30,99,1,0
            """
        )
    )
    return OrderHistoryStore(orders_csv, op_csv)


def test_orders_construct(tiny_orders: OrderHistoryStore) -> None:
    # Only 'prior' rows kept by default: 4 orders, 2 users.
    assert tiny_orders.n_orders == 4
    assert tiny_orders.n_users == 2


def test_get_user_orders_newest_first(tiny_orders: OrderHistoryStore) -> None:
    orders = tiny_orders.get_user_orders(1)
    assert len(orders) == 2
    # Newest first = order_number desc.
    assert orders[0]["order_number"] == 2
    assert orders[1]["order_number"] == 1
    # Order 11 had items [1, 3] (in cart order)
    items = orders[0]["items"]
    assert [it["product_id"] for it in items] == [1, 3]
    assert items[0]["reordered"] is True   # product 1, reordered=1
    assert items[1]["reordered"] is False  # product 3, reordered=0
    assert orders[0]["n_items"] == 2
    assert orders[0]["days_since_prior_order"] == pytest.approx(15.0)
    # First order has days_since_prior_order=None.
    assert orders[1]["days_since_prior_order"] is None


def test_get_user_orders_unknown_user(tiny_orders: OrderHistoryStore) -> None:
    assert tiny_orders.get_user_orders(99999) == []


def test_get_order(tiny_orders: OrderHistoryStore) -> None:
    o = tiny_orders.get_order(10)
    assert o is not None
    assert o["order_id"] == 10
    assert o["n_items"] == 2
    assert tiny_orders.get_order(99999) is None
    # Train orders were filtered out.
    assert tiny_orders.get_order(30) is None


def test_reorder_items_in_order(tiny_orders: OrderHistoryStore) -> None:
    # Order 10: items added in order [1 (add_to_cart_order=1), 2 (=2)]
    assert tiny_orders.reorder_items(10) == [1, 2]
    # Order 11: items added in order [1, 3]
    assert tiny_orders.reorder_items(11) == [1, 3]
    # Unknown order returns [].
    assert tiny_orders.reorder_items(99999) == []


def test_buy_again_top(tiny_orders: OrderHistoryStore) -> None:
    # User 1 bought: product 1 (orders 10, 11) = 2x, product 2 (order 10) = 1x,
    # product 3 (order 11) = 1x.
    top = tiny_orders.buy_again_top(user_id=1, top_k=10)
    assert top[0] == 1  # most frequent
    assert set(top) == {1, 2, 3}

    # User 2 bought: product 2 (orders 20, 21) = 2x, product 3 (order 20) = 1x.
    top2 = tiny_orders.buy_again_top(user_id=2, top_k=10)
    assert top2[0] == 2
    assert set(top2) == {2, 3}

    # top_k limits the result.
    assert tiny_orders.buy_again_top(user_id=1, top_k=1) == [1]
    # Unknown user → [].
    assert tiny_orders.buy_again_top(user_id=99999) == []


def test_user_cap_filters(tmp_path: Path) -> None:
    """user_cap=1 should drop the lower-activity user (user 2)."""
    orders_csv = tmp_path / "orders.csv"
    op_csv = tmp_path / "op.csv"
    orders_csv.write_text(
        textwrap.dedent(
            """\
            order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
            10,1,prior,1,0,8,
            11,1,prior,2,3,9,15.0
            12,1,prior,3,3,9,5.0
            20,2,prior,1,1,10,
            """
        )
    )
    op_csv.write_text(
        textwrap.dedent(
            """\
            order_id,product_id,add_to_cart_order,reordered
            10,1,1,0
            11,1,1,1
            12,2,1,0
            20,3,1,0
            """
        )
    )
    store = OrderHistoryStore(orders_csv, op_csv, user_cap=1)
    # Only user 1 should remain (more orders than user 2).
    assert store.n_users == 1
    assert store.get_user_orders(1) != []
    assert store.get_user_orders(2) == []
