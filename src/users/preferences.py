"""SQLite-backed dietary preferences per user.

A user's preferences are two parallel lists of attribute IDs from
:mod:`src.search.attributes`:

* ``dietary_attributes`` — positive constraints. Every product surfaced
  in search MUST have ALL of these attributes (e.g. ``["vegan",
  "gluten-free"]`` keeps only products tagged with both).
* ``excluded_attributes`` — negative constraints. Products that match
  ANY of these are dropped (e.g. ``["high-sugar"]`` filters out
  high-sugar items; or you might list specific allergens / categories
  the user wants to avoid).

The store is single-row-per-user (PRIMARY KEY ``user_id``). Setting
preferences replaces the previous values rather than merging. Pass
``None`` or empty lists to clear.

Schema::

    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id              INTEGER PRIMARY KEY,
        dietary_attributes   TEXT NOT NULL,  -- JSON list[str]
        excluded_attributes  TEXT NOT NULL,  -- JSON list[str]
        updated_at           REAL NOT NULL
    );

The class supports the context-manager protocol and uses ``stdlib
sqlite3`` only (no external deps).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

# Default location, relative to the project root. Created on demand.
DEFAULT_PREFS_DB = Path("data/processed/preferences.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_attributes(values: Iterable[str] | None) -> list[str]:
    """Coerce a user-supplied attribute list into a clean, deduplicated
    list of non-empty strings.

    ``None`` is treated as "no constraints" and becomes ``[]``. Whitespace-
    only entries and non-string values are silently dropped. Original
    ordering is preserved for the first occurrence of each ID, which gives
    the caller stable round-tripping behavior in tests.
    """
    if not values:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PreferenceStore:
    """SQLite-backed dietary preferences per user.

    Two lists per user:
        dietary_attributes: positive — every product must have ALL of these
                           (uses the same attribute IDs as
                           ``src/search/attributes.py``:
                           ``organic``, ``gluten-free``, ``vegan``,
                           ``high-protein``, etc.)
        excluded_attributes: negative — products must NOT match any of these
                            (e.g. user with a peanut allergy excludes
                            products with ``nut-free`` missing, OR user
                            excludes ``high-sugar``).
    """

    # --------------------------------------------------------------
    # Construction / lifecycle
    # --------------------------------------------------------------
    def __init__(self, db_path: str | Path | None = None):
        """Open or create the database.

        Parameters
        ----------
        db_path:
            Filesystem path to the SQLite database. ``None`` defaults
            to :data:`DEFAULT_PREFS_DB`. The parent directory is created
            if it does not exist. Pass ``":memory:"`` for tests that
            don't want to touch the filesystem.
        """
        if db_path is None:
            db_path = DEFAULT_PREFS_DB
        # ``:memory:`` is a magic SQLite path; do not touch the FS for it.
        path_str = str(db_path)
        if path_str != ":memory:":
            parent = Path(path_str).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)

        # ``check_same_thread=False`` keeps things flexible if the caller
        # accesses the store from a worker thread (e.g. FastAPI).
        self._conn: sqlite3.Connection = sqlite3.connect(
            path_str, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id              INTEGER PRIMARY KEY,
                dietary_attributes   TEXT NOT NULL,
                excluded_attributes  TEXT NOT NULL,
                updated_at           REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    # --------------------------------------------------------------
    # Read / write
    # --------------------------------------------------------------
    def set_preferences(
        self,
        user_id: int,
        dietary_attributes: list[str] | None = None,
        excluded_attributes: list[str] | None = None,
    ) -> None:
        """Replace the user's preferences.

        Both lists are normalized (deduplicated, whitespace stripped,
        non-strings dropped). ``None`` is equivalent to passing an empty
        list, i.e. clears that list. Passing empty lists on BOTH sides
        removes the user's row entirely so they no longer count as
        "having preferences" (see :meth:`list_users_with_prefs`).
        """
        dietary = _normalize_attributes(dietary_attributes)
        excluded = _normalize_attributes(excluded_attributes)

        # If both lists are empty we treat it as a clear: drop the row.
        # This keeps list_users_with_prefs() honest — a user with no
        # constraints is indistinguishable from a user who never set
        # preferences, so they should not show up.
        if not dietary and not excluded:
            self._conn.execute(
                "DELETE FROM user_preferences WHERE user_id = ?",
                (int(user_id),),
            )
            self._conn.commit()
            return

        now = time.time()
        self._conn.execute(
            """
            INSERT INTO user_preferences
                (user_id, dietary_attributes, excluded_attributes, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                dietary_attributes  = excluded.dietary_attributes,
                excluded_attributes = excluded.excluded_attributes,
                updated_at          = excluded.updated_at
            """,
            (
                int(user_id),
                json.dumps(dietary),
                json.dumps(excluded),
                now,
            ),
        )
        self._conn.commit()

    def get_preferences(self, user_id: int) -> dict:
        """Return ``{"dietary_attributes": [...], "excluded_attributes": [...]}``.

        Both lists are empty when the user has no row in the table.
        """
        cur = self._conn.execute(
            """
            SELECT dietary_attributes, excluded_attributes
            FROM user_preferences
            WHERE user_id = ?
            """,
            (int(user_id),),
        )
        row = cur.fetchone()
        if row is None:
            return {"dietary_attributes": [], "excluded_attributes": []}

        # Defensive JSON parse — if a row got corrupted we'd rather return
        # empty than crash the caller's search request.
        try:
            dietary = json.loads(row["dietary_attributes"])
            if not isinstance(dietary, list):
                dietary = []
        except (TypeError, ValueError, json.JSONDecodeError):
            dietary = []
        try:
            excluded = json.loads(row["excluded_attributes"])
            if not isinstance(excluded, list):
                excluded = []
        except (TypeError, ValueError, json.JSONDecodeError):
            excluded = []

        return {
            "dietary_attributes": [str(a) for a in dietary],
            "excluded_attributes": [str(a) for a in excluded],
        }

    def list_users_with_prefs(self) -> list[int]:
        """Return the set of user_ids that currently have preferences set,
        as a list sorted ascending. Users whose row was deleted (either
        explicitly or by clearing all constraints) are excluded.
        """
        cur = self._conn.execute(
            "SELECT user_id FROM user_preferences ORDER BY user_id ASC"
        )
        return [int(r["user_id"]) for r in cur.fetchall()]

    # --------------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------


def apply_preferences_to_search(
    products: list[dict],
    preferences: dict,
) -> list[dict]:
    """Filter a list of search-result product dicts by a user's preferences.

    Required attributes (``preferences["dietary_attributes"]``): every
    product MUST have ALL of these in its ``attributes`` field (a list
    of attribute IDs). Products missing the ``attributes`` field, or
    missing any of the required IDs, are filtered out.

    Excluded attributes (``preferences["excluded_attributes"]``): products
    whose ``attributes`` field contains ANY of these are filtered out.

    If both lists are empty the input is returned unchanged (the same
    list ordering, original dict references — no copying).

    Parameters
    ----------
    products:
        A list of dicts as returned by the search engine. Each may carry
        an ``attributes`` key whose value is an iterable of attribute IDs.
    preferences:
        Output of :meth:`PreferenceStore.get_preferences`. Missing keys
        are treated as empty lists.

    Returns
    -------
    list[dict]
        The filtered products, preserving original order.
    """
    if not isinstance(preferences, dict):
        return list(products)

    required = _normalize_attributes(preferences.get("dietary_attributes"))
    excluded = _normalize_attributes(preferences.get("excluded_attributes"))

    # Fast path: no constraints means no filtering at all.
    if not required and not excluded:
        return list(products)

    excluded_set = set(excluded)
    required_set = set(required)

    filtered: list[dict] = []
    for product in products:
        # Coerce attributes to a set of strings. Accept list / tuple /
        # set / generator; reject other types (treat as missing).
        raw_attrs = product.get("attributes") if isinstance(product, dict) else None
        if raw_attrs is None:
            # No attributes field at all.
            if required_set:
                # User requires SOMETHING — cannot prove this product
                # has it, so drop it.
                continue
            # No requirements; an absent attributes field also means
            # no excluded match is possible, so keep the product.
            filtered.append(product)
            continue

        if isinstance(raw_attrs, (list, tuple, set, frozenset)):
            attrs = {str(a) for a in raw_attrs}
        else:
            # Unknown shape — treat as missing.
            if required_set:
                continue
            filtered.append(product)
            continue

        # Required: all of them must be present.
        if required_set and not required_set.issubset(attrs):
            continue
        # Excluded: none of them may be present.
        if excluded_set and (attrs & excluded_set):
            continue

        filtered.append(product)

    return filtered


__all__ = [
    "DEFAULT_PREFS_DB",
    "PreferenceStore",
    "apply_preferences_to_search",
]
