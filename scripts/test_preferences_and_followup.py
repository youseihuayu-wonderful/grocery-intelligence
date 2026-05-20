"""End-to-end sanity check for dietary preferences + conversational follow-up.

Run from the repo root::

    source venv/bin/activate
    python scripts/test_preferences_and_followup.py

Steps:
  1. Create a temporary :class:`PreferenceStore`.
  2. Set preferences for user 42 (dietary=['organic', 'low-sugar']).
  3. Read them back and print.
  4. Build a tiny synthetic product list with ``attributes`` fields
     and run :func:`apply_preferences_to_search`; print survivors.
  5. Exercise :func:`conversational_search_followup`:
       - If ``OPENAI_API_KEY`` is set in env, call the live API for two
         sample turns.
       - Otherwise, monkey-patch a fake OpenAI client returning
         hand-crafted JSON and demonstrate the parsing pipeline.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Make ``src`` importable when invoked as ``python scripts/...``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.users.preferences import (  # noqa: E402
    PreferenceStore,
    apply_preferences_to_search,
)


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
# Steps 1-4: preferences round-trip + filter demo
# ----------------------------------------------------------------------


def step_1_to_4_preferences() -> None:
    _hr("Step 1-2 — Create PreferenceStore (temp dir), set user 42 prefs")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "preferences.db"
        print(f"DB path: {db_path}")

        with PreferenceStore(db_path) as store:
            store.set_preferences(
                user_id=42,
                dietary_attributes=["organic", "low-sugar"],
                excluded_attributes=[],
            )

            _hr("Step 3 — Read back user 42's preferences")
            prefs = store.get_preferences(user_id=42)
            print(f"User 42 preferences: {prefs}")
            print(f"Users with prefs:   {store.list_users_with_prefs()}")

            _hr("Step 4 — apply_preferences_to_search on 5 fake products")

            products: list[dict] = [
                {
                    "product_id": 1,
                    "product_name": "Organic Greek Yogurt",
                    "attributes": ["organic", "high-protein", "low-sugar"],
                },
                {
                    "product_id": 2,
                    "product_name": "Regular Yogurt",
                    "attributes": ["high-sugar"],
                },
                {
                    "product_id": 3,
                    "product_name": "Organic Vegan Yogurt",
                    "attributes": ["organic", "vegan", "low-sugar"],
                },
                {
                    "product_id": 4,
                    "product_name": "Sugar-Free Almond Milk",
                    "attributes": ["sugar-free", "low-sugar", "dairy-free"],
                },
                {
                    "product_id": 5,
                    "product_name": "Mystery Product (no attributes)",
                    # No 'attributes' field.
                },
            ]

            print("Input products:")
            for p in products:
                attrs = p.get("attributes", "<missing>")
                print(
                    f"  id={p['product_id']:>2}  "
                    f"{p['product_name']:<35}  attrs={attrs}"
                )

            survivors = apply_preferences_to_search(products, prefs)

            print("\nSurvivors after preference filter "
                  f"(require {prefs['dietary_attributes']}, "
                  f"exclude {prefs['excluded_attributes']}):")
            if not survivors:
                print("  (none)")
            for p in survivors:
                print(
                    f"  id={p['product_id']:>2}  {p['product_name']}  "
                    f"attrs={p.get('attributes')}"
                )


# ----------------------------------------------------------------------
# Step 5: conversational follow-up
# ----------------------------------------------------------------------


class _FakeCompletions:
    """A scripted fake — returns a different JSON payload on each call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def create(self, **kwargs: Any):
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = self._responses[-1]
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeOpenAIClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def step_5_followup() -> None:
    _hr("Step 5 — conversational_search_followup")

    # Import here so we get the live module (and so an import error
    # surfaces here instead of at the top).
    from src.models.llm import conversational_search_followup

    use_live = bool(os.getenv("OPENAI_API_KEY"))
    if use_live:
        print("OPENAI_API_KEY found — calling the real LLM.")
        client = None
    else:
        print("OPENAI_API_KEY not set — using a scripted fake client.")
        import json as _json

        responses = [
            # Turn 1 response: refinement (add organic to yogurt).
            _json.dumps(
                {
                    "interpreted_query": "yogurt",
                    "interpreted_filters": {"attributes": ["organic"]},
                    "clarification": "Searching yogurt with the organic filter.",
                }
            ),
            # Turn 2 response: sort hint (cheaper).
            _json.dumps(
                {
                    "interpreted_query": "yogurt",
                    "interpreted_filters": {
                        "attributes": ["organic"],
                        "sort_by": "price",
                    },
                    "clarification": "Sorting organic yogurt by price ascending.",
                }
            ),
            # Turn 3 response: topic shift to bread.
            _json.dumps(
                {
                    "interpreted_query": "bread",
                    "interpreted_filters": {},
                    "clarification": "Switching to bread search.",
                }
            ),
        ]
        client = _FakeOpenAIClient(responses)

    turns = [
        # (previous_query, previous_filters, user_followup, label)
        ("yogurt", {}, "make it organic", "Turn 1 — refinement"),
        (
            "yogurt",
            {"attributes": ["organic"]},
            "cheaper one",
            "Turn 2 — sort hint",
        ),
        (
            "yogurt",
            {"attributes": ["organic"], "sort_by": "price"},
            "show me bread instead",
            "Turn 3 — topic shift",
        ),
    ]

    for prev_query, prev_filters, followup, label in turns:
        print(f"\n[{label}]")
        print(f"  prev_query    : {prev_query!r}")
        print(f"  prev_filters  : {prev_filters}")
        print(f"  user_followup : {followup!r}")
        try:
            result = conversational_search_followup(
                previous_query=prev_query,
                previous_filters=prev_filters,
                user_followup=followup,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  -> ERROR: {type(exc).__name__}: {exc}")
            continue
        print(f"  -> interpreted_query   : {result['interpreted_query']!r}")
        print(f"  -> interpreted_filters : {result['interpreted_filters']}")
        print(f"  -> clarification       : {result['clarification']!r}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> int:
    step_1_to_4_preferences()
    step_5_followup()
    _hr("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
