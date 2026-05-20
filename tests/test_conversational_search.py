"""Tests for :func:`src.models.llm.conversational_search_followup`.

We mock the OpenAI client entirely so no network call is made; the
tests verify the prompt structure and the response-parsing contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.models.llm import conversational_search_followup


# ----------------------------------------------------------------------
# Tiny fake OpenAI client
# ----------------------------------------------------------------------


class _FakeCompletions:
    """Captures the call and returns a hard-coded JSON content string."""

    def __init__(self, content: str, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise_exc = raise_exc
        self.last_call_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any):
        self.last_call_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    """Quacks like ``openai.OpenAI`` for the bits we use."""

    def __init__(self, content: str = "{}", raise_exc: Exception | None = None) -> None:
        self._completions = _FakeCompletions(content, raise_exc)
        self.chat = _FakeChat(self._completions)

    @property
    def last_call_kwargs(self) -> dict[str, Any] | None:
        return self._completions.last_call_kwargs


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_prompt_contains_previous_query_and_followup() -> None:
    """The user message must carry both previous_query and user_followup
    so the model has the context it needs to decide between refinement
    and topic-shift."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "yogurt",
                "interpreted_filters": {"attributes": ["organic"]},
                "clarification": "Searching yogurt with organic filter.",
            }
        )
    )
    conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="make it organic",
        client=fake,
    )
    kwargs = fake.last_call_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}

    messages = kwargs["messages"]
    # System + user.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert "yogurt" in user_content
    assert "make it organic" in user_content


def test_refinement_merges_into_prev_filters() -> None:
    """A refinement preserves the previous query and the model's filters
    get parsed through to ``interpreted_filters``."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "almond milk",
                "interpreted_filters": {
                    "attributes": ["organic"],
                    "sort_by": "price",
                },
                "clarification": "Sorting organic almond milk by price ascending.",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="almond milk",
        previous_filters={"attributes": ["organic"]},
        user_followup="cheaper one",
        client=fake,
    )
    assert result["interpreted_query"] == "almond milk"
    assert result["interpreted_filters"]["attributes"] == ["organic"]
    assert result["interpreted_filters"]["sort_by"] == "price"
    assert "price" in result["clarification"].lower()


def test_topic_shift_resets_filters() -> None:
    """If the model returns a new query and an empty filter object, we
    propagate that — the search engine should treat it as a fresh
    search."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "bread",
                "interpreted_filters": {},
                "clarification": "Switching to bread search.",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={"attributes": ["organic"]},
        user_followup="show me bread instead",
        client=fake,
    )
    assert result["interpreted_query"] == "bread"
    assert result["interpreted_filters"] == {}
    assert "bread" in result["clarification"].lower()


def test_malformed_json_returns_fallback() -> None:
    """A model that returns broken JSON should NOT crash the caller — we
    return a graceful fallback that keeps the previous query."""
    fake = _FakeClient(content="not valid json at all }{")
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={"attributes": ["organic"]},
        user_followup="something",
        client=fake,
    )
    assert result["interpreted_query"] == "yogurt"
    assert result["interpreted_filters"] == {}
    assert isinstance(result["clarification"], str)
    assert result["clarification"]  # non-empty


def test_non_dict_response_returns_fallback() -> None:
    """If the model returns a JSON list (not an object) we fall back."""
    fake = _FakeClient(content=json.dumps(["yogurt", "organic"]))
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="make it organic",
        client=fake,
    )
    assert result["interpreted_query"] == "yogurt"
    assert result["interpreted_filters"] == {}


def test_client_exception_returns_fallback() -> None:
    """A network/client error should not propagate — return fallback."""
    fake = _FakeClient(raise_exc=RuntimeError("boom"))
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="make it organic",
        client=fake,
    )
    assert result["interpreted_query"] == "yogurt"
    assert result["interpreted_filters"] == {}


def test_unknown_sort_by_dropped() -> None:
    """Unknown sort_by values are filtered out (not echoed through)."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "yogurt",
                "interpreted_filters": {
                    "attributes": ["organic"],
                    "sort_by": "popularity_descending_madeup",
                },
                "clarification": "Searching organic yogurt.",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="organic ones",
        client=fake,
    )
    assert result["interpreted_filters"].get("sort_by") is None
    assert result["interpreted_filters"]["attributes"] == ["organic"]


def test_attributes_deduplicated() -> None:
    """Duplicate attributes in the model response are collapsed."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "yogurt",
                "interpreted_filters": {
                    "attributes": ["organic", "organic", " ORGANIC ", "vegan"],
                },
                "clarification": "ok",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="organic vegan ones",
        client=fake,
    )
    attrs = result["interpreted_filters"]["attributes"]
    # Deduped and normalized to lower.
    assert attrs.count("organic") == 1
    assert "vegan" in attrs


def test_missing_interpreted_query_defaults_to_previous() -> None:
    """If the model forgets interpreted_query we default to previous_query."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_filters": {"attributes": ["organic"]},
                "clarification": "ok",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="yogurt",
        previous_filters={},
        user_followup="organic",
        client=fake,
    )
    assert result["interpreted_query"] == "yogurt"


def test_returns_expected_shape() -> None:
    """Every successful call returns the same three top-level keys."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "milk",
                "interpreted_filters": {},
                "clarification": "ok",
            }
        )
    )
    result = conversational_search_followup(
        previous_query="milk",
        previous_filters={},
        user_followup="any",
        client=fake,
    )
    assert set(result.keys()) == {
        "interpreted_query",
        "interpreted_filters",
        "clarification",
    }
    assert isinstance(result["interpreted_query"], str)
    assert isinstance(result["interpreted_filters"], dict)
    assert isinstance(result["clarification"], str)


def test_previous_filters_included_in_user_message() -> None:
    """The previous_filters dict must be serialized into the user prompt
    so the model can see what was already active."""
    fake = _FakeClient(
        content=json.dumps(
            {
                "interpreted_query": "almond milk",
                "interpreted_filters": {"attributes": ["organic"]},
                "clarification": "ok",
            }
        )
    )
    conversational_search_followup(
        previous_query="almond milk",
        previous_filters={"attributes": ["organic"], "sort_by": "price"},
        user_followup="anything else?",
        client=fake,
    )
    user_content = fake.last_call_kwargs["messages"][1]["content"]
    assert "organic" in user_content
    assert "price" in user_content
