"""Tests for the query understanding module (spelling + autocomplete)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.search.query_understanding import (
    AutocompleteSuggester,
    QueryVocabulary,
    SpellingCorrector,
    levenshtein,
)

DATA_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "product_catalog.parquet"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip(f"Catalog not found at {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def vocabulary(catalog: pd.DataFrame) -> QueryVocabulary:
    return QueryVocabulary(catalog)


@pytest.fixture(scope="module")
def corrector(vocabulary: QueryVocabulary) -> SpellingCorrector:
    return SpellingCorrector(vocabulary)


@pytest.fixture(scope="module")
def suggester(vocabulary: QueryVocabulary) -> AutocompleteSuggester:
    return AutocompleteSuggester(vocabulary)


# ---------------------------------------------------------------------------
# Tiny helper
# ---------------------------------------------------------------------------


def test_levenshtein_basic_cases():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("yogrt", "yogurt") == 1
    assert levenshtein("kitten", "sitting") == 3
    # Early-exit honoured.
    assert levenshtein("kitten", "sitting", max_distance=1) > 1


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocabulary_extracts_thousands_of_unigrams(vocabulary: QueryVocabulary):
    assert len(vocabulary.unigrams) >= 1000
    # Bigrams and full names should also be plentiful.
    assert len(vocabulary.bigrams) >= 1000
    assert len(vocabulary.full_names) >= 1000
    # Categories and departments come from the catalog metadata.
    assert vocabulary.categories, "expected non-empty category list"
    assert vocabulary.departments, "expected non-empty department list"


def test_vocabulary_skips_stopwords_and_short_tokens(vocabulary: QueryVocabulary):
    for stopword in ("the", "and", "of", "with"):
        assert stopword not in vocabulary.unigrams
    # No single-character or empty tokens.
    for word in vocabulary.unigrams:
        assert len(word) >= 2
        assert word == word.lower()


def test_vocab_words_returns_sorted_unique_list(vocabulary: QueryVocabulary):
    words = vocabulary.vocab_words()
    assert len(words) == len(set(words))
    assert words == sorted(words)
    # Common grocery vocabulary should be present.
    for expected in ("yogurt", "banana", "organic", "milk"):
        assert expected in words


def test_vocabulary_builds_in_under_five_seconds(catalog: pd.DataFrame):
    start = time.perf_counter()
    QueryVocabulary(catalog)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"vocabulary build took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Spelling corrector
# ---------------------------------------------------------------------------


def test_correct_yogurt_typo(corrector: SpellingCorrector):
    assert corrector.correct("yogrt") == "yogurt"


def test_correct_multi_word_query(corrector: SpellingCorrector):
    assert corrector.correct("organik strawberries") == "organic strawberries"


def test_correct_leaves_correct_word_alone(corrector: SpellingCorrector):
    assert corrector.correct("banana") == "banana"


def test_correct_returns_input_when_no_good_candidate(
    corrector: SpellingCorrector,
):
    assert corrector.correct("xyzqwerty") == "xyzqwerty"


def test_correct_does_not_change_short_stopwords(corrector: SpellingCorrector):
    # Stopword should be preserved verbatim.
    assert corrector.correct("the") == "the"
    # And it should survive embedded inside a longer phrase too.
    assert "the" in corrector.correct("the milk").split()


def test_correct_keeps_short_words_safe(corrector: SpellingCorrector):
    # Short real words should not be coerced into other vocab entries just
    # because they happen to differ by a single letter.
    assert corrector.correct("cat") in {"cat", "can"}  # cat may not be in vocab
    # But 'milk' is definitely in vocab and must stay milk.
    assert corrector.correct("milk") == "milk"


def test_suggest_corrections_returns_alternatives(corrector: SpellingCorrector):
    suggestions = corrector.suggest_corrections("yogrt", top_k=3)
    assert isinstance(suggestions, list)
    assert len(suggestions) <= 3
    assert "yogurt" in suggestions


def test_suggest_corrections_empty_for_correct_query(corrector: SpellingCorrector):
    # If the query needs no correction, suggest_corrections returns [].
    assert corrector.suggest_corrections("banana", top_k=3) == []


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


def test_suggest_yog_prefix(suggester: AutocompleteSuggester):
    suggestions = suggester.suggest("yog", top_k=8)
    assert suggestions, "expected suggestions for 'yog'"
    for s in suggestions:
        assert s["text"].lower().startswith("yog")
        assert s["type"] in {"product", "category", "keyword"}
        assert isinstance(s["score"], int)
    texts = [s["text"].lower() for s in suggestions]
    # 'yogurt' should appear in the top half of the suggestion list.
    assert any("yogurt" in t for t in texts[: max(1, len(texts) // 2 + 1)])


def test_suggest_bana_prefix(suggester: AutocompleteSuggester):
    suggestions = suggester.suggest("bana", top_k=8)
    assert suggestions
    assert any("banana" in s["text"].lower() for s in suggestions)


def test_suggest_empty_prefix_returns_popular_categories(
    suggester: AutocompleteSuggester,
):
    suggestions = suggester.suggest("", top_k=5)
    assert suggestions, "empty prefix should fall back to popular categories"
    assert all(s["type"] == "category" for s in suggestions)


def test_suggest_unknown_prefix_returns_empty(suggester: AutocompleteSuggester):
    assert suggester.suggest("xyzqwerty", top_k=8) == []


def test_suggest_respects_top_k(suggester: AutocompleteSuggester):
    assert len(suggester.suggest("milk", top_k=3)) <= 3
    assert len(suggester.suggest("milk", top_k=8)) <= 8


def test_suggest_case_insensitive(suggester: AutocompleteSuggester):
    lower = suggester.suggest("yog", top_k=5)
    upper = suggester.suggest("YOG", top_k=5)
    assert [s["text"] for s in lower] == [s["text"] for s in upper]
