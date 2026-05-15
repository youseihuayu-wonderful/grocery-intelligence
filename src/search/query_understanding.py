"""Query understanding: spelling correction and prefix autocomplete.

Builds a popularity-weighted vocabulary from the product catalog and exposes:

* `QueryVocabulary` — unigrams, bigrams, full product names, categories and
  departments, each weighted by ``order_count`` so that popular spellings and
  popular products outrank rare ones.
* `SpellingCorrector` — Levenshtein-based spelling correction over the
  vocabulary. Picks the popularity-weighted nearest neighbour within a small
  edit budget (1 for short words, 2 for longer ones) so we never coerce
  "cat" -> "can".
* `AutocompleteSuggester` — prefix-based "as you type" suggestions ranked by
  ``order_count``, mixing single keywords, full product names and category
  names.

Pure Python, no extra runtime dependency. Levenshtein is implemented as a
classic two-row dynamic-programming routine.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "with",
        "for",
        "in",
        "to",
        "on",
        "&",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case the text and pull out alphanumeric word tokens."""

    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def _is_kept_token(token: str) -> bool:
    """Token survives vocabulary filtering."""

    if len(token) < 2:
        return False
    if token in STOPWORDS:
        return False
    return True


def levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Classic Levenshtein edit distance with optional early-exit.

    Returns ``max_distance + 1`` (or the true distance) when ``max_distance``
    is provided and the distance is known to exceed it. Pure stdlib.
    """

    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if max_distance is not None and abs(la - lb) > max_distance:
        return max_distance + 1

    # Make `a` the shorter string to keep the working row small.
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    previous = list(range(la + 1))
    for j in range(1, lb + 1):
        current = [j] + [0] * la
        bj = b[j - 1]
        row_min = current[0]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            current[i] = min(
                previous[i] + 1,        # deletion
                current[i - 1] + 1,     # insertion
                previous[i - 1] + cost,  # substitution
            )
            if current[i] < row_min:
                row_min = current[i]
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current

    return previous[la]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class QueryVocabulary:
    """Popularity-weighted vocabulary derived from the product catalog."""

    def __init__(self, catalog: pd.DataFrame):
        if "product_name" not in catalog.columns:
            raise ValueError("catalog must have a 'product_name' column")
        if "order_count" not in catalog.columns:
            raise ValueError("catalog must have an 'order_count' column")

        self.unigrams: dict[str, int] = defaultdict(int)
        self.bigrams: dict[str, int] = defaultdict(int)
        self.full_names: list[tuple[str, int]] = []
        # Internal lower-cased shadow used by the autocomplete suggester.
        self._full_names_lower: list[tuple[str, int, str]] = []

        names = catalog["product_name"].fillna("").astype(str).tolist()
        # Treat NaN / negative counts as zero contribution but still index the
        # word so it can be used for spelling.
        raw_counts = catalog["order_count"].fillna(0).astype(int).tolist()

        for name, count in zip(names, raw_counts):
            if not name:
                continue
            weight = max(int(count), 0) + 1  # +1 so vocab tokens always exist
            tokens = [tok for tok in _tokenize(name) if _is_kept_token(tok)]
            seen_unigrams: set[str] = set()
            for tok in tokens:
                if tok in seen_unigrams:
                    continue
                seen_unigrams.add(tok)
                self.unigrams[tok] += weight
            for first, second in zip(tokens, tokens[1:]):
                bigram = f"{first} {second}"
                self.bigrams[bigram] += weight
            self.full_names.append((name, max(int(count), 0)))
            self._full_names_lower.append(
                (name.lower(), max(int(count), 0), name)
            )

        # Materialise so callers see plain dicts (and not defaultdicts that
        # silently create keys on access).
        self.unigrams = dict(self.unigrams)
        self.bigrams = dict(self.bigrams)

        if "category" in catalog.columns:
            cats = catalog["category"].dropna().astype(str).unique().tolist()
        else:
            cats = []
        if "department" in catalog.columns:
            deps = (
                catalog["department"].dropna().astype(str).unique().tolist()
            )
        else:
            deps = []
        self.categories: list[str] = sorted({c for c in cats if c})
        self.departments: list[str] = sorted({d for d in deps if d})

        # Pre-compute popularity-weighted category and department lists for
        # the empty-prefix case in the autocomplete suggester.
        cat_scores: dict[str, int] = defaultdict(int)
        dep_scores: dict[str, int] = defaultdict(int)
        if "category" in catalog.columns:
            for cat, count in zip(
                catalog["category"].fillna("").astype(str),
                raw_counts,
            ):
                if cat:
                    cat_scores[cat] += max(int(count), 0)
        if "department" in catalog.columns:
            for dep, count in zip(
                catalog["department"].fillna("").astype(str),
                raw_counts,
            ):
                if dep:
                    dep_scores[dep] += max(int(count), 0)
        self._category_popularity: list[tuple[str, int]] = sorted(
            cat_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        self._department_popularity: list[tuple[str, int]] = sorted(
            dep_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        # Cached for SpellingCorrector — list is fine, we always scan it.
        self._vocab_words: list[str] = sorted(self.unigrams.keys())

    def vocab_words(self) -> list[str]:
        """Return the sorted list of unique unigrams."""

        return list(self._vocab_words)


# ---------------------------------------------------------------------------
# Spelling correction
# ---------------------------------------------------------------------------


def _allowed_distance(word: str) -> int:
    """Per-word edit-distance budget.

    Short words (<=4 chars) get a tight budget of 1 so we never coerce
    "cat" -> "can". Longer words may absorb up to 2 edits, which is what
    real-world typos like "almnd" -> "almond" need.
    """

    return 1 if len(word) <= 4 else 2


class SpellingCorrector:
    """Levenshtein-based correction over the QueryVocabulary."""

    def __init__(self, vocabulary: QueryVocabulary):
        self.vocab = vocabulary
        # Bucket vocabulary by length to prune the candidate set: only words
        # whose length is within +/- max_edit_distance can possibly match.
        self._words_by_length: dict[int, list[str]] = defaultdict(list)
        for word in vocabulary.vocab_words():
            self._words_by_length[len(word)].append(word)
        # Plain dict for predictable iteration cost.
        self._words_by_length = dict(self._words_by_length)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_words(self, word: str, max_distance: int) -> Iterable[str]:
        target_len = len(word)
        for delta in range(-max_distance, max_distance + 1):
            bucket = self._words_by_length.get(target_len + delta)
            if bucket:
                yield from bucket

    def _best_candidate(
        self, word: str, max_distance: int
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        """Return ``(best_word, scored_candidates)``.

        ``scored_candidates`` is a list of ``(candidate, distance, weight)``
        tuples (one per vocab word within budget) that callers can re-rank.
        """

        scored: list[tuple[str, int, int]] = []
        for cand in self._candidate_words(word, max_distance):
            dist = levenshtein(word, cand, max_distance)
            if dist <= max_distance:
                weight = self.vocab.unigrams.get(cand, 0)
                scored.append((cand, dist, weight))
        if not scored:
            return None, scored
        # Prefer smaller edit distance, then larger popularity, then the
        # candidate that shares the same first letter as the input (a common
        # human typing pattern).
        first_char = word[:1]
        scored.sort(
            key=lambda item: (
                item[1],
                -item[2],
                0 if item[0][:1] == first_char else 1,
                item[0],
            )
        )
        return scored[0][0], scored

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(self, query: str, max_edit_distance: int = 2) -> str:
        """Return a corrected version of ``query`` (may be identical)."""

        if not query:
            return query
        tokens = _tokenize(query)
        if not tokens:
            return query

        corrected: list[str] = []
        for token in tokens:
            # Stopwords and very short tokens are left untouched — we never
            # try to "correct" "the".
            if not _is_kept_token(token):
                corrected.append(token)
                continue
            if token in self.vocab.unigrams:
                # If the in-vocab token is very rare and there is a vastly
                # more popular near-neighbour, prefer the popular spelling.
                # This rescues real typos that happen to appear in obscure
                # product abbreviations (e.g. "Almnd Granola").
                corrected.append(self._maybe_replace_rare(token))
                continue

            budget = min(_allowed_distance(token), max_edit_distance)
            best, _ = self._best_candidate(token, budget)
            corrected.append(best if best is not None else token)

        return " ".join(corrected)

    # ------------------------------------------------------------------
    # Rare-word override
    # ------------------------------------------------------------------

    # A token whose popularity weight is below this threshold is considered
    # "rare" — likely a typo-ish abbreviation that slipped into the catalog.
    _RARE_WEIGHT_THRESHOLD = 1000
    # Replacement candidate must be this many times more popular than the
    # rare token to override it.
    _RARE_REPLACEMENT_RATIO = 10

    def _maybe_replace_rare(self, token: str) -> str:
        weight = self.vocab.unigrams.get(token, 0)
        if weight >= self._RARE_WEIGHT_THRESHOLD:
            return token
        budget = min(_allowed_distance(token), 2)
        _, scored = self._best_candidate(token, budget)
        # Skip the token itself; find the best alternative.
        for cand, _dist, cand_weight in scored:
            if cand == token:
                continue
            if cand_weight >= weight * self._RARE_REPLACEMENT_RATIO:
                return cand
            break  # candidates are sorted; first non-self is the best
        return token

    def suggest_corrections(self, query: str, top_k: int = 3) -> list[str]:
        """Return up to ``top_k`` alternative corrections (de-duplicated)."""

        if top_k <= 0 or not query:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        # Per-position candidate slates (each a list of token strings).
        slates: list[list[str]] = []
        any_alternative = False
        for token in tokens:
            if not _is_kept_token(token) or token in self.vocab.unigrams:
                slates.append([token])
                continue
            budget = min(_allowed_distance(token), 2)
            _, scored = self._best_candidate(token, budget)
            if not scored:
                slates.append([token])
                continue
            any_alternative = True
            slate = [cand for cand, _dist, _weight in scored[: max(top_k, 3)]]
            slates.append(slate)

        if not any_alternative:
            return []

        # Greedy combinatorial expansion: pair the n-th choice across slates.
        max_len = max(len(s) for s in slates)
        seen: set[str] = set()
        results: list[str] = []
        for i in range(max_len):
            phrase_tokens = [slate[min(i, len(slate) - 1)] for slate in slates]
            phrase = " ".join(phrase_tokens)
            if phrase not in seen:
                seen.add(phrase)
                results.append(phrase)
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


class AutocompleteSuggester:
    """Popularity-ranked prefix suggestions."""

    def __init__(self, vocabulary: QueryVocabulary):
        self.vocab = vocabulary
        # Stable, sorted views so the lru_cache results don't depend on
        # iteration order. Sorted by descending popularity, then alphabetically.
        self._unigrams_sorted: list[tuple[str, int]] = sorted(
            vocabulary.unigrams.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        # _full_names_lower is already (lower, count, original)
        self._full_names_sorted: list[tuple[str, int, str]] = sorted(
            vocabulary._full_names_lower,
            key=lambda t: (-t[1], t[0]),
        )
        self._categories_sorted: list[tuple[str, int]] = list(
            vocabulary._category_popularity
        )
        self._departments_sorted: list[tuple[str, int]] = list(
            vocabulary._department_popularity
        )

    # ------------------------------------------------------------------
    # Internal: pre-cached top categories used for empty-prefix output.
    # ------------------------------------------------------------------

    def _popular_seed(self, top_k: int) -> list[dict]:
        seeds: list[dict] = []
        for name, score in self._category_or_dep_seed():
            seeds.append({"text": name, "type": "category", "score": int(score)})
            if len(seeds) >= top_k:
                break
        return seeds

    def _category_or_dep_seed(self) -> list[tuple[str, int]]:
        # Categories first (more specific) then departments to fill out.
        seen: set[str] = set()
        combined: list[tuple[str, int]] = []
        for name, score in self._categories_sorted:
            if name not in seen:
                seen.add(name)
                combined.append((name, score))
        for name, score in self._departments_sorted:
            if name not in seen:
                seen.add(name)
                combined.append((name, score))
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, prefix: str, top_k: int = 8) -> list[dict]:
        """Return up to ``top_k`` autocomplete suggestions."""

        if top_k <= 0:
            return []
        norm = (prefix or "").strip().lower()
        return self._suggest_cached(norm, top_k)

    @lru_cache(maxsize=1024)
    def _suggest_cached(self, prefix: str, top_k: int) -> list[dict]:
        if not prefix:
            return self._popular_seed(top_k)

        # 1. Unigram keyword matches.
        keyword_hits: list[dict] = []
        for word, score in self._unigrams_sorted:
            if word.startswith(prefix):
                keyword_hits.append(
                    {"text": word, "type": "keyword", "score": int(score)}
                )
                if len(keyword_hits) >= top_k * 4:
                    break

        # 2. Full product name matches.
        product_hits: list[dict] = []
        for lower, count, original in self._full_names_sorted:
            if lower.startswith(prefix):
                product_hits.append(
                    {
                        "text": original,
                        "type": "product",
                        "score": int(count),
                    }
                )
                if len(product_hits) >= top_k * 4:
                    break

        # 3. Category matches.
        category_hits: list[dict] = []
        for name, score in self._categories_sorted:
            if name.lower().startswith(prefix):
                category_hits.append(
                    {
                        "text": name,
                        "type": "category",
                        "score": int(score),
                    }
                )

        combined: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()
        for item in keyword_hits + product_hits + category_hits:
            key = (item["text"].lower(), item["type"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            combined.append(item)

        # Rank: higher score first, then by length (shorter = more general).
        combined.sort(key=lambda item: (-item["score"], len(item["text"])))
        return combined[:top_k]
