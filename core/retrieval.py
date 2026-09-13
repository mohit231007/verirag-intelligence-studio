"""Deterministic lexical retrieval, fusion and reranking helpers.

The module deliberately avoids another hosted service or heavyweight cross-encoder so the
public demo can compare dense-only and hybrid retrieval at zero incremental inference cost.
The reranker is transparent: lexical BM25, reciprocal-rank fusion, query-term coverage and
exact-phrase bonuses are combined with dense similarity. It is an interpretable reranker,
not a learned cross-encoder.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return BM25 scores normalized to 0..1 for a small in-session corpus."""

    if not documents:
        return []
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)

    tokenized = [tokenize(document) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = max(1.0, sum(lengths) / len(lengths))
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    scores: list[float] = []
    corpus_size = len(documents)
    for tokens, length in zip(tokenized, lengths, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += idf * frequency * (k1 + 1.0) / max(denominator, 1e-12)
        scores.append(score)

    maximum = max(scores, default=0.0)
    if maximum <= 0.0:
        return [0.0] * len(scores)
    return [round(score / maximum, 6) for score in scores]


def query_term_coverage(query: str, document: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    document_terms = set(tokenize(document))
    return len(query_terms & document_terms) / len(query_terms)


def phrase_bonus(query: str, document: str) -> float:
    cleaned_query = " ".join(tokenize(query))
    cleaned_document = " ".join(tokenize(document))
    if len(cleaned_query) < 5:
        return 0.0
    return 1.0 if cleaned_query in cleaned_document else 0.0


def rrf_score(dense_rank: int | None, lexical_rank: int | None, *, k: int = 60) -> float:
    """Reciprocal-rank fusion score for up to two ranked lists."""

    score = 0.0
    if dense_rank is not None:
        score += 1.0 / (k + dense_rank)
    if lexical_rank is not None:
        score += 1.0 / (k + lexical_rank)
    return score


def transparent_rerank_score(
    query: str,
    document: str,
    *,
    dense_score: float,
    lexical_score: float,
    rrf: float,
    dense_weight: float = 0.60,
    rerank_weight: float = 0.20,
) -> float:
    """Blend dense, lexical and transparent reranking signals into a 0..1 score."""

    dense_weight = max(0.0, min(1.0, dense_weight))
    rerank_weight = max(0.0, min(1.0, rerank_weight))
    lexical_weight = 1.0 - dense_weight
    base = dense_weight * max(0.0, dense_score) + lexical_weight * max(0.0, lexical_score)

    coverage = query_term_coverage(query, document)
    exact = phrase_bonus(query, document)
    # RRF values are small by design. Scale a two-list top-rank score close to one.
    normalized_rrf = min(1.0, rrf * 30.5)
    transparent = 0.55 * coverage + 0.25 * exact + 0.20 * normalized_rrf
    return round(max(0.0, min(1.0, (1.0 - rerank_weight) * base + rerank_weight * transparent)), 6)
