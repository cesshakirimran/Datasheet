"""Rerankers turn (query, chunk) pairs into a 0..1 relevance score.

The reranker score is also the retrieval-relevance signal R used by the confidence score.
"""
from __future__ import annotations

import numpy as np

from .vectorstore import Hit


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NoReranker:
    """Keeps the vector order; relevance = cosine similarity clipped to [0, 1]."""

    name = "none"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        for h in hits:
            h.rerank_score = float(min(max(h.vector_score, 0.0), 1.0))
        return sorted(hits, key=lambda h: -h.rerank_score)


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = np.asarray(self.model.predict([(query, h.chunk.text) for h in hits]), dtype=float).reshape(-1)
        if scores.min() < 0.0 or scores.max() > 1.0:  # raw logits -> probabilities
            scores = _sigmoid(scores)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: -h.rerank_score)


def build_reranker(cfg: dict):
    provider = cfg.get("provider", "none")
    if provider == "cross-encoder":
        return CrossEncoderReranker(cfg["model"])
    if provider == "none":
        return NoReranker()
    raise ValueError(f"Unknown reranker provider: {provider}")
