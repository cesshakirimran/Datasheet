"""Confidence scoring.

Three signals, each in [0, 1]:
  R  retrieval relevance   mean reranker score of the top passages (does the knowledge base cover the question?)
  S  semantic similarity   how close each answer sentence is to the passages in embedding space
  F  faithfulness          share of answer claims verified against the passages

Combined score:
  weighted:  C = wR*R + wS*S + wF*F            (weights sum to 1)
  logistic:  C = sigmoid(b + aR*R + aS*S + aF*F) (fitted on the calibration split; C is then a probability
                                                   that the answer is NOT hallucinated)
Decision with threshold tau and band:
  C >= tau          -> answer
  tau-band <= C < tau -> answer with a low-confidence flag
  C < tau-band      -> abstain
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .textutils import split_sentences, strip_citations
from .vectorstore import Hit

SIGNALS = ("retrieval", "similarity", "faithfulness")


@dataclass
class Signals:
    retrieval: float
    similarity: float
    faithfulness: float
    confidence: float = 0.0

    def vector(self) -> list[float]:
        return [self.retrieval, self.similarity, self.faithfulness]

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


def retrieval_signal(hits: list[Hit], k: int = 3) -> float:
    if not hits:
        return 0.0
    top = sorted((h.rerank_score for h in hits), reverse=True)[:k]
    return float(np.mean(top))


def similarity_signal(answer: str, hits: list[Hit], embedder: Embedder) -> float:
    """Mean over answer sentences of the best cosine similarity to any retrieved passage."""
    sentences = split_sentences(strip_citations(answer))
    if not sentences or not hits:
        return 0.0
    ctx = [h.vector for h in hits if h.vector is not None]
    ctx_mat = np.vstack(ctx) if ctx else embedder.embed_documents([h.chunk.text for h in hits])
    sent_mat = embedder.embed_documents(sentences)
    sims = sent_mat @ ctx_mat.T
    return float(np.clip(sims.max(axis=1), 0.0, 1.0).mean())


class ConfidenceScorer:
    def __init__(self, cfg: dict, calibration_path: Path | None = None) -> None:
        self.method = cfg.get("method", "weighted")
        w = cfg.get("weights", {})
        self.weights = {k: float(w.get(k, 1 / 3)) for k in SIGNALS}
        self.threshold = float(cfg.get("threshold", 0.6))
        self.flag_band = float(cfg.get("flag_band", 0.0))
        self.logistic: dict | None = None
        self.source = "config"
        if calibration_path is not None and calibration_path.exists():
            self.load_calibration(calibration_path)

    def load_calibration(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.weights = data["weights"]
        self.logistic = data.get("logistic")
        key = "threshold_logistic" if self.method == "logistic" else "threshold_weighted"
        self.threshold = float(data.get(key, self.threshold))
        self.source = str(path)

    def combine(self, retrieval: float, similarity: float, faithfulness: float) -> float:
        x = {"retrieval": retrieval, "similarity": similarity, "faithfulness": faithfulness}
        if self.method == "logistic":
            if not self.logistic:
                raise RuntimeError("confidence.method=logistic requires a calibration file (run `calibrate`)")
            z = self.logistic["intercept"] + sum(self.logistic["coef"][k] * x[k] for k in SIGNALS)
            return 1.0 / (1.0 + math.exp(-z))
        total = sum(self.weights.values()) or 1.0
        return sum(self.weights[k] * x[k] for k in SIGNALS) / total

    def decide(self, confidence: float) -> str:
        if confidence >= self.threshold:
            return "answer"
        if confidence >= self.threshold - self.flag_band:
            return "flag"
        return "abstain"

    def describe(self) -> dict:
        return {"method": self.method, "weights": self.weights, "threshold": self.threshold,
                "flag_band": self.flag_band, "logistic": self.logistic, "source": self.source}
