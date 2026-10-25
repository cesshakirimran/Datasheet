"""Fits the confidence score on the calibration split.

Input: judged rows of configurations A and B on the calibration split (their pre-gate answers are
exactly what the gate sees in C and D). Each row has the signals R, S, F and the judge label.

Output (calibration.json):
  - logistic regression  P(grounded) = sigmoid(b + aR*R + aS*S + aF*F)
  - weights for the weighted method = positive logistic coefficients normalised to sum to 1
    (signals are on comparable 0..1 scales, so coefficients are comparable)
  - a threshold tau for each method, chosen with the configured strategy:
      coverage: lowest hallucination rate among kept answers while the gate keeps >= min_coverage of the
                answers given to answerable questions
      youden:   maximise TPR - FPR for detecting grounded answers
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..confidence import SIGNALS
from .metrics import auroc


def _rows_for_fit(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r["model_abstained"] and r["judge"].get("hallucinated") is not None]


def choose_threshold(scores: np.ndarray, grounded: np.ndarray, answerable: np.ndarray,
                     n_answerable_total: int, strategy: str, min_coverage: float) -> float:
    candidates = np.unique(np.concatenate([[0.0], scores]))
    best_tau, best_key = 0.0, None
    for tau in candidates:
        keep = scores >= tau
        if strategy == "youden":
            tpr = (keep & (grounded == 1)).sum() / max((grounded == 1).sum(), 1)
            fpr = (keep & (grounded == 0)).sum() / max((grounded == 0).sum(), 1)
            key = (tpr - fpr, -tau)
        else:
            coverage = (keep & answerable).sum() / max(n_answerable_total, 1)
            if coverage + 1e-9 < min_coverage:
                continue
            risk = ((keep & (grounded == 0)).sum() / keep.sum()) if keep.any() else 0.0
            key = (-risk, coverage)
        if best_key is None or key > best_key:
            best_tau, best_key = float(tau), key
    return best_tau


def fit_calibration(rows: list[dict], strategy: str = "coverage", min_coverage: float = 0.8,
                    seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression

    fit_rows = _rows_for_fit(rows)
    if len(fit_rows) < 10:
        raise ValueError(f"Only {len(fit_rows)} judged answers in the calibration split; need at least 10.")
    X = np.array([[r["signals"][k] for k in SIGNALS] for r in fit_rows], float)
    y = np.array([0 if r["judge"]["hallucinated"] else 1 for r in fit_rows])
    answerable = np.array([bool(r["answerable"]) for r in fit_rows])
    # coverage = share of answers to answerable questions that the gate lets through
    n_answerable_total = int(answerable.sum())

    if len(set(y)) < 2:
        coef = np.ones(len(SIGNALS))
        intercept = 0.0
        note = "only one class in calibration data; equal weights used"
    else:
        model = LogisticRegression(C=1.0, class_weight="balanced", random_state=seed, max_iter=1000)
        model.fit(X, y)
        coef, intercept = model.coef_[0], float(model.intercept_[0])
        note = ""

    pos = np.clip(coef, 0, None)
    weights = (pos / pos.sum()) if pos.sum() > 0 else np.full(len(SIGNALS), 1 / len(SIGNALS))
    weighted_scores = X @ weights
    logistic_scores = 1 / (1 + np.exp(-(intercept + X @ coef)))

    tau_w = choose_threshold(weighted_scores, y, answerable, n_answerable_total, strategy, min_coverage)
    tau_l = choose_threshold(logistic_scores, y, answerable, n_answerable_total, strategy, min_coverage)

    return {
        "n_fit": int(len(fit_rows)),
        "grounded_share": float(y.mean()),
        "strategy": strategy,
        "min_coverage": min_coverage,
        "weights": {k: round(float(w), 4) for k, w in zip(SIGNALS, weights)},
        "logistic": {"intercept": round(intercept, 6), "coef": {k: round(float(c), 6) for k, c in zip(SIGNALS, coef)}},
        "threshold_weighted": round(tau_w, 4),
        "threshold_logistic": round(tau_l, 4),
        "auroc_fit": {
            "weighted": auroc(list(y), list(weighted_scores)),
            "logistic": auroc(list(y), list(logistic_scores)),
            **{k: auroc(list(y), list(X[:, i])) for i, k in enumerate(SIGNALS)},
        },
        "note": note,
    }


def save_calibration(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v),
                    encoding="utf-8")
