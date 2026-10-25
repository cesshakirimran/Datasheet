"""Metrics and statistical tests for comparing configurations.

Per-question rows (one per question x configuration) are produced by runner.py and contain:
  answerable, answered, model_abstained, decision, judge (Judgement dict), signals, latency_s, llm_calls, ...

Definitions
  coverage                   answered / answerable questions
  accuracy                   correct answers / answerable questions (abstentions count as not correct)
  selective accuracy         correct / answered answerable questions
  hallucination rate         answered AND hallucinated / all questions
  selective halluc. rate     answered AND hallucinated / answered questions
  unanswerable abstention    abstained / unanswerable questions (higher is better)
  hallucination reduction    (rate_baseline - rate_config) / rate_baseline
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else None


def hallucinated_final(row: dict) -> bool:
    return bool(row["answered"] and row["judge"].get("hallucinated"))


def correct_final(row: dict) -> float:
    if not row["answerable"]:
        return 1.0 if not row["answered"] else 0.0
    if not row["answered"]:
        return 0.0
    return float(row["judge"].get("correctness_score") or 0.0)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    answered_ans = [r for r in ans if r["answered"]]
    halluc = [r for r in rows if hallucinated_final(r)]
    out = {
        "n_questions": n,
        "coverage": len(answered_ans) / len(ans) if ans else None,
        "accuracy": _mean(correct_final(r) for r in ans),
        "selective_accuracy": _mean(correct_final(r) for r in answered_ans),
        "faithfulness": _mean(r["judge"].get("faithfulness") for r in answered),
        "answer_relevancy": _mean(r["judge"].get("relevancy") for r in answered),
        "hallucination_rate": len(halluc) / n if n else None,
        "selective_hallucination_rate": (len(halluc) / len(answered)) if answered else None,
        "unanswerable_abstention": (sum(not r["answered"] for r in unans) / len(unans)) if unans else None,
        "flagged_share": (sum(r["decision"] == "flag" for r in rows) / n) if n else None,
        "corrected_share": (sum(bool(r.get("corrected")) for r in rows) / n) if n else None,
        "mean_latency_s": _mean(r["latency_s"] for r in rows),
        "p95_latency_s": float(np.percentile([r["latency_s"] for r in rows], 95)) if rows else None,
        "mean_llm_calls": _mean(r["llm_calls"] for r in rows),
        "mean_tokens": _mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
    }
    out.update(confidence_quality(rows))
    return out


# ------------------------------------------------------------------ confidence quality

def _pregate_scored(rows: list[dict]) -> list[dict]:
    """Rows where the pre-gate answer was a real answer that the judge labelled."""
    return [r for r in rows if not r["model_abstained"] and r["judge"].get("hallucinated") is not None]


def auroc(labels: list[int], scores: list[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def expected_calibration_error(probs: list[float], labels: list[int], bins: int = 10) -> float | None:
    if not probs:
        return None
    probs_a, labels_a = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs_a >= lo) & ((probs_a < hi) if hi < 1 else (probs_a <= hi))
        if mask.any():
            ece += mask.mean() * abs(probs_a[mask].mean() - labels_a[mask].mean())
    return float(ece)


def confidence_quality(rows: list[dict]) -> dict:
    scored = _pregate_scored(rows)
    y = [0 if r["judge"]["hallucinated"] else 1 for r in scored]  # 1 = grounded
    out = {"n_scored": len(scored)}
    for key in ("confidence", "retrieval", "similarity", "faithfulness"):
        out[f"auroc_{key}"] = auroc(y, [r["signals"][key] for r in scored])
    out["ece_confidence"] = expected_calibration_error([r["signals"]["confidence"] for r in scored], y)
    rc = risk_coverage(rows)
    out["aurc"] = rc["aurc"]
    return out


def risk_coverage(rows: list[dict]) -> dict:
    """Selective-prediction curve: answer only the top-x% most confident pre-gate answers."""
    scored = sorted(_pregate_scored(rows), key=lambda r: -r["signals"]["confidence"])
    if not scored:
        return {"coverage": [], "risk": [], "aurc": None}
    errs = np.array([1.0 if r["judge"]["hallucinated"] else 0.0 for r in scored])
    k = np.arange(1, len(scored) + 1)
    risk = np.cumsum(errs) / k
    coverage = k / len(scored)
    return {"coverage": coverage.tolist(), "risk": risk.tolist(), "aurc": float(np.mean(risk))}


def reliability_bins(rows: list[dict], bins: int = 10) -> list[dict]:
    scored = _pregate_scored(rows)
    probs = np.array([r["signals"]["confidence"] for r in scored])
    y = np.array([0 if r["judge"]["hallucinated"] else 1 for r in scored])
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if mask.any():
            out.append({"bin_mid": (lo + hi) / 2, "mean_conf": float(probs[mask].mean()),
                        "grounded_rate": float(y[mask].mean()), "n": int(mask.sum())})
    return out


# ------------------------------------------------------------------ paired comparisons

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(base: list[dict], other: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired comparison of two configurations on the same questions."""
    b = {r["question_id"]: r for r in base}
    o = {r["question_id"]: r for r in other}
    ids = sorted(set(b) & set(o))
    hb = np.array([hallucinated_final(b[i]) for i in ids], float)
    ho = np.array([hallucinated_final(o[i]) for i in ids], float)
    diff = ho - hb  # negative = fewer hallucinations
    # McNemar exact test on discordant pairs
    n01 = int(((hb == 1) & (ho == 0)).sum())  # fixed by `other`
    n10 = int(((hb == 0) & (ho == 1)).sum())  # introduced by `other`
    mcnemar_p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0
    # faithfulness on questions both answered
    both = [i for i in ids if b[i]["answered"] and o[i]["answered"]
            and b[i]["judge"].get("faithfulness") is not None and o[i]["judge"].get("faithfulness") is not None]
    fb = np.array([b[i]["judge"]["faithfulness"] for i in both])
    fo = np.array([o[i]["judge"]["faithfulness"] for i in both])
    wilcoxon_p = None
    if len(both) > 0 and np.any(fo - fb != 0):
        wilcoxon_p = float(stats.wilcoxon(fo, fb, zero_method="wilcox").pvalue)
    rate_b = float(hb.mean()) if len(ids) else float("nan")
    rate_o = float(ho.mean()) if len(ids) else float("nan")
    ci = bootstrap_ci(diff, n_boot, seed)
    return {
        "n_paired": len(ids),
        "halluc_rate_base": rate_b,
        "halluc_rate_other": rate_o,
        "halluc_rate_diff": rate_o - rate_b,
        "halluc_rate_diff_ci95": list(ci),
        "hallucination_reduction": ((rate_b - rate_o) / rate_b) if rate_b > 0 else None,
        "fixed": n01,
        "introduced": n10,
        "mcnemar_p": mcnemar_p,
        "n_both_answered": len(both),
        "faithfulness_diff": float((fo - fb).mean()) if len(both) else None,
        "wilcoxon_p_faithfulness": wilcoxon_p,
    }
