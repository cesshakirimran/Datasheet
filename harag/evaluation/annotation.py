"""Human validation of the LLM judge.

1. `export_sample` writes a CSV for annotators WITHOUT the judge's labels (blind annotation) and a
   separate key file with the judge labels.
2. Annotators fill `human_hallucinated` with 1 (at least one claim not supported by the context) or 0.
3. `agreement` computes Cohen's kappa and accuracy of the judge against the human labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def export_sample(run_files: list[Path], out_dir: Path, n: int = 100, seed: int = 42) -> tuple[Path, Path]:
    rows = []
    for path in run_files:
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if not r["model_abstained"] and r["judge"].get("hallucinated") is not None]
    # de-duplicate identical answers (configs often share the same pre-gate answer)
    seen, unique = set(), []
    for r in rows:
        key = (r["question_id"], r["pre_gate_answer"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rng = random.Random(seed)
    pos = [r for r in unique if r["judge"]["hallucinated"]]
    neg = [r for r in unique if not r["judge"]["hallucinated"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    sample = pos[:half] + neg[: n - min(half, len(pos))]
    sample = sample[:n]
    rng.shuffle(sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    blind, key = [], []
    for i, r in enumerate(sample, start=1):
        item = f"item{i:03d}"
        blind.append({
            "item": item, "question": r["question"],
            "context": "\n\n".join(f"[{k}] {h['text']}" for k, h in enumerate(r["retrieved"], start=1)),
            "answer": r["pre_gate_answer"], "human_hallucinated": "", "human_notes": "",
        })
        key.append({"item": item, "question_id": r["question_id"], "config": r["config"],
                    "judge_hallucinated": int(bool(r["judge"]["hallucinated"])),
                    "judge_faithfulness": r["judge"].get("faithfulness")})
    blind_path, key_path = out_dir / "annotation_sample.csv", out_dir / "annotation_key.csv"
    pd.DataFrame(blind).to_csv(blind_path, index=False)
    pd.DataFrame(key).to_csv(key_path, index=False)
    return blind_path, key_path


def agreement(annotated: Path, key: Path, second_annotator: Path | None = None) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    a = pd.read_csv(annotated)
    k = pd.read_csv(key)
    df = a.merge(k, on="item")
    df = df[df["human_hallucinated"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        raise ValueError("No annotated rows: fill the human_hallucinated column with 0 or 1.")
    human = df["human_hallucinated"].astype(float).astype(int)
    judge = df["judge_hallucinated"].astype(int)
    out = {
        "n": int(len(df)),
        "cohen_kappa_judge_vs_human": float(cohen_kappa_score(human, judge)),
        "accuracy_judge_vs_human": float((human == judge).mean()),
        "confusion_matrix [human rows 0/1 x judge cols 0/1]": confusion_matrix(human, judge, labels=[0, 1]).tolist(),
        "human_hallucination_share": float(human.mean()),
    }
    if second_annotator is not None:
        b = pd.read_csv(second_annotator)[["item", "human_hallucinated"]].rename(
            columns={"human_hallucinated": "h2"})
        both = df.merge(b, on="item")
        both = both[both["h2"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        if not both.empty:
            out["cohen_kappa_human_vs_human"] = float(cohen_kappa_score(
                both["human_hallucinated"].astype(float).astype(int), both["h2"].astype(float).astype(int)))
    return out
