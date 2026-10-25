"""Runs configurations over a question set, judges the answers and stores one JSON line per
question x configuration. Runs are resumable: questions already in the output file are skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..pipeline import RAGSystem


def load_questions(path: Path, split: str | None = None, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if split and split != "all":
        rows = [r for r in rows if r.get("split", "test") == split]
    return rows[:limit] if limit else rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_config(rag: RAGSystem, judge, questions: list[dict], config: str, out_path: Path,
               resume: bool = True, progress: bool = True) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["question_id"]: r for r in _read_jsonl(out_path)} if resume else {}
    if not resume and out_path.exists():
        out_path.unlink()
    todo = [q for q in questions if q["id"] not in done]
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, start=1):
            res = rag.answer(q["question"], config=config, analysis=True, question_id=q["id"])
            contexts = [h["text"] for h in res.retrieved]
            judgement = judge.judge(q["question"], res.pre_gate_answer, contexts, q.get("reference"))
            row = {
                **res.to_dict(),
                "answered": res.answered,
                "answerable": bool(q.get("answerable", True)),
                "type": q.get("type", ""),
                "split": q.get("split", ""),
                "reference": q.get("reference"),
                "source_docs": q.get("source_docs", []),
                "retrieved_docs": sorted({h["doc_id"] for h in res.retrieved}),
                "judge": judgement.to_dict(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[q["id"]] = row
            if progress:
                eta = (time.time() - t0) / i * (len(todo) - i)
                sys.stderr.write(f"\r  config {config}: {i}/{len(todo)} questions (eta {eta:5.0f}s)   ")
                sys.stderr.flush()
    if progress and todo:
        sys.stderr.write("\n")
    wanted = {q["id"] for q in questions}
    return [done[i] for i in sorted(done) if i in wanted]


def retrieval_hit_rate(rows: list[dict]) -> float | None:
    """Share of answerable questions for which at least one source document was retrieved."""
    relevant = [r for r in rows if r["answerable"] and r.get("source_docs")]
    if not relevant:
        return None
    return sum(bool(set(r["source_docs"]) & set(r["retrieved_docs"])) for r in relevant) / len(relevant)
