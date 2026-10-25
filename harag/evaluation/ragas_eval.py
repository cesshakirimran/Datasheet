"""Optional: score saved runs with the official RAGAS library (cross-check of the built-in judge).

Install the extra dependencies first:
    pip install -r requirements-ragas.txt

Written for the ragas 0.2.x API (EvaluationDataset + metric classes). RAGAS changes its API between
minor versions; if an import fails, check the installed version's documentation and adjust the imports
below. The built-in judge (harag/evaluation/judge.py) implements the same metric definitions and does
not depend on RAGAS.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def run_ragas(run_file: Path, settings, out_csv: Path) -> pd.DataFrame:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"RAGAS is not installed or its API changed ({exc}). See requirements-ragas.txt.") from exc

    with run_file.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r["answered"]]
    samples = []
    for r in rows:
        s = {"user_input": r["question"], "retrieved_contexts": [h["text"] for h in r["retrieved"]],
             "response": r["final_answer"]}
        if r.get("reference"):
            s["reference"] = r["reference"]
        samples.append(s)
    dataset = EvaluationDataset.from_list(samples)

    judge = settings["judge"]
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge["model"], base_url=judge["base_url"],
                                         api_key=judge.get("api_key") or "none", temperature=0))
    emb_cfg = settings["embeddings"]
    emb_model = emb_cfg["model"] if emb_cfg.get("provider") == "openai" else "nomic-embed-text"
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=emb_model, base_url=emb_cfg.get("base_url", judge["base_url"]),
                                                      api_key=emb_cfg.get("api_key") or "none",
                                                      check_embedding_ctx_length=False))
    metrics = [Faithfulness(), ResponseRelevancy()]
    if all("reference" in s for s in samples):
        metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)
    df = result.to_pandas()
    df.insert(0, "question_id", [r["question_id"] for r in rows])
    df.insert(1, "config", [r["config"] for r in rows])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
