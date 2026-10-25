"""Streamlit demo of the hallucination-aware RAG knowledge assistant.

    streamlit run app.py                       # uses config.yaml
    streamlit run app.py -- --config config.offline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from harag.config import load_settings
from harag.pipeline import CONFIGS, RAGSystem, ingest

ROOT = Path(__file__).parent


def _config_path() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.config


@st.cache_resource(show_spinner="Loading models and index…")
def get_system(config_path: str) -> RAGSystem:
    return RAGSystem(load_settings(ROOT / config_path))


def example_questions(settings) -> list[str]:
    path = settings.path("questions_file")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l)["question"] for l in f if l.strip()]


DECISION_STYLE = {
    "answer": ("✅ Answer", "#2a9d6f"),
    "flag": ("⚠️ Answer with low confidence: verify before relying on it", "#e8a33d"),
    "abstain": ("🛑 Abstained: no reliable answer in the knowledge base", "#c0392b"),
}
LABEL_ICON = {"supported": "✅ supported", "unsupported": "❔ unsupported", "contradicted": "❌ contradicted"}


def show_result(r, threshold: float) -> None:
    title, color = DECISION_STYLE[r.decision]
    st.markdown(f"<div style='padding:6px 10px;border-left:5px solid {color};font-weight:600'>{title}</div>",
                unsafe_allow_html=True)
    st.write("")
    st.markdown(r.final_answer)

    s = r.signals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence C", f"{s['confidence']:.2f}", help=f"Threshold τ = {threshold:.2f}")
    c2.metric("R · retrieval", f"{s['retrieval']:.2f}", help="Reranker relevance of the top passages")
    c3.metric("S · similarity", f"{s['similarity']:.2f}", help="Answer–context semantic similarity")
    c4.metric("F · faithfulness", f"{s['faithfulness']:.2f}", help="Share of claims verified against the context")
    st.progress(min(max(float(s["confidence"]), 0.0), 1.0))
    st.caption(f"{r.latency_s:.2f} s · {r.llm_calls} LLM calls · stages: "
               + ", ".join(f"{k} {v:.2f}s" for k, v in r.timings.items()))

    if r.corrected:
        with st.expander("Draft answer before correction"):
            st.markdown(r.draft_answer)
    claims = r.verification.get("claims", [])
    if claims:
        with st.expander(f"Claim verification ({sum(c['label'] == 'supported' for c in claims)}/{len(claims)} supported)",
                         expanded=r.decision != "answer"):
            st.dataframe(pd.DataFrame([{"claim": c["claim"], "verdict": LABEL_ICON.get(c["label"], c["label"]),
                                        "score": c["score"]} for c in claims]),
                         hide_index=True, use_container_width=True)
    with st.expander("Sources"):
        for i, h in enumerate(r.retrieved, start=1):
            st.markdown(f"**[{i}] {h['doc_id']} › {h['section']}** · relevance {h['rerank_score']:.2f}")
            st.text(h["text"])


def page_ask(rag: RAGSystem, settings) -> None:
    examples = example_questions(settings)
    choice = st.selectbox("Example questions", ["(type your own)"] + examples)
    question = st.text_input("Ask the knowledge base", value="" if choice == "(type your own)" else choice)
    compare = st.toggle("Compare with baseline RAG (A)", value=True)
    config = st.radio("Configuration", list(CONFIGS), index=3, horizontal=True,
                      format_func=lambda c: CONFIGS[c]["label"])
    if st.button("Ask", type="primary", disabled=not question.strip()):
        configs = ["A", config] if compare and config != "A" else [config]
        cols = st.columns(len(configs))
        for col, cfg in zip(cols, configs):
            with col:
                st.subheader(CONFIGS[cfg]["label"])
                with st.spinner("Thinking…"):
                    try:
                        res = rag.answer(question.strip(), cfg, analysis=True)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Error: {exc}")
                        continue
                show_result(res, rag.scorer.threshold)


def page_results(settings) -> None:
    out = settings.path("results_dir") / "test"
    summary = out / "summary.csv"
    if not summary.exists():
        st.info("No evaluation results yet. Run `python -m harag calibrate` and `python -m harag evaluate`.")
        return
    df = pd.read_csv(summary)
    cols = ["config", "hallucination_rate", "selective_hallucination_rate", "faithfulness", "answer_relevancy",
            "accuracy", "coverage", "unanswerable_abstention", "auroc_confidence", "mean_latency_s", "mean_llm_calls"]
    st.dataframe(df[[c for c in cols if c in df.columns]], hide_index=True, use_container_width=True)
    for fig in ("metrics_by_config", "risk_coverage", "signal_auroc", "reliability", "cost_latency"):
        p = out / "figures" / f"{fig}.png"
        if p.exists():
            st.image(str(p))
    pq = out / "per_question.csv"
    if pq.exists():
        with st.expander("Per-question results"):
            st.dataframe(pd.read_csv(pq), use_container_width=True)


def page_kb(settings, config_path: str) -> None:
    corpus = settings.path("corpus_dir")
    docs = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"})
    st.write(f"**{len(docs)} documents** in `{corpus.relative_to(settings.root)}`")
    st.dataframe(pd.DataFrame([{"document": p.name, "size_kb": round(p.stat().st_size / 1024, 1)} for p in docs]),
                 hide_index=True, use_container_width=True)
    uploads = st.file_uploader("Add documents (.md, .txt, .pdf, .docx)", accept_multiple_files=True,
                               type=["md", "txt", "pdf", "docx"])
    if uploads and st.button("Save and re-index"):
        for up in uploads:
            (corpus / up.name).write_bytes(up.getbuffer())
        with st.spinner("Indexing…"):
            info = ingest(settings)
        get_system.clear()
        st.success(f"Indexed {info['documents']} documents into {info['chunks']} chunks.")


def main() -> None:
    st.set_page_config(page_title="Hallucination-aware RAG", page_icon="🔎", layout="wide")
    config_path = _config_path()
    settings = load_settings(ROOT / config_path)
    st.title("Hallucination-aware enterprise knowledge assistant")
    st.caption(f"Generator `{settings['llm'].get('model')}` · verifier `{settings['verification'].get('method')}` · "
               f"config `{config_path}`")
    try:
        rag = get_system(config_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        if st.button("Build index now"):
            with st.spinner("Indexing…"):
                ingest(settings)
            st.rerun()
        return

    with st.sidebar:
        st.header("Confidence gate")
        rag.scorer.threshold = st.slider("Threshold τ", 0.0, 1.0, float(rag.scorer.threshold), 0.01)
        rag.scorer.flag_band = st.slider("Low-confidence band", 0.0, 0.5, float(rag.scorer.flag_band), 0.01,
                                         help="Answers with τ − band ≤ C < τ are shown with a warning")
        st.caption(f"Method: {rag.scorer.method} · weights: "
                   + ", ".join(f"{k[0].upper()}={v:.2f}" for k, v in rag.scorer.weights.items()))
        st.caption(f"Source: {Path(rag.scorer.source).name}")

    tab_ask, tab_results, tab_kb = st.tabs(["Ask", "Evaluation results", "Knowledge base"])
    with tab_ask:
        page_ask(rag, settings)
    with tab_results:
        page_results(settings)
    with tab_kb:
        page_kb(settings, config_path)


main()
