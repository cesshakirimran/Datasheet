# Thesis overview

**Topic:** Design and Evaluation of a Hallucination-Aware Retrieval-Augmented Knowledge Base System for Enterprise Knowledge Management

**Programme:** Business Informatics MSc, Széchenyi István University · Department of Informatics
**Supervisor:** Erdős Ferenc, associate professor
**Period:** 2025/2026 autumn – 2026/2027 spring · **Language:** English

## Subtasks from the data sheet and where this repository covers them

| # | Subtask | Covered by |
| --- | --- | --- |
| 1 | Literature review of RAG, enterprise knowledge management and hallucination detection methods | Thesis text (not in this repository) |
| 2 | Analysis of existing RAG architectures and confidence-scoring approaches | Thesis text (not in this repository) |
| 3 | Design of a hallucination-aware RAG-based knowledge management system | [Architecture](PROJECT_DOCUMENTATION.md#architecture) |
| 4 | Prototype with semantic retrieval and answer generation | `harag/documents.py`, `embeddings.py`, `vectorstore.py`, `reranker.py`, `pipeline.py` |
| 5 | Context-grounded verification and LLM-as-a-judge evaluation | `harag/verifier.py`, `harag/evaluation/judge.py` |
| 6 | Confidence scoring from retrieval relevance, semantic similarity and RAGAS metrics | `harag/confidence.py`, `harag/evaluation/calibration.py`, `harag/evaluation/ragas_eval.py` |
| 7 | Comparison of baseline and hallucination-aware RAG | Configurations A–D, `harag/evaluation/runner.py` |
| 8 | Evaluation with faithfulness, answer relevance and hallucination reduction | `harag/evaluation/metrics.py`, `report.py`, `annotation.py` |
| 9 | Analysis of results and recommendations for enterprise KM systems | Uses the report produced by `python -m harag evaluate` |

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the full technical documentation.
