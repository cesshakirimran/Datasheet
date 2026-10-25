# Hallucination-Aware RAG Knowledge Base (prototype)

[![Tests](https://github.com/cesshakirimran/Datasheet/actions/workflows/tests.yml/badge.svg)](https://github.com/cesshakirimran/Datasheet/actions/workflows/tests.yml)

A retrieval-augmented question-answering system for enterprise knowledge management. It verifies its own
answers against the source documents, scores its confidence, and abstains instead of guessing. It also
includes a complete evaluation harness that compares a baseline RAG pipeline with the hallucination-aware
variants.

Everything runs locally with open models (Ollama + Hugging Face), or with any OpenAI-compatible API.

Documentation: [project documentation](docs/PROJECT_DOCUMENTATION.md) · [thesis overview and subtask mapping](docs/THESIS_OVERVIEW.md)

```
question ─► retrieve (dense vectors) ─► rerank (cross-encoder) ─► generate answer with citations      ← A: baseline
                                                                       │
                                     extract claims ─► verify each claim against the passages (NLI or LLM judge)
                                                                       │
                                     rewrite unsupported claims (correction)                           ← B
                                                                       │
                                     confidence C = f(R retrieval, S similarity, F faithfulness)
                                     C ≥ τ → answer   ·   C < τ → abstain                              ← C (gate only), D (all)
```

## What is in the project

| Component | Files | What it does |
| --- | --- | --- |
| Ingestion and chunking | `harag/documents.py` | Loads .md/.txt/.pdf/.docx and chunks them by section, keeping titles for citations |
| Semantic retrieval | `harag/embeddings.py`, `harag/vectorstore.py`, `harag/reranker.py` | bge embeddings, exact cosine search, bge cross-encoder reranker |
| Generation | `harag/pipeline.py`, `harag/prompts.py`, `harag/llm.py` | Grounded answers with [n] citations; any OpenAI-compatible LLM; SQLite response cache |
| Context-grounded verification | `harag/verifier.py` | Atomic claim extraction + verification by NLI (DeBERTa-v3), LLM-as-a-judge, or lexical overlap |
| Hallucination mitigation | `harag/pipeline.py` | Rewrites answers so unsupported claims are removed (config B/D) |
| Confidence scoring | `harag/confidence.py`, `harag/evaluation/calibration.py` | R + S + F combined (weighted or logistic), weights and threshold fitted on a calibration split |
| Evaluation | `harag/evaluation/*` | LLM judge (RAGAS-style faithfulness, answer relevancy, correctness), hallucination rate, coverage, AUROC, ECE, risk–coverage, McNemar/Wilcoxon/bootstrap tests, figures, Markdown report |
| Judge validation | `harag/evaluation/annotation.py` | Blind human-annotation sample + Cohen's kappa |
| RAGAS cross-check | `harag/evaluation/ragas_eval.py` | Optional scoring of saved runs with the official RAGAS library |
| Demo UI | `app.py` | Streamlit app: ask, compare baseline vs hallucination-aware side by side, view claims and sources, browse results, upload documents |
| Sample data | `data/corpus/`, `data/eval/` | 10 policy documents of a fictional company + 86 questions (66 answerable with reference answers, 20 unanswerable) |

## The four configurations

| | Retrieval + rerank + generation | Claim verification + correction | Confidence gate (abstain) |
| --- | --- | --- | --- |
| **A** Baseline RAG | ✓ | | |
| **B** + Verification & correction | ✓ | ✓ | |
| **C** + Confidence gate | ✓ | (verification used only for the score) | ✓ |
| **D** Hallucination-aware RAG | ✓ | ✓ | ✓ |

All four share the same retriever, prompt, models and questions, so differences come only from the
added components. B and C are ablations that show what each component contributes.

---

## 1. Quick check without any models (2 minutes)

The offline profile uses a deterministic fake LLM that deliberately injects errors, TF-IDF retrieval and a
lexical verifier. It only proves the code runs end to end. **Its numbers are not research results.**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: conda env create -f environment.yml && conda activate harag

bash scripts/run_offline_demo.sh   # ingest -> ask -> calibrate -> evaluate (offline profile)
python -m pytest -q
```

On Windows without bash, run the four `python -m harag --config config.offline.yaml ...` commands listed in
`scripts/run_offline_demo.sh` one by one.

## 2. Real setup

### 2.1 Models

Install [Ollama](https://ollama.com) and pull a generator and a judge model. The judge should differ from
the generator and ideally be stronger:

```bash
ollama pull llama3.1:8b     # generator (~5 GB, runs on 16 GB RAM laptops)
ollama pull qwen2.5:14b     # judge (~9 GB); on weaker machines use qwen2.5:7b
```

The embedding, reranker and NLI models (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`,
`cross-encoder/nli-deberta-v3-base`) download automatically from Hugging Face on first use (~1.5 GB).

**Using an API instead of Ollama.** Edit `llm` and/or `judge` in `config.yaml`:

```yaml
judge:
  provider: openai-compatible
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
```

Any OpenAI-compatible server works the same way (Azure OpenAI, Groq, Together, vLLM, LM Studio).

### 2.2 Run the full experiment

```bash
python -m harag check          # verifies the LLM endpoints and models load
python -m harag ingest         # builds data/index from data/corpus
python -m harag ask "How long are customer support tickets kept?" --compare

python -m harag calibrate      # runs A and B on the calibration split, fits weights + threshold
python -m harag evaluate       # runs A, B, C, D on the test split, judges, writes the report
streamlit run app.py           # demo UI
```

`bash scripts/run_experiment.sh` runs check → ingest → calibrate → evaluate in one go and also saves the
package versions and Ollama model list to `results/` for reproducibility. After `pip install -e .` the
command `harag ...` works as a shortcut for `python -m harag ...`.

Output in `results/test/`:

- `report.md`: all tables (summary, paired tests, signal AUROC, calibration)
- `summary.csv`, `summary.json`, `per_question.csv`
- `figures/`: `metrics_by_config.png`, `risk_coverage.png`, `signal_auroc.png`, `reliability.png`, `cost_latency.png`
- `runs/<config>.jsonl`: everything about every answer (retrieved passages, claims, verdicts, signals, timings)

Every LLM response is cached in `.cache/llm_cache.sqlite`, so rerunning is fast and gives identical
results. Use `--limit 10` for a quick trial and `--fresh` to discard previous runs (the cache is kept).
Runs restart automatically when the settings change.

### 2.3 Validate the judge with human labels

Results that rely on an LLM judge need evidence that the judge agrees with humans:

```bash
python -m harag annotate-export --n 100                       # blind sample + separate key file
# fill the human_hallucinated column with 1/0 in results/annotation/annotation_sample.csv
python -m harag kappa results/annotation/annotation_sample.csv results/annotation/annotation_key.csv
# with a second annotator: add --second other_annotator.csv
```

### 2.4 Optional: RAGAS cross-check

```bash
pip install -r requirements-ragas.txt      # or: pip install -e ".[ragas]"
python -m harag ragas -c A     # and -c D; writes results/test/ragas_<config>.csv
```

This is written for the ragas 0.2.x API. If an import fails with a newer version, adjust the imports in
`harag/evaluation/ragas_eval.py`. The built-in judge implements the same metric definitions without the
dependency.

---

## 3. Metrics

| Metric | Definition |
| --- | --- |
| Faithfulness | supported claims / all claims, judged against the retrieved passages (RAGAS definition) |
| Hallucination | an answer with at least one claim not supported by the retrieved passages |
| Hallucination rate | shown hallucinated answers / all questions (abstaining is never a hallucination) |
| Hallucination rate (answered) | shown hallucinated answers / shown answers |
| Hallucination reduction | (rate of A − rate of X) / rate of A |
| Answer relevancy | mean cosine similarity between the question and questions the judge generates from the answer (RAGAS definition) |
| Accuracy | judge agreement with the reference answer on answerable questions (abstention = not correct) |
| Coverage | answerable questions that receive an answer |
| Abstains on unanswerable | unanswerable questions where the system abstains |
| AUROC of C | how well the confidence score separates grounded from hallucinated answers (0.5 = chance) |
| ECE | expected calibration error of C (meaningful for the logistic method) |
| AURC | area under the risk–coverage curve (lower is better) |
| Latency, LLM calls, tokens | cost per question; cached calls are replayed with their originally measured latency |

Paired statistics against A: McNemar exact test on hallucinated answers, a bootstrap 95% CI for the
difference in hallucination rate, and a Wilcoxon signed-rank test on faithfulness.

**Runtime confidence vs. offline evaluation.** The confidence score C is computed live for every answer
from cheap signals (reranker score, embeddings, NLI verifier). The judge runs only offline, on saved
answers, with a different and stronger model. The calibration split and the test split do not overlap.

## 4. Using your own documents

1. Put documents (.md, .txt, .pdf, .docx) in `data/corpus/` (or point `corpus_dir` somewhere else) and run `ingest`.
2. For non-English documents (e.g. Hungarian), set `embeddings.model: BAAI/bge-m3`, `query_prefix: ""`,
   `reranker.model: BAAI/bge-reranker-v2-m3`, and use a multilingual NLI model or `verification.method: llm`.
3. Write questions in `data/eval/questions.jsonl`, one JSON object per line:

```json
{"id": "q001", "question": "...", "reference": "expert answer or null", "answerable": true, "type": "answerable", "source_docs": ["file.md"], "split": "test"}
```

Use `"answerable": false, "reference": null` for questions the documents do not answer, and put roughly a
third of the questions in `"split": "calib"`. `data/eval/make_questions.py` shows how the sample set was built.

## 5. Settings worth experimenting with

All in `config.yaml`:

- `retrieval.top_n`: passages given to the generator
- `chunking.chunk_size` / `overlap`
- `verification.method`: `nli` (cheap, local) vs `llm` (judge-style); compare them as an extra ablation
- `confidence.method`: `weighted` vs `logistic`; `calibration.strategy`: `coverage` vs `youden`; `min_coverage`
- `confidence.flag_band`: show borderline answers with a warning instead of abstaining

## 6. Repository structure

```
harag/                      Python package
  config.py                 YAML config with inheritance and ${ENV} expansion
  documents.py              loading + section-aware chunking
  embeddings.py             sentence-transformers / OpenAI-compatible / TF-IDF embedders
  vectorstore.py            persistent NumPy vector index
  reranker.py               cross-encoder reranker
  llm.py                    OpenAI-compatible client with cache and cost tracking; demo LLM for tests
  prompts.py                every prompt used (quote these in an appendix)
  verifier.py               claim extraction + NLI / LLM / lexical verifiers
  confidence.py             R, S, F signals and the combined score C
  pipeline.py               RAGSystem and configurations A-D
  cli.py                    command-line interface
  evaluation/
    judge.py                LLM-as-a-judge (faithfulness, relevancy, correctness)
    runner.py               runs configurations over questions (resumable)
    metrics.py              metrics + statistical tests
    calibration.py          fits weights and threshold
    report.py               tables, figures, Markdown report
    annotation.py           human validation of the judge (Cohen's kappa)
    ragas_eval.py           optional RAGAS cross-check
app.py                      Streamlit demo UI
config.yaml                 main configuration (Ollama + Hugging Face models)
config.offline.yaml         offline smoke-test profile
data/
  corpus/                   sample enterprise documents (fictional company)
  eval/                     evaluation questions + the script that builds them
docs/
  PROJECT_DOCUMENTATION.md  technical documentation
  THESIS_OVERVIEW.md        thesis topic and subtask-to-code mapping
  architecture.png          architecture diagram
scripts/
  run_offline_demo.sh       end-to-end smoke test without models
  run_experiment.sh         full experiment with real models
tests/                      unit tests (no models needed)
.github/workflows/tests.yml CI: lint, unit tests and offline smoke test on every push
requirements-core.txt       core dependencies (used by CI)
requirements.txt            full dependencies (core + models + UI)
requirements-ragas.txt      optional RAGAS cross-check
environment.yml             Conda environment
pyproject.toml              package metadata (pip install -e .)
```

Generated folders (`data/index*/`, `results*/`, `.cache/`) are ignored by git.

## 7. Reproducibility notes

- Temperature 0, fixed seed, and a cached LLM response for every prompt. Record the exact model tags
  (`ollama list`) and package versions (`pip freeze > results/environment.txt`) with your results.
- Ollama model tags can be updated in place. Pin them or note the digest.
- The sample company and its documents are fictional and were written for this prototype.

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot reach LLM endpoint` | Start Ollama (`ollama serve`) and check `base_url`; run `python -m harag check` |
| `HTTP 400 ... response_format` | The client retries without JSON mode automatically; or set `json_mode: false` |
| Newer OpenAI models reject `max_tokens`/`temperature` | Use models such as gpt-4o-mini, or edit the payload in `harag/llm.py` |
| `Index was built with ... Re-run ingest` | The embedding model changed; run `ingest` again |
| Slow NLI on CPU | Lower `retrieval.top_n`, or switch `verification.method` to `llm` |
| Calibration warns "only one class" | The calibration split has no hallucinated (or no grounded) answers; add harder or unanswerable questions |
