"""The RAG system and its four experimental configurations.

  A  Baseline RAG                       retrieve -> rerank -> generate
  B  + Verification & correction        ... -> extract claims -> verify -> rewrite unsupported claims
  C  + Confidence gate                  ... -> extract claims -> verify -> score R,S,F -> answer / flag / abstain
  D  Hallucination-aware RAG (full)     B and C combined

All configurations share the same retriever, generator prompt and models, so differences in the
results come only from the hallucination-aware components.

With analysis=True the confidence signals are also computed for configurations that do not use
them (A, B). This "shadow" work is excluded from latency and cost, and lets the evaluation
measure how well the confidence score predicts hallucinations for every configuration.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import prompts
from .confidence import ConfidenceScorer, Signals, retrieval_signal, similarity_signal
from .config import Settings
from .embeddings import build_embedder
from .llm import BaseLLM, LLMStats, build_llm
from .reranker import build_reranker
from .textutils import ABSTAIN_MESSAGE, ABSTAIN_TOKEN, is_abstention
from .vectorstore import Hit, VectorStore
from .verifier import ClaimExtractor, Verification, build_verifier

CONFIGS = {
    "A": {"label": "A: Baseline RAG", "correct": False, "gate": False},
    "B": {"label": "B: + Verification & correction", "correct": True, "gate": False},
    "C": {"label": "C: + Confidence gate", "correct": False, "gate": True},
    "D": {"label": "D: Hallucination-aware RAG", "correct": True, "gate": True},
}


@dataclass
class RAGResult:
    question: str
    config: str
    final_answer: str
    decision: str                 # answer | flag | abstain
    model_abstained: bool         # generator (or correction) itself said it cannot answer
    draft_answer: str             # first generated answer
    pre_gate_answer: str          # answer after optional correction, before the confidence gate
    corrected: bool
    retrieved: list[dict]
    verification: dict            # claims + faithfulness of the pre-gate answer
    signals: dict                 # retrieval, similarity, faithfulness, confidence
    timings: dict                 # seconds per stage (counted stages only)
    latency_s: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    question_id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.decision in ("answer", "flag")

    def to_dict(self) -> dict:
        return asdict(self)


class _Meter:
    """Measures a stage: wall time with cached LLM calls replaced by their originally measured latency."""

    def __init__(self, llms: list[BaseLLM]) -> None:
        self.llms = llms
        self.seconds = 0.0
        self.stats = LLMStats()

    def __enter__(self) -> "_Meter":
        self._t0 = time.perf_counter()
        self._before = [l.stats.copy() for l in self.llms]
        return self

    def __exit__(self, *exc) -> None:
        elapsed = time.perf_counter() - self._t0
        for llm, before in zip(self.llms, self._before):
            d = llm.stats.minus(before)
            elapsed += d.seconds - d.wall
            for k in ("calls", "cached_calls", "seconds", "wall", "prompt_tokens", "completion_tokens"):
                setattr(self.stats, k, getattr(self.stats, k) + getattr(d, k))
        self.seconds = max(elapsed, 0.0)


class RAGSystem:
    def __init__(self, settings: Settings, load_index: bool = True) -> None:
        self.settings = settings
        cache = settings.path("cache_path")
        self.embedder = build_embedder(settings["embeddings"])
        self.reranker = build_reranker(settings["reranker"])
        self.llm = build_llm(settings["llm"], cache, settings.get("seed", 42))
        vcfg = settings["verification"]
        self.extractor = ClaimExtractor(self.llm, vcfg.get("claim_extraction", "llm"))
        self.verifier = build_verifier(vcfg, self.llm)
        self.scorer = ConfidenceScorer(settings["confidence"], self._calibration_path())
        self.store: VectorStore | None = None
        if load_index:
            self.store = VectorStore.load(settings.path("index_dir"), self.embedder)

    def _calibration_path(self) -> Path | None:
        p = self.settings["confidence"].get("calibration_file")
        if not p:
            return None
        p = Path(p)
        return p if p.is_absolute() else self.settings.root / p

    # ------------------------------------------------------------------ stages

    def retrieve(self, question: str) -> list[Hit]:
        r = self.settings["retrieval"]
        qvec = self.embedder.embed_query(question)
        hits = self.store.search(qvec, int(r.get("top_k", 8)))
        hits = self.reranker.rerank(question, hits)
        return hits[: int(r.get("top_n", 4))]

    def generate(self, question: str, contexts: list[str]) -> str:
        return self.llm.chat(
            [{"role": "system", "content": prompts.GENERATE_SYSTEM},
             {"role": "user", "content": prompts.GENERATE_USER.format(
                 context=prompts.format_context(contexts), question=question)}],
            task="generate", meta={"question": question, "contexts": contexts},
        ).strip()

    def verify(self, question: str, answer: str, contexts: list[str]) -> Verification:
        if is_abstention(answer):
            return Verification([])
        claims = self.extractor.extract(question, answer)
        return self.verifier.verify(claims, contexts)

    def correct(self, question: str, answer: str, contexts: list[str], unsupported: list[str]) -> str:
        return self.llm.chat(
            [{"role": "system", "content": prompts.CORRECT_SYSTEM},
             {"role": "user", "content": prompts.CORRECT_USER.format(
                 context=prompts.format_context(contexts), question=question, answer=answer,
                 unsupported="\n".join(f"- {c}" for c in unsupported))}],
            task="correct", meta={"question": question, "answer": answer, "contexts": contexts,
                                  "unsupported": unsupported},
        ).strip()

    # ------------------------------------------------------------------ full run

    def answer(self, question: str, config: str = "D", analysis: bool = True, question_id: str = "") -> RAGResult:
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {list(CONFIGS)}")
        if self.store is None:
            raise RuntimeError("Index not loaded")
        spec = CONFIGS[config]
        llms = [self.llm]
        timings: dict[str, float] = {}
        counted = LLMStats()

        def run(stage: str, fn, count: bool = True):
            with _Meter(llms) as m:
                out = fn()
            if count:
                timings[stage] = timings.get(stage, 0.0) + m.seconds
                for k in ("calls", "cached_calls", "prompt_tokens", "completion_tokens"):
                    setattr(counted, k, getattr(counted, k) + getattr(m.stats, k))
            return out

        hits = run("retrieval", lambda: self.retrieve(question))
        contexts = [h.chunk.text for h in hits]
        draft = run("generation", lambda: self.generate(question, contexts))
        answer = draft
        corrected = False
        needs_verification = spec["correct"] or spec["gate"]
        verification = Verification([])
        if needs_verification or analysis:
            verification = run("verification", lambda: self.verify(question, answer, contexts), count=needs_verification)

        if spec["correct"]:
            rounds = int(self.settings["verification"].get("max_correction_rounds", 1))
            for _ in range(rounds):
                if is_abstention(answer) or not verification.unsupported:
                    break
                answer = run("correction", lambda: self.correct(question, answer, contexts, verification.unsupported))
                corrected = True
                verification = run("verification", lambda: self.verify(question, answer, contexts))

        model_abstained = is_abstention(answer)

        def score() -> Signals:
            r = retrieval_signal(hits)
            if model_abstained:
                return Signals(r, 0.0, 0.0, 0.0)
            s = similarity_signal(answer, hits, self.embedder)
            f = verification.faithfulness
            return Signals(r, s, f, self.scorer.combine(r, s, f))

        signals = run("scoring", score, count=spec["gate"]) if (spec["gate"] or analysis) else Signals(0, 0, 0, 0)

        if model_abstained:
            decision = "abstain"
        elif spec["gate"]:
            decision = self.scorer.decide(signals.confidence)
        else:
            decision = "answer"
        final = ABSTAIN_MESSAGE if decision == "abstain" else answer

        return RAGResult(
            question=question, config=config, final_answer=final, decision=decision,
            model_abstained=model_abstained, draft_answer=draft, pre_gate_answer=answer, corrected=corrected,
            retrieved=[h.to_dict() for h in hits], verification=verification.to_dict(), signals=signals.to_dict(),
            timings={k: round(v, 4) for k, v in timings.items()}, latency_s=round(sum(timings.values()), 4),
            llm_calls=counted.calls + counted.cached_calls, prompt_tokens=counted.prompt_tokens,
            completion_tokens=counted.completion_tokens, question_id=question_id,
        )


def ingest(settings: Settings) -> dict:
    """Load documents, chunk, embed and save the index."""
    from .documents import build_chunks

    ch = settings["chunking"]
    chunks = build_chunks(settings.path("corpus_dir"), int(ch.get("chunk_size", 180)), int(ch.get("overlap", 30)))
    embedder = build_embedder(settings["embeddings"])
    store = VectorStore.build(chunks, embedder)
    store.save(settings.path("index_dir"), embedder, {"chunking": dict(ch)})
    docs = sorted({c.doc_id for c in chunks})
    return {"documents": len(docs), "chunks": len(chunks), "index_dir": str(settings.path("index_dir"))}
