"""Offline evaluation judges (LLM-as-a-judge).

The judge grades the answer each configuration produced *before* the confidence gate, so the
same judgement serves two purposes:
  1. system metrics: a shown answer is hallucinated if its pre-gate answer is;
  2. confidence analysis: how well the score C separates hallucinated from grounded answers.

Metric definitions (following RAGAS, Es et al. 2023):
  faithfulness      = supported claims / all claims, judged against the retrieved context
  hallucinated      = at least one claim not supported by the retrieved context
  answer relevancy  = mean cosine similarity between the question and N questions generated from the answer
  correctness       = agreement with the expert reference answer (correct / partially / incorrect)

Use a judge model that differs from (and is stronger than) the generator to limit self-preference
bias, and validate it against human labels (see `annotate-export` and `kappa`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .. import prompts
from ..embeddings import Embedder
from ..llm import BaseLLM
from ..textutils import content_tokens, is_abstention, numbers_in, split_sentences, strip_citations
from ..verifier import ClaimExtractor, LexicalVerifier, LLMVerifier

CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


@dataclass
class Judgement:
    abstained: bool
    hallucinated: bool | None = None
    faithfulness: float | None = None
    relevancy: float | None = None
    correctness: str | None = None
    correctness_score: float | None = None
    claims: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(a @ b, 0.0, 1.0))


class LLMJudge:
    name = "llm"

    def __init__(self, llm: BaseLLM, embedder: Embedder, n_questions: int = 3) -> None:
        self.llm = llm
        self.embedder = embedder
        self.n_questions = n_questions
        self.extractor = ClaimExtractor(llm, "llm")
        self.verifier = LLMVerifier(llm)

    def judge(self, question: str, answer: str, contexts: list[str], reference: str | None) -> Judgement:
        if is_abstention(answer):
            j = Judgement(abstained=True)
            if reference:
                j.correctness, j.correctness_score = "incorrect", 0.0
            return j
        claims = self.extractor.extract(question, answer)
        ver = self.verifier.verify(claims, contexts)
        j = Judgement(
            abstained=False,
            hallucinated=bool(ver.unsupported),
            faithfulness=round(ver.faithfulness, 4),
            claims=[c.to_dict() for c in ver.claims],
            relevancy=self.relevancy(question, answer),
        )
        if reference:
            j.correctness, j.correctness_score = self.correctness(question, answer, reference)
        return j

    def relevancy(self, question: str, answer: str) -> float:
        data = self.llm.chat_json(
            [{"role": "system", "content": prompts.RELEVANCY_SYSTEM.format(n=self.n_questions)},
             {"role": "user", "content": prompts.RELEVANCY_USER.format(answer=strip_citations(answer))}],
            task="relevancy_questions", meta={"answer": answer},
        )
        generated = [str(q) for q in data.get("questions", []) if str(q).strip()]
        if not generated:
            return 0.0
        vecs = self.embedder.embed_documents([question] + generated)
        return round(float(np.mean([_cosine(vecs[0], v) for v in vecs[1:]])), 4)

    def correctness(self, question: str, answer: str, reference: str) -> tuple[str, float]:
        data = self.llm.chat_json(
            [{"role": "system", "content": prompts.CORRECTNESS_SYSTEM},
             {"role": "user", "content": prompts.CORRECTNESS_USER.format(
                 question=question, reference=reference, answer=strip_citations(answer))}],
            task="correctness", meta={"question": question, "answer": answer, "reference": reference},
        )
        label = str(data.get("label", "incorrect")).strip().lower()
        if label not in CORRECTNESS_SCORE:
            label = "incorrect"
        return label, CORRECTNESS_SCORE[label]


class LexicalJudge:
    """Offline judge for smoke tests: word/number overlap instead of an LLM. Not valid for thesis results."""

    name = "lexical"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.verifier = LexicalVerifier(threshold=0.7)

    def judge(self, question: str, answer: str, contexts: list[str], reference: str | None) -> Judgement:
        if is_abstention(answer):
            j = Judgement(abstained=True)
            if reference:
                j.correctness, j.correctness_score = "incorrect", 0.0
            return j
        claims = split_sentences(strip_citations(answer))
        ver = self.verifier.verify(claims, contexts)
        vecs = self.embedder.embed_documents([question, strip_citations(answer)])
        j = Judgement(
            abstained=False, hallucinated=bool(ver.unsupported), faithfulness=round(ver.faithfulness, 4),
            claims=[c.to_dict() for c in ver.claims], relevancy=round(_cosine(vecs[0], vecs[1]), 4),
        )
        if reference:
            ref_nums, ans_nums = numbers_in(reference), numbers_in(strip_citations(answer))
            ref_toks, ans_toks = set(content_tokens(reference)), set(content_tokens(answer))
            recall = len(ref_toks & ans_toks) / max(len(ref_toks), 1)
            nums_ok = ref_nums <= ans_nums
            if recall >= 0.5 and nums_ok:
                label = "correct"
            elif recall >= 0.3 and (nums_ok or not ref_nums):
                label = "partially_correct"
            else:
                label = "incorrect"
            j.correctness, j.correctness_score = label, CORRECTNESS_SCORE[label]
        return j


def build_judge(settings, embedder: Embedder):
    from ..llm import build_llm

    cfg = settings["judge"]
    if cfg.get("provider") == "demo":
        return LexicalJudge(embedder)
    return LLMJudge(build_llm(cfg, settings.path("cache_path"), settings.get("seed", 42)), embedder)
