"""Context-grounded verification: split an answer into claims and check each claim against the
retrieved context.

Three verifiers share one interface:
  - NLIVerifier      : a natural-language-inference cross-encoder (DeBERTa-v3 MNLI). Cheap, local, no LLM call.
  - LLMVerifier      : LLM-as-a-judge, one call per answer that labels all claims.
  - LexicalVerifier  : word/number overlap; offline fallback and a simple baseline for the thesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import prompts
from .llm import BaseLLM
from .textutils import content_tokens, numbers_in, split_sentences, strip_citations


@dataclass
class ClaimVerdict:
    claim: str
    label: str          # supported | unsupported | contradicted
    score: float        # support strength 0..1
    evidence: str = ""  # passage number or quote

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verification:
    claims: list[ClaimVerdict]

    @property
    def faithfulness(self) -> float:
        """Share of claims supported by the context (RAGAS-style faithfulness); 1.0 if there are no claims."""
        if not self.claims:
            return 1.0
        return sum(c.label == "supported" for c in self.claims) / len(self.claims)

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.claims if c.label != "supported"]

    def to_dict(self) -> dict:
        return {"faithfulness": round(self.faithfulness, 4), "claims": [c.to_dict() for c in self.claims]}


# ------------------------------------------------------------------ claim extraction

class ClaimExtractor:
    def __init__(self, llm: BaseLLM | None, method: str = "llm") -> None:
        self.llm = llm
        self.method = method

    def extract(self, question: str, answer: str) -> list[str]:
        clean = strip_citations(answer)
        if not clean:
            return []
        if self.method == "llm" and self.llm is not None:
            data = self.llm.chat_json(
                [{"role": "system", "content": prompts.EXTRACT_CLAIMS_SYSTEM},
                 {"role": "user", "content": prompts.EXTRACT_CLAIMS_USER.format(question=question, answer=clean)}],
                task="extract_claims", meta={"question": question, "answer": answer},
            )
            claims = [str(c).strip() for c in data.get("claims", []) if str(c).strip()]
            if claims:
                return claims
        return split_sentences(clean)  # fallback / "sentences" method


# ------------------------------------------------------------------ verifiers

def lexical_support(claim: str, contexts: list[str]) -> float:
    """Best fraction of the claim's content words found in a window of 1-2 context sentences.

    A window only counts if it also contains every number in the claim, because wrong numbers
    are the most common kind of hallucination in policy documents.
    """
    claim = strip_citations(claim)
    claim_toks = set(content_tokens(claim))
    if not claim_toks:
        return 1.0
    claim_nums = numbers_in(claim)
    best = 0.0
    for ctx in contexts:
        sents = split_sentences(ctx)
        for i in range(len(sents)):
            window = " ".join(sents[i : i + 2])
            if claim_nums and not claim_nums <= numbers_in(window):
                continue
            toks = set(content_tokens(window))
            best = max(best, len(claim_toks & toks) / len(claim_toks))
    return best


class LexicalVerifier:
    name = "lexical"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def verify(self, claims: list[str], contexts: list[str]) -> Verification:
        out = []
        for c in claims:
            s = lexical_support(c, contexts)
            out.append(ClaimVerdict(c, "supported" if s >= self.threshold else "unsupported", round(s, 4)))
        return Verification(out)


class NLIVerifier:
    """Claim is supported if some passage (or sentence window) entails it with probability >= threshold."""

    name = "nli"

    def __init__(self, model: str, threshold: float = 0.5) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model)
        self.threshold = threshold
        id2label = getattr(getattr(self.model, "model", None), "config", None)
        id2label = getattr(id2label, "id2label", None) or {0: "contradiction", 1: "entailment", 2: "neutral"}
        labels = {int(k): str(v).lower() for k, v in id2label.items()}
        self.entail_idx = next(i for i, l in labels.items() if l.startswith("entail"))
        self.contra_idx = next(i for i, l in labels.items() if l.startswith("contra"))

    @staticmethod
    def _premises(contexts: list[str]) -> list[str]:
        premises = []
        for ctx in contexts:
            premises.append(ctx)  # whole passage
            sents = split_sentences(ctx)
            for i in range(0, len(sents), 2):  # short windows help NLI models trained on short premises
                premises.append(" ".join(sents[i : i + 3]))
        return premises

    def verify(self, claims: list[str], contexts: list[str]) -> Verification:
        if not claims:
            return Verification([])
        premises = self._premises(contexts)
        pairs = [(p, strip_citations(c)) for c in claims for p in premises]
        logits = np.asarray(self.model.predict(pairs, batch_size=32), dtype=float)
        if logits.ndim == 1:
            raise RuntimeError("NLI model must output 3 classes (contradiction, entailment, neutral)")
        if not np.allclose(logits.sum(axis=1), 1.0, atol=1e-3):
            logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            logits = logits / logits.sum(axis=1, keepdims=True)
        probs = logits.reshape(len(claims), len(premises), -1)
        out = []
        for ci, claim in enumerate(claims):
            entail = probs[ci, :, self.entail_idx]
            contra = probs[ci, :, self.contra_idx]
            j = int(entail.argmax())
            if entail[j] >= self.threshold:
                label = "supported"
            elif contra.max() >= 0.5:
                label = "contradicted"
            else:
                label = "unsupported"
            out.append(ClaimVerdict(claim, label, round(float(entail[j]), 4), premises[j][:200]))
        return Verification(out)


class LLMVerifier:
    """LLM-as-a-judge verifier: one call labels all claims of an answer."""

    name = "llm"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def verify(self, claims: list[str], contexts: list[str]) -> Verification:
        if not claims:
            return Verification([])
        data = self.llm.chat_json(
            [{"role": "system", "content": prompts.VERIFY_SYSTEM},
             {"role": "user", "content": prompts.VERIFY_USER.format(
                 context=prompts.format_context(contexts), claims=prompts.format_claims(claims))}],
            task="verify_claims", meta={"claims": claims, "contexts": contexts},
        )
        by_id = {}
        for v in data.get("verdicts", []):
            try:
                by_id[int(v.get("id"))] = v
            except (TypeError, ValueError):
                continue
        out = []
        for i, claim in enumerate(claims, start=1):
            v = by_id.get(i, {})
            label = str(v.get("label", "unsupported")).lower().strip()
            if label not in {"supported", "unsupported", "contradicted"}:
                label = "unsupported"  # unparseable verdicts are treated conservatively
            out.append(ClaimVerdict(claim, label, 1.0 if label == "supported" else 0.0, str(v.get("evidence", ""))))
        return Verification(out)


def build_verifier(cfg: dict, llm: BaseLLM | None):
    method = cfg.get("method", "nli")
    if method == "nli":
        return NLIVerifier(cfg["nli_model"], float(cfg.get("support_threshold", 0.5)))
    if method == "llm":
        if llm is None:
            raise ValueError("verification.method=llm needs an LLM")
        return LLMVerifier(llm)
    if method == "lexical":
        return LexicalVerifier()
    raise ValueError(f"Unknown verification method: {method}")
