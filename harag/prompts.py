"""All prompts in one place, so they can be quoted in the thesis appendix and changed consistently."""
from __future__ import annotations

from .textutils import ABSTAIN_TOKEN

# ------------------------------------------------------------------ generation (all configurations)

GENERATE_SYSTEM = f"""You are an enterprise knowledge assistant. Answer the employee's question using ONLY the numbered context passages.
Rules:
- Every factual sentence must end with the number(s) of the passage(s) that support it, e.g. [1] or [2, 3].
- Do not use outside knowledge, and do not guess numbers, names, dates or amounts.
- Be concise: at most 4 sentences.
- If the passages do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}"""

GENERATE_USER = """Context passages:
{context}

Question: {question}

Answer:"""

# ------------------------------------------------------------------ claim extraction

EXTRACT_CLAIMS_SYSTEM = """You split an answer into atomic factual claims.
Each claim must be a short, self-contained sentence that can be checked on its own (resolve pronouns,
keep numbers and units exactly). Ignore citation markers like [1]. Do not add information.
Return JSON: {"claims": ["...", "..."]}"""

EXTRACT_CLAIMS_USER = """Question: {question}
Answer: {answer}

Return the JSON object."""

# ------------------------------------------------------------------ claim verification (LLM-as-a-judge)

VERIFY_SYSTEM = """You are a strict fact-checker. For each claim, decide whether it is fully supported by the context.
Labels:
- "supported": the context states it or it follows directly from the context.
- "contradicted": the context states something incompatible with it.
- "unsupported": the context does not contain enough information (even if the claim may be true in general).
Judge only against the context, not world knowledge. Numbers, dates and conditions must match exactly.
Return JSON: {"verdicts": [{"id": 1, "label": "supported|unsupported|contradicted", "evidence": "short quote or empty"}]}"""

VERIFY_USER = """Context:
{context}

Claims:
{claims}

Return the JSON object with one verdict per claim id."""

# ------------------------------------------------------------------ correction (hallucination mitigation)

CORRECT_SYSTEM = f"""You revise a draft answer so that it contains only information supported by the context passages.
Remove or fix every claim listed as unsupported. Keep supported content and its citation numbers [n].
Do not add new facts. If nothing supported remains that answers the question, reply with exactly: {ABSTAIN_TOKEN}"""

CORRECT_USER = """Context passages:
{context}

Question: {question}

Draft answer:
{answer}

Claims NOT supported by the context:
{unsupported}

Revised answer:"""

# ------------------------------------------------------------------ evaluation judge

RELEVANCY_SYSTEM = """Given an answer, write {n} different questions that this answer would be a direct response to.
If the answer is evasive or says it does not know, return an empty list.
Return JSON: {{"questions": ["...", "..."]}}"""

RELEVANCY_USER = """Answer: {answer}

Return the JSON object."""

CORRECTNESS_SYSTEM = """You grade an answer against a reference answer written by a domain expert.
Labels:
- "correct": contains the key facts of the reference and nothing that contradicts it.
- "partially_correct": contains some key facts but misses or garbles others.
- "incorrect": wrong, contradicts the reference, or does not answer.
Extra details not in the reference are fine unless they contradict it.
Return JSON: {"label": "correct|partially_correct|incorrect", "reason": "one sentence"}"""

CORRECTNESS_USER = """Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Return the JSON object."""


def format_context(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))


def format_claims(claims: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
