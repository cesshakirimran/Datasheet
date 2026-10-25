"""Small text helpers shared by several modules (no external dependencies)."""
from __future__ import annotations

import re

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

ABSTAIN_MESSAGE = (
    "I could not find a reliable answer to this question in the knowledge base. "
    "Please check the relevant policy owner or rephrase the question."
)

_STOP = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being this that these those
    it its into than then so such can could may might must shall should will would do does did done have has had
    i you he she we they me my your our their his her them us what which who whom whose when where why how
    not no yes all any each every some per also only just very about above below up down out over under again
    there here both either neither more most other own same too within without after before during between""".split()
)

_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|approx|incl|no|nr|Ltd|Kft|Dr|Mr|Ms|Mrs|St)\.$", re.IGNORECASE)


_SENT_END = re.compile(r"[.!?](?:\s*\[\d+(?:\s*,\s*\d+)*\])?(?=\s+[A-Z0-9\"'(\[]|\s*$)")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter.

    Keeps decimals (0.30), common abbreviations and list items intact, and keeps a trailing
    citation marker such as "[2]" with the sentence it belongs to.
    """
    pieces: list[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        start = 0
        for m in _SENT_END.finditer(line):
            candidate = line[start : m.end()].strip()
            if _ABBREV.search(line[start : m.start() + 1].strip()):
                continue
            if candidate:
                pieces.append(candidate)
            start = m.end()
        rest = line[start:].strip()
        if rest:
            pieces.append(rest)
    return [p for p in pieces if len(p) > 2]


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ž]+|\d+(?:[.,]\d+)*", text.lower())


def stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in _STOP and (len(t) > 1 or t.isdigit())]


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def is_abstention(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    a = answer.strip().lower()
    if ABSTAIN_TOKEN.lower() in a:
        return True
    patterns = (
        "i don't know", "i do not know", "cannot be answered", "can't be answered",
        "does not contain", "doesn't contain", "not mentioned in the context", "no information",
        "not provided in the context", "context does not", "i could not find",
    )
    return len(a) < 300 and any(p in a for p in patterns)
