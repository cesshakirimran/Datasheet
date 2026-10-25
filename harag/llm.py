"""LLM clients.

`OpenAICompatibleLLM` talks to any server implementing POST /v1/chat/completions
(Ollama, OpenAI, Azure OpenAI, vLLM, LM Studio, Groq, Together, ...). It uses only the
standard library, caches every response in SQLite, and records the latency and token
usage of each call, so experiments are reproducible and cost can be reported.

`DemoLLM` is a deterministic, extractive test double used by the offline profile and the
unit tests. It deliberately injects hallucinations into some answers. Never report its
numbers as research results.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .textutils import ABSTAIN_TOKEN, content_tokens, numbers_in, split_sentences, strip_citations


@dataclass
class LLMStats:
    calls: int = 0              # calls that hit the model
    cached_calls: int = 0       # calls answered from cache
    seconds: float = 0.0        # model latency (for cached calls: latency measured when first computed)
    wall: float = 0.0           # real time spent inside chat() in this run (small for cache hits)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def copy(self) -> "LLMStats":
        return LLMStats(**self.__dict__)

    def minus(self, other: "LLMStats") -> "LLMStats":
        return LLMStats(**{k: getattr(self, k) - getattr(other, k) for k in self.__dict__})

    @property
    def total_calls(self) -> int:
        return self.calls + self.cached_calls


class ResponseCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, latency REAL, "
                "prompt_tokens INTEGER, completion_tokens INTEGER)"
            )
            self._conn.commit()

    def get(self, key: str):
        if self._conn is None:
            return None
        with self._lock:
            return self._conn.execute(
                "SELECT response, latency, prompt_tokens, completion_tokens FROM cache WHERE key=?", (key,)
            ).fetchone()

    def put(self, key: str, response: str, latency: float, pt: int, ct: int) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (key, response, latency, pt, ct))
            self._conn.commit()


class BaseLLM:
    model: str = "base"

    def __init__(self) -> None:
        self.stats = LLMStats()

    def chat(self, messages: list[dict], *, task: str = "", meta: dict | None = None, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        raise NotImplementedError

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        text = self.chat(messages, json_mode=True, **kwargs)
        return parse_json(text)


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"items": value}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.0,
                 max_tokens: int = 512, json_mode: bool = True, timeout: int = 180,
                 cache: ResponseCache | None = None, retries: int = 3) -> None:
        super().__init__()
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.supports_json_mode = json_mode
        self.timeout = timeout
        self.cache = cache or ResponseCache(None)
        self.retries = retries

    def chat(self, messages, *, task="", meta=None, json_mode=False, temperature=None, max_tokens=None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode and self.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        t_start = time.perf_counter()
        try:
            return self._chat(payload, key)
        finally:
            self.stats.wall += time.perf_counter() - t_start

    def _chat(self, payload: dict, key: str) -> str:
        hit = self.cache.get(key)
        if hit is not None:
            response, latency, pt, ct = hit
            self.stats.cached_calls += 1
            self.stats.seconds += latency or 0.0
            self.stats.prompt_tokens += pt or 0
            self.stats.completion_tokens += ct or 0
            return response

        body = json.dumps(payload).encode()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.url, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                latency = time.perf_counter() - t0
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                self.stats.calls += 1
                self.stats.seconds += latency
                self.stats.prompt_tokens += pt
                self.stats.completion_tokens += ct
                self.cache.put(key, text, latency, pt, ct)
                return text
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = RuntimeError(f"HTTP {exc.code} from {self.url}: {detail}")
                if exc.code == 400 and "response_format" in payload:
                    payload.pop("response_format")  # endpoint does not support JSON mode
                    body = json.dumps(payload).encode()
                    continue
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = RuntimeError(f"Cannot reach LLM endpoint {self.url}: {exc}")
            time.sleep(2 ** attempt)
        raise last_error or RuntimeError("LLM call failed")


class DemoLLM(BaseLLM):
    """Deterministic extractive stand-in for an LLM (offline profile and unit tests only)."""

    model = "demo-extractive"

    def __init__(self, hallucination_rate: float = 0.35, seed: int = 42) -> None:
        super().__init__()
        self.hallucination_rate = hallucination_rate
        self.seed = seed

    def _rand(self, text: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{text}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def chat(self, messages, *, task="", meta=None, json_mode=False, temperature=None, max_tokens=None) -> str:
        self.stats.calls += 1
        meta = meta or {}
        if task == "generate":
            return self._generate(meta["question"], meta["contexts"])
        if task == "extract_claims":
            return json.dumps({"claims": split_sentences(strip_citations(meta["answer"]))})
        if task == "verify_claims":
            from .verifier import lexical_support  # local import avoids a cycle

            verdicts = []
            for i, claim in enumerate(meta["claims"], start=1):
                score = lexical_support(claim, meta["contexts"])
                verdicts.append({"id": i, "label": "supported" if score >= 0.6 else "unsupported", "evidence": ""})
            return json.dumps({"verdicts": verdicts})
        if task == "correct":
            bad = {strip_citations(c).lower() for c in meta["unsupported"]}
            kept = [s for s in split_sentences(meta["answer"]) if strip_citations(s).lower() not in bad]
            return " ".join(kept) if kept else ABSTAIN_TOKEN
        if task == "relevancy_questions":
            return json.dumps({"questions": []})
        return ABSTAIN_TOKEN

    def _generate(self, question: str, contexts: list[str]) -> str:
        q = set(content_tokens(question))
        best: list[tuple[float, int, str]] = []
        for idx, ctx in enumerate(contexts, start=1):
            body = ctx.split("\n", 1)[-1]
            for sent in split_sentences(body):
                toks = set(content_tokens(sent))
                if not toks:
                    continue
                overlap = len(q & toks) / max(len(q), 1)
                best.append((overlap, idx, sent))
        best.sort(key=lambda t: -t[0])
        if not best or best[0][0] < 0.34:
            return ABSTAIN_TOKEN
        top = best[0]
        answer = [f"{top[2].rstrip('.')}. [{top[1]}]"]
        if len(best) > 1 and best[1][0] >= 0.75 * top[0] and best[1][2] != top[2]:
            answer.append(f"{best[1][2].rstrip('.')}. [{best[1][1]}]")
        # inject a hallucination; more likely when the evidence is weak (as with real models)
        p = self.hallucination_rate if top[0] >= 0.6 else min(1.0, 2 * self.hallucination_rate)
        if self._rand(question) < p:
            answer[0] = self._perturb(answer[0], question)
        return " ".join(answer)

    def _perturb(self, sentence: str, question: str) -> str:
        nums = sorted(numbers_in(strip_citations(sentence)), key=len, reverse=True)
        if nums and self._rand(question + "num") < 0.7:
            n = nums[0]
            try:
                new = str(int(float(n)) + 5) if "." not in n else f"{float(n) * 2:.2f}"
            except ValueError:
                new = n + "0"
            return re.sub(rf"(?<![\d.]){re.escape(n)}(?![\d.])", new, sentence, count=1)
        cite = re.search(r"\[\d+\]", sentence)
        extra = "This rule also applies to contractors and interns, with approval from the CEO"
        return f"{sentence} {extra}. {cite.group(0) if cite else ''}".strip()


def build_llm(cfg: dict, cache_path: Path | None, seed: int = 42) -> BaseLLM:
    provider = cfg.get("provider", "openai-compatible")
    if provider == "demo":
        return DemoLLM(cfg.get("hallucination_rate", 0.35), seed)
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(
            base_url=cfg["base_url"], model=cfg["model"], api_key=cfg.get("api_key", ""),
            temperature=float(cfg.get("temperature", 0.0)), max_tokens=int(cfg.get("max_tokens", 512)),
            json_mode=bool(cfg.get("json_mode", True)), timeout=int(cfg.get("timeout", 180)),
            cache=ResponseCache(cache_path),
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
