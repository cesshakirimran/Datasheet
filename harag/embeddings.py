"""Embedding back-ends. All return L2-normalised float32 matrices, so dot product = cosine similarity."""
from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> None:  # only needed by corpus-fitted embedders (TF-IDF)
        pass

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, folder: Path) -> None:
        pass

    def load(self, folder: Path) -> None:
        pass

    def describe(self) -> dict:
        return {"provider": self.name}


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str, query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import: heavy dependency

        self.model_name = model
        self.query_prefix = query_prefix or ""
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return _normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + text])[0]

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model_name}


class OpenAICompatibleEmbedder(Embedder):
    """Any /v1/embeddings endpoint (OpenAI, Ollama with nomic-embed-text, vLLM, ...)."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 120) -> None:
        self.model_name = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            body = json.dumps({"model": self.model_name, "input": texts[i : i + 64]}).encode()
            req = urllib.request.Request(self.url, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return _normalise(np.array(out))

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model_name}


class TfidfEmbedder(Embedder):
    """Offline fallback: TF-IDF vectors fitted on the corpus. Lexical, not semantic; for smoke tests only."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        self.fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedder must be fitted (run `ingest`) before use")
        return _normalise(self.vectorizer.transform(texts).toarray())

    def save(self, folder: Path) -> None:
        with (folder / "tfidf.pkl").open("wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, folder: Path) -> None:
        with (folder / "tfidf.pkl").open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.fitted = True


def build_embedder(cfg: dict) -> Embedder:
    provider = cfg.get("provider", "sentence-transformers")
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg["model"], cfg.get("query_prefix", ""))
    if provider == "openai":
        return OpenAICompatibleEmbedder(cfg["model"], cfg["base_url"], cfg.get("api_key", ""))
    if provider == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embeddings provider: {provider}")
