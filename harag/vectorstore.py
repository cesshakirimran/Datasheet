"""A small persistent vector store (NumPy, exact cosine search).

Exact search is fast enough for tens of thousands of chunks and keeps the experiments
deterministic. For larger corpora the same interface can wrap ChromaDB, Qdrant or FAISS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .documents import Chunk, load_chunks, save_chunks
from .embeddings import Embedder


@dataclass
class Hit:
    chunk: Chunk
    vector_score: float        # cosine similarity query <-> chunk
    rerank_score: float = 0.0  # 0..1 relevance from the reranker (filled later)
    vector: np.ndarray | None = None  # chunk embedding (used for the similarity signal)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "vector_score": round(float(self.vector_score), 4),
            "rerank_score": round(float(self.rerank_score), 4),
            "text": self.chunk.text,
        }


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorStore":
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        return cls(chunks, embedder.embed_documents(texts))

    def search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        scores = self.vectors @ query_vec.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), vector=self.vectors[i]) for i in order]

    def save(self, folder: Path, embedder: Embedder, extra_meta: dict | None = None) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        save_chunks(self.chunks, folder / "chunks.jsonl")
        embedder.save(folder)
        meta = {"embedder": embedder.describe(), "n_chunks": len(self.chunks), **(extra_meta or {})}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> "VectorStore":
        if not (folder / "vectors.npy").exists():
            raise FileNotFoundError(f"No index in {folder}. Run `python -m harag ingest` first.")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.describe():
            raise RuntimeError(
                f"Index was built with {meta['embedder']} but config uses {embedder.describe()}. Re-run ingest."
            )
        embedder.load(folder)
        return cls(load_chunks(folder / "chunks.jsonl"), np.load(folder / "vectors.npy"))
