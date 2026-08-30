import json
from pathlib import Path

import numpy as np


class VectorStore:
    """An in-memory semantic index over text chunks, each tagged with source metadata for citations."""

    def __init__(self):
        self.chunks: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[str], metadatas: list[dict], embeddings: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.metadatas.extend(metadatas)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top_k most similar chunks, each with its source metadata and similarity score."""
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            {"text": self.chunks[i], "score": float(scores[i]), **self.metadatas[i]}
            for i in top_indices
        ]

    def save(self, dir_path: str) -> None:
        """Persist the index to disk so it doesn't need to be rebuilt on every run."""
        dest = Path(dir_path)
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "embeddings.npy", self.embeddings)
        records = [{"text": chunk, **metadata} for chunk, metadata in zip(self.chunks, self.metadatas)]
        (dest / "records.json").write_text(json.dumps(records, indent=2))

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        dest = Path(dir_path)
        records = json.loads((dest / "records.json").read_text())
        store = cls()
        store.chunks = [record["text"] for record in records]
        store.metadatas = [{k: v for k, v in record.items() if k != "text"} for record in records]
        store.embeddings = np.load(dest / "embeddings.npy")
        return store
