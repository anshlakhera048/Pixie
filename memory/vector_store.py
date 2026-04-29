"""FAISS-backed vector memory for semantic retrieval.

Uses sentence-transformers for embedding and FAISS IndexFlatL2 for
nearest-neighbour search.  Falls back to the naive in-memory store
from ``memory.memory`` if the dependencies are not installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from memory.memory import BaseVectorMemory

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = Path(".pixie_data")


class FAISSVectorMemory(BaseVectorMemory):
    """Production vector store backed by FAISS and sentence-transformers.

    Parameters
    ----------
    model_name:
        HuggingFace sentence-transformers model id.
    persist_dir:
        If set, the FAISS index and text store are saved/loaded from this
        directory automatically.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        persist_dir: Path | None = DEFAULT_PERSIST_DIR,
    ) -> None:
        log.info("Loading embedding model '%s' …", model_name)
        self._encoder = SentenceTransformer(model_name)
        self._dimension: int = self._encoder.get_embedding_dimension()
        self._persist_dir = persist_dir

        # Parallel arrays: FAISS stores vectors, we store raw text + metadata
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self._dimension)

        # Attempt to load persisted state
        if self._persist_dir:
            self._load()

        log.info(
            "FAISSVectorMemory ready (dim=%d, entries=%d)",
            self._dimension,
            self._index.ntotal,
        )

    # ------------------------------------------------------------------
    # BaseVectorMemory interface
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Embed and store a text chunk."""
        vec = self._encode(text)
        self._index.add(vec)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        self._maybe_persist()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k most semantically similar entries."""
        if self._index.ntotal == 0:
            return []

        vec = self._encode(query)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self._texts[idx],
                "metadata": self._metadata[idx],
                "distance": float(dist),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist index + texts to disk."""
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._persist_dir / "faiss.index"
        texts_path = self._persist_dir / "texts.npy"
        meta_path = self._persist_dir / "metadata.npy"

        faiss.write_index(self._index, str(index_path))
        np.save(str(texts_path), np.array(self._texts, dtype=object), allow_pickle=True)
        np.save(str(meta_path), np.array(self._metadata, dtype=object), allow_pickle=True)
        log.info("Persisted %d vectors to %s", self._index.ntotal, self._persist_dir)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted index if it exists."""
        if not self._persist_dir:
            return
        index_path = self._persist_dir / "faiss.index"
        texts_path = self._persist_dir / "texts.npy"
        meta_path = self._persist_dir / "metadata.npy"

        if index_path.exists() and texts_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                self._texts = list(np.load(str(texts_path), allow_pickle=True))
                if meta_path.exists():
                    self._metadata = list(np.load(str(meta_path), allow_pickle=True))
                else:
                    self._metadata = [{} for _ in self._texts]
                log.info("Loaded %d vectors from %s", self._index.ntotal, self._persist_dir)
            except Exception:
                log.warning("Failed to load persisted index; starting fresh", exc_info=True)

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single string into a 2-D float32 array for FAISS."""
        vec = self._encoder.encode([text], convert_to_numpy=True)
        return vec.astype(np.float32)

    def _maybe_persist(self) -> None:
        """Auto-save every 10 additions."""
        if self._persist_dir and self._index.ntotal % 10 == 0:
            self.save()
