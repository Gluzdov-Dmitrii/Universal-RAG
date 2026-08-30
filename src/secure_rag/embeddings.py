from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import EmbeddingConfig
from .hf_policy import hf_local_files_only


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def model_version(self) -> str: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig, cache_root: Path) -> None:
        local_files_only = hf_local_files_only()
        from sentence_transformers import SentenceTransformer

        self._config = config
        self._model = SentenceTransformer(
            config.model_id,
            revision=config.revision,
            cache_folder=str(cache_root),
            device=_resolve_device(config.device),
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        actual = int(self._model.get_embedding_dimension())
        if actual != config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: config={config.dimension}, model={actual}"
            )

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model_version(self) -> str:
        return f"{self._config.model_id}@{self._config.revision}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([f"{self._config.passage_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"{self._config.query_prefix}{text}"])[0]


class HashingEmbedder:
    """Small deterministic embedder for tests and offline smoke checks."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_version(self) -> str:
        return f"test-hashing-v1:{self.dimension}"

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[\w-]+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimension
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._one(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)
