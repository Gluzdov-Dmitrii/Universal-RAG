from __future__ import annotations

from dataclasses import replace

import numpy as np

from secure_rag.config import load_config
from secure_rag.events import PipelineEvent
from secure_rag.retrieval import Retriever


class RecordingEmbedder:
    dimension = 4
    model_version = "recording-v1"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return np.ones(self.dimension, dtype=np.float32)


class EmptyVectorStore:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] = {}

    def search(self, *_args, **kwargs) -> list:
        self.search_kwargs = kwargs
        return []


class EmptyManifest:
    pass


def test_retrieval_normalizes_terminal_punctuation_before_embedding(tmp_path) -> None:
    base = load_config()
    source = (tmp_path / "source").resolve()
    source.mkdir()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    embedder = RecordingEmbedder()
    vector_store = EmptyVectorStore()
    events: list[PipelineEvent] = []

    hits = Retriever(
        config,
        embedder,
        EmptyManifest(),
        vector_store,
    ).search("  тестовый   запрос?  ", on_event=events.append)

    assert hits == []
    assert embedder.queries == ["тестовый запрос"]
    assert vector_store.search_kwargs["hnsw_ef"] == 512
    assert vector_store.search_kwargs["exact_search"] is False
    completed = next(
        event
        for event in events
        if event.stage == "retrieval.query_embedding" and event.status == "completed"
    )
    assert completed.details["query_normalized"] is True
