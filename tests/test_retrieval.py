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
    @staticmethod
    def search(*_args, **_kwargs) -> list:
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
    events: list[PipelineEvent] = []

    hits = Retriever(
        config,
        embedder,
        EmptyManifest(),
        EmptyVectorStore(),
    ).search("  тестовый   запрос?  ", on_event=events.append)

    assert hits == []
    assert embedder.queries == ["тестовый запрос"]
    completed = next(
        event
        for event in events
        if event.stage == "retrieval.query_embedding" and event.status == "completed"
    )
    assert completed.details["query_normalized"] is True
