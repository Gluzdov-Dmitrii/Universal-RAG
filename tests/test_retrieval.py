from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import secure_rag.retrieval.service as retrieval_module
from secure_rag.config import load_config
from secure_rag.domain.models import DocumentRecord, RetrievalHit, TextLocation
from secure_rag.ingestion.chunking import chunk_text
from secure_rag.ingestion.extractors import ExtractedDocument, extract_document, file_sha256
from secure_rag.ingestion.manifest import ManifestStore
from secure_rag.ingestion.signatures import compute_index_signature
from secure_rag.orchestration.events import PipelineEvent
from secure_rag.retrieval.embeddings import HashingEmbedder
from secure_rag.retrieval.service import Retriever


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


def test_retriever_derives_page_from_char_offsets_without_reindex_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    path = source / "report.pdf"
    path.write_bytes(b"stable-placeholder")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
        ingestion=replace(base.ingestion, persist_extracted_text_cache=False),
    )
    embedder = RecordingEmbedder()
    signature = compute_index_signature(config, embedder.model_version)
    revision = file_sha256(path)
    text = "первая страница\n\n[PAGE_BREAK]\n\nнужный фрагмент"
    second_page_start = text.index("нужный")
    extracted = ExtractedDocument(
        text,
        page_count=2,
        locations=(
            TextLocation(0, second_page_start, "page", "1"),
            TextLocation(second_page_start, len(text), "page", "2"),
        ),
    )
    monkeypatch.setattr(retrieval_module, "extract_document", lambda *_args: extracted)
    stat = path.stat()
    record = DocumentRecord(
        document_id="doc-legacy",
        # The synchronized source root moved after indexing. Relative identity remains
        # stable and retrieval must resolve it against the current configured root.
        source_path=tmp_path / "retired-source" / "report.pdf",
        relative_path="report.pdf",
        extension=".pdf",
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        revision=revision,
        status="indexed",
        indexed_revision=revision,
        index_signature=signature,
    )

    class LegacyVectorStore:
        @staticmethod
        def search(*_args, **_kwargs):
            return [
                SimpleNamespace(
                    id="chunk-legacy",
                    score=0.9,
                    payload={
                        "document_id": record.document_id,
                        "revision": revision,
                        "chunk_id": "chunk-legacy",
                        "ordinal": 1,
                        "char_start": second_page_start,
                        "char_end": len(text),
                        "embedding_version": embedder.model_version,
                        "index_signature": signature,
                    },
                )
            ]

    with ManifestStore(config.manifest_path) as manifest:
        manifest.upsert_document(record)
        hits = Retriever(config, embedder, manifest, LegacyVectorStore()).search("фрагмент")

    assert len(hits) == 1
    assert hits[0].location_kind == "page"
    assert hits[0].location_start == "2"
    assert hits[0].location_end is None
    assert hits[0].source_path == path


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


def test_retriever_bounds_verified_source_documents(tmp_path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
        retrieval=replace(base.retrieval, max_source_documents=2),
    )
    embedder = RecordingEmbedder()
    signature = compute_index_signature(config, embedder.model_version)
    records = []
    points = []
    with ManifestStore(config.manifest_path) as manifest:
        for index in range(3):
            path = source / f"document-{index}.txt"
            text = f"Документ номер {index}"
            path.write_text(text, encoding="utf-8")
            stat = path.stat()
            revision = file_sha256(path)
            record = DocumentRecord(
                document_id=f"doc-{index}",
                source_path=path,
                relative_path=path.name,
                extension=".txt",
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                revision=revision,
                status="indexed",
                indexed_revision=revision,
                index_signature=signature,
            )
            records.append(record)
            manifest.upsert_document(record)
            points.append(
                SimpleNamespace(
                    id=f"chunk-{index}",
                    score=1.0 - index / 10,
                    payload={
                        "document_id": record.document_id,
                        "revision": revision,
                        "chunk_id": f"chunk-{index}",
                        "ordinal": 0,
                        "char_start": 0,
                        "char_end": len(text),
                        "embedding_version": embedder.model_version,
                        "index_signature": signature,
                    },
                )
            )

        class ThreeDocumentVectorStore:
            @staticmethod
            def search(*_args, **_kwargs):
                return points

        hits = Retriever(config, embedder, manifest, ThreeDocumentVectorStore()).search(
            "документ"
        )

    assert [hit.document_id for hit in hits] == [record.document_id for record in records[:2]]


def test_retriever_expands_authorized_citation_to_adjacent_chunks(tmp_path) -> None:
    base = load_config()
    source = (tmp_path / "source").resolve()
    source.mkdir()
    runtime = (tmp_path / "runtime").resolve()
    path = source / "table.txt"
    path.write_text("header\n" + "value " * 80, encoding="utf-8")
    config = replace(
        base,
        paths=replace(base.paths, source_root=source, runtime_root=runtime),
        ingestion=replace(
            base.ingestion,
            chunk_chars=80,
            chunk_overlap_chars=10,
            min_chunk_chars=10,
        ),
    )
    extracted = extract_document(path, config.ingestion)
    revision = file_sha256(path)
    chunks = chunk_text(
        extracted.text,
        "doc-1",
        revision,
        config.ingestion.chunk_chars,
        config.ingestion.chunk_overlap_chars,
        config.ingestion.min_chunk_chars,
        text_locations=(TextLocation(0, len(extracted.text), "page", "7"),),
    )
    assert len(chunks) >= 3
    stat = path.stat()
    record = DocumentRecord(
        document_id="doc-1",
        source_path=path,
        relative_path="table.txt",
        extension=".txt",
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        revision=revision,
        status="indexed",
        indexed_revision=revision,
        index_signature="sig",
    )
    with ManifestStore(runtime / "manifest" / "documents.sqlite") as manifest:
        manifest.upsert_document(record)
        manifest.replace_chunks(record.document_id, chunks)
        middle = chunks[1]
        hit = RetrievalHit(
            chunk_id=middle.chunk_id,
            document_id=record.document_id,
            revision=revision,
            score=0.9,
            start=middle.start,
            end=middle.end,
            text=middle.text,
            source_name=path.name,
            source_path=path,
            source_type="txt",
            ordinal=middle.ordinal,
            index_signature="sig",
        )
        expanded = Retriever(
            config,
            HashingEmbedder(),
            manifest,
            EmptyVectorStore(),
        ).expand_adjacent([hit], radius=1)

    assert [item.ordinal for item in expanded] == [0, 2]
    assert all(item.source_path == path for item in expanded)
    assert all(item.location_kind == "page" for item in expanded)
    assert all(item.location_start == "7" for item in expanded)
