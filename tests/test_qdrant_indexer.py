from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, replace

import httpx
import numpy as np
import pytest
from filelock import FileLock
from openpyxl import Workbook
from qdrant_client.http.exceptions import ResponseHandlingException

import secure_rag.ingestion.indexer as indexer_module
import secure_rag.retrieval.vector_store as vector_store_module
from secure_rag.config import load_config
from secure_rag.domain.models import ChunkRecord
from secure_rag.ingestion.chunking import chunk_text
from secure_rag.ingestion.extractors import ExtractionError, extract_document
from secure_rag.ingestion.indexer import Indexer
from secure_rag.ingestion.manifest import ManifestStore
from secure_rag.ingestion.signatures import compute_index_signature
from secure_rag.retrieval.embeddings import HashingEmbedder
from secure_rag.retrieval.service import Retriever
from secure_rag.retrieval.vector_store import (
    UPSERT_BATCH_SIZE,
    QdrantStore,
    QdrantTransientWriteError,
)


def test_index_signature_changes_with_qdrant_target(monkeypatch) -> None:
    monkeypatch.delenv("SECURE_RAG_QDRANT_MODE", raising=False)
    monkeypatch.delenv("SECURE_RAG_QDRANT_URL", raising=False)
    server = load_config()
    embedded = replace(
        server,
        qdrant=replace(server.qdrant, mode="embedded", url=None),
    )
    other_collection = replace(
        server,
        qdrant=replace(server.qdrant, collection_name="another_collection"),
    )

    server_signature = compute_index_signature(server, "test-embedding")
    assert server_signature != compute_index_signature(embedded, "test-embedding")
    assert server_signature != compute_index_signature(other_collection, "test-embedding")

    changed_passage_prefix = replace(
        server,
        embedding=replace(server.embedding, passage_prefix="changed-passage: "),
    )
    changed_query_prefix = replace(
        server,
        embedding=replace(server.embedding, query_prefix="changed-query: "),
    )
    assert server_signature != compute_index_signature(
        changed_passage_prefix, "test-embedding"
    )
    assert server_signature != compute_index_signature(
        changed_query_prefix, "test-embedding"
    )

    transport_tuned = replace(
        server,
        qdrant=replace(
            server.qdrant,
            timeout_seconds=15,
            write_max_attempts=1,
            retry_backoff_seconds=0.0,
        ),
    )
    assert server_signature == compute_index_signature(
        transport_tuned, "test-embedding"
    )


def test_qdrant_transport_defaults_and_environment(monkeypatch) -> None:
    for name in (
        "SECURE_RAG_QDRANT_TIMEOUT_SECONDS",
        "SECURE_RAG_QDRANT_WRITE_MAX_ATTEMPTS",
        "SECURE_RAG_QDRANT_RETRY_BACKOFF_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = load_config().qdrant
    assert defaults.timeout_seconds == 60
    assert defaults.write_max_attempts == 3
    assert defaults.retry_backoff_seconds == 0.25

    monkeypatch.setenv("SECURE_RAG_QDRANT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("SECURE_RAG_QDRANT_WRITE_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("SECURE_RAG_QDRANT_RETRY_BACKOFF_SECONDS", "0")
    overridden = load_config().qdrant
    assert overridden.timeout_seconds == 45
    assert overridden.write_max_attempts == 2
    assert overridden.retry_backoff_seconds == 0.0


def test_server_store_applies_global_client_timeout(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)

        def collection_exists(self, _name) -> bool:
            return False

        def create_collection(self, **_kwargs) -> None:
            pass

        def update_collection(self, **_kwargs) -> None:
            pass

        def create_payload_index(self, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(vector_store_module, "QdrantClient", FakeClient)
    with QdrantStore(
        tmp_path / "unused",
        "test_collection",
        8,
        url="http://127.0.0.1:6333",
        timeout_seconds=37,
    ):
        pass

    assert observed["timeout"] == 37


def test_incremental_index_and_filtered_retrieval(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "safe.txt"
    source_file.write_text(
        "Испытание турбины показало вибрацию на расчетном режиме.",
        encoding="utf-8",
    )
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)
    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(config.qdrant_path, "test_collection", embedder.dimension) as store,
    ):
        first = Indexer(config, embedder, manifest, store).run()
        second = Indexer(config, embedder, manifest, store).run()
        hits = Retriever(config, embedder, manifest, store).search("вибрация турбины")
        assert first.indexed == 1
        assert first.chunks == 1
        assert second.skipped == 1
        assert store.count() == 1
        assert len(hits) == 1
        assert "турбины" in hits[0].text

        changed_config = replace(
            config,
            ingestion=replace(config.ingestion, chunk_chars=400, chunk_overlap_chars=60),
        )
        changed = Indexer(changed_config, embedder, manifest, store).run()
        assert changed.indexed == 1
        assert changed.skipped == 0

        old_stat = source_file.stat()
        source_file.write_bytes(b"X" * old_stat.st_size)
        os.utime(source_file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        assert Retriever(changed_config, embedder, manifest, store).search("вибрация") == []

        repaired = Indexer(changed_config, embedder, manifest, store).run()
        assert repaired.indexed == 1
        source_file.unlink()
        preserved = Indexer(changed_config, embedder, manifest, store).run()
        assert preserved.failed == 0
        assert store.count() == 1
        removed = Indexer(changed_config, embedder, manifest, store).run(
            prune_missing=True
        )
        assert removed.failed == 0
        assert store.count() == 0


def test_new_xlsx_persists_sheet_location_in_manifest_and_qdrant(tmp_path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Расчёты"
    sheet.append(["Показатель", "Значение"])
    sheet.append(["Масса", 125])
    workbook.save(source / "report.xlsx")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "source_location_collection",
            embedder.dimension,
        ) as store,
    ):
        report = Indexer(config, embedder, manifest, store).run()
        manifest_location = manifest.connection.execute(
            """
            SELECT location_kind, location_start, location_end
            FROM chunks
            """
        ).fetchone()
        points, _ = store.client.scroll(
            collection_name=store.collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

    assert report.indexed == 1
    assert dict(manifest_location) == {
        "location_kind": "sheet",
        "location_start": "Расчёты",
        "location_end": None,
    }
    assert len(points) == 1
    assert points[0].payload["location_kind"] == "sheet"
    assert points[0].payload["location_start"] == "Расчёты"
    assert points[0].payload["location_end"] is None


def test_index_progress_is_aggregate_and_log_safe(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret_filename = "sensitive-customer-name.txt"
    secret_content = "Commercial secret value 123456"
    (source / secret_filename).write_text(secret_content, encoding="utf-8")

    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)
    events = []
    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(config.qdrant_path, "progress_collection", embedder.dimension) as store,
    ):
        report = Indexer(config, embedder, manifest, store).run(
            progress=events.append,
            progress_every_files=1,
            progress_every_seconds=3600,
        )

    assert report.indexed == 1
    assert [event.phase for event in events] == ["started", "running", "complete"]
    assert events[-1].discovered == 1
    assert events[-1].indexed == 1
    assert events[-1].chunks == 1
    serialized = json.dumps([asdict(event) for event in events])
    assert secret_filename not in serialized
    assert secret_content not in serialized
    assert str(tmp_path) not in serialized


def test_failed_reindex_is_retried_instead_of_marked_as_skipped(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text(
        "Расчётная вибрация ротора на контрольном режиме.",
        encoding="utf-8",
    )
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    changed_config = replace(
        config,
        ingestion=replace(config.ingestion, chunk_chars=400, chunk_overlap_chars=60),
    )
    changed_config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(changed_config.manifest_path) as manifest,
        QdrantStore(
            changed_config.qdrant_path,
            "recovery_collection",
            embedder.dimension,
        ) as store,
    ):
        assert Indexer(config, embedder, manifest, store).run().indexed == 1
        original_extract = indexer_module.extract_document

        def fail_extract(*_args, **_kwargs):
            raise ExtractionError("synthetic_failure")

        monkeypatch.setattr(indexer_module, "extract_document", fail_extract)
        failed = Indexer(changed_config, embedder, manifest, store).run()
        assert failed.failed == 1
        # Extraction failed before vector replacement began, so the last usable
        # version remains available in Qdrant until a replacement succeeds.
        assert store.count() == 1

        monkeypatch.setattr(indexer_module, "extract_document", original_extract)
        repaired = Indexer(changed_config, embedder, manifest, store).run()
        assert repaired.indexed == 1
        assert repaired.skipped == 0
        assert len(
            Retriever(changed_config, embedder, manifest, store).search("вибрация")
        ) == 1


def test_second_index_writer_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text("Безопасный тест", encoding="utf-8")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)
    lock_path = config.locks_path / "index-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        FileLock(str(lock_path)),
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(config.qdrant_path, "locked_collection", embedder.dimension) as store,
    ):
        with pytest.raises(RuntimeError, match="index_build_already_running"):
            Indexer(config, embedder, manifest, store).run()


def test_stale_lock_file_does_not_block_index_after_process_exit(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text("Безопасный тест", encoding="utf-8")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    lock_path = config.locks_path / "index-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("left by terminated process", encoding="utf-8")
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "stale_lock_collection",
            embedder.dimension,
        ) as store,
    ):
        report = Indexer(config, embedder, manifest, store).run()

    assert report.indexed == 1


def test_starting_build_marks_stale_running_build_as_interrupted(tmp_path) -> None:
    manifest_path = tmp_path / "manifest" / "documents.sqlite"

    with ManifestStore(manifest_path) as manifest:
        manifest.start_build("stale-build", "embedding", "config")
        manifest.start_build("replacement-build", "embedding", "config")
        rows = manifest.connection.execute(
            """
            SELECT build_id, status, finished_at
            FROM builds
            ORDER BY started_at, build_id
            """
        ).fetchall()

    by_id = {str(row["build_id"]): row for row in rows}
    assert by_id["stale-build"]["status"] == "interrupted"
    assert by_id["stale-build"]["finished_at"] is not None
    assert by_id["replacement-build"]["status"] == "running"
    assert by_id["replacement-build"]["finished_at"] is None


def test_preflight_failure_finishes_build_as_failed(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "preflight_failure_collection",
            embedder.dimension,
        ) as store,
    ):
        def fail_count(*_args, **_kwargs):
            raise RuntimeError("synthetic_qdrant_failure")

        monkeypatch.setattr(store, "count_signature", fail_count)
        with pytest.raises(RuntimeError, match="synthetic_qdrant_failure"):
            Indexer(config, embedder, manifest, store).run()
        row = manifest.connection.execute(
            "SELECT status, finished_at FROM builds"
        ).fetchone()

    assert row["status"] == "failed"
    assert row["finished_at"] is not None


def test_keyboard_interrupt_closes_build_and_releases_lock(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    def interrupt_scan(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(indexer_module, "iter_source_files", interrupt_scan)
    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "interrupt_collection",
            embedder.dimension,
        ) as store,
    ):
        with pytest.raises(KeyboardInterrupt):
            Indexer(config, embedder, manifest, store).run()
        row = manifest.connection.execute(
            "SELECT status, finished_at FROM builds"
        ).fetchone()

    assert row["status"] == "failed"
    assert row["finished_at"] is not None
    lock = FileLock(str(config.locks_path / "index-build.lock"))
    with lock.acquire(timeout=0):
        pass


def test_oversized_file_is_rejected_before_hashing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "large.txt").write_bytes(b"larger-than-zero")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
        ingestion=replace(base.ingestion, max_file_mb=0),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    def unexpected_hash(*_args, **_kwargs):
        raise AssertionError("oversized source must not be hashed")

    monkeypatch.setattr(indexer_module, "file_sha256", unexpected_hash)
    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(config.qdrant_path, "size_collection", embedder.dimension) as store,
    ):
        report = Indexer(config, embedder, manifest, store).run()
    assert report.failed == 1
    assert report.indexed == 0


def test_qdrant_upsert_is_split_into_bounded_batches(tmp_path, monkeypatch) -> None:
    dimension = 8
    chunks = [
        ChunkRecord(
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk-{index}")),
            document_id="a" * 32,
            revision="b" * 64,
            ordinal=index,
            start=index * 10,
            end=index * 10 + 10,
            text="test chunk",
        )
        for index in range(UPSERT_BATCH_SIZE * 2 + 1)
    ]
    vectors = np.ones((len(chunks), dimension), dtype=np.float32)

    with QdrantStore(tmp_path / "qdrant", "batch_collection", dimension) as store:
        original_upsert = store.client.upsert
        observed_sizes: list[int] = []

        def counted_upsert(*args, **kwargs):
            observed_sizes.append(len(kwargs["points"]))
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(store.client, "upsert", counted_upsert)
        store.upsert(
            chunks,
            vectors,
            build_id="build",
            embedding_version="embedding",
            index_signature="signature",
        )
        assert store.count() == len(chunks)

    assert observed_sizes == [UPSERT_BATCH_SIZE, UPSERT_BATCH_SIZE, 1]


@pytest.mark.parametrize("failure_kind", ["timeout", "transport"])
def test_qdrant_upsert_retries_transient_failures(
    tmp_path, monkeypatch, failure_kind
) -> None:
    dimension = 8
    chunk = ChunkRecord(
        chunk_id=str(uuid.uuid4()),
        document_id="a" * 32,
        revision="b" * 64,
        ordinal=0,
        start=0,
        end=10,
        text="safe synthetic chunk",
    )
    vector = np.ones((1, dimension), dtype=np.float32)

    with QdrantStore(
        tmp_path / "qdrant",
        f"retry_{failure_kind}_collection",
        dimension,
        timeout_seconds=17,
        write_max_attempts=3,
        retry_backoff_seconds=0,
    ) as store:
        original_upsert = store.client.upsert
        calls = 0

        def flaky_upsert(*args, **kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["timeout"] == 17
            if calls < 3:
                request = httpx.Request("PUT", "http://127.0.0.1:6333")
                source = (
                    httpx.ReadTimeout("synthetic", request=request)
                    if failure_kind == "timeout"
                    else httpx.ConnectError("synthetic", request=request)
                )
                raise ResponseHandlingException(source)
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(store.client, "upsert", flaky_upsert)
        store.upsert(
            [chunk],
            vector,
            build_id="build",
            embedding_version="embedding",
            index_signature="signature",
        )
        assert calls == 3
        assert store.count() == 1


def test_qdrant_delete_document_retries_transient_failure(
    tmp_path, monkeypatch
) -> None:
    dimension = 8
    chunk = ChunkRecord(
        chunk_id=str(uuid.uuid4()),
        document_id="a" * 32,
        revision="b" * 64,
        ordinal=0,
        start=0,
        end=10,
        text="safe synthetic chunk",
    )
    vector = np.ones((1, dimension), dtype=np.float32)

    with QdrantStore(
        tmp_path / "qdrant",
        "retry_delete_collection",
        dimension,
        timeout_seconds=19,
        write_max_attempts=3,
        retry_backoff_seconds=0,
    ) as store:
        store.upsert(
            [chunk],
            vector,
            build_id="build",
            embedding_version="embedding",
            index_signature="signature",
        )
        original_delete = store.client.delete
        calls = 0

        def flaky_delete(*args, **kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["timeout"] == 19
            if calls < 3:
                request = httpx.Request("POST", "http://127.0.0.1:6333")
                raise ResponseHandlingException(
                    httpx.ConnectError("synthetic", request=request)
                )
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(store.client, "delete", flaky_delete)
        store.delete_document(chunk.document_id)
        assert calls == 3
        assert store.count() == 0


def test_indexer_classifies_exhausted_qdrant_timeout(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text("Безопасный тест", encoding="utf-8")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "timeout_classification_collection",
            embedder.dimension,
            write_max_attempts=3,
            retry_backoff_seconds=0,
        ) as store,
    ):
        calls = 0

        def always_timeout(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            request = httpx.Request("PUT", "http://127.0.0.1:6333")
            raise ResponseHandlingException(
                httpx.ReadTimeout("synthetic", request=request)
            )

        monkeypatch.setattr(store.client, "upsert", always_timeout)
        report = Indexer(config, embedder, manifest, store).run()
        row = manifest.connection.execute(
            "SELECT status, error_code FROM documents"
        ).fetchone()

        assert calls == 3
        assert report.failed == 1
        assert store.count() == 0
        assert row["status"] == "failed"
        assert row["error_code"] == "qdrant_timeout"


def test_exhausted_transport_error_is_safe(tmp_path, monkeypatch) -> None:
    with QdrantStore(
        tmp_path / "qdrant",
        "transport_classification_collection",
        8,
        write_max_attempts=3,
        retry_backoff_seconds=0,
    ) as store:
        calls = 0

        def always_disconnected(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            request = httpx.Request("POST", "http://127.0.0.1:6333")
            raise ResponseHandlingException(
                httpx.ConnectError("sensitive upstream detail", request=request)
            )

        monkeypatch.setattr(store.client, "delete", always_disconnected)
        with pytest.raises(QdrantTransientWriteError) as caught:
            store.delete_document("a" * 32)

    assert calls == 3
    assert caught.value.code == "qdrant_transport"
    assert str(caught.value) == "qdrant_transport"
    assert "sensitive" not in str(caught.value)


def test_failed_mid_batch_upsert_removes_partial_document_points(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text(
        " ".join(f"расчёт-{index}" for index in range(500)),
        encoding="utf-8",
    )
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "failed_batch_collection",
            embedder.dimension,
        ) as store,
    ):
        original_upsert = store.client.upsert
        upsert_calls = 0

        def fail_second_upsert(*args, **kwargs):
            nonlocal upsert_calls
            upsert_calls += 1
            if upsert_calls == 2:
                raise RuntimeError("synthetic_mid_batch_failure")
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(vector_store_module, "UPSERT_BATCH_SIZE", 1)
        monkeypatch.setattr(store.client, "upsert", fail_second_upsert)

        report = Indexer(config, embedder, manifest, store).run()

        document = manifest.all_documents()[0]
        row = manifest.connection.execute(
            "SELECT status, error_code FROM documents WHERE document_id = ?",
            (document.document_id,),
        ).fetchone()
        assert upsert_calls == 2
        assert report.failed == 1
        assert report.indexed == 0
        assert store.count() == 0
        assert row["status"] == "failed"
        assert row["error_code"] == "internal_error"


def test_manifest_does_not_skip_after_qdrant_collection_loss(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.txt").write_text(
        "Контрольный расчёт вибрации.",
        encoding="utf-8",
    )
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(config.qdrant_path, "loss_collection", embedder.dimension) as store,
    ):
        assert Indexer(config, embedder, manifest, store).run().indexed == 1
        assert manifest.indexed_chunk_count(
            compute_index_signature(config, embedder.model_version)
        ) == 1
        store.delete_document(manifest.all_documents()[0].document_id)
        assert store.count() == 0
        recovered = Indexer(config, embedder, manifest, store).run()

        assert recovered.indexed == 1
        assert recovered.skipped == 0
        assert recovered.details["fast_skip_safe"] is False
        assert store.count() == 1


def test_partial_document_upsert_is_replaced_without_deleting_other_signature(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "document.txt"
    source_file.write_text(
        " ".join(f"расчёт-{index}" for index in range(500)),
        encoding="utf-8",
    )
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source.resolve(),
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    embedder = HashingEmbedder(64)
    signature = compute_index_signature(config, embedder.model_version)

    with (
        ManifestStore(config.manifest_path) as manifest,
        QdrantStore(
            config.qdrant_path,
            "partial_upsert_collection",
            embedder.dimension,
        ) as store,
    ):
        assert Indexer(config, embedder, manifest, store).run().indexed == 1
        document = manifest.all_documents()[0]
        extracted = extract_document(source_file, config.ingestion)
        chunks = chunk_text(
            extracted.text,
            document_id=document.document_id,
            revision=document.revision,
            chunk_chars=config.ingestion.chunk_chars,
            overlap_chars=config.ingestion.chunk_overlap_chars,
            min_chunk_chars=config.ingestion.min_chunk_chars,
            access_group=config.retrieval.access_group,
        )
        assert len(chunks) > 1
        vectors = embedder.embed_passages([chunk.text for chunk in chunks])

        # Simulate process death after only the first Qdrant batch/point was
        # committed. The document status was already committed as extracting.
        store.delete_document(document.document_id)
        store.upsert(
            chunks[:1],
            vectors[:1],
            build_id="interrupted-build",
            embedding_version=embedder.model_version,
            index_signature=signature,
        )
        manifest.upsert_document(replace(document, status="extracting"))

        unrelated = ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            document_id="f" * 32,
            revision="e" * 64,
            ordinal=0,
            start=0,
            end=10,
            text="unrelated",
        )
        store.upsert(
            [unrelated],
            np.ones((1, embedder.dimension), dtype=np.float32),
            build_id="other-build",
            embedding_version=embedder.model_version,
            index_signature="other-signature",
        )

        recovered = Indexer(config, embedder, manifest, store).run()

        assert recovered.indexed == 1
        assert recovered.skipped == 0
        assert recovered.details["fast_skip_safe"] is False
        assert store.count_signature(embedder.model_version, signature) == len(chunks)
        assert store.count() == len(chunks) + 1
        assert manifest.get_document(document.document_id).status == "indexed"
