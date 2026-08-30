from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace

from filelock import FileLock, Timeout

from .chunking import chunk_text
from .config import AppConfig
from .embeddings import Embedder
from .extracted_cache import ExtractedTextCache, is_cacheable_extension
from .extractors import (
    ExtractionError,
    extract_document,
    file_sha256,
    iter_source_files,
    stable_document_id,
)
from .manifest import ManifestStore
from .models import DocumentRecord, IndexProgress, IndexReport
from .signatures import compute_index_signature
from .vector_store import QdrantStore, QdrantTransientWriteError


class Indexer:
    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        manifest: ManifestStore,
        vector_store: QdrantStore,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.manifest = manifest
        self.vector_store = vector_store
        self.extracted_cache = ExtractedTextCache(
            config.paths.runtime_root / "extracted-text-cache"
        )

    def run(
        self,
        max_files: int | None = None,
        *,
        progress: Callable[[IndexProgress], None] | None = None,
        progress_every_files: int = 25,
        progress_every_seconds: float = 10.0,
        prune_missing: bool = False,
    ) -> IndexReport:
        lock_path = self.config.paths.runtime_root / "locks" / "index-build.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError("index_build_already_running") from None
        try:
            return self._run_unlocked(
                max_files=max_files,
                progress=progress,
                progress_every_files=progress_every_files,
                progress_every_seconds=progress_every_seconds,
                prune_missing=prune_missing,
            )
        finally:
            lock.release()

    def _run_unlocked(
        self,
        max_files: int | None = None,
        *,
        progress: Callable[[IndexProgress], None] | None = None,
        progress_every_files: int = 25,
        progress_every_seconds: float = 10.0,
        prune_missing: bool = False,
    ) -> IndexReport:
        if progress_every_files < 1:
            raise ValueError("progress_every_files must be positive")
        if progress_every_seconds <= 0:
            raise ValueError("progress_every_seconds must be positive")

        build_id = str(uuid.uuid4())
        started_at = time.monotonic()
        last_progress_at = started_at
        last_progress_count = 0
        config_version = (
            f"schema={self.config.schema_version};"
            f"chunk={self.config.ingestion.chunk_chars};"
            f"overlap={self.config.ingestion.chunk_overlap_chars}"
        )
        self.manifest.start_build(build_id, self.embedder.model_version, config_version)
        discovered = indexed = skipped = failed = chunk_count = 0
        seen_document_ids: set[str] = set()
        try:
            index_signature = compute_index_signature(
                self.config, self.embedder.model_version
            )
            manifest_signature_chunks = self.manifest.indexed_chunk_count(index_signature)
            qdrant_signature_points = self.vector_store.count_signature(
                self.embedder.model_version,
                index_signature,
            )
            fast_skip_safe = manifest_signature_chunks == qdrant_signature_points
        except Exception:
            # A Qdrant/preflight failure happens before the main loop's finally
            # block. Record the attempt as failed instead of leaving a false
            # `running` build behind until the next process starts.
            self.manifest.finish_build(
                build_id,
                "failed",
                indexed_count=indexed,
                failed_count=failed,
                chunk_count=chunk_count,
            )
            raise

        def emit_progress(phase: str) -> None:
            elapsed = max(0.0, time.monotonic() - started_at)
            event = IndexProgress(
                build_id=build_id,
                phase=phase,
                discovered=discovered,
                indexed=indexed,
                skipped=skipped,
                failed=failed,
                chunks=chunk_count,
                elapsed_seconds=round(elapsed, 3),
                files_per_second=round(discovered / elapsed, 3) if elapsed else 0.0,
            )
            if progress is not None:
                try:
                    progress(event)
                except Exception:
                    # Observability must never change indexing correctness.
                    pass

        emit_progress("started")

        # Keep a terminal status available even for BaseException subclasses such
        # as KeyboardInterrupt. The finally block can then close the build row
        # before FileLock is released while the interrupt itself still propagates.
        status = "failed"
        try:
            files = iter_source_files(
                self.config.paths.source_root,
                self.config.ingestion.supported_extensions,
                max_files=max_files,
            )
            for path in files:
                discovered += 1
                relative = path.relative_to(self.config.paths.source_root).as_posix()
                document_id = stable_document_id(relative)
                seen_document_ids.add(document_id)
                vector_replacement_started = False
                try:
                    stat = path.stat()
                    previous = self.manifest.get_document(document_id)
                    if stat.st_size > self.config.ingestion.max_file_mb * 1024 * 1024:
                        raise ExtractionError("file_too_large")
                    revision = file_sha256(path)
                    if (
                        fast_skip_safe
                        and previous is not None
                        and previous.size == stat.st_size
                        and previous.mtime_ns == stat.st_mtime_ns
                        and previous.status == "indexed"
                        and previous.indexed_revision == revision
                        and previous.revision == revision
                        and previous.index_signature == index_signature
                    ):
                        skipped += 1
                        continue

                    record = DocumentRecord(
                        document_id=document_id,
                        source_path=path,
                        relative_path=relative,
                        extension=path.suffix.lower(),
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        revision=revision,
                        status="extracting",
                        indexed_revision=previous.indexed_revision if previous else None,
                        index_signature=index_signature,
                    )
                    self.manifest.upsert_document(record)
                    cacheable = (
                        self.config.ingestion.persist_extracted_text_cache
                        and is_cacheable_extension(path.suffix)
                    )
                    extracted = None
                    loaded_from_cache = False
                    if cacheable:
                        try:
                            extracted = self.extracted_cache.load(
                                document_id,
                                revision,
                                path.suffix,
                                self.config.ingestion,
                            )
                            loaded_from_cache = extracted is not None
                        except ValueError:
                            extracted = None
                    if extracted is None:
                        extracted = extract_document(path, self.config.ingestion)
                    post_stat = path.stat()
                    if (
                        post_stat.st_size != stat.st_size
                        or post_stat.st_mtime_ns != stat.st_mtime_ns
                        or file_sha256(path) != revision
                    ):
                        raise ExtractionError("source_changed_during_index")
                    if cacheable and not loaded_from_cache:
                        try:
                            self.extracted_cache.store(
                                document_id,
                                revision,
                                path.suffix,
                                self.config.ingestion,
                                extracted,
                            )
                        except (OSError, ValueError):
                            # Index correctness does not depend on this performance cache.
                            pass
                    chunks = chunk_text(
                        extracted.text,
                        document_id=document_id,
                        revision=revision,
                        chunk_chars=self.config.ingestion.chunk_chars,
                        overlap_chars=self.config.ingestion.chunk_overlap_chars,
                        min_chunk_chars=self.config.ingestion.min_chunk_chars,
                        access_group=self.config.retrieval.access_group,
                    )
                    if not chunks:
                        raise ExtractionError("no_chunks")

                    vectors = self.embedder.embed_passages([chunk.text for chunk in chunks])
                    # Delete by opaque document ID, not only by manifest-known chunk IDs.
                    # This also cleans points orphaned by a prior crash after Qdrant upsert.
                    # Set the guard before the request: a timeout may mean that Qdrant
                    # committed the delete even though the client did not receive a reply.
                    vector_replacement_started = True
                    self.vector_store.delete_document(document_id)
                    self.vector_store.upsert(
                        chunks,
                        vectors,
                        build_id=build_id,
                        embedding_version=self.embedder.model_version,
                        index_signature=index_signature,
                    )
                    self.manifest.replace_chunks(document_id, chunks)
                    self.manifest.upsert_document(
                        replace(record, status="indexed", indexed_revision=revision)
                    )
                    indexed += 1
                    chunk_count += len(chunks)
                except ExtractionError as exc:
                    failed += 1
                    if vector_replacement_started:
                        self._cleanup_failed_replacement(document_id)
                    prior = self.manifest.get_document(document_id)
                    if prior is not None:
                        self.manifest.upsert_document(
                            replace(prior, status="failed"),
                            error_code=exc.code,
                        )
                except QdrantTransientWriteError as exc:
                    failed += 1
                    if vector_replacement_started:
                        self._cleanup_failed_replacement(document_id)
                    prior = self.manifest.get_document(document_id)
                    if prior is not None:
                        self.manifest.upsert_document(
                            replace(prior, status="failed"),
                            error_code=exc.code,
                        )
                except Exception:
                    failed += 1
                    if vector_replacement_started:
                        self._cleanup_failed_replacement(document_id)
                    prior = self.manifest.get_document(document_id)
                    if prior is not None:
                        self.manifest.upsert_document(
                            replace(prior, status="failed"),
                            error_code="internal_error",
                        )
                finally:
                    now = time.monotonic()
                    files_since_progress = discovered - last_progress_count
                    seconds_since_progress = now - last_progress_at
                    if (
                        files_since_progress >= progress_every_files
                        or seconds_since_progress >= progress_every_seconds
                    ):
                        emit_progress("running")
                        last_progress_at = now
                        last_progress_count = discovered
            if max_files is None and prune_missing:
                self._remove_missing(seen_document_ids)
            status = "complete" if failed == 0 else "complete_with_errors"
        except Exception:
            status = "failed"
            emit_progress("failed")
            raise
        finally:
            self.manifest.finish_build(
                build_id,
                status,
                indexed_count=indexed,
                failed_count=failed,
                chunk_count=chunk_count,
            )

        emit_progress(status)

        report = IndexReport(
            build_id=build_id,
            discovered=discovered,
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            chunks=chunk_count,
            details={
                "embedding_version": self.embedder.model_version,
                "config_version": config_version,
                "source_content_logged": False,
                "manifest_signature_chunks_at_start": manifest_signature_chunks,
                "qdrant_signature_points_at_start": qdrant_signature_points,
                "fast_skip_safe": fast_skip_safe,
            },
        )
        self._write_safe_report(report)
        return report

    def _cleanup_failed_replacement(self, document_id: str) -> None:
        """Best-effort removal of points from an incomplete document replacement."""
        try:
            self.vector_store.delete_document(document_id)
        except Exception:
            # Preserve the original safe failure code. A later build detects the
            # manifest/Qdrant count mismatch and retries the document.
            pass

    def _remove_missing(self, seen_document_ids: set[str]) -> None:
        for record in self.manifest.all_documents():
            if record.document_id in seen_document_ids or record.status == "deleted":
                continue
            self.vector_store.delete_document(record.document_id)
            self.manifest.replace_chunks(record.document_id, [])
            self.manifest.upsert_document(
                replace(
                    record,
                    status="deleted",
                    indexed_revision=None,
                    index_signature=None,
                ),
                error_code="source_missing",
            )

    def _write_safe_report(self, report: IndexReport) -> None:
        path = self.config.paths.runtime_root / "reports" / f"index-{report.build_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
