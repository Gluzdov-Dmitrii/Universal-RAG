from __future__ import annotations

import json
import uuid
from dataclasses import asdict, replace

from .chunking import chunk_text
from .config import AppConfig
from .embeddings import Embedder
from .extractors import (
    ExtractionError,
    extract_document,
    file_sha256,
    iter_source_files,
    stable_document_id,
)
from .manifest import ManifestStore
from .models import DocumentRecord, IndexReport
from .signatures import compute_index_signature
from .vector_store import QdrantStore


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

    def run(self, max_files: int | None = None) -> IndexReport:
        build_id = str(uuid.uuid4())
        config_version = (
            f"schema={self.config.schema_version};"
            f"chunk={self.config.ingestion.chunk_chars};"
            f"overlap={self.config.ingestion.chunk_overlap_chars}"
        )
        self.manifest.start_build(build_id, self.embedder.model_version, config_version)
        discovered = indexed = skipped = failed = chunk_count = 0
        seen_document_ids: set[str] = set()
        index_signature = compute_index_signature(self.config, self.embedder.model_version)

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
                try:
                    stat = path.stat()
                    previous = self.manifest.get_document(document_id)
                    revision = file_sha256(path)
                    if (
                        previous is not None
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
                    if (
                        previous is not None
                        and previous.indexed_revision == revision
                        and previous.index_signature == index_signature
                    ):
                        self.manifest.upsert_document(
                            replace(record, status="indexed", indexed_revision=revision)
                        )
                        skipped += 1
                        continue

                    extracted = extract_document(path, self.config.ingestion)
                    post_stat = path.stat()
                    if (
                        post_stat.st_size != stat.st_size
                        or post_stat.st_mtime_ns != stat.st_mtime_ns
                        or file_sha256(path) != revision
                    ):
                        raise ExtractionError("source_changed_during_index")
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
                    old_ids = self.manifest.old_chunk_ids(document_id)
                    self.vector_store.delete_points(old_ids)
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
                    prior = self.manifest.get_document(document_id)
                    if prior is not None:
                        self.manifest.upsert_document(
                            replace(prior, status="failed"),
                            error_code=exc.code,
                        )
                except Exception:
                    failed += 1
                    prior = self.manifest.get_document(document_id)
                    if prior is not None:
                        self.manifest.upsert_document(
                            replace(prior, status="failed"),
                            error_code="internal_error",
                        )
            if max_files is None:
                self._remove_missing(seen_document_ids)
            status = "complete" if failed == 0 else "complete_with_errors"
        except Exception:
            status = "failed"
            raise
        finally:
            self.manifest.finish_build(
                build_id,
                status,
                indexed_count=indexed,
                failed_count=failed,
                chunk_count=chunk_count,
            )

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
            },
        )
        self._write_safe_report(report)
        return report

    def _remove_missing(self, seen_document_ids: set[str]) -> None:
        for record in self.manifest.all_documents():
            if record.document_id in seen_document_ids or record.status == "deleted":
                continue
            old_ids = self.manifest.old_chunk_ids(record.document_id)
            self.vector_store.delete_points(old_ids)
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
