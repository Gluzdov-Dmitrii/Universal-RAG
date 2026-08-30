from __future__ import annotations

from .config import AppConfig
from .embeddings import Embedder
from .events import EventCallback, timed_stage
from .extracted_cache import ExtractedTextCache, ExtractedTextCacheBackend
from .extractors import ExtractionError, extract_document, file_sha256
from .manifest import ManifestStore
from .models import RetrievalHit
from .signatures import compute_index_signature
from .vector_store import QdrantStore


class Retriever:
    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        manifest: ManifestStore,
        vector_store: QdrantStore,
        extracted_cache: ExtractedTextCacheBackend | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.manifest = manifest
        self.vector_store = vector_store
        self.extracted_cache = extracted_cache
        if self.extracted_cache is None and config.ingestion.persist_extracted_text_cache:
            self.extracted_cache = ExtractedTextCache(
                config.paths.runtime_root / "extracted-text-cache"
            )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []
        requested_top_k = top_k or self.config.retrieval.top_k
        expected_signature = compute_index_signature(
            self.config, self.embedder.model_version
        )
        with timed_stage(
            on_event,
            "retrieval.query_embedding",
            "Токенизация и embedding запроса",
            {"embedding_dimension": self.embedder.dimension},
        ):
            vector = self.embedder.embed_query(query)
        with timed_stage(
            on_event,
            "retrieval.ann",
            "ANN-поиск в Qdrant",
            {"top_k": requested_top_k},
        ) as details:
            points = self.vector_store.search(
                vector,
                top_k=requested_top_k,
                access_group=self.config.retrieval.access_group,
                embedding_version=self.embedder.model_version,
                index_signature=expected_signature,
                score_threshold=self.config.retrieval.score_threshold,
            )
            details["candidates"] = len(points)
        text_cache: dict[str, str] = {}
        hits: list[RetrievalHit] = []
        with timed_stage(
            on_event,
            "retrieval.source_read",
            "Чтение и проверка найденных исходных документов",
            {"candidate_count": len(points)},
        ) as details:
            source_hashes = 0
            extractions = 0
            cache_hits = 0
            cache_misses = 0
            cache_writes = 0
            cache_write_failures = 0
            cache_bypassed = 0
            for point in points:
                payload = point.payload or {}
                document_id = str(payload.get("document_id", ""))
                document = self.manifest.get_document(document_id)
                if document is None or document.status != "indexed":
                    continue
                if str(payload.get("revision", "")) != document.indexed_revision:
                    continue
                if str(payload.get("embedding_version", "")) != self.embedder.model_version:
                    continue
                if str(payload.get("index_signature", "")) != expected_signature:
                    continue
                try:
                    resolved_path = document.source_path.resolve()
                    if not resolved_path.is_relative_to(self.config.paths.source_root):
                        continue
                    stat = resolved_path.stat()
                    if stat.st_size != document.size or stat.st_mtime_ns != document.mtime_ns:
                        continue
                    if document_id not in text_cache:
                        source_hashes += 1
                        source_revision = file_sha256(resolved_path)
                        verified_stat = resolved_path.stat()
                        if (
                            verified_stat.st_size != stat.st_size
                            or verified_stat.st_mtime_ns != stat.st_mtime_ns
                            or source_revision != document.revision
                        ):
                            continue

                        cacheable = (
                            self.extracted_cache is not None
                            and self.extracted_cache.allows_extension(document.extension)
                        )
                        extracted_document = None
                        if cacheable:
                            try:
                                extracted_document = self.extracted_cache.load(
                                    document_id,
                                    document.revision,
                                    document.extension,
                                    self.config.ingestion,
                                )
                            except ValueError:
                                # A legacy/invalid opaque id is not allowed to become a path.
                                extracted_document = None
                            if extracted_document is None:
                                cache_misses += 1
                            else:
                                cache_hits += 1
                        else:
                            cache_bypassed += 1

                        if extracted_document is None:
                            extractions += 1
                            extracted_document = extract_document(
                                resolved_path, self.config.ingestion
                            )
                            post_stat = resolved_path.stat()
                            source_hashes += 1
                            if (
                                post_stat.st_size != verified_stat.st_size
                                or post_stat.st_mtime_ns != verified_stat.st_mtime_ns
                                or file_sha256(resolved_path) != document.revision
                            ):
                                continue
                            if cacheable:
                                try:
                                    stored = self.extracted_cache.store(
                                        document_id,
                                        document.revision,
                                        document.extension,
                                        self.config.ingestion,
                                        extracted_document,
                                    )
                                    if stored is False:
                                        cache_bypassed += 1
                                    else:
                                        cache_writes += 1
                                except (OSError, ValueError):
                                    # Cache availability must not change retrieval correctness.
                                    cache_write_failures += 1
                        text_cache[document_id] = extracted_document.text
                except (OSError, ExtractionError):
                    continue
                start = int(payload.get("char_start", -1))
                end = int(payload.get("char_end", -1))
                text = text_cache[document_id]
                if start < 0 or end <= start or end > len(text):
                    continue
                hits.append(
                    RetrievalHit(
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        document_id=document_id,
                        revision=str(payload["revision"]),
                        score=float(point.score),
                        start=start,
                        end=end,
                        text=text[start:end],
                        source_name=document.source_path.name,
                        build_id=str(payload.get("build_id", "")),
                        embedding_version=str(payload.get("embedding_version", "")),
                        index_signature=str(payload.get("index_signature", "")),
                    )
                )
            details["documents_read"] = len(text_cache)
            details["accepted_hits"] = len(hits)
            details["source_hashes"] = source_hashes
            details["extractions"] = extractions
            details["extracted_cache_hits"] = cache_hits
            details["extracted_cache_misses"] = cache_misses
            details["extracted_cache_writes"] = cache_writes
            details["extracted_cache_write_failures"] = cache_write_failures
            details["extracted_cache_bypassed"] = cache_bypassed
        return hits
