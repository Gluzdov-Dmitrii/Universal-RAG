from __future__ import annotations

import numpy as np

from ..config import AppConfig
from ..domain.models import RetrievalHit
from ..ingestion.cache import ExtractedTextCache, ExtractedTextCacheBackend
from ..ingestion.chunking import source_location_for_span
from ..ingestion.extractors import (
    ExtractedDocument,
    ExtractionError,
    extract_document,
    file_sha256,
)
from ..ingestion.manifest import ManifestStore
from ..ingestion.signatures import compute_index_signature
from ..orchestration.events import EventCallback, timed_stage
from .embeddings import Embedder, normalize_retrieval_query
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
            self.extracted_cache = ExtractedTextCache(config.extracted_text_cache_path)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []
        normalized_query = normalize_retrieval_query(query)
        if not normalized_query:
            return []
        requested_top_k = top_k or self.config.retrieval.top_k
        expected_signature = compute_index_signature(
            self.config, self.embedder.model_version
        )
        with timed_stage(
            on_event,
            "retrieval.query_embedding",
            "Токенизация и embedding запроса",
            {
                "embedding_dimension": self.embedder.dimension,
                "query_normalized": normalized_query != query,
            },
        ):
            vector = self.embedder.embed_query(normalized_query)
        with timed_stage(
            on_event,
            "retrieval.ann",
            "ANN-поиск в Qdrant",
            {
                "top_k": requested_top_k,
                "hnsw_ef": self.config.retrieval.hnsw_ef,
                "exact_search": self.config.retrieval.exact_search,
            },
        ) as details:
            points = self.vector_store.search(
                vector,
                top_k=requested_top_k,
                access_group=self.config.retrieval.access_group,
                embedding_version=self.embedder.model_version,
                index_signature=expected_signature,
                score_threshold=self.config.retrieval.score_threshold,
                hnsw_ef=self.config.retrieval.hnsw_ef,
                exact_search=self.config.retrieval.exact_search,
            )
            details["candidates"] = len(points)
        text_cache: dict[str, ExtractedDocument] = {}
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
                        text_cache[document_id] = extracted_document
                except (OSError, ExtractionError):
                    continue
                start = int(payload.get("char_start", -1))
                end = int(payload.get("char_end", -1))
                extracted_document = text_cache[document_id]
                text = extracted_document.text
                if start < 0 or end <= start or end > len(text):
                    continue
                location_kind, location_start, location_end = source_location_for_span(
                    extracted_document.locations,
                    start,
                    end,
                )
                if not location_kind:
                    location_kind = str(payload.get("location_kind", ""))
                    location_start = (
                        str(payload["location_start"])
                        if payload.get("location_start") is not None
                        else None
                    )
                    location_end = (
                        str(payload["location_end"])
                        if payload.get("location_end") is not None
                        else None
                    )
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
                        source_path=resolved_path,
                        source_type=document.extension.lower().lstrip("."),
                        ordinal=int(payload.get("ordinal", -1)),
                        build_id=str(payload.get("build_id", "")),
                        embedding_version=str(payload.get("embedding_version", "")),
                        index_signature=str(payload.get("index_signature", "")),
                        location_kind=location_kind,
                        location_start=location_start,
                        location_end=location_end,
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

    def query_similarity(self, original_query: str, rewritten_query: str) -> float:
        """Keep LLM-generated retrieval queries anchored to the user's intent."""

        normalized_original = normalize_retrieval_query(original_query)
        normalized_rewritten = normalize_retrieval_query(rewritten_query)
        if not normalized_original or not normalized_rewritten:
            return 0.0
        original = self.embedder.embed_query(normalized_original)
        rewritten = self.embedder.embed_query(normalized_rewritten)
        denominator = float(np.linalg.norm(original) * np.linalg.norm(rewritten))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(original, rewritten) / denominator)

    def expand_adjacent(
        self,
        hits: list[RetrievalHit],
        radius: int,
        *,
        on_event: EventCallback | None = None,
    ) -> list[RetrievalHit]:
        """Rehydrate neighboring chunks for citations already authorized by retrieval."""

        if radius <= 0:
            return []
        expanded: list[RetrievalHit] = []
        seen: set[str] = set()
        with timed_stage(
            on_event,
            "retrieval.expand",
            "Чтение соседних фрагментов выбранных источников",
            {"candidate_count": len(hits), "adjacent_radius": radius},
        ) as details:
            for hit in hits:
                if hit.ordinal < 0 or hit.chunk_id in seen:
                    continue
                document = self.manifest.get_document(hit.document_id)
                if (
                    document is None
                    or document.status != "indexed"
                    or document.indexed_revision != hit.revision
                    or document.index_signature != hit.index_signature
                ):
                    continue
                try:
                    path = document.source_path.resolve(strict=True)
                    if not path.is_relative_to(self.config.paths.source_root):
                        continue
                    before = path.stat()
                    if (
                        before.st_size != document.size
                        or before.st_mtime_ns != document.mtime_ns
                    ):
                        continue
                    if file_sha256(path) != document.revision:
                        continue
                    extracted = extract_document(path, self.config.ingestion)
                    after = path.stat()
                    if (
                        after.st_size != before.st_size
                        or after.st_mtime_ns != before.st_mtime_ns
                        or file_sha256(path) != document.revision
                    ):
                        continue
                except (OSError, ExtractionError):
                    continue
                locations = self.manifest.adjacent_chunks(
                    hit.document_id,
                    hit.revision,
                    hit.ordinal,
                    radius,
                )
                for location in locations:
                    if (
                        location.chunk_id == hit.chunk_id
                        or location.chunk_id in seen
                        or location.start < 0
                        or location.end <= location.start
                        or location.end > len(extracted.text)
                    ):
                        continue
                    seen.add(location.chunk_id)
                    location_kind, location_start, location_end = (
                        source_location_for_span(
                            extracted.locations,
                            location.start,
                            location.end,
                        )
                    )
                    if not location_kind:
                        location_kind = location.location_kind
                        location_start = location.location_start
                        location_end = location.location_end
                    expanded.append(
                        RetrievalHit(
                            chunk_id=location.chunk_id,
                            document_id=hit.document_id,
                            revision=hit.revision,
                            score=hit.score,
                            start=location.start,
                            end=location.end,
                            text=extracted.text[location.start : location.end],
                            source_name=path.name,
                            source_path=path,
                            source_type=document.extension.lower().lstrip("."),
                            ordinal=location.ordinal,
                            build_id=hit.build_id,
                            embedding_version=hit.embedding_version,
                            index_signature=hit.index_signature,
                            location_kind=location_kind,
                            location_start=location_start,
                            location_end=location_end,
                        )
                    )
            details["accepted_hits"] = len(expanded)
        return expanded
