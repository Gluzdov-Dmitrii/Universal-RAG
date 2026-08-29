from __future__ import annotations

from .config import AppConfig
from .embeddings import Embedder
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
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.manifest = manifest
        self.vector_store = vector_store

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        if not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        points = self.vector_store.search(
            vector,
            top_k=top_k or self.config.retrieval.top_k,
            access_group=self.config.retrieval.access_group,
            score_threshold=self.config.retrieval.score_threshold,
        )
        text_cache: dict[str, str] = {}
        expected_signature = compute_index_signature(
            self.config, self.embedder.model_version
        )
        hits: list[RetrievalHit] = []
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
                    extracted = extract_document(resolved_path, self.config.ingestion).text
                    post_stat = resolved_path.stat()
                    if (
                        post_stat.st_size != stat.st_size
                        or post_stat.st_mtime_ns != stat.st_mtime_ns
                        or file_sha256(resolved_path) != document.revision
                    ):
                        continue
                    text_cache[document_id] = extracted
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
        return hits
