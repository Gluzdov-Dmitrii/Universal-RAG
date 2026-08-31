from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import httpx
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .models import ChunkRecord

UPSERT_BATCH_SIZE = 512
_TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_TIMEOUT_HTTP_STATUSES = {408, 504}
_T = TypeVar("_T")


class QdrantTransientWriteError(RuntimeError):
    """Safe terminal error after retrying a transient Qdrant write."""

    def __init__(self, code: str) -> None:
        if code not in {"qdrant_timeout", "qdrant_transport"}:
            raise ValueError("Unsupported Qdrant transient error code")
        self.code = code
        super().__init__(code)


def _transient_error_code(error: Exception) -> str | None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (httpx.TimeoutException, TimeoutError)):
            return "qdrant_timeout"
        if isinstance(current, (httpx.TransportError, ConnectionError)):
            return "qdrant_transport"
        if isinstance(current, UnexpectedResponse):
            status = current.status_code
            if status in _TIMEOUT_HTTP_STATUSES:
                return "qdrant_timeout"
            if status in _TRANSIENT_HTTP_STATUSES:
                return "qdrant_transport"
        if isinstance(current, ResponseHandlingException):
            pending.append(current.source)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


class QdrantStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        dimension: int,
        url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        write_max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= write_max_attempts <= 10:
            raise ValueError("write_max_attempts must be between 1 and 10")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        self.collection_name = collection_name
        self.dimension = dimension
        self.server_mode = bool(url)
        self.timeout_seconds = timeout_seconds
        self.write_max_attempts = write_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout_seconds,
            )
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        self._ensure_collection()

    def _write_with_retry(self, operation: Callable[[], _T]) -> _T:
        for attempt in range(self.write_max_attempts):
            try:
                return operation()
            except Exception as exc:
                code = _transient_error_code(exc)
                if code is None:
                    raise
                if attempt + 1 >= self.write_max_attempts:
                    raise QdrantTransientWriteError(code) from None
                delay = self.retry_backoff_seconds * (2**attempt)
                if delay:
                    time.sleep(delay)
        raise AssertionError("unreachable")

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
        else:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or vectors.size != self.dimension:
                raise ValueError("Existing Qdrant collection has a different vector dimension")
        if self.server_mode:
            self.client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
            for field_name, field_schema in (
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("access_group", models.PayloadSchemaType.KEYWORD),
                ("goz", models.PayloadSchemaType.BOOL),
                ("is_final", models.PayloadSchemaType.BOOL),
                ("embedding_version", models.PayloadSchemaType.KEYWORD),
                ("index_signature", models.PayloadSchemaType.KEYWORD),
            ):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def delete_points(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        selector = models.PointIdsList(points=list(point_ids))
        self._write_with_retry(
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
                wait=True,
                timeout=self.timeout_seconds,
            )
        )

    def delete_document(self, document_id: str) -> None:
        if not document_id:
            raise ValueError("document_id must not be empty")
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        )
        self._write_with_retry(
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
                wait=True,
                timeout=self.timeout_seconds,
            )
        )

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: np.ndarray,
        build_id: str,
        embedding_version: str,
        index_signature: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors have different lengths")
        for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch_chunks = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
            batch_vectors = vectors[batch_start : batch_start + UPSERT_BATCH_SIZE]
            points = [
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector.tolist(),
                    payload={
                        "document_id": chunk.document_id,
                        "revision": chunk.revision,
                        "chunk_id": chunk.chunk_id,
                        "ordinal": chunk.ordinal,
                        "char_start": chunk.start,
                        "char_end": chunk.end,
                        "access_group": chunk.access_group,
                        "goz": chunk.goz,
                        "is_final": chunk.is_final,
                        "build_id": build_id,
                        "embedding_version": embedding_version,
                        "index_signature": index_signature,
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self._write_with_retry(
                lambda batch_points=points: self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                    wait=True,
                    timeout=self.timeout_seconds,
                )
            )

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        access_group: str,
        embedding_version: str,
        index_signature: str,
        score_threshold: float | None = None,
        hnsw_ef: int | None = None,
        exact_search: bool = False,
    ) -> list[models.ScoredPoint]:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="access_group",
                    match=models.MatchValue(value=access_group),
                ),
                models.FieldCondition(key="goz", match=models.MatchValue(value=False)),
                models.FieldCondition(key="is_final", match=models.MatchValue(value=True)),
                models.FieldCondition(
                    key="embedding_version",
                    match=models.MatchValue(value=embedding_version),
                ),
                models.FieldCondition(
                    key="index_signature",
                    match=models.MatchValue(value=index_signature),
                ),
            ]
        )
        search_params = (
            models.SearchParams(hnsw_ef=hnsw_ef, exact=exact_search)
            if self.server_mode
            else None
        )
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        )
        return list(result.points)

    def count(self) -> int:
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def count_signature(self, embedding_version: str, index_signature: str) -> int:
        signature_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="embedding_version",
                    match=models.MatchValue(value=embedding_version),
                ),
                models.FieldCondition(
                    key="index_signature",
                    match=models.MatchValue(value=index_signature),
                ),
            ]
        )
        return int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=signature_filter,
                exact=True,
            ).count
        )
