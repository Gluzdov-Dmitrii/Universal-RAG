from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from .models import ChunkRecord


class QdrantStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        dimension: int,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
            )
            return
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams) or vectors.size != self.dimension:
            raise ValueError("Existing Qdrant collection has a different vector dimension")

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def delete_points(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
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
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        access_group: str,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="access_group",
                    match=models.MatchValue(value=access_group),
                ),
                models.FieldCondition(key="goz", match=models.MatchValue(value=False)),
                models.FieldCondition(key="is_final", match=models.MatchValue(value=True)),
            ]
        )
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return list(result.points)

    def count(self) -> int:
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )
