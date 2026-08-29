from __future__ import annotations

import os
from dataclasses import replace

from secure_rag.config import load_config
from secure_rag.embeddings import HashingEmbedder
from secure_rag.indexer import Indexer
from secure_rag.manifest import ManifestStore
from secure_rag.retrieval import Retriever
from secure_rag.vector_store import QdrantStore


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
        removed = Indexer(changed_config, embedder, manifest, store).run()
        assert removed.failed == 0
        assert store.count() == 0
