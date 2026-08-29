from __future__ import annotations

from .bridge import BridgeManager
from .config import AppConfig
from .manifest import ManifestStore
from .models import BridgeResult, MarkerState
from .providers import StubProvider
from .retrieval import Retriever
from .sanitization.core import PrivacyGateway, merge_spans


class SecureRagPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        gateway: PrivacyGateway,
        manifest: ManifestStore,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.gateway = gateway
        self.manifest = manifest
        self.bridge = BridgeManager(config)

    def run(
        self,
        question: str,
        provider: str = "manual",
        top_k: int | None = None,
    ) -> BridgeResult:
        if not question.strip():
            raise ValueError("Question is empty")
        if provider not in {"manual", "stub"}:
            raise ValueError("Unsupported provider")

        hits = self.retriever.search(question, top_k=top_k)
        state = MarkerState()
        raw_fields = [question, *[hit.text for hit in hits]]
        detected = [self.gateway.detect_spans(text) for text in raw_fields]
        known: list[tuple[str, str, int]] = []
        for text, spans in zip(raw_fields, detected, strict=True):
            for span in spans:
                known.append((span.label, text[span.start : span.end], span.priority))
        completed_spans = [
            merge_spans([*spans, *self.gateway.propagate_known(text, known)])
            for text, spans in zip(raw_fields, detected, strict=True)
        ]
        sanitized_question = self.gateway.sanitize_with_spans(
            "question", question, state, completed_spans[0]
        ).text
        contexts: list[dict[str, str | float]] = []
        for index, hit in enumerate(hits, start=1):
            sanitized_text = self.gateway.sanitize_with_spans(
                f"chunk:{hit.chunk_id}",
                hit.text,
                state,
                completed_spans[index],
            ).text
            source_ref = self.gateway.sanitize_filename(hit.source_name, state)
            contexts.append(
                {
                    "document_id": hit.document_id,
                    "chunk_id": hit.chunk_id,
                    "citation_ref": f"R{index:03d}",
                    "score": hit.score,
                    "source_ref": source_ref,
                    "text": sanitized_text,
                }
            )

        versions = {
            "app": "0.1.0",
            "index_build_ids": ",".join(sorted({hit.build_id for hit in hits if hit.build_id}))
            or self.manifest.latest_build_id(),
            "embedding": self.retriever.embedder.model_version,
            "qdrant_collection": self.config.qdrant.collection_name,
            "sanitizer": getattr(self.gateway.detector, "name", "unknown"),
        }
        result = self.bridge.create(
            sanitized_question,
            contexts,
            state,
            versions=versions,
            provider=provider,
        )
        if provider == "stub":
            marked_answer = StubProvider().answer(sanitized_question, contexts)
            PrivacyGateway.validate_outbound(marked_answer, state)
            self.bridge.restore_text(result.request_id, marked_answer)
        return result
