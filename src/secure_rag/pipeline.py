from __future__ import annotations

from pathlib import Path

from .attachments import attachment_hits, resolve_attachment_path
from .bridge import BridgeManager
from .config import AppConfig
from .embeddings import QUERY_NORMALIZATION_VERSION
from .events import EventCallback, PipelineEvent, emit_event, timed_stage
from .manifest import ManifestStore
from .models import BridgeResult, MarkerState, RetrievalHit
from .providers import (
    LocalCodexProvider,
    OpenAIResponsesProvider,
    StubProvider,
    resolve_provider_name,
)
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
        provider: str = "auto",
        top_k: int | None = None,
        attachment_path: str | Path | None = None,
        on_event: EventCallback | None = None,
    ) -> BridgeResult:
        with timed_stage(
            on_event,
            "request.validate",
            "Проверка запроса и локального файла",
            {
                "question_chars": len(question),
                "attachment_provided": bool(str(attachment_path or "").strip()),
            },
        ) as details:
            if not question.strip():
                raise ValueError("Question is empty")
            resolved_provider = resolve_provider_name(provider)
            details["provider"] = resolved_provider
            attachment = (
                resolve_attachment_path(self.config, attachment_path)
                if attachment_path and str(attachment_path).strip()
                else None
            )
            if attachment is not None:
                details["attachment_extension"] = attachment.suffix.lower()

        with timed_stage(
            on_event,
            "retrieval.total",
            "Поиск релевантного контекста",
        ) as details:
            if on_event is None:
                hits = self.retriever.search(question, top_k=top_k)
            else:
                hits = self.retriever.search(
                    question,
                    top_k=top_k,
                    on_event=on_event,
                )
            details["index_hits"] = len(hits)

        if attachment is not None:
            direct_hits = attachment_hits(
                self.config,
                self.retriever.embedder,
                question,
                attachment,
                top_k=top_k or self.config.retrieval.top_k,
                on_event=on_event,
            )
            hits = self._merge_hits(direct_hits, hits)

        state = MarkerState()
        raw_fields = [question, *[hit.text for hit in hits]]
        with timed_stage(
            on_event,
            "sanitizer.detect",
            "NER и regex: поиск чувствительных сущностей",
            {"fields": len(raw_fields)},
        ) as details:
            detected = [self.gateway.detect_spans(text) for text in raw_fields]
            details["detected_spans"] = sum(len(spans) for spans in detected)

        with timed_stage(
            on_event,
            "sanitizer.mark",
            "Распространение сущностей и установка маркеров",
        ) as details:
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
            details["marker_count"] = len(state.marker_to_value)

        versions = {
            "app": "0.1.0",
            "index_build_ids": ",".join(sorted({hit.build_id for hit in hits if hit.build_id}))
            or self.manifest.latest_build_id(),
            "embedding": self.retriever.embedder.model_version,
            "query_normalization": QUERY_NORMALIZATION_VERSION,
            "qdrant_collection": self.config.qdrant.collection_name,
            "sanitizer": getattr(self.gateway.detector, "name", "unknown"),
        }
        with timed_stage(
            on_event,
            "outbound.prepare",
            "Формирование и проверка маркированного запроса",
            {"context_count": len(contexts)},
        ):
            result = self.bridge.create(
                sanitized_question,
                contexts,
                state,
                versions=versions,
                provider=resolved_provider,
            )
        if resolved_provider != "manual":
            with timed_stage(
                on_event,
                "provider.call",
                "Получение маркированного ответа от provider",
                {"provider": resolved_provider},
            ) as details:
                if resolved_provider == "responses":
                    marked_answer = OpenAIResponsesProvider().answer_payload(
                        result.codex_input.read_text(encoding="utf-8"),
                        result.request_id,
                    )
                elif resolved_provider == "codex-local":
                    marked_answer = LocalCodexProvider(
                        forbidden_roots=(
                            self.config.repo_root,
                            self.config.paths.source_root,
                            self.config.paths.runtime_root,
                        )
                    ).answer_payload(
                        result.codex_input.read_text(encoding="utf-8"),
                        result.request_id,
                    )
                elif resolved_provider == "stub":
                    marked_answer = StubProvider().answer(sanitized_question, contexts)
                else:  # pragma: no cover - guarded by resolve_provider_name
                    raise ValueError("unsupported_automatic_provider")
                # Persist the already-validated marked response before the stage's
                # completion callback. A UI rerun or power loss can then resume the
                # local demarker without sending the provider request again.
                self.bridge.stage_response(result.request_id, marked_answer)
                details["output_chars"] = len(marked_answer)
            with timed_stage(
                on_event,
                "demarker.restore",
                "Проверка маркеров и локальное восстановление ответа",
            ):
                self.bridge.restore_staged(result.request_id)
        else:
            emit_event(
                on_event,
                PipelineEvent(
                    stage="provider.awaiting",
                    label="Маркированный запрос ожидает внешнего ответа",
                    status="info",
                    details={"provider": resolved_provider, "automatic": False},
                ),
            )
        return result

    @staticmethod
    def _merge_hits(
        primary: list[RetrievalHit],
        secondary: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        merged: list[RetrievalHit] = []
        seen: set[str] = set()
        for hit in [*primary, *secondary]:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            merged.append(hit)
        return merged
