from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .. import __version__
from ..config import AppConfig
from ..domain.models import BridgeResult, MarkerState, RetrievalHit
from ..ingestion.manifest import ManifestStore
from ..llm.adapters import StubProvider
from ..llm.bridge import BridgeManager
from ..llm.contracts import Provider
from ..llm.control import RetrievalRequest, parse_retrieval_request
from ..llm.factory import ProviderFactory, resolve_provider_name
from ..retrieval.attachments import attachment_hits, resolve_attachment_path
from ..retrieval.embeddings import QUERY_NORMALIZATION_VERSION
from ..retrieval.service import Retriever
from ..sanitization.core import PrivacyGateway
from .context import ContextAssembler
from .events import EventCallback, PipelineEvent, emit_event, timed_stage
from .parents import ParentContextBuilder

_INSUFFICIENT_CONTEXT = (
    "Недостаточно данных в доступных локальных документах после исчерпания "
    "итераций поиска. Уточните вопрос или добавьте документы.\n"
)


class SecureRagPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        gateway: PrivacyGateway,
        manifest: ManifestStore,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.gateway = gateway
        self.context_assembler = ContextAssembler(gateway)
        self.manifest = manifest
        self.bridge = BridgeManager(config)
        self.provider_factory = provider_factory or ProviderFactory(config)

    def run(
        self,
        question: str,
        provider: str = "auto",
        top_k: int | None = None,
        attachment_path: str | Path | None = None,
        on_event: EventCallback | None = None,
        conversation_context: str = "",
    ) -> BridgeResult:
        normalized_context = conversation_context.strip()
        retrieval_question = question
        provider_question = question
        if normalized_context:
            retrieval_question = (
                f"{question}\n\nПредыдущий контекст диалога:\n{normalized_context}"
            )
            provider_question = (
                "Предыдущий контекст диалога:\n"
                f"{normalized_context}\n\nТекущий вопрос пользователя:\n{question}"
            )
        with timed_stage(
            on_event,
            "request.validate",
            "Проверка запроса и локального файла",
            {
                "question_chars": len(question),
                "conversation_context_chars": len(normalized_context),
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
            hits = self.retriever.search(
                retrieval_question,
                top_k=top_k,
                on_event=on_event,
            )
            details["index_hits"] = len(hits)

        if attachment is not None:
            direct_hits = attachment_hits(
                self.config,
                self.retriever.embedder,
                retrieval_question,
                attachment,
                top_k=top_k or self.config.retrieval.top_k,
                on_event=on_event,
            )
            hits = self._merge_hits(direct_hits, hits)
        hits = hits[: self.config.retrieval.max_contexts]
        parent_builder = ParentContextBuilder(self.config)
        context_hits = parent_builder.build(hits, on_event=on_event)

        state = MarkerState()
        sanitized_question, contexts = self.context_assembler.sanitize(
            provider_question,
            context_hits,
            state,
            on_event,
        )
        sources = self.context_assembler.sources(context_hits)
        iterative_enabled = self.config.retrieval.iterative_enabled and resolved_provider in {
            "responses",
            "codex-local",
        }
        max_iterations = self.config.retrieval.max_iterations if iterative_enabled else 1
        versions = {
            "app": __version__,
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
                sources=sources,
                iteration=1,
                max_iterations=max_iterations,
                iterative_enabled=iterative_enabled,
            )

        if resolved_provider == "manual":
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

        if resolved_provider == "stub":
            marked_answer = StubProvider().answer(sanitized_question, contexts)
            self._persist_and_restore(result, marked_answer, on_event, iteration=1)
            return result

        automatic_provider = self._create_provider(resolved_provider)
        iteration = 1
        while True:
            final_answer: str | None = None
            with timed_stage(
                on_event,
                "provider.call",
                "Оценка контекста и получение маркированного ответа",
                {"provider": resolved_provider, "iteration": iteration},
            ) as details:
                marked_answer = automatic_provider.answer_payload(
                    result.codex_input.read_text(encoding="utf-8"),
                    result.request_id,
                )
                PrivacyGateway.validate_outbound(marked_answer, state)
                control = parse_retrieval_request(
                    marked_answer,
                    max_queries=self.config.retrieval.max_queries_per_iteration,
                )
                details["output_chars"] = len(marked_answer)
                details["retrieval_requested"] = control is not None
                if control is None:
                    self.bridge.stage_response(result.request_id, marked_answer)
                    final_answer = marked_answer
                elif iteration >= max_iterations:
                    self.bridge.stage_response(result.request_id, _INSUFFICIENT_CONTEXT)
                    final_answer = _INSUFFICIENT_CONTEXT

            if final_answer is not None:
                with timed_stage(
                    on_event,
                    "demarker.restore",
                    "Проверка маркеров и локальное восстановление ответа",
                ):
                    self.bridge.restore_staged(result.request_id)
                return replace(
                    result,
                    retrieved_count=len(contexts),
                    marker_count=len(state.marker_to_value),
                    sources=sources,
                    iterations=iteration,
                )
            assert control is not None

            new_hits = self._execute_retrieval_request(
                retrieval_question,
                context_hits,
                control,
                state,
                top_k=top_k,
                on_event=on_event,
            )
            previous_count = len(hits)
            hits = self._merge_hits(hits, new_hits)[: self.config.retrieval.max_contexts]
            iteration = (
                iteration + 1 if len(hits) > previous_count else max_iterations
            )
            context_hits = parent_builder.build(hits, on_event=on_event)
            sanitized_question, contexts = self.context_assembler.sanitize(
                provider_question,
                context_hits,
                state,
                on_event,
            )
            sources = self.context_assembler.sources(context_hits)
            with timed_stage(
                on_event,
                "outbound.update",
                "Обновление маркированного контекста",
                {"context_count": len(contexts), "iteration": iteration},
            ):
                self.bridge.update_prepared(
                    result,
                    sanitized_question,
                    contexts,
                    state,
                    sources,
                    iteration=iteration,
                    max_iterations=max_iterations,
                )

    def _execute_retrieval_request(
        self,
        original_question: str,
        current_context_hits: list[RetrievalHit],
        request: RetrievalRequest,
        state: MarkerState,
        *,
        top_k: int | None,
        on_event: EventCallback | None,
    ) -> list[RetrievalHit]:
        with timed_stage(
            on_event,
            "retrieval.iteration",
            "Уточняющий multi-query retrieval",
            {
                "query_count": len(request.queries),
                "expand_count": len(request.expand_citations),
            },
        ) as details:
            discovered: list[RetrievalHit] = []
            accepted_queries = 0
            for sanitized_query in request.queries:
                raw_query = PrivacyGateway.restore(
                    sanitized_query,
                    state,
                    fail_on_unknown=True,
                )
                similarity = self.retriever.query_similarity(original_question, raw_query)
                if similarity < self.config.retrieval.rewrite_min_similarity:
                    continue
                accepted_queries += 1
                discovered.extend(
                    self.retriever.search(raw_query, top_k=top_k, on_event=on_event)
                )
            citation_hits = []
            for citation in request.expand_citations:
                index = int(citation[1:]) - 1
                if (
                    0 <= index < len(current_context_hits)
                    and current_context_hits[index].context_scope != "whole_document"
                ):
                    citation_hits.append(current_context_hits[index])
            if citation_hits:
                discovered.extend(
                    self.retriever.expand_adjacent(
                        citation_hits,
                        self.config.retrieval.adjacent_chunk_radius,
                        on_event=on_event,
                    )
                )
            details["accepted_queries"] = accepted_queries
            details["accepted_hits"] = len(discovered)
        return discovered

    def _persist_and_restore(
        self,
        result: BridgeResult,
        marked_answer: str,
        on_event: EventCallback | None,
        *,
        iteration: int,
    ) -> None:
        with timed_stage(
            on_event,
            "provider.call",
            "Получение маркированного ответа от provider",
            {
                "provider": "stub",
                "iteration": iteration,
                "output_chars": len(marked_answer),
            },
        ):
            self.bridge.stage_response(result.request_id, marked_answer)
        with timed_stage(
            on_event,
            "demarker.restore",
            "Проверка маркеров и локальное восстановление ответа",
        ):
            self.bridge.restore_staged(result.request_id)

    def _create_provider(self, name: str) -> Provider:
        return self.provider_factory.create(name)

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
