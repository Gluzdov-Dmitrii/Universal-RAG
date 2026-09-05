from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from secure_rag.api.chat_state import ChatStateStore, ConversationKey, StateMessage
from secure_rag.composition import create_embedder, create_gateway, create_vector_store
from secure_rag.config import AppConfig, load_config
from secure_rag.domain.models import BridgeResult, DocumentSource, MarkerState
from secure_rag.ingestion.cache import ProcessMemoryExtractedTextCache
from secure_rag.ingestion.manifest import ManifestStore
from secure_rag.orchestration.context import ProcessMemorySpanCache
from secure_rag.orchestration.events import EventCallback, JsonlEventLog, PipelineEvent
from secure_rag.orchestration.pipeline import SecureRagPipeline
from secure_rag.retrieval.service import Retriever

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ID = "universal-rag"
_MODEL_CREATED = int(time.time())
_RAM_CACHE_MAX_ENTRIES = 16
_RAM_CACHE_MAX_TOTAL_CHARS = 20_000_000
_RAM_CACHE_MAX_ENTRY_CHARS = 5_000_000
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SOURCE_SECTION = "\n\nИсточники на сервере:"
_ROLE_LABELS = {
    "system": "Инструкции агента/проекта",
    "user": "Пользователь",
    "assistant": "Ассистент",
    "tool": "Инструмент",
}
_PROGRESS_END_MARKER = "<!-- universal-rag-progress-end -->"
_PIPELINE_ERROR_MARKER = "<!-- universal-rag-error -->"
_LOGGER = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    stream: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class OpenWebUIIdentity:
    user_id: str
    chat_id: str


@dataclass(slots=True)
class RuntimeResources:
    config: AppConfig
    embedder: Any
    gateway: Any
    extracted_text_cache: ProcessMemoryExtractedTextCache
    span_cache: ProcessMemorySpanCache


class RagRuntime:
    """Own long-lived model resources and serialize access to the local pipeline."""

    def __init__(self) -> None:
        self._resources: RuntimeResources | None = None
        self._resource_lock = threading.Lock()
        self._request_lock = threading.Lock()

    def _create_resources(self) -> RuntimeResources:
        default_config = str(_REPO_ROOT / "config" / "app.yaml")
        config = load_config(os.getenv("SECURE_RAG_CONFIG", default_config))
        config.ensure_runtime()
        embedder = create_embedder(config)
        gateway = create_gateway(
            config,
            mode=os.getenv("SECURE_RAG_NER_MODE", "all").strip().lower(),
        )
        probe = MarkerState()
        inflected = gateway.mark_literal("Ерофеева", "PER", probe)
        nominative = gateway.mark_literal("Ерофеев", "PER", probe)
        if inflected != nominative or not probe.marker_to_aliases:
            raise RuntimeError("sanitizer_runtime_version_mismatch")
        extracted_text_cache = ProcessMemoryExtractedTextCache(
            max_entries=_RAM_CACHE_MAX_ENTRIES,
            max_total_chars=_RAM_CACHE_MAX_TOTAL_CHARS,
            max_entry_chars=min(
                _RAM_CACHE_MAX_ENTRY_CHARS,
                config.ingestion.max_extracted_chars,
            ),
        )
        span_cache = ProcessMemorySpanCache()
        return RuntimeResources(config, embedder, gateway, extracted_text_cache, span_cache)

    def resources(self) -> RuntimeResources:
        if self._resources is None:
            with self._resource_lock:
                if self._resources is None:
                    self._resources = self._create_resources()
        return self._resources

    def run(
        self,
        question: str,
        *,
        conversation_context: str = "",
        retrieval_context: str = "",
        on_event: EventCallback | None = None,
    ) -> BridgeResult:
        # The first server profile has one GPU/model set. Serial execution avoids concurrent
        # model loads and makes request-scoped Marker Vault ownership unambiguous.
        with self._request_lock:
            resources = self.resources()
            config = resources.config
            manifest = ManifestStore(config.manifest_path)
            store = create_vector_store(config, resources.embedder.dimension)
            try:
                retriever = Retriever(
                    config,
                    resources.embedder,
                    manifest,
                    store,
                    extracted_cache=resources.extracted_text_cache,
                )
                pipeline = SecureRagPipeline(
                    config,
                    retriever,
                    resources.gateway,
                    manifest,
                    span_cache=resources.span_cache,
                )
                return pipeline.run(
                    question,
                    provider=os.getenv("SECURE_RAG_PROVIDER", "auto").strip().lower(),
                    conversation_context=conversation_context,
                    retrieval_context=retrieval_context,
                    on_event=on_event,
                )
            finally:
                store.close()
                manifest.close()


runtime = RagRuntime()
_state_stores: dict[Path, ChatStateStore] = {}
_state_stores_lock = threading.Lock()
app = FastAPI(
    title="Universal RAG API",
    version="0.3.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _authenticate(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("SECURE_RAG_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="api_key_not_configured")
    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_api_key")


def _decode_jwt_segment(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (ValueError, UnicodeError) as error:
        raise HTTPException(status_code=401, detail="invalid_user_identity") from error


def _verify_identity_jwt(token: str) -> str:
    secret = os.getenv("SECURE_RAG_IDENTITY_JWT_SECRET") or os.getenv(
        "SECURE_RAG_API_KEY", ""
    )
    if not secret:
        raise HTTPException(status_code=503, detail="identity_key_not_configured")
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="invalid_user_identity")
    header_segment, payload_segment, signature_segment = parts
    try:
        header = json.loads(_decode_jwt_segment(header_segment))
        payload = json.loads(_decode_jwt_segment(payload_segment))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise HTTPException(status_code=401, detail="invalid_user_identity") from error
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise HTTPException(status_code=401, detail="invalid_user_identity")
    if not isinstance(payload, dict) or payload.get("iss") != "open-webui":
        raise HTTPException(status_code=401, detail="invalid_user_identity")
    signed = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.digest(secret.encode("utf-8"), signed, "sha256")
    supplied_signature = _decode_jwt_segment(signature_segment)
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise HTTPException(status_code=401, detail="invalid_user_identity")
    now = int(time.time())
    try:
        issued_at = int(payload["iat"])
        expires_at = int(payload["exp"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=401, detail="invalid_user_identity") from error
    if issued_at > now + 60 or expires_at < now - 30 or expires_at <= issued_at:
        raise HTTPException(status_code=401, detail="expired_user_identity")
    user_id = str(payload.get("sub", "")).strip()
    if not _IDENTIFIER_PATTERN.fullmatch(user_id):
        raise HTTPException(status_code=401, detail="invalid_user_identity")
    return user_id


def _openwebui_identity(
    user_jwt: Annotated[
        str | None,
        Header(alias="X-OpenWebUI-User-Jwt"),
    ] = None,
    chat_id: Annotated[
        str | None,
        Header(alias="X-OpenWebUI-Chat-Id"),
    ] = None,
) -> OpenWebUIIdentity:
    if not user_jwt or not chat_id:
        raise HTTPException(status_code=401, detail="openwebui_identity_required")
    normalized_chat_id = chat_id.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized_chat_id):
        raise HTTPException(status_code=401, detail="invalid_chat_identity")
    return OpenWebUIIdentity(
        user_id=_verify_identity_jwt(user_jwt),
        chat_id=normalized_chat_id,
    )


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None or not raw.strip() else int(raw)
    if value < 1:
        raise ValueError(f"{name}_must_be_positive")
    return value


def _chat_state_store() -> ChatStateStore:
    default_config = str(_REPO_ROOT / "config" / "app.yaml")
    config = load_config(os.getenv("SECURE_RAG_CONFIG", default_config))
    configured_path = os.getenv("SECURE_RAG_CHAT_STATE_DB")
    path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else config.chat_state_path
    )
    if path not in _state_stores:
        with _state_stores_lock:
            if path not in _state_stores:
                _state_stores[path] = ChatStateStore(
                    path,
                    max_messages=_positive_env_int(
                        "SECURE_RAG_CHAT_STATE_MAX_MESSAGES", 100
                    ),
                    max_total_chars=_positive_env_int(
                        "SECURE_RAG_CHAT_STATE_MAX_CHARS", 500_000
                    ),
                )
    return _state_stores[path]


def _message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in message.content
            if part.get("type") in {"text", "input_text"}
        ]
        return "\n".join(part for part in text_parts if part)
    return ""


def _latest_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            question = _message_text(message).strip()
            if question:
                return question
    raise HTTPException(status_code=400, detail="user_message_required")


def _state_messages(messages: list[ChatMessage]) -> tuple[StateMessage, ...]:
    return tuple(
        StateMessage(message.role, text)
        for message in messages
        if (text := _message_text(message).strip())
    )


def _history_content(message: StateMessage) -> str:
    content = message.content.strip()
    if message.role != "assistant":
        return content
    if _PIPELINE_ERROR_MARKER in content:
        return ""
    if _PROGRESS_END_MARKER in content:
        content = content.split(_PROGRESS_END_MARKER, 1)[1].strip()
        if content.startswith("Ответ:"):
            content = content.removeprefix("Ответ:").strip()
    return content.split(_SOURCE_SECTION, 1)[0].strip()


def _conversation_context(messages: tuple[StateMessage, ...]) -> str:
    latest_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == "user"),
        None,
    )
    if latest_user_index is None:
        return ""
    candidates = messages[:latest_user_index]
    max_messages = _positive_env_int("SECURE_RAG_CHAT_CONTEXT_MAX_MESSAGES", 20)
    max_chars = _positive_env_int("SECURE_RAG_CHAT_CONTEXT_MAX_CHARS", 24_000)
    blocks: list[str] = []
    used_chars = 0
    for message in reversed(candidates[-max_messages:]):
        content = _history_content(message)
        if not content:
            continue
        label = _ROLE_LABELS[message.role]
        remaining = max_chars - used_chars - len(label) - 2
        if remaining <= 0:
            break
        block = f"{label}:\n{content[-remaining:]}"
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(reversed(blocks))


def _retrieval_context(messages: tuple[StateMessage, ...]) -> str:
    """Keep follow-up retrieval anchored to recent user questions, not UI/system text."""

    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return ""
    max_messages = _positive_env_int("SECURE_RAG_RETRIEVAL_HISTORY_MESSAGES", 3)
    max_chars = _positive_env_int("SECURE_RAG_RETRIEVAL_HISTORY_CHARS", 2_000)
    prior_questions = [
        message.content.strip()
        for message in messages[:latest_user_index]
        if message.role == "user" and message.content.strip()
    ][-max_messages:]
    selected: list[str] = []
    used_chars = 0
    for question in reversed(prior_questions):
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        selected.append(question[-remaining:])
        used_chars += len(selected[-1])
    if not selected:
        return ""
    return "Предыдущие вопросы пользователя:\n" + "\n".join(
        f"- {question}" for question in reversed(selected)
    )


def _openwebui_task(request: ChatCompletionRequest) -> str | None:
    if not request.metadata:
        return None
    raw_task = request.metadata.get("task")
    if not isinstance(raw_task, str):
        return None
    task = raw_task.rsplit(".", 1)[-1].strip().lower()
    return task if re.fullmatch(r"[a-z0-9_]{1,80}", task) else None


def _format_location(kind: str, start: str, end: str | None) -> str:
    labels = {
        "page": "стр.",
        "approx_page": "примерно стр.",
        "slide": "слайд",
        "sheet": "лист",
    }
    label = labels.get(kind, kind)
    value = start if not end or end == start else f"{start}–{end}"
    return f"{label} {value}".strip()


def _format_source(source: DocumentSource) -> str:
    references = ", ".join(source.citation_refs)
    locations = ", ".join(
        _format_location(item.kind, item.start, item.end) for item in source.locations
    )
    details = " · ".join(item for item in (references, source.file_type, locations) if item)
    return f"{details} · {source.path}"


def _plain_text_block(value: str) -> str:
    # Provider output is untrusted. Indented Markdown renders it as text instead of allowing
    # provider-controlled links or images to trigger browser requests.
    return "\n".join(f"    {line}" for line in value.splitlines() or [""])


def _answer_content(result: BridgeResult) -> str:
    answer = _restored_answer(result)
    sections = [_plain_text_block(answer)]
    if result.sources:
        source_text = "\n".join(_format_source(source) for source in result.sources)
        sections.extend(("Источники на сервере:", _plain_text_block(source_text)))
    return "\n\n".join(sections)


def _restored_answer(result: BridgeResult) -> str:
    return (
        result.restored_output.read_text(encoding="utf-8")
        if result.restored_output.exists()
        else ""
    )


def _chunk_payload(
    completion_id: str,
    model: str,
    delta: dict[str, str],
    finish: str | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _stream_chunk(completion_id: str, model: str, content: str) -> str:
    return (
        "data: "
        + json.dumps(
            _chunk_payload(completion_id, model, {"content": content}),
            ensure_ascii=False,
        )
        + "\n\n"
    )


_STAGE_TITLES = {
    "request.validate": "Проверяю запрос и контекст чата",
    "retrieval.total": "Ищу релевантные фрагменты в базе документов",
    "retrieval.iteration": "Уточняю поиск по найденному контексту",
    "retrieval.parent_context": "Собираю полные версии найденных документов",
    "sanitizer.detect": "Проверяю чувствительные данные",
    "sanitizer.mark": "Формирую безопасную версию контекста",
    "outbound.prepare": "Готовлю запрос для модели",
    "outbound.update": "Обновляю контекст после дополнительного поиска",
    "provider.initialize": "Подготавливаю выбранную модель",
    "provider.call": "Формирую ответ",
    "demarker.restore": "Восстанавливаю ответ внутри корпоративного контура",
    "chat.persist": "Сохраняю состояние чата и поиска",
}


def _safe_stage_title(stage: str) -> str:
    return _STAGE_TITLES.get(stage, "Обрабатываю запрос")


def _progress_text(event: PipelineEvent) -> str | None:
    title = _safe_stage_title(event.stage)
    if event.status == "started":
        return f"- ⏳ {title}…\n"
    if event.status == "failed" and event.stage != "pipeline.total":
        return f"- ⚠️ Этап «{title}» завершился ошибкой.\n"
    if event.status == "completed" and event.stage == "retrieval.total":
        count = event.details.get("index_hits")
        if isinstance(count, int):
            return f"- ✓ Поиск завершён: найдено фрагментов — {count}.\n"
    if event.status == "completed" and event.stage == "retrieval.parent_context":
        contexts = event.details.get("contexts")
        whole_documents = event.details.get("whole_documents")
        context_chars = event.details.get("context_chars")
        if all(isinstance(value, int) for value in (contexts, whole_documents, context_chars)):
            return (
                "- ✓ Контекст подготовлен: "
                f"документов — {contexts}, целиком — {whole_documents}, "
                f"символов — {context_chars}.\n"
            )
    if event.status == "completed" and event.stage in {
        "provider.initialize",
        "demarker.restore",
        "chat.persist",
    }:
        return f"- ✓ {title}.\n"
    return None


def _event_callback(
    diagnostic_id: str,
    event_queue: queue.Queue[tuple[str, object]] | None = None,
) -> tuple[EventCallback, dict[str, str]]:
    default_config = str(_REPO_ROOT / "config" / "app.yaml")
    config = load_config(os.getenv("SECURE_RAG_CONFIG", default_config))
    configured_root = os.getenv("SECURE_RAG_PIPELINE_EVENT_DIR")
    event_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else config.diagnostics_path / "pipeline-events"
    )
    try:
        event_log = JsonlEventLog(event_root, diagnostic_id)
    except OSError:
        event_log = None
        _LOGGER.error(
            "pipeline_event_log_init_failed diagnostic_id=%s",
            diagnostic_id,
        )
    progress_state = {"stage": "pipeline.total"}

    def callback(event: PipelineEvent) -> None:
        progress_state["stage"] = event.stage
        if event_queue is not None:
            event_queue.put(("event", event))
        if event_log is not None:
            try:
                event_log(event)
            except OSError:
                # Observability must not turn an otherwise valid answer into a failed request.
                _LOGGER.error(
                    "pipeline_event_log_failed diagnostic_id=%s stage=%s",
                    diagnostic_id,
                    event.stage,
                )

    return callback, progress_state


def _execute_pipeline(
    *,
    question: str,
    conversation_context: str,
    retrieval_context: str,
    model: str,
    key: ConversationKey,
    state_store: ChatStateStore,
    on_event: EventCallback,
) -> str:
    result = runtime.run(
        question,
        conversation_context=conversation_context,
        retrieval_context=retrieval_context,
        on_event=on_event,
    )
    content = _answer_content(result)
    on_event(
        PipelineEvent(
            stage="chat.persist",
            label="Сохранение состояния чата и поиска",
            status="started",
        )
    )
    state_store.append_assistant(key, content)
    state_store.record_retrieval(
        key,
        request_id=result.request_id,
        model_id=model,
        question=question,
        sources=result.sources,
    )
    on_event(
        PipelineEvent(
            stage="chat.persist",
            label="Сохранение состояния чата и поиска",
            status="completed",
        )
    )
    return content


def _record_pipeline_failure(
    *,
    diagnostic_id: str,
    stage: str,
    error: Exception,
    on_event: EventCallback,
) -> None:
    on_event(
        PipelineEvent(
            stage="pipeline.total",
            label="Обработка запроса",
            status="failed",
            details={"error_type": type(error).__name__},
        )
    )
    _LOGGER.error(
        "pipeline_failed diagnostic_id=%s stage=%s error_type=%s",
        diagnostic_id,
        stage,
        type(error).__name__,
    )


def _stream_pipeline_response(
    *,
    completion_id: str,
    diagnostic_id: str,
    model: str,
    question: str,
    conversation_context: str,
    retrieval_context: str,
    key: ConversationKey,
    state_store: ChatStateStore,
):
    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    on_event, progress_state = _event_callback(diagnostic_id, event_queue)

    def worker() -> None:
        try:
            content = _execute_pipeline(
                question=question,
                conversation_context=conversation_context,
                retrieval_context=retrieval_context,
                model=model,
                key=key,
                state_store=state_store,
                on_event=on_event,
            )
            event_queue.put(("result", content))
        except Exception as error:
            failed_stage = progress_state["stage"]
            _record_pipeline_failure(
                diagnostic_id=diagnostic_id,
                stage=failed_stage,
                error=error,
                on_event=on_event,
            )
            event_queue.put(("error", failed_stage))

    threading.Thread(
        target=worker,
        name=f"rag-{diagnostic_id[:8]}",
        daemon=True,
    ).start()
    yield f"data: {json.dumps(_chunk_payload(completion_id, model, {'role': 'assistant'}))}\n\n"
    yield _stream_chunk(completion_id, model, "Ход выполнения:\n\n")
    while True:
        try:
            kind, payload = event_queue.get(timeout=10)
        except queue.Empty:
            yield ": keep-alive\n\n"
            continue
        if kind == "event":
            assert isinstance(payload, PipelineEvent)
            if progress := _progress_text(payload):
                yield _stream_chunk(completion_id, model, progress)
            continue
        if kind == "result":
            assert isinstance(payload, str)
            separator = f"\n{_PROGRESS_END_MARKER}\n\nОтвет:\n\n"
            yield _stream_chunk(completion_id, model, separator + payload)
            break
        assert kind == "error"
        stage = _safe_stage_title(str(payload))
        error_text = (
            f"\n{_PROGRESS_END_MARKER}\n{_PIPELINE_ERROR_MARKER}\n\n"
            "Не удалось завершить обработку запроса.\n\n"
            f"Этап: {stage}.\n"
            f"Код диагностики: `{diagnostic_id}`."
        )
        yield _stream_chunk(completion_id, model, error_text)
        break
    yield f"data: {json.dumps(_chunk_payload(completion_id, model, {}, 'stop'))}\n\n"
    yield "data: [DONE]\n\n"


def _completion_response(completion_id: str, model: str, content: str) -> JSONResponse:
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def _background_task_response(
    request: ChatCompletionRequest,
    task: str,
    completion_id: str,
    diagnostic_id: str,
) -> JSONResponse:
    # Open WebUI sends follow-up generation through the selected chat model after every
    # answer. It is UI housekeeping, not a new user turn, and must never enter RAG state.
    if task == "follow_up_generation":
        return _completion_response(completion_id, request.model, '{"follow_ups":[]}')

    question = _latest_user_question(request.messages)
    on_event, progress_state = _event_callback(diagnostic_id)
    try:
        result = runtime.run(question, on_event=on_event)
        content = _restored_answer(result).strip()
    except Exception as error:
        failed_stage = progress_state["stage"]
        _record_pipeline_failure(
            diagnostic_id=diagnostic_id,
            stage=failed_stage,
            error=error,
            on_event=on_event,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "background_task_failed",
                "task": task,
                "diagnostic_id": diagnostic_id,
            },
        ) from None
    return _completion_response(completion_id, request.model, content)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "model": _MODEL_ID}


@app.get("/v1/models", dependencies=[Depends(_authenticate)])
def models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL_ID,
                "object": "model",
                "created": _MODEL_CREATED,
                "owned_by": "universal-rag",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_authenticate)])
def chat_completions(
    request: ChatCompletionRequest,
    identity: Annotated[OpenWebUIIdentity, Depends(_openwebui_identity)],
):
    if request.model != _MODEL_ID:
        raise HTTPException(status_code=404, detail="model_not_found")
    diagnostic_id = str(uuid.uuid4())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    if task := _openwebui_task(request):
        return _background_task_response(request, task, completion_id, diagnostic_id)

    question = _latest_user_question(request.messages)
    key = ConversationKey(identity.user_id, identity.chat_id)
    try:
        state_store = _chat_state_store()
        history = state_store.sync_messages(key, _state_messages(request.messages))
        conversation_context = _conversation_context(history)
        retrieval_context = _retrieval_context(history)
    except HTTPException:
        raise
    except Exception as error:
        _LOGGER.error(
            "chat_state_failed diagnostic_id=%s error_type=%s",
            diagnostic_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "chat_state_failed", "diagnostic_id": diagnostic_id},
        ) from None

    if request.stream:
        return StreamingResponse(
            _stream_pipeline_response(
                completion_id=completion_id,
                diagnostic_id=diagnostic_id,
                model=request.model,
                question=question,
                conversation_context=conversation_context,
                retrieval_context=retrieval_context,
                key=key,
                state_store=state_store,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    on_event, progress_state = _event_callback(diagnostic_id)
    try:
        content = _execute_pipeline(
            question=question,
            conversation_context=conversation_context,
            retrieval_context=retrieval_context,
            model=request.model,
            key=key,
            state_store=state_store,
            on_event=on_event,
        )
    except Exception as error:
        failed_stage = progress_state["stage"]
        _record_pipeline_failure(
            diagnostic_id=diagnostic_id,
            stage=failed_stage,
            error=error,
            on_event=on_event,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_failed",
                "stage": _safe_stage_title(failed_stage),
                "diagnostic_id": diagnostic_id,
            },
        ) from None

    return _completion_response(completion_id, request.model, content)
