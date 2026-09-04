from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from secure_rag.composition import create_embedder, create_gateway, create_vector_store
from secure_rag.config import AppConfig, load_config
from secure_rag.domain.models import BridgeResult, DocumentSource, MarkerState
from secure_rag.ingestion.cache import ProcessMemoryExtractedTextCache
from secure_rag.ingestion.manifest import ManifestStore
from secure_rag.orchestration.pipeline import SecureRagPipeline
from secure_rag.retrieval.service import Retriever

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ID = "universal-rag"
_MODEL_CREATED = int(time.time())
_RAM_CACHE_MAX_ENTRIES = 16
_RAM_CACHE_MAX_TOTAL_CHARS = 20_000_000
_RAM_CACHE_MAX_ENTRY_CHARS = 5_000_000


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


@dataclass(slots=True)
class RuntimeResources:
    config: AppConfig
    embedder: Any
    gateway: Any
    extracted_text_cache: ProcessMemoryExtractedTextCache


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
        return RuntimeResources(config, embedder, gateway, extracted_text_cache)

    def resources(self) -> RuntimeResources:
        if self._resources is None:
            with self._resource_lock:
                if self._resources is None:
                    self._resources = self._create_resources()
        return self._resources

    def run(self, question: str) -> BridgeResult:
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
                )
                return pipeline.run(
                    question,
                    provider=os.getenv("SECURE_RAG_PROVIDER", "auto").strip().lower(),
                )
            finally:
                store.close()
                manifest.close()


runtime = RagRuntime()
app = FastAPI(
    title="Universal RAG API",
    version="0.2.0",
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
    answer = (
        result.restored_output.read_text(encoding="utf-8")
        if result.restored_output.exists()
        else ""
    )
    sections = [_plain_text_block(answer)]
    if result.sources:
        source_text = "\n".join(_format_source(source) for source in result.sources)
        sections.extend(("Источники на сервере:", _plain_text_block(source_text)))
    return "\n\n".join(sections)


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


def _stream_response(completion_id: str, model: str, content: str):
    yield f"data: {json.dumps(_chunk_payload(completion_id, model, {'role': 'assistant'}))}\n\n"
    yield (
        "data: "
        + json.dumps(
            _chunk_payload(completion_id, model, {"content": content}),
            ensure_ascii=False,
        )
        + "\n\n"
    )
    yield f"data: {json.dumps(_chunk_payload(completion_id, model, {}, 'stop'))}\n\n"
    yield "data: [DONE]\n\n"


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
def chat_completions(request: ChatCompletionRequest):
    if request.model != _MODEL_ID:
        raise HTTPException(status_code=404, detail="model_not_found")
    question = _latest_user_question(request.messages)
    try:
        result = runtime.run(question)
        content = _answer_content(result)
    except HTTPException:
        raise
    except Exception:
        # Do not return source paths, document content, raw values, or tracebacks to clients.
        raise HTTPException(status_code=500, detail="pipeline_failed") from None

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    if request.stream:
        return StreamingResponse(
            _stream_response(completion_id, request.model, content),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
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
