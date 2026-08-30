from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from time import perf_counter

import streamlit as st

from secure_rag.config import AppConfig, load_config
from secure_rag.events import EventCallback, JsonlEventLog, PipelineEvent, timed_stage
from secure_rag.extracted_cache import ProcessMemoryExtractedTextCache
from secure_rag.factory import create_embedder, create_gateway, create_vector_store
from secure_rag.manifest import ManifestStore
from secure_rag.pipeline import SecureRagPipeline
from secure_rag.retrieval import Retriever

_RAM_CACHE_MAX_ENTRIES = 16
_RAM_CACHE_MAX_TOTAL_CHARS = 20_000_000
_RAM_CACHE_MAX_ENTRY_CHARS = 5_000_000
_CONFIG_SNAPSHOT_ATTEMPTS = 3


def _config_content_hash(config_path: str) -> str:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _load_config_snapshot(config_path: str) -> tuple[AppConfig, str]:
    for _ in range(_CONFIG_SNAPSHOT_ATTEMPTS):
        hash_before = _config_content_hash(config_path)
        config = load_config(config_path)
        hash_after = _config_content_hash(config_path)
        if hash_before == hash_after:
            return config, hash_after
    raise RuntimeError("Configuration changed while it was being loaded")


@st.cache_resource(show_spinner=False)
def cached_embedder(config_path: str, config_hash: str, _config: AppConfig):
    return create_embedder(_config)


@st.cache_resource(show_spinner=False)
def cached_gateway(
    config_path: str,
    config_hash: str,
    mode: str,
    _config: AppConfig,
):
    return create_gateway(_config, mode=mode)


@st.cache_resource(show_spinner=False)
def cached_extracted_text(
    config_path: str,
    config_hash: str,
    _config: AppConfig,
) -> ProcessMemoryExtractedTextCache:
    return ProcessMemoryExtractedTextCache(
        max_entries=_RAM_CACHE_MAX_ENTRIES,
        max_total_chars=_RAM_CACHE_MAX_TOTAL_CHARS,
        max_entry_chars=min(
            _RAM_CACHE_MAX_ENTRY_CHARS,
            _config.ingestion.max_extracted_chars,
        ),
    )


def run_question(
    config_path: str,
    question: str,
    provider: str,
    ner_mode: str,
    attachment_path: str,
    on_event: EventCallback | None = None,
    *,
    config_snapshot: tuple[AppConfig, str] | None = None,
):
    with timed_stage(on_event, "runtime.config", "Загрузка и проверка конфигурации"):
        config, config_hash = config_snapshot or _load_config_snapshot(config_path)
        config.ensure_runtime()
    with timed_stage(
        on_event,
        "runtime.models",
        "Подготовка embedding- и NER-моделей",
        {"ner_mode": ner_mode},
    ):
        embedder = cached_embedder(config_path, config_hash, config)
        gateway = cached_gateway(config_path, config_hash, ner_mode, config)
        extracted_text_cache = cached_extracted_text(config_path, config_hash, config)
    with timed_stage(
        on_event,
        "runtime.storage",
        "Подключение к manifest и Qdrant",
        {"qdrant_mode": config.qdrant.mode},
    ):
        manifest = ManifestStore(config.manifest_path)
        store = create_vector_store(config, embedder.dimension)
    try:
        retriever = Retriever(
            config,
            embedder,
            manifest,
            store,
            extracted_cache=extracted_text_cache,
        )
        return SecureRagPipeline(config, retriever, gateway, manifest).run(
            question,
            provider=provider,
            attachment_path=attachment_path or None,
            on_event=on_event,
        )
    finally:
        store.close()
        manifest.close()


def _format_duration(duration_ms: float | None) -> str:
    if duration_ms is None:
        return ""
    if duration_ms < 1000:
        return f"{duration_ms:.0f} мс"
    return f"{duration_ms / 1000:.2f} с"


def _format_event(event: PipelineEvent) -> str:
    icons = {"started": "⏳", "completed": "✅", "failed": "❌", "info": "ℹ️"}
    duration = _format_duration(event.duration_ms)
    safe_details = ", ".join(f"{key}={value}" for key, value in event.details.items())
    suffix = " · ".join(item for item in (duration, safe_details) if item)
    return f"{icons[event.status]} **{event.label}**" + (f" — {suffix}" if suffix else "")


def main() -> None:
    st.set_page_config(page_title="Secure RAG", page_icon="🔒", layout="centered")

    repo_root = Path(__file__).resolve().parents[2]
    default_config = str(repo_root / "config" / "pilot.yaml")
    config_path = os.getenv("SECURE_RAG_CONFIG", default_config)
    provider = os.getenv("SECURE_RAG_PROVIDER", "auto").strip().lower()
    ner_mode = os.getenv("SECURE_RAG_NER_MODE", "all").strip().lower()

    st.title("Secure RAG")
    st.caption("Локальный поиск → маркировка → provider → локальное восстановление")

    with st.form("request_form", clear_on_submit=True):
        question = st.text_area(
            "Вопрос",
            height=130,
            placeholder="Что нужно найти или подготовить по локальным документам?",
        )
        attachment_path = st.text_input(
            "Путь к локальному файлу — необязательно",
            placeholder="папка/документ.docx",
            help=(
                "Разрешены только поддерживаемые файлы внутри каталога source_root. "
                "Сам путь не отправляется provider."
            ),
        )
        submitted = st.form_submit_button(
            "Отправить",
            type="primary",
            use_container_width=True,
        )

    answer_slot = st.container()
    log_slot = st.container()

    if submitted:
        if not question.strip():
            st.session_state.pop("last_run", None)
            answer_slot.error("Введите вопрос.")
        else:
            events: list[PipelineEvent] = []
            started = perf_counter()
            with log_slot:
                live_status = st.status("Запускаю локальный pipeline…", expanded=True)
            event_log: JsonlEventLog | None = None
            config_snapshot: tuple[AppConfig, str] | None = None
            try:
                config_snapshot = _load_config_snapshot(config_path)
                log_config = config_snapshot[0]
                event_log = JsonlEventLog(
                    log_config.paths.runtime_root / "reports" / "pipeline-events",
                    str(uuid.uuid4()),
                )
            except Exception:
                # The pipeline will report configuration/runtime errors through safe events.
                pass

            def receive_event(event: PipelineEvent) -> None:
                events.append(event)
                if event_log is not None:
                    try:
                        event_log(event)
                    except OSError:
                        pass
                if event.status == "started":
                    live_status.update(label=event.label, state="running")
                else:
                    live_status.write(_format_event(event))

            try:
                result = run_question(
                    config_path,
                    question,
                    provider,
                    ner_mode,
                    attachment_path,
                    receive_event,
                    config_snapshot=config_snapshot,
                )
                total_ms = (perf_counter() - started) * 1000
                live_status.update(
                    label=f"Pipeline завершён за {_format_duration(total_ms)}",
                    state="complete",
                    expanded=False,
                )
                restored = (
                    result.restored_output.read_text(encoding="utf-8")
                    if result.restored_output.exists()
                    else ""
                )
                st.session_state["last_run"] = {
                    "answer": restored,
                    "request_id": result.request_id,
                    "retrieved_count": result.retrieved_count,
                    "marker_count": result.marker_count,
                    "provider": result.provider,
                    "codex_input": str(result.codex_input),
                    "event_log": str(event_log.path) if event_log else "",
                    "events": events,
                    "total_ms": total_ms,
                }
            except Exception as exc:
                total_ms = (perf_counter() - started) * 1000
                live_status.update(
                    label=f"Pipeline остановлен за {_format_duration(total_ms)}",
                    state="error",
                    expanded=True,
                )
                st.session_state["last_run"] = {
                    "answer": "",
                    "error_type": type(exc).__name__,
                    "event_log": str(event_log.path) if event_log else "",
                    "events": events,
                    "total_ms": total_ms,
                }

    last_run = st.session_state.get("last_run")
    if last_run:
        with answer_slot:
            if last_run.get("answer"):
                st.subheader("Ответ")
                # The restored text contains private values and is provider-controlled.
                # Rendering Markdown could trigger remote image/link requests with those values.
                st.code(str(last_run["answer"]), language=None)
                st.caption(
                    f"request={last_run['request_id']} · provider={last_run['provider']} · "
                    f"контекстов={last_run['retrieved_count']} · "
                    f"маркеров={last_run['marker_count']}"
                )
            elif last_run.get("error_type"):
                st.error(
                    "Pipeline не завершён. "
                    f"Тип ошибки: {last_run['error_type']}. "
                    "Значения запроса и пути в лог не записаны."
                )
            else:
                st.info(
                    "Маркированный запрос подготовлен; "
                    "автоматический provider пока не вернул ответ."
                )

        with log_slot:
            with st.expander("Журнал выполнения", expanded=False):
                st.write(f"Общее время: {_format_duration(float(last_run['total_ms']))}")
                for item in last_run["events"]:
                    if item.status != "started":
                        st.markdown(_format_event(item))
                if last_run.get("codex_input"):
                    st.code(str(last_run["codex_input"]), language=None)
                if last_run.get("event_log"):
                    st.caption(f"Локальный JSONL: {last_run['event_log']}")

    with st.expander("Режим стенда", expanded=False):
        st.write(f"Configured provider: `{provider}` · Sanitizer: `{ner_mode}`")
        if provider == "auto" and not os.getenv("OPENAI_API_KEY", "").strip():
            st.caption("OPENAI_API_KEY не задан: safe auto использует локальный stub.")
        if provider == "codex-local":
            st.error(
                "codex-local — небезопасный лабораторный режим: read-only SDK "
                "не запрещает агенту читать доступные файлы."
            )
        st.caption(
            "В журнал не попадают текст запроса, путь к файлу и найденные значения сущностей. "
            "Для прямого файла разрешён только путь внутри configured source_root."
        )


if __name__ == "__main__":
    main()
