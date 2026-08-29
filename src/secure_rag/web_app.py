from __future__ import annotations

from pathlib import Path

import streamlit as st

from secure_rag.bridge import BridgeManager
from secure_rag.config import load_config
from secure_rag.factory import create_embedder, create_gateway, create_vector_store
from secure_rag.indexer import Indexer
from secure_rag.manifest import ManifestStore
from secure_rag.pipeline import SecureRagPipeline
from secure_rag.retrieval import Retriever

st.set_page_config(page_title="Secure RAG Pilot", page_icon="🔒", layout="wide")


@st.cache_resource(show_spinner="Загружаю embedding-модель…")
def cached_embedder(config_path: str):
    config = load_config(config_path)
    return create_embedder(config)


@st.cache_resource(show_spinner="Загружаю локальные NER-модели…")
def cached_gateway(config_path: str, mode: str):
    config = load_config(config_path)
    return create_gateway(config, mode=mode)


def run_index(config_path: str, max_files: int):
    config = load_config(config_path)
    config.ensure_runtime()
    embedder = cached_embedder(config_path)
    with (
        ManifestStore(config.manifest_path) as manifest,
        create_vector_store(config, embedder.dimension) as store,
    ):
        return Indexer(config, embedder, manifest, store).run(max_files=max_files)


def run_question(config_path: str, question: str, provider: str, ner_mode: str):
    config = load_config(config_path)
    config.ensure_runtime()
    embedder = cached_embedder(config_path)
    gateway = cached_gateway(config_path, ner_mode)
    with (
        ManifestStore(config.manifest_path) as manifest,
        create_vector_store(config, embedder.dimension) as store,
    ):
        retriever = Retriever(config, embedder, manifest, store)
        return SecureRagPipeline(config, retriever, gateway, manifest).run(
            question,
            provider=provider,
        )


default_config = str(Path(__file__).resolve().parents[2] / "config" / "pilot.yaml")

st.title("Secure RAG: локальный стенд")
st.caption("Files → local embeddings → Qdrant → NER/regex markers → Codex bridge → demarker")

with st.sidebar:
    config_path = st.text_input("Конфигурация", value=default_config)
    ner_mode = st.selectbox(
        "Sanitizer",
        options=("all", "legal", "collection3", "regex"),
        format_func={
            "all": "Regex + обе NER",
            "legal": "Regex + Legal NER",
            "collection3": "Regex + Collection3",
            "regex": "Только regex (быстрый тест)",
        }.get,
    )
    provider_label = st.radio(
        "Режим ответа",
        ("Файл для ручного Codex", "Локальный mock"),
    )
    max_files = st.number_input(
        "Файлов в пилотном индексе",
        min_value=1,
        max_value=5000,
        value=10,
        step=10,
    )
    if st.button("Построить/обновить индекс", use_container_width=True):
        try:
            with st.spinner("Читаю файлы батчами и считаю embeddings локально…"):
                report = run_index(config_path, int(max_files))
            st.success(
                f"Готово: indexed={report.indexed}, skipped={report.skipped}, "
                f"failed={report.failed}, chunks={report.chunks}"
            )
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    st.warning(
        "Embedded Qdrant — учебный режим для небольшого индекса. "
        "Для пилота свыше ~20 000 chunks нужен Qdrant server."
    )

question = st.text_area(
    "Вопрос",
    height=120,
    placeholder="Введите вопрос к локальным документам…",
)
if st.button("Запустить полный поток", type="primary", disabled=not question.strip()):
    provider = "manual" if provider_label.startswith("Файл") else "stub"
    try:
        with st.spinner("Retrieval и локальная маркировка…"):
            result = run_question(config_path, question, provider, ner_mode)
        st.session_state["request_id"] = result.request_id
        st.success(
            f"Request {result.request_id}: фрагментов {result.retrieved_count}, "
            f"маркеров {result.marker_count}"
        )
        st.code(str(result.codex_input), language=None)
        payload = result.codex_input.read_bytes()
        with st.expander("Просмотреть точный sanitized payload"):
            st.code(payload.decode("utf-8"), language="markdown")
        st.download_button(
            "Скачать sanitized codex_input.md",
            data=payload,
            file_name="codex_input.md",
            mime="text/markdown",
        )
        if provider == "stub" and result.restored_output.exists():
            st.subheader("Результат после локального demarker")
            st.markdown(result.restored_output.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

st.divider()
st.subheader("Импорт ответа Codex")
request_id = st.text_input(
    "Request ID",
    value=st.session_state.get("request_id", ""),
)
marked_answer = st.text_area(
    "Вставьте ответ Codex с сохранёнными markers",
    height=180,
)
if st.button("Демаркировать локально", disabled=not (request_id and marked_answer.strip())):
    try:
        config = load_config(config_path)
        output = BridgeManager(config).restore_text(request_id, marked_answer)
        restored = output.read_text(encoding="utf-8")
        st.success("Ответ восстановлен локально.")
        st.markdown(restored)
        st.download_button(
            "Скачать restored_answer.md",
            data=restored.encode("utf-8"),
            file_name="restored_answer.md",
            mime="text/markdown",
        )
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

with st.expander("Что именно проверяет этот стенд"):
    st.markdown(
        """
- Исходный запрос и документы используются только для локального retrieval.
- В Codex-файл попадают sanitized query, chunks и скрытые filenames.
- Marker map лежит рядом локально и не входит в Codex-файл.
- Неизвестный marker в ответе блокирует demarker.
- Это прототип privacy pipeline, а не сертифицированная DLP-система.
"""
    )
