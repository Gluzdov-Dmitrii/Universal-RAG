from __future__ import annotations

from dataclasses import replace

import secure_rag.orchestration.parents as parents_module
from secure_rag.config import load_config
from secure_rag.domain.models import RetrievalHit, TextLocation
from secure_rag.ingestion.extractors import ExtractedDocument, file_sha256
from secure_rag.orchestration.parents import ParentContextBuilder


def _config(tmp_path, **retrieval_overrides):
    base = load_config()
    source = (tmp_path / "source").resolve()
    source.mkdir()
    return replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
        retrieval=replace(base.retrieval, **retrieval_overrides),
    )


def _hit(path, text: str, start: int, end: int, *, source_type: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        document_id="document-1",
        revision=file_sha256(path),
        score=0.9,
        start=start,
        end=end,
        text=text[start:end],
        source_name=path.name,
        source_path=path,
        source_type=source_type,
        ordinal=3,
    )


def test_small_source_is_promoted_to_whole_document(tmp_path) -> None:
    config = _config(tmp_path)
    path = config.paths.source_root / "policy.txt"
    text = "Введение.\n\nИскомый факт.\n\nСсылки на связанные документы."
    path.write_text(text, encoding="utf-8")
    start = text.index("Искомый")

    contexts = ParentContextBuilder(config).build(
        [_hit(path, text, start, start + len("Искомый факт."), source_type="txt")]
    )

    assert len(contexts) == 1
    assert contexts[0].text == text
    assert contexts[0].context_scope == "whole_document"
    assert contexts[0].start == 0
    assert contexts[0].end == len(text)


def test_large_source_uses_complete_matching_logical_location(
    tmp_path, monkeypatch
) -> None:
    config = _config(
        tmp_path,
        whole_document_max_chars=1_000,
        logical_parent_max_chars=1_200,
        total_context_max_chars=1_200,
    )
    path = config.paths.source_root / "report.pdf"
    path.write_bytes(b"source-revision")
    first_page = "первая страница " * 100
    second_page = "заголовок второй страницы\n" + "нужный факт " * 30
    extracted_text = first_page + second_page
    second_start = len(first_page)
    monkeypatch.setattr(
        parents_module,
        "extract_document",
        lambda *_args: ExtractedDocument(
            extracted_text,
            locations=(
                TextLocation(0, second_start, "page", "1"),
                TextLocation(second_start, len(extracted_text), "page", "2"),
            ),
        ),
    )
    focus = extracted_text.index("нужный факт", second_start)

    contexts = ParentContextBuilder(config).build(
        [
            _hit(
                path,
                extracted_text,
                focus,
                focus + len("нужный факт"),
                source_type="pdf",
            )
        ]
    )

    assert contexts[0].text == second_page
    assert contexts[0].context_scope == "logical_parent"
    assert first_page not in contexts[0].text


def test_large_table_keeps_header_and_area_around_match(tmp_path) -> None:
    config = _config(
        tmp_path,
        whole_document_max_chars=1_000,
        logical_parent_max_chars=1_200,
        total_context_max_chars=1_200,
    )
    path = config.paths.source_root / "staff.csv"
    text = "employee,department,salary\n" + ("filler,value,0\n" * 300) + "TARGET,legal,42\n"
    path.write_text(text, encoding="utf-8")
    focus = text.index("TARGET")

    contexts = ParentContextBuilder(config).build(
        [_hit(path, text, focus, focus + len("TARGET,legal,42"), source_type="csv")]
    )

    assert len(contexts[0].text) <= 1_200
    assert "employee,department,salary" in contexts[0].text
    assert "TARGET,legal,42" in contexts[0].text
    assert "строки между заголовком" in contexts[0].text
    assert contexts[0].context_scope == "logical_parent"
