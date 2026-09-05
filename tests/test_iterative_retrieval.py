from __future__ import annotations

import json
from dataclasses import replace

import pytest

from secure_rag.config import load_config
from secure_rag.domain.models import EntitySpan, RetrievalHit
from secure_rag.llm.control import parse_retrieval_request
from secure_rag.orchestration.pipeline import SecureRagPipeline
from secure_rag.sanitization.core import PrivacyGateway


class PersonDetector:
    name = "person"

    @staticmethod
    def detect(text: str) -> list[EntitySpan]:
        value = "Анна Смирнова"
        start = text.find(value)
        if start < 0:
            return []
        return [
            EntitySpan(
                start=start,
                end=start + len(value),
                label="PER",
                score=1.0,
                source="person",
                priority=500,
            )
        ]


class FakeEmbedder:
    model_version = "fake@1"


class IterativeRetriever:
    embedder = FakeEmbedder()

    def __init__(self, first: RetrievalHit, second: RetrievalHit) -> None:
        self.first = first
        self.second = second
        self.queries: list[str] = []

    def search(self, query, top_k=None, *, on_event=None):
        del top_k, on_event
        self.queries.append(query)
        return [self.first] if len(self.queries) == 1 else [self.second]

    @staticmethod
    def query_similarity(original_query: str, rewritten_query: str) -> float:
        assert "[[" not in rewritten_query
        assert "Анна Смирнова" in rewritten_query
        assert original_query
        return 0.9

    @staticmethod
    def expand_adjacent(hits, radius, *, on_event=None):
        del hits, radius, on_event
        return []


class FakeManifest:
    @staticmethod
    def latest_build_id() -> str:
        return "build-test"


def test_iterative_multi_query_keeps_paths_local_and_returns_sources(
    tmp_path, monkeypatch
) -> None:
    source_root = (tmp_path / "source").resolve()
    source_root.mkdir()
    first_path = source_root / "private" / "record.docx"
    first_path.parent.mkdir()
    first_path.write_text("placeholder", encoding="utf-8")
    second_path = source_root / "finance" / "salary.xlsx"
    second_path.parent.mkdir()
    second_path.write_text("placeholder", encoding="utf-8")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source_root,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    first = RetrievalHit(
        chunk_id="00000000-0000-0000-0000-000000000001",
        document_id="doc-1",
        revision="rev-1",
        score=0.9,
        start=0,
        end=30,
        text="Анна Смирнова указана в карточке.",
        source_name=first_path.name,
        source_path=first_path,
        source_type="docx",
        ordinal=0,
        location_kind="page",
        location_start="12",
        location_end="13",
    )
    second = RetrievalHit(
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_id="doc-2",
        revision="rev-2",
        score=0.85,
        start=0,
        end=40,
        text="Заголовок таблицы: сотрудник, оклад.",
        source_name=second_path.name,
        source_path=second_path,
        source_type="xlsx",
        ordinal=1,
    )
    retriever = IterativeRetriever(first, second)

    class IterativeProvider:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        def answer_payload(self, payload: str, request_id: str) -> str:
            assert request_id
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return (
                    '<retrieval_request>{"queries":["подробности про [[PER_0001]]"],'
                    '"expand_citations":[]}</retrieval_request>\n'
                )
            return "Сотрудник [[PER_0001]] указан в источниках R001 и R002.\n"

    provider = IterativeProvider()
    monkeypatch.setattr(
        SecureRagPipeline,
        "_create_provider",
        lambda _self, _name: provider,
    )
    result = SecureRagPipeline(
        config,
        retriever,
        PrivacyGateway(PersonDetector()),
        FakeManifest(),
    ).run("Что известно про Анна Смирнова?", provider="responses")

    assert result.iterations == 2
    assert retriever.queries == [
        "Что известно про Анна Смирнова?",
        "подробности про Анна Смирнова",
    ]
    assert len(provider.payloads) == 2
    assert str(first_path) not in provider.payloads[-1]
    assert str(second_path) not in provider.payloads[-1]
    assert "File type: xlsx" in provider.payloads[-1]
    assert [source.path for source in result.sources] == [first_path, second_path]
    sources = json.loads(result.sources_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in sources] == [str(first_path), str(second_path)]
    assert sources[0]["locations"] == [
        {
            "citation_ref": "R001",
            "kind": "page",
            "start": "12",
            "end": "13",
        }
    ]
    restored = result.restored_output.read_text(encoding="utf-8")
    assert "Анна Смирнова" in restored
    assert "R001" in restored and "R002" in restored


def test_retrieval_control_is_exact_and_bounded() -> None:
    parsed = parse_retrieval_request(
        '<retrieval_request>{"queries":["a", "b"],'
        '"expand_citations":["R001"]}</retrieval_request>',
        max_queries=2,
    )
    assert parsed is not None
    assert parsed.queries == ("a", "b")
    assert parsed.expand_citations == ("R001",)

    with pytest.raises(ValueError, match="invalid_retrieval_control_envelope"):
        parse_retrieval_request(
            "explanation\n<retrieval_request>{}</retrieval_request>",
            max_queries=2,
        )
    with pytest.raises(ValueError, match="retrieval_control_limit_exceeded"):
        parse_retrieval_request(
            '<retrieval_request>{"queries":["a", "b", "c"]}</retrieval_request>',
            max_queries=2,
        )


def test_malformed_provider_retrieval_control_returns_local_answer(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = (tmp_path / "source").resolve()
    source_root.mkdir()
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source_root,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()

    class EmptyRetriever:
        embedder = FakeEmbedder()

        @staticmethod
        def search(_query, top_k=None, *, on_event=None):
            del top_k, on_event
            return []

    class MalformedProvider:
        @staticmethod
        def answer_payload(_payload: str, _request_id: str) -> str:
            return "explanation\n<retrieval_request>{}</retrieval_request>"

    monkeypatch.setattr(
        SecureRagPipeline,
        "_create_provider",
        lambda _self, _name: MalformedProvider(),
    )
    events = []
    result = SecureRagPipeline(
        config,
        EmptyRetriever(),
        PrivacyGateway(PersonDetector()),
        FakeManifest(),
    ).run("Простой вопрос", provider="responses", on_event=events.append)

    assert "Недостаточно данных" in result.restored_output.read_text(encoding="utf-8")
    completed_call = next(
        event
        for event in events
        if event.stage == "provider.call" and event.status == "completed"
    )
    assert completed_call.details["retrieval_control_invalid"] is True


def test_iterative_retrieval_stops_with_local_insufficient_answer(
    tmp_path, monkeypatch
) -> None:
    source_root = (tmp_path / "source").resolve()
    source_root.mkdir()
    path = source_root / "only.txt"
    path.write_text("placeholder", encoding="utf-8")
    base = load_config()
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_root=source_root,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )
    config.ensure_runtime()
    hit = RetrievalHit(
        chunk_id="00000000-0000-0000-0000-000000000010",
        document_id="doc-only",
        revision="rev",
        score=0.8,
        start=0,
        end=20,
        text="Контекст без ответа.",
        source_name=path.name,
        source_path=path,
        source_type="txt",
        ordinal=0,
    )
    retriever = IterativeRetriever(hit, hit)

    class AlwaysSearchProvider:
        calls = 0

        def answer_payload(self, payload: str, request_id: str) -> str:
            del payload, request_id
            self.calls += 1
            return (
                '<retrieval_request>{"queries":["[[PER_0001]]"],'
                '"expand_citations":[]}</retrieval_request>\n'
            )

    provider = AlwaysSearchProvider()
    monkeypatch.setattr(
        SecureRagPipeline,
        "_create_provider",
        lambda _self, _name: provider,
    )
    result = SecureRagPipeline(
        config,
        retriever,
        PrivacyGateway(PersonDetector()),
        FakeManifest(),
    ).run("Что известно про Анна Смирнова?", provider="responses")

    assert provider.calls == 2
    assert result.iterations == config.retrieval.max_iterations
    restored = result.restored_output.read_text(encoding="utf-8")
    assert "Недостаточно данных" in restored
