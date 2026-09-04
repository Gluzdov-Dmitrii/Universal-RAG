from __future__ import annotations

import sys
from types import ModuleType

import pytest

from secure_rag.config import EmbeddingConfig, NerModelConfig
from secure_rag.infrastructure.huggingface import HF_LOCAL_ONLY_ENV, hf_local_files_only
from secure_rag.retrieval.embeddings import SentenceTransformerEmbedder
from secure_rag.sanitization.ner import TransformersNerDetector


def _set_local_only_environment(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(HF_LOCAL_ONLY_ENV, raising=False)
    else:
        monkeypatch.setenv(HF_LOCAL_ONLY_ENV, value)


@pytest.mark.parametrize(
    ("environment_value", "expected"), [(None, True), ("1", True), ("0", False)]
)
def test_sentence_transformer_receives_local_only_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, environment_value: str | None, expected: bool
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured.update(kwargs)

        @staticmethod
        def get_embedding_dimension() -> int:
            return 3

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    _set_local_only_environment(monkeypatch, environment_value)

    SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_id="test/embedding",
            revision="fixed-revision",
            dimension=3,
            batch_size=2,
            device="cpu",
            query_prefix="",
            passage_prefix="",
        ),
        tmp_path,
    )

    assert captured["local_files_only"] is expected


@pytest.mark.parametrize(
    ("environment_value", "expected"), [(None, True), ("1", True), ("0", False)]
)
def test_ner_loaders_receive_local_only_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch, environment_value: str | None, expected: bool
) -> None:
    tokenizer_kwargs: dict[str, object] = {}
    model_kwargs: dict[str, object] = {}

    class FakeTokenizer:
        model_max_length = 512

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_source: str, **kwargs: object) -> FakeTokenizer:
            tokenizer_kwargs["model_source"] = model_source
            tokenizer_kwargs.update(kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_source: str, **kwargs: object) -> object:
            model_kwargs["model_source"] = model_source
            model_kwargs.update(kwargs)
            return object()

    fake_module = ModuleType("transformers")
    fake_module.AutoTokenizer = FakeAutoTokenizer
    fake_module.AutoModelForTokenClassification = FakeAutoModel
    fake_module.pipeline = lambda *args, **kwargs: lambda text: []
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    _set_local_only_environment(monkeypatch, environment_value)

    TransformersNerDetector(
        NerModelConfig(
            name="test-ner",
            model_id="test/ner",
            revision="fixed-revision",
            local_path=None,
            enabled=True,
            threshold=0.5,
            device="cpu",
            priority=100,
        ),
        tmp_path,
    )

    assert tokenizer_kwargs["local_files_only"] is expected
    assert model_kwargs["local_files_only"] is expected


def test_invalid_local_only_environment_value_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_value = "invalid-secret-value"
    monkeypatch.setenv(HF_LOCAL_ONLY_ENV, sensitive_value)

    with pytest.raises(ValueError) as caught:
        hf_local_files_only()

    assert HF_LOCAL_ONLY_ENV in str(caught.value)
    assert sensitive_value not in str(caught.value)
