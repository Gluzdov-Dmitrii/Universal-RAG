from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import secure_rag.web_app as web_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP_PATH = PROJECT_ROOT / "src" / "secure_rag" / "web_app.py"


def _copy_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "pilot.yaml"
    config_path.parent.mkdir()
    config_path.write_bytes((PROJECT_ROOT / "config" / "pilot.yaml").read_bytes())
    return config_path


def _clear_resource_caches() -> None:
    web_app.cached_embedder.clear()
    web_app.cached_gateway.clear()
    web_app.cached_extracted_text.clear()


def test_config_snapshot_retries_until_hash_matches_loaded_file(monkeypatch) -> None:
    first_config = object()
    stable_config = object()
    hashes = iter(("before-change", "after-change", "stable", "stable"))
    configs = iter((first_config, stable_config))
    load_calls = 0

    monkeypatch.setattr(web_app, "_config_content_hash", lambda _path: next(hashes))

    def counted_load(_path):
        nonlocal load_calls
        load_calls += 1
        return next(configs)

    monkeypatch.setattr(web_app, "load_config", counted_load)

    config, config_hash = web_app._load_config_snapshot("pilot.yaml")

    assert config is stable_config
    assert config_hash == "stable"
    assert load_calls == 2


def test_resource_caches_invalidate_when_config_contents_change(
    tmp_path, monkeypatch
) -> None:
    config_path = _copy_config(tmp_path)
    created_embedders: list[object] = []
    created_gateways: list[object] = []
    embedder_configs: list[object] = []
    gateway_configs: list[object] = []

    def create_embedder(config):
        embedder_configs.append(config)
        resource = object()
        created_embedders.append(resource)
        return resource

    def create_gateway(config, *, mode):
        assert mode == "regex"
        gateway_configs.append(config)
        resource = object()
        created_gateways.append(resource)
        return resource

    monkeypatch.setattr(web_app, "create_embedder", create_embedder)
    monkeypatch.setattr(web_app, "create_gateway", create_gateway)
    _clear_resource_caches()
    try:
        first_config, first_hash = web_app._load_config_snapshot(str(config_path))
        first_embedder = web_app.cached_embedder(
            str(config_path), first_hash, first_config
        )
        first_gateway = web_app.cached_gateway(
            str(config_path), first_hash, "regex", first_config
        )
        first_text_cache = web_app.cached_extracted_text(
            str(config_path), first_hash, first_config
        )

        assert (
            web_app.cached_embedder(str(config_path), first_hash, first_config)
            is first_embedder
        )
        assert (
            web_app.cached_gateway(
                str(config_path), first_hash, "regex", first_config
            )
            is first_gateway
        )
        assert (
            web_app.cached_extracted_text(str(config_path), first_hash, first_config)
            is first_text_cache
        )

        config_path.write_bytes(config_path.read_bytes() + b"\n# cache key change\n")
        changed_config, changed_hash = web_app._load_config_snapshot(str(config_path))
        changed_embedder = web_app.cached_embedder(
            str(config_path), changed_hash, changed_config
        )
        changed_gateway = web_app.cached_gateway(
            str(config_path), changed_hash, "regex", changed_config
        )
        changed_text_cache = web_app.cached_extracted_text(
            str(config_path), changed_hash, changed_config
        )

        assert changed_hash != first_hash
        assert changed_embedder is not first_embedder
        assert changed_gateway is not first_gateway
        assert changed_text_cache is not first_text_cache
        assert created_embedders == [first_embedder, changed_embedder]
        assert created_gateways == [first_gateway, changed_gateway]
        assert embedder_configs == [first_config, changed_config]
        assert gateway_configs == [first_config, changed_config]
    finally:
        _clear_resource_caches()


def test_run_question_reuses_supplied_config_snapshot(tmp_path, monkeypatch) -> None:
    config_path = _copy_config(tmp_path)
    config_snapshot = web_app._load_config_snapshot(str(config_path))
    config, config_hash = config_snapshot
    embedder = SimpleNamespace(dimension=8)
    gateway = object()
    text_cache = object()
    manifest = SimpleNamespace(closed=False)
    store = SimpleNamespace(closed=False)
    observed_configs: list[object] = []

    def unexpected_reload(_path):
        raise AssertionError("a request must reuse its already-loaded config")

    def cached_embedder(path, digest, supplied_config):
        assert path == str(config_path)
        assert digest == config_hash
        observed_configs.append(supplied_config)
        return embedder

    def cached_gateway(path, digest, mode, supplied_config):
        assert path == str(config_path)
        assert digest == config_hash
        assert mode == "regex"
        observed_configs.append(supplied_config)
        return gateway

    def cached_extracted_text(path, digest, supplied_config):
        assert path == str(config_path)
        assert digest == config_hash
        observed_configs.append(supplied_config)
        return text_cache

    def create_store(supplied_config, dimension):
        assert supplied_config is config
        assert dimension == embedder.dimension
        store.close = lambda: setattr(store, "closed", True)
        return store

    def create_manifest(path):
        assert path == config.manifest_path
        manifest.close = lambda: setattr(manifest, "closed", True)
        return manifest

    def create_retriever(
        supplied_config,
        supplied_embedder,
        supplied_manifest,
        supplied_store,
        *,
        extracted_cache,
    ):
        assert supplied_config is config
        assert supplied_embedder is embedder
        assert supplied_manifest is manifest
        assert supplied_store is store
        assert extracted_cache is text_cache
        return object()

    expected_result = object()

    class Pipeline:
        def __init__(self, supplied_config, retriever, supplied_gateway, supplied_manifest):
            assert supplied_config is config
            assert retriever is not None
            assert supplied_gateway is gateway
            assert supplied_manifest is manifest

        def run(self, question, *, provider, attachment_path, on_event):
            assert question == "test question"
            assert provider == "stub"
            assert attachment_path is None
            assert on_event is None
            return expected_result

    monkeypatch.setattr(web_app, "_load_config_snapshot", unexpected_reload)
    monkeypatch.setattr(web_app, "cached_embedder", cached_embedder)
    monkeypatch.setattr(web_app, "cached_gateway", cached_gateway)
    monkeypatch.setattr(web_app, "cached_extracted_text", cached_extracted_text)
    monkeypatch.setattr(web_app, "ManifestStore", create_manifest)
    monkeypatch.setattr(web_app, "create_vector_store", create_store)
    monkeypatch.setattr(web_app, "Retriever", create_retriever)
    monkeypatch.setattr(web_app, "SecureRagPipeline", Pipeline)

    result = web_app.run_question(
        str(config_path),
        "test question",
        "stub",
        "regex",
        "",
        config_snapshot=config_snapshot,
    )

    assert result is expected_result
    assert observed_configs == [config, config, config]
    assert manifest.closed
    assert store.closed


def test_app_form_clears_and_keeps_single_neutral_submit_control() -> None:
    app = AppTest.from_file(str(WEB_APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Отправить"]
    form = next(child for child in app.main.children.values() if child.type == "form")
    assert form.proto.form.clear_on_submit
    assert len(app.text_input) == 1
    placeholder = app.text_input[0].placeholder
    assert "RAG TEST" not in placeholder
    assert "D:" not in placeholder

    app.text_input[0].set_value("папка/документ.docx")
    app.button[0].click().run()

    assert not app.exception
    assert app.text_area[0].value == ""
    assert [error.value for error in app.error] == ["Введите вопрос."]


def test_app_renders_provider_answer_as_plain_text() -> None:
    answer = '**private**\n<img src="https://example.test/leak">'
    app = AppTest.from_file(str(WEB_APP_PATH), default_timeout=10).run()
    app.session_state["last_run"] = {
        "answer": answer,
        "request_id": "request-id",
        "retrieved_count": 1,
        "marker_count": 2,
        "provider": "stub",
        "codex_input": "",
        "event_log": "",
        "events": [],
        "total_ms": 1.0,
    }

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Отправить"]
    assert [(code.value, code.language) for code in app.code] == [
        (answer, "plaintext")
    ]
    assert all(answer not in markdown.value for markdown in app.markdown)
