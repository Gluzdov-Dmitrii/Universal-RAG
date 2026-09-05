from __future__ import annotations

from dataclasses import replace

from secure_rag.config import load_config


def test_runtime_artifacts_are_grouped_by_lifetime(tmp_path) -> None:
    base = load_config()
    runtime_root = (tmp_path / "runtime").resolve()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=runtime_root),
    )

    config.ensure_runtime()

    assert config.manifest_path == runtime_root / "data" / "manifest" / "documents.sqlite"
    assert config.qdrant_path == runtime_root / "data" / "qdrant"
    assert config.chat_state_path == runtime_root / "data" / "sessions" / "chat-state.sqlite"
    assert config.requests_path == runtime_root / "state" / "requests"
    assert config.marker_vault_path == runtime_root / "state" / "marker-vault"
    assert config.model_cache_path == runtime_root / "cache" / "models"
    assert config.extracted_text_cache_path == runtime_root / "cache" / "extracted-text"
    assert config.reports_path == runtime_root / "diagnostics" / "reports"
    assert config.logs_path == runtime_root / "diagnostics" / "logs"
    assert config.locks_path == runtime_root / "run" / "locks"
    assert config.pids_path == runtime_root / "run" / "pids"
    assert config.codex_sandbox_path == runtime_root / "tmp" / "codex-sandbox"

    assert {path.name for path in runtime_root.iterdir()} == {
        "cache",
        "data",
        "diagnostics",
        "run",
        "state",
        "tmp",
    }
