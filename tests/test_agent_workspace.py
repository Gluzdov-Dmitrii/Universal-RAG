from __future__ import annotations

from dataclasses import replace

import pytest

from secure_rag.config import load_config
from secure_rag.llm.factory import ProviderFactory
from secure_rag.llm.workspace import load_agent_instructions


def _workspace_config(tmp_path):
    base = load_config()
    workspace = (tmp_path / "agent-workspace").resolve()
    rule = workspace / ".cursor" / "rules" / "rag.mdc"
    skill = workspace / ".cursor" / "skills" / "rag" / "SKILL.md"
    rule.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    rule.write_text("trusted rule", encoding="utf-8")
    skill.write_text("trusted skill", encoding="utf-8")
    config = replace(
        base,
        generation=replace(
            base.generation,
            agent_workspace_root=workspace,
            instruction_files=(
                ".cursor/rules/rag.mdc",
                ".cursor/skills/rag/SKILL.md",
            ),
            max_instruction_chars=1_000,
        ),
    )
    return config, workspace


def test_only_allowlisted_agent_instructions_are_loaded(tmp_path) -> None:
    config, workspace = _workspace_config(tmp_path)
    (workspace / "not-allowlisted.md").write_text("must stay out", encoding="utf-8")

    loaded = load_agent_instructions(config)

    assert "trusted rule" in loaded
    assert "trusted skill" in loaded
    assert "must stay out" not in loaded
    assert str(workspace) not in loaded


def test_agent_workspace_must_not_overlap_private_roots(tmp_path) -> None:
    config, _workspace = _workspace_config(tmp_path)
    overlapping = replace(
        config,
        generation=replace(
            config.generation,
            agent_workspace_root=config.repo_root,
            instruction_files=("README.md",),
        ),
    )

    with pytest.raises(ValueError, match="^agent_workspace_overlaps_private_root$"):
        load_agent_instructions(overlapping)


def test_agent_instruction_path_cannot_escape_workspace(tmp_path) -> None:
    config, _workspace = _workspace_config(tmp_path)
    escaped = replace(
        config,
        generation=replace(
            config.generation,
            instruction_files=("../outside.md",),
        ),
    )

    with pytest.raises(ValueError, match="^agent_instruction_path_invalid$"):
        load_agent_instructions(escaped)


def test_provider_factory_injects_workspace_instructions_without_tools(
    tmp_path, monkeypatch
) -> None:
    config, _workspace = _workspace_config(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = ProviderFactory(config).create("responses")

    assert "trusted rule" in provider.instructions
    assert "trusted skill" in provider.instructions
    assert "do not grant file or tool access" in provider.instructions


def test_provider_factory_binds_local_codex_to_agent_project(
    tmp_path, monkeypatch
) -> None:
    config, workspace = _workspace_config(tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL", "1")

    provider = ProviderFactory(config).create("codex-local")

    assert provider.project_root == workspace
