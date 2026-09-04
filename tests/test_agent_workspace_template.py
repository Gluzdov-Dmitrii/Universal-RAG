from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from secure_rag.config import load_config


def test_rag_agent_workspace_manifest_is_complete_and_safe() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_root = repo_root / "llm-workspaces" / "rag-test"
    manifest = json.loads(
        (template_root / ".rag-workspace-manifest.json").read_text(encoding="utf-8")
    )
    managed = tuple(str(item) for item in manifest["managed_files"])

    assert manifest["schema_version"] == 1
    assert manifest["component"] == "secure-rag-user-agent"
    assert len(managed) == len(set(managed))
    assert set(load_config().generation.instruction_files).issubset(managed)

    for raw_relative in managed:
        relative = PurePosixPath(raw_relative)
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = (template_root / Path(*relative.parts)).resolve(strict=True)
        assert path.is_file()
        assert path.is_relative_to(template_root.resolve())


def test_rag_agent_skill_is_concise_and_discoverable() -> None:
    skill_path = (
        Path(__file__).resolve().parents[1]
        / "llm-workspaces"
        / "rag-test"
        / ".cursor"
        / "skills"
        / "answering-with-secure-rag"
        / "SKILL.md"
    )
    content = skill_path.read_text(encoding="utf-8")

    assert len(content.splitlines()) < 500
    assert content.startswith("---\nname: answering-with-secure-rag\n")
    assert "description:" in content
    assert "<retrieval_request>" in content
