from __future__ import annotations

from pathlib import Path, PurePosixPath

from ..config import AppConfig

_ALLOWED_INSTRUCTION_SUFFIXES = frozenset({".md", ".mdc"})


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def load_agent_instructions(config: AppConfig) -> str:
    """Load only explicitly allowlisted instructions from an external client workspace."""

    configured_root = config.generation.agent_workspace_root
    configured_files = config.generation.instruction_files
    if configured_root is None or not configured_files:
        return ""

    try:
        root = configured_root.resolve(strict=True)
    except OSError:
        raise RuntimeError("agent_workspace_unavailable") from None
    if not root.is_dir():
        raise RuntimeError("agent_workspace_unavailable")

    for forbidden in (
        config.repo_root.resolve(),
        config.paths.source_root.resolve(),
        config.paths.runtime_root.resolve(),
    ):
        if _overlaps(root, forbidden):
            raise ValueError("agent_workspace_overlaps_private_root")

    sections: list[str] = []
    total_chars = 0
    for raw_relative in configured_files:
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() not in _ALLOWED_INSTRUCTION_SUFFIXES
        ):
            raise ValueError("agent_instruction_path_invalid")
        try:
            path = (root / Path(*relative.parts)).resolve(strict=True)
        except OSError:
            raise RuntimeError("agent_instruction_unavailable") from None
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("agent_instruction_path_invalid")
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimeError("agent_instruction_unavailable") from None
        total_chars += len(content)
        if total_chars > config.generation.max_instruction_chars:
            raise ValueError("agent_instructions_too_large")
        sections.append(f"## {relative.as_posix()}\n\n{content}")

    return "\n\n".join(sections)
