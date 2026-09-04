from __future__ import annotations

import os

from ..config import AppConfig
from .adapters import LocalCodexProvider, OpenAIResponsesProvider
from .contracts import Provider
from .prompts import with_agent_instructions
from .workspace import load_agent_instructions

SUPPORTED_PROVIDERS = frozenset(
    {"auto", "responses", "codex-local", "manual", "stub"}
)


def resolve_provider_name(requested: str) -> str:
    """Resolve the safe default without constructing a network client."""

    name = requested.strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        raise ValueError("unsupported_provider")
    if name == "auto":
        return "responses" if os.getenv("OPENAI_API_KEY", "").strip() else "stub"
    return name


class ProviderFactory:
    """Composition boundary for substituting cloud, Codex, or local LLM adapters."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def create(self, name: str) -> Provider:
        instructions = with_agent_instructions(load_agent_instructions(self.config))
        if name == "responses":
            return OpenAIResponsesProvider(instructions=instructions)
        if name == "codex-local":
            return LocalCodexProvider(
                forbidden_roots=(
                    self.config.repo_root,
                    self.config.paths.source_root,
                    self.config.paths.runtime_root,
                ),
                project_root=self.config.generation.agent_workspace_root,
                instructions=instructions,
            )
        raise ValueError("unsupported_automatic_provider")
