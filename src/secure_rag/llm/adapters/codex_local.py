from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from ..prompts import PROVIDER_INSTRUCTIONS


class LocalCodexProvider:
    """Explicitly unsafe local SDK adapter; read-only does not constrain file reads."""

    name = "codex-local"

    def __init__(
        self,
        forbidden_roots: Iterable[Path] = (),
        *,
        project_root: Path | None = None,
        instructions: str = PROVIDER_INSTRUCTIONS,
    ) -> None:
        if os.getenv("SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise RuntimeError("codex_local_requires_explicit_unsafe_opt_in")

        configured_root = os.getenv("SECURE_RAG_CODEX_SANDBOX_ROOT", "").strip()
        if configured_root:
            root = Path(configured_root).expanduser().resolve()
        else:
            local_app_data = os.getenv("LOCALAPPDATA", "").strip()
            base = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
            root = (base / "SecureRagCodexSandbox").resolve()

        resolved_forbidden_roots = tuple(forbidden.resolve() for forbidden in forbidden_roots)
        for forbidden_root in resolved_forbidden_roots:
            if (
                root == forbidden_root
                or root.is_relative_to(forbidden_root)
                or forbidden_root.is_relative_to(root)
            ):
                raise ValueError("codex_local_sandbox_overlaps_private_root")
        self.sandbox_root = root
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        self.project_root: Path | None = None
        if project_root is not None:
            try:
                resolved_project_root = project_root.resolve(strict=True)
            except OSError:
                raise RuntimeError("codex_local_project_unavailable") from None
            if not resolved_project_root.is_dir():
                raise RuntimeError("codex_local_project_unavailable")
            for forbidden_root in resolved_forbidden_roots:
                if (
                    resolved_project_root == forbidden_root
                    or resolved_project_root.is_relative_to(forbidden_root)
                    or forbidden_root.is_relative_to(resolved_project_root)
                ):
                    raise ValueError("codex_local_project_overlaps_private_root")
            self.project_root = resolved_project_root

        self.model = os.getenv("SECURE_RAG_CODEX_MODEL", "gpt-5.6-luna").strip()
        if not self.model:
            raise ValueError("codex_local_model_missing")

        persist_value = os.getenv("SECURE_RAG_CODEX_PERSIST_THREADS", "0").strip().lower()
        if persist_value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("codex_local_persist_threads_invalid")
        self.persist_threads = persist_value in {"1", "true", "yes"}
        self._request_calls: dict[str, int] = {}
        self._request_threads: dict[str, str] = {}
        self.instructions = instructions

    def answer_payload(self, payload: str, request_id: str) -> str:
        if not payload.strip():
            raise ValueError("sanitized_provider_payload_empty")
        call_number = self._request_calls.get(request_id, 0) + 1
        self._request_calls[request_id] = call_number
        workspace = self.project_root
        if workspace is None:
            workspace = (self.sandbox_root / f"{request_id}-{call_number:03d}").resolve()
            if not workspace.is_relative_to(self.sandbox_root):
                raise ValueError("codex_local_workspace_escape")
            try:
                workspace.mkdir(parents=False, exist_ok=False)
            except OSError:
                raise RuntimeError("codex_local_workspace_create_failed") from None

        try:
            from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

            instructions = (
                f"{self.instructions} "
                "Do not use tools, shell commands, network access, or read files."
            )
            with Codex(CodexConfig(cwd=str(workspace))) as codex:
                thread_options = {
                    "cwd": str(workspace),
                    "model": self.model,
                    "sandbox": Sandbox.read_only,
                    "approval_mode": ApprovalMode.deny_all,
                    "developer_instructions": instructions,
                }
                thread_id = self._request_threads.get(request_id)
                if self.persist_threads and thread_id is not None:
                    # Keep iterative retrieval in one visible conversation. The updated
                    # payload is still complete and remains bound to this request's markers.
                    thread = codex.thread_resume(thread_id, **thread_options)
                else:
                    thread = codex.thread_start(
                        ephemeral=not self.persist_threads,
                        **thread_options,
                    )
                    if self.persist_threads:
                        # One thread per request prevents unrelated marker namespaces from
                        # mixing while leaving an auditable task in the configured project.
                        self._request_threads[request_id] = thread.id
                        thread.set_name(f"Secure RAG {request_id[:8]}")
                result = thread.run(payload)
        except Exception:
            # The local SDK can include payloads and paths in its errors.
            raise RuntimeError("codex_local_provider_failed") from None
        answer = getattr(result, "final_response", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("codex_local_provider_empty_response")
        return answer.strip() + "\n"
