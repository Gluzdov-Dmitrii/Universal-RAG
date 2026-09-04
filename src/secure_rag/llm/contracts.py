from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    """Replaceable text-generation boundary used by orchestration."""

    name: str

    def answer_payload(self, payload: str, request_id: str) -> str: ...
