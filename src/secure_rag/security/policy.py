from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Fail-closed access policy applied inside vector search."""

    access_group: str
    goz: bool = False
    is_final: bool = True

    def __post_init__(self) -> None:
        if not self.access_group.strip():
            raise ValueError("access_group must not be empty")
