from __future__ import annotations

import json
import re
from dataclasses import dataclass

_CONTROL_RE = re.compile(
    r"\A<retrieval_request>(?P<payload>\{.*\})</retrieval_request>\s*\Z",
    flags=re.DOTALL,
)
_CITATION_RE = re.compile(r"^R[0-9]{3}$")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    queries: tuple[str, ...]
    expand_citations: tuple[str, ...]


def parse_retrieval_request(
    text: str,
    *,
    max_queries: int,
    max_query_chars: int = 300,
    max_citations: int = 5,
) -> RetrievalRequest | None:
    """Parse the only provider-controlled command accepted by the local pipeline."""

    match = _CONTROL_RE.fullmatch(text.strip())
    if match is None:
        if "<retrieval_request>" in text or "</retrieval_request>" in text:
            raise ValueError("invalid_retrieval_control_envelope")
        return None
    try:
        payload = json.loads(match.group("payload"))
    except (TypeError, json.JSONDecodeError):
        raise ValueError("invalid_retrieval_control_json") from None
    if not isinstance(payload, dict) or set(payload) - {"queries", "expand_citations"}:
        raise ValueError("invalid_retrieval_control_fields")
    raw_queries = payload.get("queries", [])
    raw_citations = payload.get("expand_citations", [])
    if not isinstance(raw_queries, list) or not isinstance(raw_citations, list):
        raise ValueError("invalid_retrieval_control_lists")
    if len(raw_queries) > max_queries or len(raw_citations) > max_citations:
        raise ValueError("retrieval_control_limit_exceeded")

    queries: list[str] = []
    for value in raw_queries:
        if not isinstance(value, str):
            raise ValueError("invalid_retrieval_query")
        if any(ord(char) < 32 for char in value):
            raise ValueError("invalid_retrieval_query")
        query = " ".join(value.split())
        if not query or len(query) > max_query_chars:
            raise ValueError("invalid_retrieval_query")
        if query not in queries:
            queries.append(query)

    citations: list[str] = []
    for value in raw_citations:
        if not isinstance(value, str) or not _CITATION_RE.fullmatch(value):
            raise ValueError("invalid_retrieval_citation")
        if value not in citations:
            citations.append(value)
    if not queries and not citations:
        raise ValueError("empty_retrieval_control")
    return RetrievalRequest(tuple(queries), tuple(citations))
