from __future__ import annotations

import os
from typing import Any

from ..prompts import PROVIDER_INSTRUCTIONS


class OpenAIResponsesProvider:
    """Automatic cloud provider through a no-tools Responses API call."""

    name = "responses"

    def __init__(
        self,
        client: Any | None = None,
        *,
        instructions: str = PROVIDER_INSTRUCTIONS,
    ) -> None:
        self.model = os.getenv("SECURE_RAG_OPENAI_MODEL", "gpt-5-mini").strip()
        if not self.model:
            raise ValueError("openai_model_missing")
        self.instructions = instructions

        if client is not None:
            self.client = client
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("openai_api_key_missing")
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                max_retries=0,
            )
        except Exception:
            raise RuntimeError("openai_responses_client_init_failed") from None

    def answer_payload(self, payload: str, request_id: str) -> str:
        if not payload.strip():
            raise ValueError("sanitized_provider_payload_empty")
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=payload,
                metadata={"request_id": request_id},
                store=False,
            )
        except Exception:
            # Provider exception strings can contain request data; expose only a safe code.
            raise RuntimeError("openai_responses_provider_failed") from None
        answer = getattr(response, "output_text", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("openai_responses_provider_empty_response")
        return answer.strip() + "\n"
