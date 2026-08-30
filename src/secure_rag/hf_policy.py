from __future__ import annotations

import os

HF_LOCAL_ONLY_ENV = "SECURE_RAG_HF_LOCAL_ONLY"


def hf_local_files_only() -> bool:
    """Keep Hugging Face loaders offline unless a setup run explicitly opts out."""
    raw = os.getenv(HF_LOCAL_ONLY_ENV)
    if raw is None or not raw.strip():
        return True
    normalized = raw.strip()
    if normalized == "1":
        return True
    if normalized == "0":
        return False
    raise ValueError(f"Invalid boolean environment value: {HF_LOCAL_ONLY_ENV}")
