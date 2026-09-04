from __future__ import annotations

import os

HF_LOCAL_ONLY_ENV = "SECURE_RAG_HF_LOCAL_ONLY"
HF_OFFLINE_ENVIRONMENTS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")


def _enable_library_offline_mode() -> None:
    # Some transitive Hugging Face helpers do not consistently forward
    # local_files_only. Enforce the same boundary at the library level.
    for name in HF_OFFLINE_ENVIRONMENTS:
        os.environ[name] = "1"


def hf_local_files_only() -> bool:
    """Keep Hugging Face loaders offline unless a setup run explicitly opts out."""
    raw = os.getenv(HF_LOCAL_ONLY_ENV)
    if raw is None or not raw.strip():
        _enable_library_offline_mode()
        return True
    normalized = raw.strip()
    if normalized == "1":
        _enable_library_offline_mode()
        return True
    if normalized == "0":
        return False
    raise ValueError(f"Invalid boolean environment value: {HF_LOCAL_ONLY_ENV}")
