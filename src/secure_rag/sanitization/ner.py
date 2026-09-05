from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import NerModelConfig
from ..domain.models import EntitySpan
from ..infrastructure.huggingface import hf_local_files_only


class SpanDetector(Protocol):
    name: str

    def detect(self, text: str) -> list[EntitySpan]: ...


LABEL_ALIASES = {
    "PERSON": "PER",
    "PERS": "PER",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "LOCATION": "LOC",
}


def normalize_label(label: str) -> str:
    label = label.upper().strip()
    if label.startswith("B-") or label.startswith("I-"):
        label = label[2:]
    return LABEL_ALIASES.get(label, label)


def _pipeline_device(value: str) -> int | str:
    if value != "auto":
        if value == "cpu":
            return -1
        if value.startswith("cuda"):
            return 0 if ":" not in value else int(value.split(":", 1)[1])
        return value
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


class TransformersNerDetector:
    def __init__(self, config: NerModelConfig, cache_root: Path) -> None:
        local_files_only = hf_local_files_only()
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        self.name = config.name
        self.threshold = config.threshold
        self.priority = config.priority
        self.allowed_labels = {
            normalize_label(label) for label in config.allowed_labels if label.strip()
        }
        self.min_chars = config.min_chars
        model_source = (
            str(config.local_path)
            if config.local_path is not None and config.local_path.is_dir()
            else config.model_id
        )
        if model_source != config.model_id:
            required = (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            )
            if any(not (config.local_path / name).is_file() for name in required):
                raise ValueError(f"Local NER bundle is incomplete: {config.name}")
            try:
                from safetensors import safe_open

                with safe_open(config.local_path / "model.safetensors", framework="pt"):
                    pass
            except Exception as exc:
                raise ValueError(f"Local NER bundle is invalid: {config.name}") from exc
        revision = None if model_source != config.model_id else config.revision
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=revision,
            cache_dir=str(cache_root),
            use_fast=True,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        tokenizer.model_max_length = min(int(tokenizer.model_max_length), 512)
        self._tokenizer = tokenizer
        self._window_tokens = max(32, tokenizer.model_max_length - 2)
        self._overlap_tokens = min(64, self._window_tokens // 4)
        model = AutoModelForTokenClassification.from_pretrained(
            model_source,
            revision=revision,
            cache_dir=str(cache_root),
            trust_remote_code=False,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=_pipeline_device(config.device),
        )

    def detect(self, text: str) -> list[EntitySpan]:
        return self.detect_many([text])[0]

    def detect_many(self, texts: list[str]) -> list[list[EntitySpan]]:
        """Run all token windows as GPU batches while preserving field offsets."""

        results: list[dict[tuple[int, int, str], EntitySpan]] = [
            {} for _ in texts
        ]
        windows: list[tuple[int, int, int, str]] = []
        for text_index, text in enumerate(texts):
            if not text.strip():
                continue
            windows.extend(
                (text_index, window_start, window_end, text[window_start:window_end])
                for window_start, window_end in self._character_windows(text)
            )
        if not windows:
            return [[] for _ in texts]
        raw_predictions = self._pipeline(
            [window[3] for window in windows],
            batch_size=16,
        )
        if raw_predictions and isinstance(raw_predictions[0], dict):
            raw_predictions = [raw_predictions]
        for (text_index, window_start, _window_end, _text), predictions in zip(
            windows,
            raw_predictions,
            strict=True,
        ):
            best = results[text_index]
            for prediction in predictions:
                score = float(prediction.get("score", 0.0))
                if score < self.threshold:
                    continue
                start = window_start + int(prediction["start"])
                end = window_start + int(prediction["end"])
                if end <= start:
                    continue
                label = normalize_label(
                    str(
                        prediction.get("entity_group")
                        or prediction.get("entity")
                        or "ENTITY"
                    )
                )
                if self.allowed_labels and label not in self.allowed_labels:
                    continue
                value = text[start:end]
                if sum(character.isalnum() for character in value) < self.min_chars:
                    continue
                span = EntitySpan(
                    start=start,
                    end=end,
                    label=label,
                    score=score,
                    source=self.name,
                    priority=self.priority,
                )
                key = (start, end, label)
                if key not in best or best[key].score < score:
                    best[key] = span
        return [list(best.values()) for best in results]

    def _character_windows(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if int(end) > int(start)
        ]
        if not offsets:
            return [(0, len(text))]
        windows: list[tuple[int, int]] = []
        token_start = 0
        while token_start < len(offsets):
            token_end = min(token_start + self._window_tokens, len(offsets))
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
            windows.append((char_start, char_end))
            if token_end >= len(offsets):
                break
            token_start = token_end - self._overlap_tokens
        return windows


class EnsembleDetector:
    def __init__(self, detectors: list[SpanDetector]) -> None:
        self.detectors = detectors
        self.name = "ensemble"

    def detect(self, text: str) -> list[EntitySpan]:
        return self.detect_many([text])[0]

    def detect_many(self, texts: list[str]) -> list[list[EntitySpan]]:
        spans: list[list[EntitySpan]] = [[] for _ in texts]
        for detector in self.detectors:
            bulk_detect = getattr(detector, "detect_many", None)
            detected = (
                bulk_detect(texts)
                if callable(bulk_detect)
                else [detector.detect(text) for text in texts]
            )
            for target, values in zip(spans, detected, strict=True):
                target.extend(values)
        return spans
