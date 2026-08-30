from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Protocol

from .config import IngestionConfig
from .extractors import EXTRACTOR_VERSION, ExtractedDocument

CACHE_SCHEMA_VERSION = 1
CACHEABLE_EXTENSIONS = frozenset({".docx", ".pdf", ".pptx", ".xlsx"})
_DOCUMENT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{64}\Z")

type MemoryCacheKey = tuple[str, str, str, str]


class ExtractedTextCacheBackend(Protocol):
    def allows_extension(self, extension: str) -> bool: ...

    def load(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
    ) -> ExtractedDocument | None: ...

    def store(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
        document: ExtractedDocument,
    ) -> object: ...


def is_cacheable_extension(extension: str) -> bool:
    """Limit duplicated source text to formats that are expensive to extract."""

    return extension.casefold() in CACHEABLE_EXTENSIONS


def extraction_policy_signature(config: IngestionConfig, extension: str) -> str:
    """Identify every setting that can change the cached extraction result."""

    payload = {
        "extractor": EXTRACTOR_VERSION,
        "extension": extension.casefold(),
        "max_extracted_chars": config.max_extracted_chars,
        "max_file_mb": config.max_file_mb,
        "pdf_max_pages": config.pdf_max_pages,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_key(document_id: str, revision: str) -> None:
    if not _DOCUMENT_ID_RE.fullmatch(document_id):
        raise ValueError("Invalid cache document id")
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError("Invalid cache revision")


class ExtractedTextCache:
    """Local, opaque-keyed cache for normalized extracted document text.

    Cache files contain source-derived text and therefore live only below the ignored
    runtime directory. A single gzip envelope is replaced atomically, so readers never
    observe a partially-written metadata/text pair.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def allows_extension(self, extension: str) -> bool:
        return is_cacheable_extension(extension)

    def path_for(self, document_id: str, revision: str) -> Path:
        _validate_key(document_id, revision)
        return self.root / document_id[:2] / document_id / f"{revision}.json.gz"

    def load(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
    ) -> ExtractedDocument | None:
        path = self.path_for(document_id, revision)
        if not path.is_file():
            return None

        # JSON escaping and UTF-8 need at most a small multiple of the character cap.
        # The bound also prevents a locally-corrupted gzip file from expanding without limit.
        expanded_limit = config.max_extracted_chars * 6 + 65_536
        try:
            with gzip.open(path, "rb") as stream:
                encoded = stream.read(expanded_limit + 1)
            if len(encoded) > expanded_limit:
                raise ValueError("cache_payload_too_large")
            payload = json.loads(encoded.decode("utf-8"))
            text = payload["text"]
            if not isinstance(text, str) or not text or len(text) > config.max_extracted_chars:
                raise ValueError("cache_text_invalid")
            expected = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "document_id": document_id,
                "revision": revision,
                "extension": extension.casefold(),
                "policy_signature": extraction_policy_signature(config, extension),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                raise ValueError("cache_metadata_invalid")
            page_count = payload.get("page_count")
            if page_count is not None and (not isinstance(page_count, int) or page_count < 0):
                raise ValueError("cache_page_count_invalid")
            truncated = payload.get("truncated")
            if not isinstance(truncated, bool):
                raise ValueError("cache_truncated_invalid")
            return ExtractedDocument(
                text=text,
                page_count=page_count,
                truncated=truncated,
            )
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            # Invalid cache data is never trusted. Removing it lets the caller regenerate
            # the entry from the verified source without exposing any content in logs.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def store(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
        document: ExtractedDocument,
    ) -> Path:
        if not document.text or len(document.text) > config.max_extracted_chars:
            raise ValueError("Extracted text violates max_extracted_chars")
        path = self.path_for(document_id, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "document_id": document_id,
            "revision": revision,
            "extension": extension.casefold(),
            "policy_signature": extraction_policy_signature(config, extension),
            "text_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            **asdict(document),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = gzip.compress(encoded, compresslevel=6, mtime=0)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".cache-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path


class ProcessMemoryExtractedTextCache:
    """Thread-safe bounded LRU for one application's process lifetime.

    The cache never opens a file and never creates a directory. Its key binds an
    opaque document ID and source revision to the extraction policy. Eviction is
    governed by both entry count and total character count; an entry above the
    per-entry bound is skipped instead of displacing the whole cache.

    This is a performance cache, not an authorization boundary. Callers must verify
    the source revision and access policy before every load.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        max_total_chars: int,
        max_entry_chars: int,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_total_chars < 1:
            raise ValueError("max_total_chars must be positive")
        if max_entry_chars < 1:
            raise ValueError("max_entry_chars must be positive")
        if max_entry_chars > max_total_chars:
            raise ValueError("max_entry_chars must not exceed max_total_chars")
        self.max_entries = max_entries
        self.max_total_chars = max_total_chars
        self.max_entry_chars = max_entry_chars
        self._entries: OrderedDict[MemoryCacheKey, ExtractedDocument] = OrderedDict()
        self._total_chars = 0
        self._lock = RLock()

    def allows_extension(self, extension: str) -> bool:
        return bool(extension)

    def load(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
    ) -> ExtractedDocument | None:
        key = self._key(document_id, revision, extension, config)
        with self._lock:
            document = self._entries.get(key)
            if document is not None:
                self._entries.move_to_end(key)
            return document

    def store(
        self,
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
        document: ExtractedDocument,
    ) -> bool:
        if not document.text or len(document.text) > config.max_extracted_chars:
            raise ValueError("Extracted text violates max_extracted_chars")
        key = self._key(document_id, revision, extension, config)
        size = len(document.text)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_chars -= len(previous.text)
            if size > self.max_entry_chars or size > self.max_total_chars:
                return False
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._total_chars + size > self.max_total_chars
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_chars -= len(evicted.text)
            self._entries[key] = document
            self._total_chars += size
            return True

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_chars(self) -> int:
        with self._lock:
            return self._total_chars

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_chars = 0

    @staticmethod
    def _key(
        document_id: str,
        revision: str,
        extension: str,
        config: IngestionConfig,
    ) -> MemoryCacheKey:
        _validate_key(document_id, revision)
        normalized_extension = extension.casefold()
        return (
            document_id,
            revision,
            normalized_extension,
            extraction_policy_signature(config, normalized_extension),
        )
