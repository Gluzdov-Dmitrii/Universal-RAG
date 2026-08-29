from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from .config import IngestionConfig


class ExtractionError(RuntimeError):
    """An extraction failure carrying a log-safe error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    text: str
    page_count: int | None = None
    truncated: bool = False


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_document_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and blank:
            continue
        normalized.append(line)
        blank = is_blank
    return "\n".join(normalized).strip()


def _read_plain_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("text_encoding")


def _extract_html(path: Path) -> str:
    raw = _read_plain_text(path)
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def _extract_docx(path: Path) -> str:
    document = Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)
        for paragraph in section.footer.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)
    return "\n\n".join(parts)


def _extract_pdf(path: Path, max_pages: int) -> tuple[str, int, bool]:
    reader = PdfReader(path)
    count = len(reader.pages)
    limit = min(count, max_pages)
    pages: list[str] = []
    for page in reader.pages[:limit]:
        pages.append(page.extract_text() or "")
    return "\n\n[PAGE_BREAK]\n\n".join(pages), count, count > limit


def _extract_json(path: Path) -> str:
    raw = _read_plain_text(path)
    value = json.loads(raw)
    return json.dumps(value, ensure_ascii=False, indent=2)


def extract_document(path: Path, config: IngestionConfig) -> ExtractedDocument:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExtractionError("stat_failed") from exc
    if size > config.max_file_mb * 1024 * 1024:
        raise ExtractionError("file_too_large")

    extension = path.suffix.lower()
    try:
        if extension in {".txt", ".md", ".csv", ".xml"}:
            raw = _read_plain_text(path)
            page_count = None
            truncated = False
        elif extension in {".htm", ".html"}:
            raw = _extract_html(path)
            page_count = None
            truncated = False
        elif extension == ".json":
            raw = _extract_json(path)
            page_count = None
            truncated = False
        elif extension == ".docx":
            raw = _extract_docx(path)
            page_count = None
            truncated = False
        elif extension == ".pdf":
            raw, page_count, truncated = _extract_pdf(path, config.pdf_max_pages)
        else:
            raise ExtractionError("unsupported_extension")
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("extract_failed") from exc

    text = normalize_text(raw)
    if not text:
        raise ExtractionError("empty_text")
    if len(text) > config.max_extracted_chars:
        text = text[: config.max_extracted_chars]
        truncated = True
    return ExtractedDocument(text=text, page_count=page_count, truncated=truncated)


def iter_source_files(
    source_root: Path,
    supported_extensions: Iterable[str],
    max_files: int | None = None,
) -> Iterable[Path]:
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Configured source root does not exist")
    supported = {item.lower() for item in supported_extensions}
    yielded = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if max_files is not None and yielded >= max_files:
            break
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(root) or not resolved.is_file():
            continue
        if resolved.suffix.lower() not in supported:
            continue
        yielded += 1
        yield resolved
