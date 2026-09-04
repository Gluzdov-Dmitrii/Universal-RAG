from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document
from lxml import etree, html
from openpyxl import load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader

from ..config import IngestionConfig
from ..domain.models import TextLocation

EXTRACTOR_VERSION = "extractors-v4-lxml-html"
GENERATED_WEBHELP_DIRECTORIES = frozenset({"whdata", "whgdata", "whxdata"})


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
    locations: tuple[TextLocation, ...] = ()


# OOXML files are ZIP containers. Libraries stream worksheet cells where possible, but
# they still need to parse the archive directory and selected XML parts. These bounds
# reject pathological containers before openpyxl/python-pptx see them. They intentionally
# sit below the configured broad source-file cap.
_OOXML_MAX_ENTRIES = 10_000
_OOXML_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_OOXML_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_OOXML_MAX_COMPRESSION_RATIO = 250
_XLSX_MAX_VISITED_CELLS = 2_000_000
_XLSX_MAX_VISITED_ROWS = 250_000
_DOCX_MAX_APP_PROPERTIES_BYTES = 1024 * 1024
_DOCX_MAX_SAVED_PAGES = 10_000
_PARSER_DIAGNOSTIC_LOCK = threading.RLock()


@contextmanager
def _suppress_parser_diagnostics(logger_namespace: str) -> Iterator[None]:
    """Keep source-derived third-party diagnostics out of structured CLI output.

    pypdf logs recoverable parser problems and openpyxl emits warnings while still
    returning useful text. Their messages can quote document content, so they must not
    share stdout/stderr with the indexer's JSONL progress stream. The lock keeps the
    temporary process-wide warning and logger changes well-defined if extraction later
    becomes concurrent.
    """

    with _PARSER_DIAGNOSTIC_LOCK:
        namespace_logger = logging.getLogger(logger_namespace)
        previous_handlers = namespace_logger.handlers[:]
        previous_propagate = namespace_logger.propagate
        known_children = [
            logger
            for name, logger in logging.Logger.manager.loggerDict.items()
            if name.startswith(f"{logger_namespace}.")
            and isinstance(logger, logging.Logger)
        ]
        previous_disabled = [(logger, logger.disabled) for logger in known_children]

        # The NullHandler also prevents logging.lastResort from writing to stderr.
        namespace_logger.handlers = [logging.NullHandler()]
        namespace_logger.propagate = False
        for logger, _ in previous_disabled:
            logger.disabled = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            for logger, disabled in previous_disabled:
                logger.disabled = disabled
            namespace_logger.handlers = previous_handlers
            namespace_logger.propagate = previous_propagate


class _BoundedParts:
    """Collect extraction output without retaining more than the configured limit."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.parts: list[str] = []
        self.length = 0
        self.truncated = False

    def add(self, value: str, separator: str = "\n") -> bool:
        if not value:
            return True
        prefix = separator if self.parts else ""
        remaining = self.max_chars - self.length
        combined = prefix + value
        if len(combined) > remaining:
            if remaining > 0:
                self.parts.append(combined[:remaining])
                self.length += remaining
            self.truncated = True
            return False
        self.parts.append(combined)
        self.length += len(combined)
        return True

    def text(self) -> str:
        return "".join(self.parts)


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
    # Some damaged PDFs expose lone UTF-16 surrogate code points through pypdf.
    # Rust-backed Hugging Face tokenizers and UTF-8 artifact writers reject them.
    # U+FFFD is a one-code-point replacement, so chunk offsets remain stable.
    text = re.sub(r"[\ud800-\udfff]", "\ufffd", text)
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
    parser = html.HTMLParser(no_network=True, recover=True)
    document = html.document_fromstring(raw, parser=parser)
    # Keep tail text after an excluded element: in
    # ``visible<script>hidden</script>visible`` both visible fragments matter.
    etree.strip_elements(
        document,
        "script",
        "style",
        "noscript",
        with_tail=False,
    )
    return "\n".join(document.itertext())


def _docx_saved_page_count(path: Path) -> int | None:
    """Read Word's last saved page count without claiming current-layout accuracy."""

    namespace = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"
    try:
        with ZipFile(path) as archive:
            info = archive.getinfo("docProps/app.xml")
            if info.file_size > _DOCX_MAX_APP_PROPERTIES_BYTES:
                return None
            payload = archive.read("docProps/app.xml")
        parser = etree.XMLParser(
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            recover=False,
            huge_tree=False,
        )
        root = etree.fromstring(payload, parser=parser)
        element = root.find(f"{namespace}Pages")
        if element is None or element.text is None:
            return None
        value = int(element.text.strip())
        return value if 1 <= value <= _DOCX_MAX_SAVED_PAGES else None
    except (BadZipFile, KeyError, OSError, ValueError, etree.XMLSyntaxError):
        return None


def _extract_docx(path: Path, max_chars: int) -> tuple[str, int | None, bool]:
    _validate_ooxml_archive(path)
    page_count = _docx_saved_page_count(path)
    document = Document(path)
    output = _BoundedParts(max_chars)
    for paragraph in document.paragraphs:
        if paragraph.text.strip() and not output.add(paragraph.text, separator="\n\n"):
            return output.text(), page_count, True
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells) and not output.add("\t".join(cells), separator="\n\n"):
                return output.text(), page_count, True
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            if paragraph.text.strip() and not output.add(
                paragraph.text, separator="\n\n"
            ):
                return output.text(), page_count, True
        for paragraph in section.footer.paragraphs:
            if paragraph.text.strip() and not output.add(
                paragraph.text, separator="\n\n"
            ):
                return output.text(), page_count, True
    return output.text(), page_count, output.truncated


def _extract_pdf(
    path: Path,
    max_pages: int,
    max_chars: int,
) -> tuple[str, int, bool, tuple[int, ...]]:
    reader = PdfReader(path)
    count = len(reader.pages)
    limit = min(count, max_pages)
    output = _BoundedParts(max_chars)
    included_pages: list[int] = []
    for page_number, page in enumerate(reader.pages[:limit], start=1):
        parts_before = len(output.parts)
        if not output.add(
            page.extract_text() or "",
            separator="\n\n[PAGE_BREAK]\n\n",
        ):
            if len(output.parts) > parts_before:
                included_pages.append(page_number)
            break
        if len(output.parts) > parts_before:
            included_pages.append(page_number)
    return (
        output.text(),
        count,
        count > limit or output.truncated,
        tuple(included_pages),
    )


def _marker_locations(
    text: str,
    pattern: re.Pattern[str],
    kind: str,
) -> tuple[TextLocation, ...]:
    """Turn stable extraction markers into offsets in the normalized text."""

    markers = list(pattern.finditer(text))
    return tuple(
        TextLocation(
            start=marker.start(),
            end=(markers[index + 1].start() if index + 1 < len(markers) else len(text)),
            kind=kind,
            value=marker.group(1).strip(),
        )
        for index, marker in enumerate(markers)
        if marker.start() < len(text)
    )


def _pdf_locations(text: str, page_numbers: tuple[int, ...]) -> tuple[TextLocation, ...]:
    """Map the unchanged PAGE_BREAK extraction text back to physical PDF pages."""

    page_breaks = list(re.finditer(r"(?m)^\[PAGE_BREAK\]$", text))
    boundaries = [0, *[marker.end() for marker in page_breaks], len(text)]
    return tuple(
        TextLocation(
            start=boundaries[index],
            end=boundaries[index + 1],
            kind="page",
            value=str(page_number),
        )
        for index, page_number in enumerate(page_numbers)
        if index + 1 < len(boundaries) and boundaries[index + 1] > boundaries[index]
    )


def _approximate_page_locations(
    text: str,
    page_count: int | None,
) -> tuple[TextLocation, ...]:
    """Estimate DOCX pages from Word's saved count and normalized text position."""

    if not text or page_count is None or page_count < 1:
        return ()
    effective_count = min(page_count, len(text))
    return tuple(
        TextLocation(
            start=(len(text) * index) // effective_count,
            end=(len(text) * (index + 1)) // effective_count,
            kind="approx_page",
            value=str((page_count * index) // effective_count + 1),
        )
        for index in range(effective_count)
    )


def _validate_ooxml_archive(path: Path) -> None:
    """Reject malformed or unexpectedly expansive OOXML ZIP containers."""

    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > _OOXML_MAX_ENTRIES:
                raise ExtractionError("office_archive_too_many_entries")

            total_uncompressed = 0
            for member in members:
                # We never extract members to disk, but rejecting traversal-style names
                # also keeps the accepted OOXML shape narrow and auditable.
                normalized_name = member.filename.replace("\\", "/")
                path_parts = normalized_name.split("/")
                if (
                    normalized_name.startswith("/")
                    or any(part == ".." for part in path_parts)
                    or "\x00" in normalized_name
                ):
                    raise ExtractionError("office_archive_unsafe_name")
                if member.flag_bits & 0x1:
                    raise ExtractionError("office_archive_encrypted")
                if member.file_size > _OOXML_MAX_MEMBER_BYTES:
                    raise ExtractionError("office_archive_member_too_large")
                total_uncompressed += member.file_size
                if total_uncompressed > _OOXML_MAX_UNCOMPRESSED_BYTES:
                    raise ExtractionError("office_archive_too_large")
                if member.file_size and (
                    member.compress_size == 0
                    or member.file_size / member.compress_size > _OOXML_MAX_COMPRESSION_RATIO
                ):
                    raise ExtractionError("office_archive_suspicious_ratio")
    except ExtractionError:
        raise
    except (BadZipFile, OSError) as exc:
        raise ExtractionError("office_archive_invalid") from exc


def _format_cell_value(value: object) -> str | None:
    """Render only scalar values returned by openpyxl in a stable text form."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        if value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(value)
    if isinstance(value, str):
        # OOXML itself rejects these controls. Stripping them here also covers
        # workbooks produced by non-conforming generators.
        cleaned = ILLEGAL_CHARACTERS_RE.sub("", value)
        return cleaned if cleaned.strip() else None
    return None


def _extract_xlsx(path: Path, max_chars: int) -> tuple[str, bool]:
    _validate_ooxml_archive(path)
    output = _BoundedParts(max_chars)
    visited_cells = 0
    visited_rows = 0
    workbook = load_workbook(
        filename=path,
        read_only=True,
        data_only=True,
        keep_links=False,
        rich_text=False,
    )
    try:
        for worksheet in workbook.worksheets:
            if not output.add(f"[SHEET] {worksheet.title}", separator="\n\n"):
                break
            for row in worksheet.iter_rows():
                visited_rows += 1
                visited_cells += len(row)
                if (
                    visited_rows > _XLSX_MAX_VISITED_ROWS
                    or visited_cells > _XLSX_MAX_VISITED_CELLS
                ):
                    output.truncated = True
                    break
                values = [
                    formatted
                    for cell in row
                    if (formatted := _format_cell_value(cell.value)) is not None
                ]
                if values and not output.add("\t".join(values)):
                    break
            if output.truncated:
                break
    finally:
        workbook.close()
    return output.text(), output.truncated


def _iter_pptx_shape_text(shape: object) -> Iterator[str]:
    if getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
        for child in getattr(shape, "shapes", ()):
            yield from _iter_pptx_shape_text(child)
        return
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            values = [normalize_text(cell.text) for cell in row.cells]
            if any(values):
                yield "\t".join(values)
        return
    if getattr(shape, "has_text_frame", False):
        value = normalize_text(shape.text)
        if value:
            yield value


def _extract_pptx(path: Path, max_chars: int) -> tuple[str, int, bool]:
    _validate_ooxml_archive(path)
    presentation = Presentation(path)
    output = _BoundedParts(max_chars)
    slide_count = len(presentation.slides)
    for slide_number, slide in enumerate(presentation.slides, start=1):
        if not output.add(f"[SLIDE {slide_number}]", separator="\n\n"):
            break
        for value in (
            text for shape in slide.shapes for text in _iter_pptx_shape_text(shape)
        ):
            if not output.add(value):
                break
        if output.truncated:
            break
        if slide.has_notes_slide:
            notes = normalize_text(slide.notes_slide.notes_text_frame.text)
            if notes:
                if not output.add("[NOTES]") or not output.add(notes):
                    break
    return output.text(), slide_count, output.truncated


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
    pdf_page_numbers: tuple[int, ...] = ()
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
            raw, page_count, truncated = _extract_docx(
                path,
                config.max_extracted_chars,
            )
        elif extension == ".xlsx":
            with _suppress_parser_diagnostics("openpyxl"):
                raw, truncated = _extract_xlsx(path, config.max_extracted_chars)
            page_count = None
        elif extension == ".pptx":
            raw, page_count, truncated = _extract_pptx(
                path,
                config.max_extracted_chars,
            )
        elif extension == ".pdf":
            with _suppress_parser_diagnostics("pypdf"):
                raw, page_count, truncated, pdf_page_numbers = _extract_pdf(
                    path,
                    config.pdf_max_pages,
                    config.max_extracted_chars,
                )
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
    if extension == ".docx":
        locations = _approximate_page_locations(text, page_count)
    elif extension == ".pdf":
        locations = _pdf_locations(text, pdf_page_numbers)
    elif extension == ".pptx":
        locations = _marker_locations(
            text,
            re.compile(r"(?m)^\[SLIDE (\d+)\]$"),
            "slide",
        )
    elif extension == ".xlsx":
        locations = _marker_locations(
            text,
            re.compile(r"(?m)^\[SHEET\] (.+)$"),
            "sheet",
        )
    else:
        locations = ()
    return ExtractedDocument(
        text=text,
        page_count=page_count,
        truncated=truncated,
        locations=locations,
    )


def iter_source_files(
    source_root: Path,
    supported_extensions: Iterable[str],
    max_files: int | None = None,
) -> Iterable[Path]:
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Configured source root does not exist")
    supported = {item.lower() for item in supported_extensions}
    candidates: list[Path] = []

    # Nextcloud can remove placeholders while a scan is in progress. os.walk's
    # onerror hook lets the inventory skip that directory and continue instead
    # of aborting the entire build before the first document is processed.
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=lambda _error: None,
        followlinks=False,
    ):
        directory_names[:] = sorted(
            (
                name
                for name in directory_names
                if name.casefold() not in GENERATED_WEBHELP_DIRECTORIES
            ),
            key=str.casefold,
        )
        for file_name in sorted(file_names, key=str.casefold):
            candidate = Path(directory, file_name)
            try:
                resolved = candidate.resolve(strict=True)
                is_file = resolved.is_file()
            except OSError:
                continue
            if not is_file or not resolved.is_relative_to(root):
                continue
            relative = resolved.relative_to(root)
            if any(
                part.casefold() in GENERATED_WEBHELP_DIRECTORIES
                for part in relative.parts[:-1]
            ):
                continue
            if resolved.suffix.lower() in supported:
                candidates.append(resolved)

    for yielded, candidate in enumerate(
        sorted(candidates, key=lambda item: str(item).casefold())
    ):
        if max_files is not None and yielded >= max_files:
            break
        yield candidate
