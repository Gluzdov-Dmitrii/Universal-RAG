from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from ..composition import create_embedder, create_gateway, create_vector_store
from ..config import AppConfig, load_config
from ..domain.models import IndexProgress
from ..ingestion.indexer import Indexer
from ..ingestion.manifest import ManifestStore
from ..llm.bridge import BridgeManager
from ..orchestration.pipeline import SecureRagPipeline
from ..retrieval.service import Retriever

_SAFE_ERROR_CODE_RE = re.compile(r"^[a-z0-9_]{1,80}$")
WEB_APP_PATH = Path(__file__).with_name("web.py")


def _safe_print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _safe_jsonl(value: dict[str, object]) -> None:
    """Emit one machine-readable record without source-derived content."""

    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def _safe_progress(event: IndexProgress) -> None:
    _safe_jsonl(asdict(event))


def _index(
    config: AppConfig,
    max_files: int | None,
    progress_every_files: int = 25,
    progress_every_seconds: float = 10.0,
    prune_missing: bool = False,
) -> int:
    config.ensure_runtime()
    embedder = create_embedder(config)
    with (
        ManifestStore(config.manifest_path) as manifest,
        create_vector_store(config, embedder.dimension) as vector_store,
    ):
        report = Indexer(config, embedder, manifest, vector_store).run(
            max_files=max_files,
            progress=_safe_progress,
            progress_every_files=progress_every_files,
            progress_every_seconds=progress_every_seconds,
            prune_missing=prune_missing,
        )
    safe = asdict(report)
    # Index progress is commonly redirected to a .jsonl audit log. Keep the final
    # report on one line as well, so every emitted line can be parsed independently.
    _safe_jsonl(safe)
    return 0 if report.failed == 0 else 2


def _ask(
    config: AppConfig,
    question: str,
    provider: str,
    ner_mode: str,
    top_k: int | None,
) -> int:
    config.ensure_runtime()
    embedder = create_embedder(config)
    gateway = create_gateway(config, mode=ner_mode)
    with (
        ManifestStore(config.manifest_path) as manifest,
        create_vector_store(config, embedder.dimension) as vector_store,
    ):
        retriever = Retriever(config, embedder, manifest, vector_store)
        result = SecureRagPipeline(config, retriever, gateway, manifest).run(
            question,
            provider=provider,
            top_k=top_k,
        )
    _safe_print(
        {
            "request_id": result.request_id,
            "provider": result.provider,
            "retrieved_count": result.retrieved_count,
            "marker_count": result.marker_count,
            "codex_input": str(result.codex_input),
            "codex_output": str(result.codex_output),
            "restored_output": (
                str(result.restored_output) if result.restored_output.exists() else None
            ),
            "sources_path": str(result.sources_path),
            "sources": [
                {
                    "citation_refs": list(source.citation_refs),
                    "path": str(source.path),
                    "file_type": source.file_type,
                    "best_score": source.best_score,
                    "locations": [asdict(location) for location in source.locations],
                }
                for source in result.sources
            ],
            "iterations": result.iterations,
            "raw_content_printed": False,
        }
    )
    return 0


def _doctor(config: AppConfig) -> int:
    config.ensure_runtime()
    source_exists = config.paths.source_root.is_dir()
    with ManifestStore(config.manifest_path) as manifest:
        stats = manifest.stats()
    _safe_print(
        {
            "config_schema": config.schema_version,
            "source_root_exists": source_exists,
            "runtime_writable": config.paths.runtime_root.is_dir(),
            "python": sys.version.split()[0],
            "embedding_model": (
                f"{config.embedding.model_id}@{config.embedding.revision}"
            ),
            "embedding_dimension": config.embedding.dimension,
            "qdrant_mode": config.qdrant.mode,
            "manifest": stats,
            "checks_do_not_load_models": True,
        }
    )
    return 0 if source_exists else 2


def _prepare_models(config: AppConfig, ner_mode: str) -> int:
    """Download or validate all model artifacts without indexing source documents."""

    config.ensure_runtime()
    embedder = create_embedder(config)
    create_gateway(config, mode=ner_mode)
    _safe_print(
        {
            "status": "models_ready",
            "embedding_model": embedder.model_version,
            "embedding_dimension": embedder.dimension,
            "ner_mode": ner_mode,
            "model_cache": str(config.model_cache_path),
            "source_documents_read": False,
        }
    )
    return 0


def _read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if args.question:
        return str(args.question)
    raise ValueError("Use a positional question or --question-file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secure-rag",
        description="Local RAG → sanitizer → Codex file bridge → demarker",
    )
    parser.add_argument("--config", default=None, help="Path to pilot.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check paths and local manifest without loading models")

    models_parser = subparsers.add_parser(
        "prepare-models",
        help="Download or validate embedding and NER models without indexing documents",
    )
    models_parser.add_argument(
        "--ner",
        choices=("all", "regex", "legal", "collection3"),
        default="all",
    )

    index_parser = subparsers.add_parser("index", help="Incrementally index local files")
    index_parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Supported files to inspect; use 0 for all",
    )
    index_parser.add_argument(
        "--progress-every-files",
        type=int,
        default=25,
        help="Emit aggregate progress after this many inspected files",
    )
    index_parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=10.0,
        help="Emit aggregate progress after this many seconds (checked between files)",
    )
    index_parser.add_argument(
        "--prune-missing",
        action="store_true",
        help=(
            "Delete points for unseen documents after a complete authoritative scan; "
            "leave disabled for network or partially available sources"
        ),
    )

    ask_parser = subparsers.add_parser("ask", help="Prepare a sanitized Codex request")
    ask_parser.add_argument("question", nargs="?")
    ask_parser.add_argument("--question-file")
    ask_parser.add_argument(
        "--provider",
        choices=("auto", "responses", "codex-local", "manual", "stub"),
        default="auto",
    )
    ask_parser.add_argument(
        "--ner",
        choices=("all", "regex", "legal", "collection3"),
        default="all",
    )
    ask_parser.add_argument("--top-k", type=int, default=None)

    demark_parser = subparsers.add_parser("demark", help="Restore a marked Codex response")
    demark_parser.add_argument("request_id")
    demark_parser.add_argument("--input", type=Path, default=None)

    subparsers.add_parser("serve", help="Start the local Streamlit chat")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            return _doctor(config)
        if args.command == "prepare-models":
            return _prepare_models(config, args.ner)
        if args.command == "index":
            max_files = None if args.max_files == 0 else args.max_files
            if max_files is not None and max_files < 1:
                raise ValueError("--max-files must be positive or zero")
            return _index(
                config,
                max_files,
                args.progress_every_files,
                args.progress_every_seconds,
                args.prune_missing,
            )
        if args.command == "ask":
            return _ask(
                config,
                _read_question(args),
                provider=args.provider,
                ner_mode=args.ner,
                top_k=args.top_k,
            )
        if args.command == "demark":
            output = BridgeManager(config).restore_file(args.request_id, args.input)
            _safe_print({"restored_output": str(output), "content_printed": False})
            return 0
        if args.command == "serve":
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(WEB_APP_PATH),
                    "--server.address",
                    "127.0.0.1",
                ]
            )
    except Exception as exc:
        _safe_print(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_code": (
                    str(exc) if _SAFE_ERROR_CODE_RE.fullmatch(str(exc)) else "operation_failed"
                ),
                "raw_document_content_printed": False,
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
