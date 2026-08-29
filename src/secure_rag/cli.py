from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .bridge import BridgeManager
from .config import AppConfig, load_config
from .factory import create_embedder, create_gateway, create_vector_store
from .indexer import Indexer
from .manifest import ManifestStore
from .pipeline import SecureRagPipeline
from .retrieval import Retriever


def _safe_print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _index(config: AppConfig, max_files: int | None) -> int:
    config.ensure_runtime()
    embedder = create_embedder(config)
    with (
        ManifestStore(config.manifest_path) as manifest,
        create_vector_store(config, embedder.dimension) as vector_store,
    ):
        report = Indexer(config, embedder, manifest, vector_store).run(max_files=max_files)
    safe = asdict(report)
    _safe_print(safe)
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

    index_parser = subparsers.add_parser("index", help="Incrementally index local files")
    index_parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Supported files to inspect; use 0 for all",
    )

    ask_parser = subparsers.add_parser("ask", help="Prepare a sanitized Codex request")
    ask_parser.add_argument("question", nargs="?")
    ask_parser.add_argument("--question-file")
    ask_parser.add_argument("--provider", choices=("manual", "stub"), default="manual")
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
        if args.command == "index":
            max_files = None if args.max_files == 0 else args.max_files
            if max_files is not None and max_files < 1:
                raise ValueError("--max-files must be positive or zero")
            return _index(config, max_files)
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
            app_path = Path(__file__).with_name("web_app.py")
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app_path),
                    "--server.address",
                    "127.0.0.1",
                ]
            )
    except Exception as exc:
        _safe_print(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "raw_document_content_printed": False,
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
