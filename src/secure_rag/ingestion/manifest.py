from __future__ import annotations

import sqlite3
from pathlib import Path

from ..domain.models import ChunkLocation, ChunkRecord, DocumentRecord


class ManifestStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                extension TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                revision TEXT NOT NULL,
                status TEXT NOT NULL,
                indexed_revision TEXT,
                index_signature TEXT,
                error_code TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                revision TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                location_kind TEXT NOT NULL DEFAULT '',
                location_start TEXT,
                location_end TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_chunks_document
                ON chunks(document_id, revision);
            CREATE TABLE IF NOT EXISTS builds (
                build_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                status TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                config_version TEXT NOT NULL,
                indexed_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "index_signature" not in columns:
            self.connection.execute("ALTER TABLE documents ADD COLUMN index_signature TEXT")
        chunk_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "location_kind" not in chunk_columns:
            self.connection.execute(
                "ALTER TABLE chunks ADD COLUMN location_kind TEXT NOT NULL DEFAULT ''"
            )
        if "location_start" not in chunk_columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN location_start TEXT")
        if "location_end" not in chunk_columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN location_end TEXT")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ManifestStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            document_id=row["document_id"],
            source_path=Path(row["source_path"]),
            relative_path=row["relative_path"],
            extension=row["extension"],
            size=row["size"],
            mtime_ns=row["mtime_ns"],
            revision=row["revision"],
            status=row["status"],
            indexed_revision=row["indexed_revision"],
            index_signature=row["index_signature"],
        )

    def upsert_document(self, record: DocumentRecord, error_code: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT INTO documents (
                document_id, source_path, relative_path, extension, size, mtime_ns,
                revision, status, indexed_revision, error_code
                , index_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                source_path=excluded.source_path,
                relative_path=excluded.relative_path,
                extension=excluded.extension,
                size=excluded.size,
                mtime_ns=excluded.mtime_ns,
                revision=excluded.revision,
                status=excluded.status,
                indexed_revision=excluded.indexed_revision,
                index_signature=excluded.index_signature,
                error_code=excluded.error_code,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                record.document_id,
                str(record.source_path),
                record.relative_path,
                record.extension,
                record.size,
                record.mtime_ns,
                record.revision,
                record.status,
                record.indexed_revision,
                error_code,
                record.index_signature,
            ),
        )
        self.connection.commit()

    def old_chunk_ids(self, document_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def adjacent_chunks(
        self,
        document_id: str,
        revision: str,
        ordinal: int,
        radius: int,
    ) -> list[ChunkLocation]:
        if radius < 0:
            raise ValueError("radius must not be negative")
        rows = self.connection.execute(
            """
            SELECT chunk_id, document_id, revision, ordinal, char_start, char_end,
                   location_kind, location_start, location_end
            FROM chunks
            WHERE document_id = ? AND revision = ? AND ordinal BETWEEN ? AND ?
            ORDER BY ordinal
            """,
            (document_id, revision, ordinal - radius, ordinal + radius),
        ).fetchall()
        return [
            ChunkLocation(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                revision=str(row["revision"]),
                ordinal=int(row["ordinal"]),
                start=int(row["char_start"]),
                end=int(row["char_end"]),
                location_kind=str(row["location_kind"] or ""),
                location_start=(
                    str(row["location_start"])
                    if row["location_start"] is not None
                    else None
                ),
                location_end=(
                    str(row["location_end"])
                    if row["location_end"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def all_documents(self) -> list[DocumentRecord]:
        rows = self.connection.execute("SELECT document_id FROM documents").fetchall()
        result: list[DocumentRecord] = []
        for row in rows:
            record = self.get_document(str(row["document_id"]))
            if record is not None:
                result.append(record)
        return result

    def replace_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, revision, ordinal, char_start, char_end,
                    location_kind, location_start, location_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.revision,
                        chunk.ordinal,
                        chunk.start,
                        chunk.end,
                        chunk.location_kind,
                        chunk.location_start,
                        chunk.location_end,
                    )
                    for chunk in chunks
                ],
            )

    def start_build(self, build_id: str, embedding_version: str, config_version: str) -> None:
        # Indexer holds the process-wide build FileLock before calling this method.
        # Therefore any older row still marked as running belongs to a process that
        # terminated without reaching finish_build (for example after power loss).
        # Closing those rows in the same transaction as the new row keeps build
        # history truthful without treating a merely persistent lock file as active.
        with self.connection:
            self.connection.execute(
                """
                UPDATE builds
                SET finished_at=CURRENT_TIMESTAMP, status='interrupted'
                WHERE status='running'
                """
            )
            self.connection.execute(
                """
                INSERT INTO builds (build_id, status, embedding_version, config_version)
                VALUES (?, 'running', ?, ?)
                """,
                (build_id, embedding_version, config_version),
            )

    def finish_build(
        self,
        build_id: str,
        status: str,
        indexed_count: int,
        failed_count: int,
        chunk_count: int,
    ) -> None:
        self.connection.execute(
            """
            UPDATE builds
            SET finished_at=CURRENT_TIMESTAMP, status=?, indexed_count=?,
                failed_count=?, chunk_count=?
            WHERE build_id=?
            """,
            (status, indexed_count, failed_count, chunk_count, build_id),
        )
        self.connection.commit()

    def stats(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
        ).fetchall()
        result = {f"documents_{row['status']}": int(row["count"]) for row in rows}
        chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        result["chunks"] = int(chunk_count)
        return result

    def indexed_chunk_count(self, index_signature: str) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks AS c
            JOIN documents AS d ON d.document_id = c.document_id
            WHERE d.status = 'indexed'
              AND d.index_signature = ?
              AND d.indexed_revision = d.revision
            """,
            (index_signature,),
        ).fetchone()
        return int(row[0])

    def latest_build_id(self) -> str | None:
        row = self.connection.execute(
            """
            SELECT build_id FROM builds
            WHERE status IN ('complete', 'complete_with_errors')
            ORDER BY finished_at DESC
            LIMIT 1
            """
        ).fetchone()
        return None if row is None else str(row["build_id"])
