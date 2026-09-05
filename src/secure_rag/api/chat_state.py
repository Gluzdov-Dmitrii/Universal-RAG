from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from secure_rag.domain.models import DocumentSource

_SCHEMA_VERSION = 1
_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True, slots=True)
class ConversationKey:
    user_id: str
    chat_id: str


@dataclass(frozen=True, slots=True)
class StateMessage:
    role: str
    content: str


class ChatStateStore:
    """Durable, server-side conversation and retrieval state owned by one backend."""

    def __init__(
        self,
        path: Path,
        *,
        max_messages: int = 100,
        max_total_chars: int = 500_000,
    ) -> None:
        if max_messages < 1 or max_total_chars < 1:
            raise ValueError("chat_state_limits_must_be_positive")
        self.path = path
        self.max_messages = max_messages
        self.max_total_chars = max_total_chars
        self._schema_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    self._create_schema(connection)
                    self._schema_ready = True
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS chat_sessions (
                user_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, chat_id)
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                user_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, chat_id, position),
                FOREIGN KEY (user_id, chat_id)
                    REFERENCES chat_sessions (user_id, chat_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS retrieval_runs (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                question TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id, chat_id)
                    REFERENCES chat_sessions (user_id, chat_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_retrieval_runs_scope_created
                ON retrieval_runs (user_id, chat_id, created_at DESC);
            """
        )
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version not in {0, _SCHEMA_VERSION}:
            raise RuntimeError(f"unsupported_chat_state_schema:{current_version}")
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        connection.commit()

    def sync_messages(
        self,
        key: ConversationKey,
        messages: Iterable[StateMessage],
    ) -> tuple[StateMessage, ...]:
        with self._write_lock:
            return self._sync_messages(key, messages)

    def _sync_messages(
        self,
        key: ConversationKey,
        messages: Iterable[StateMessage],
    ) -> tuple[StateMessage, ...]:
        incoming = self._trim(messages)
        now = int(time.time())
        with self._connect() as connection:
            existing = self._read_messages(connection, key)
            # Open WebUI normally sends its complete transcript. A one-message request is
            # treated as an incremental client so reconnects can still use persisted state.
            if len(incoming) == 1 and existing:
                merged = existing if incoming[0] == existing[-1] else (*existing, incoming[0])
                normalized = self._trim(merged)
            else:
                normalized = incoming
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    user_id, chat_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, chat_id)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (*self._scope(key), now, now),
            )
            connection.execute(
                """
                DELETE FROM chat_messages
                WHERE user_id = ? AND chat_id = ?
                """,
                self._scope(key),
            )
            connection.executemany(
                """
                INSERT INTO chat_messages (
                    user_id, chat_id, position, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (*self._scope(key), position, message.role, message.content, now)
                    for position, message in enumerate(normalized)
                ),
            )
        return normalized

    def append_assistant(self, key: ConversationKey, content: str) -> None:
        message = StateMessage("assistant", content.strip())
        if not message.content:
            return
        with self._write_lock:
            with self._connect() as connection:
                existing = self._read_messages(connection, key)
            self._sync_messages(key, (*existing, message))

    def history(self, key: ConversationKey) -> tuple[StateMessage, ...]:
        with self._connect() as connection:
            return self._read_messages(connection, key)

    def record_retrieval(
        self,
        key: ConversationKey,
        *,
        request_id: str,
        model_id: str,
        question: str,
        sources: Sequence[DocumentSource],
    ) -> None:
        serialized_sources = json.dumps(
            [
                {
                    "document_id": source.document_id,
                    "citation_refs": list(source.citation_refs),
                    "path": str(source.path),
                    "file_type": source.file_type,
                    "best_score": source.best_score,
                }
                for source in sources
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._write_lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO retrieval_runs (
                        request_id, user_id, chat_id, model_id,
                        question, sources_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        *self._scope(key),
                        model_id,
                        question,
                        serialized_sources,
                        int(time.time()),
                    ),
                )

    def _trim(self, messages: Iterable[StateMessage]) -> tuple[StateMessage, ...]:
        normalized = tuple(
            StateMessage(message.role, message.content.strip())
            for message in messages
            if message.role in _ROLES and message.content.strip()
        )
        selected: list[StateMessage] = []
        total_chars = 0
        for message in reversed(normalized[-self.max_messages :]):
            remaining = self.max_total_chars - total_chars
            if remaining <= 0:
                break
            content = message.content[-remaining:]
            selected.append(StateMessage(message.role, content))
            total_chars += len(content)
        return tuple(reversed(selected))

    @staticmethod
    def _scope(key: ConversationKey) -> tuple[str, str]:
        return key.user_id, key.chat_id

    @classmethod
    def _read_messages(
        cls,
        connection: sqlite3.Connection,
        key: ConversationKey,
    ) -> tuple[StateMessage, ...]:
        rows = connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE user_id = ? AND chat_id = ?
            ORDER BY position
            """,
            cls._scope(key),
        ).fetchall()
        return tuple(StateMessage(str(row["role"]), str(row["content"])) for row in rows)
