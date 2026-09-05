from __future__ import annotations

import sqlite3
from pathlib import Path

from secure_rag.api.chat_state import ChatStateStore, ConversationKey, StateMessage
from secure_rag.domain.models import DocumentSource


def test_chat_state_persists_history_and_retrieval_per_user(tmp_path: Path) -> None:
    path = tmp_path / "state" / "chat-state.sqlite"
    store = ChatStateStore(path)
    alice = ConversationKey("alice", "chat-1")
    bob = ConversationKey("bob", "chat-1")

    store.sync_messages(alice, [StateMessage("user", "вопрос Alice")])
    store.append_assistant(alice, "ответ Alice")
    store.sync_messages(bob, [StateMessage("user", "вопрос Bob")])
    store.record_retrieval(
        alice,
        request_id="request-1",
        model_id="universal-rag",
        question="вопрос Alice",
        sources=(
            DocumentSource(
                citation_refs=("R001",),
                document_id="doc-1",
                path=Path(r"D:\Nextcloud\doc.pdf"),
                file_type="pdf",
                best_score=0.91,
            ),
        ),
    )

    reopened = ChatStateStore(path)
    assert reopened.history(alice) == (
        StateMessage("user", "вопрос Alice"),
        StateMessage("assistant", "ответ Alice"),
    )
    assert reopened.history(bob) == (StateMessage("user", "вопрос Bob"),)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT user_id, chat_id, model_id, sources_json FROM retrieval_runs"
        ).fetchone()
    assert row[:3] == ("alice", "chat-1", "universal-rag")
    assert '"document_id":"doc-1"' in row[3]


def test_single_message_request_extends_existing_state_but_full_history_replaces_it(
    tmp_path: Path,
) -> None:
    store = ChatStateStore(tmp_path / "chat-state.sqlite")
    key = ConversationKey("user", "chat")

    store.sync_messages(key, [StateMessage("user", "one")])
    incremental = store.sync_messages(key, [StateMessage("user", "two")])
    replaced = store.sync_messages(
        key,
        [StateMessage("system", "agent"), StateMessage("user", "edited")],
    )

    assert incremental == (StateMessage("user", "one"), StateMessage("user", "two"))
    assert replaced == (
        StateMessage("system", "agent"),
        StateMessage("user", "edited"),
    )
