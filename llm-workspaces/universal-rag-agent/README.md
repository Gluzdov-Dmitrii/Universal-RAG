# Universal RAG agent workspace

This directory is a deployed copy of the versioned Secure RAG user-agent component. It contains
only agent instructions and no Secure RAG implementation, runtime database, vector storage, or
source documents.

Managed behavior is defined in:

- `AGENTS.md`
- `.cursor/rules/rag-assistant.mdc`
- `.cursor/skills/answering-with-secure-rag/SKILL.md`
- `.cursor/agents/rag-answerer.md`

The canonical template belongs to the secure-rag repository under
`llm-workspaces/universal-rag-agent/`. Deploy or verify this copy with
`scripts/sync-rag-agent-workspace.ps1` from that repository.

The project directory is an instruction and organization boundary, not a filesystem security
boundary. Confidential inference should run with the LLM and Secure RAG backend separated by a
narrow interface and, when required, by different virtual or physical machines.
