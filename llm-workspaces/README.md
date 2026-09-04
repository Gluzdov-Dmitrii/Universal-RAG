# LLM workspace templates

This directory contains versioned, deployable client workspaces for LLM components. It does
not contain model weights, runtime databases, source documents, or request artifacts.

`rag-test/` is the canonical source for the user-facing Secure RAG agent. Deploy it with:

```powershell
.\scripts\sync-rag-agent-workspace.ps1 -Mode Push `
    -TargetRoot C:\Dev\LLM\Codex_RAG_Test
```

Use `-Mode Check` in CI or before a demo to detect drift. The sync operation overwrites only
files listed in the template manifest and never deletes additional target files.
