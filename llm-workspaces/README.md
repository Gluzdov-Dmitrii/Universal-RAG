# LLM workspace templates

This directory contains versioned, deployable client workspaces for LLM components. It does
not contain model weights, runtime databases, source documents, or request artifacts.

`universal-rag-agent/` is the canonical source for the provider-side agent workspace. Deploy it with:

```powershell
.\scripts\sync-rag-agent-workspace.ps1 -Mode Push `
    -TargetRoot C:\Services\Universal-RAG-Agent
```

Use `-Mode Check` in CI or before a demo to detect drift. The sync operation overwrites only
files listed in the template manifest and never deletes additional target files.
