# Qdrant server

Qdrant runs in Docker, while the Python/Streamlit application runs on Windows.
Only the REST port is exposed and only on localhost. Vector data persists in the
Docker named volume `secure_rag_qdrant_storage`.

## One-command local launcher

Run these commands from the repository root:

```powershell
# Start Docker Desktop if necessary, start/health-check Qdrant, then start Streamlit.
.\scripts\start-local.ps1

# Show app/Qdrant health, tracked Streamlit PID and current log files.
.\scripts\status-local.ps1

# Stop only the native Streamlit process. Qdrant remains available.
.\scripts\stop-local.ps1

# Stop Streamlit and Qdrant, preserving the Docker named volume and vector index.
.\scripts\stop-local.ps1 -StopQdrant
```

The launcher is idempotent. It stores only process metadata in
`runtime/run/pids/streamlit.json` and writes Streamlit stdout/stderr to timestamped files in
`runtime/diagnostics/logs`. Before stopping a PID, the stop script verifies the executable and full
Streamlit app path. A process on port 8501 that was not started by this launcher is
reported but never stopped.

After a Windows restart, run `start-local.ps1` again. OS auto-start is intentionally not
installed by this prototype; it can be added later as an explicit Task Scheduler step.
Streamlit remains native on Windows because the current PyTorch/CUDA environment, local
source path, marker vault and provider boundary are host concerns. Only Qdrant belongs in
Docker at this stage.

## Manual Qdrant control

From the repository root:

```powershell
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:6333/readyz
```

Control commands:

```powershell
# Stop containers; keep the vector index.
docker compose -f deploy/docker-compose.yml stop

# Start the existing containers again.
docker compose -f deploy/docker-compose.yml start

# Follow Qdrant service logs.
docker compose -f deploy/docker-compose.yml logs -f --tail 100 qdrant

# Remove containers/network; keep the named volume.
docker compose -f deploy/docker-compose.yml down
```

Do not add `--volumes` to `docker compose down` unless you intentionally want to
delete the complete server index. `restart: unless-stopped` restarts Qdrant after
Docker Desktop starts, except when the service was explicitly stopped.

The default `config/pilot.yaml` targets this server. Server vectors live only in the Docker
named volume; `runtime/data/qdrant` is created only if embedded mode is explicitly selected.
The Qdrant target is part of the index signature, so switching backends safely re-embeds
documents instead of incorrectly skipping them.

For a future remote server, set `SECURE_RAG_QDRANT_URL` and keep the API key only in
`SECURE_RAG_QDRANT_API_KEY`; never commit it. Localhost mode intentionally has no API
key because the port is not reachable from other machines.

Client and per-write timeout defaults to 60 seconds. A write is attempted at most three
times with a 0.25-second initial backoff. Configure these operational values in
`config/pilot.yaml` or with `SECURE_RAG_QDRANT_TIMEOUT_SECONDS`,
`SECURE_RAG_QDRANT_WRITE_MAX_ATTEMPTS` and
`SECURE_RAG_QDRANT_RETRY_BACKOFF_SECONDS`. They deliberately do not affect the index
signature because they do not change vector contents or retrieval semantics.
