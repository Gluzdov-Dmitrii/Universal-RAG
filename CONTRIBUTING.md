# Участие в разработке

## Рабочий процесс

1. Создайте короткую feature branch от актуального `main`.
2. Не добавляйте в код реальные документы, пути, имена, credentials или дампы runtime.
3. Добавьте или обновите tests вместе с изменением поведения.
4. Выполните локальные проверки:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest
$env:SECURE_RAG_API_KEY = "compose-check"
$env:OPEN_WEBUI_SECRET_KEY = "compose-check-secret"
docker compose -f deploy/docker-compose.yml config --quiet
```

5. Откройте pull request; merge в `main` выполняется только после CI и review.

## Границы изменений

- delivery code — `api`, use cases — `orchestration`, indexing — `ingestion`;
- concrete providers находятся в `llm/adapters`, сборка — в `composition.py`;
- изменение extraction/chunking/embedding/access policy требует новой index signature и
  плана миграции;
- изменение SQLite/Open WebUI schema требует backup/restore note;
- интеграция нового источника или инструмента требует реального adapter, ACL contract и tests;
- outbound к внешней LLM не может содержать raw path, filename, Marker Vault или немаркированное
  чувствительное значение.

Для tests используйте только вымышленные данные. PR с `.env`, runtime database, model weights,
Qdrant snapshot или корпоративным документом не принимается.
