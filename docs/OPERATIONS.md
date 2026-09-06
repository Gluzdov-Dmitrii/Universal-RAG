# Состояние, версии и резервные копии

## Что чем версионируется

| Объект | Механизм версии | Где хранится |
|---|---|---|
| Код и конфигурация | Git commit/tag | GitHub |
| Manifest schema | `schema_version` + migrations | Git; live SQLite на сервере |
| Извлечение и embeddings | index signature | SQLite build history + Qdrant payload |
| Vector index | collection name + Qdrant snapshot | Qdrant volume + backup storage |
| Open WebUI | pinned image tag | Compose; состояние в отдельном volume |
| История backend/retrieval | chat-state schema `user_version` | `runtime/data/sessions/chat-state.sqlite` |
| Модели | model ID + pinned revision | `runtime/cache/models` или artifact storage |
| Golden datasets/releases | будущие DVC/MLflow IDs | закрытое object storage, не Git |

Live `.sqlite`, Qdrant snapshots, Open WebUI database, документы и веса не коммитятся. Git не
подходит для постоянно меняющихся БД и может раскрыть удалённые значения через историю.

## Диагностика запросов Open WebUI

В потоковом ответе Universal RAG показывает пользователю текущий этап и количество найденных
фрагментов. Если обработка завершилась ошибкой, скопируйте из ответа код диагностики и найдите
файл `runtime/diagnostics/pipeline-events/<код>.jsonl`. В нём сохраняются только этапы,
длительности, счётчики и тип исключения — без вопроса, текста документов и путей к ним.

Для переноса журналов в отдельный служебный каталог задайте
`SECURE_RAG_PIPELINE_EVENT_DIR`. Ошибка записи диагностического журнала логируется, но не
прерывает успешно выполняющийся RAG-запрос.

Open WebUI хранит исходный transcript в своём volume. Backend хранит его серверное зеркало и
retrieval runs в `runtime/data/sessions/chat-state.sqlite`. Запросы интерфейса с
`metadata.task` изолируются и не считаются пользовательскими репликами. Журнал API и JSONL
этапов намеренно не содержат тексты запросов.

## Привязка codex-local задач к проекту Codex

Для видимых persistent задач создайте в Codex отдельный локальный проект и используйте один
и тот же абсолютный путь при развёртывании workspace и в `.env`:

```powershell
.\scripts\sync-rag-agent-workspace.ps1 `
    -Mode Push `
    -TargetRoot "C:\Dev\LLM\Codex_RAG_Test"

# .env
SECURE_RAG_AGENT_WORKSPACE_ROOT=C:\Dev\LLM\Codex_RAG_Test
SECURE_RAG_CODEX_PERSIST_THREADS=1
```

Путь должен совпадать с корнем сохранённого проекта Codex буквально. Если указать другой
каталог, SDK всё равно использует его как `cwd`, но задача останется в глобальном Recents.
После изменения `.env` перезапустите API. `projectId` может появиться в интерфейсе не в момент
создания задачи, а после завершения первого turn и обновления списка проектов.

Sanitized provider input уже хранится в
`runtime/state/requests/<request-id>/codex_input.txt`; исходные значения находятся отдельно в
Marker Vault и никогда не копируются в Codex project workspace. Перенос sanitized artifacts
в каталог проекта требует отдельной retention/encryption policy: проект Codex сам по себе не
является защищённым хранилищем.

## Создание snapshot

Сохраняйте backup на другом диске или защищённой сетевой папке вне clone:

```powershell
.\scripts\backup-state.ps1 -DestinationRoot "D:\dev tests\Universal-RAG-backups"
```

Команда делает online backup manifest и chat state через SQLite Backup API, создаёт Qdrant collection
snapshot, скачивает его из container storage и записывает `snapshot.json` со следующей связкой:

- Git commit;
- config schema;
- Qdrant collection и snapshot name;
- SHA-256 копии manifest;
- SHA-256 копии chat state, если база уже создана;
- timestamp.

Чтобы также сохранить аккаунты, настройки и историю чатов, добавьте:

```powershell
.\scripts\backup-state.ps1 `
    -DestinationRoot "D:\dev tests\Universal-RAG-backups" `
    -IncludeOpenWebUI
```

Open WebUI будет кратковременно остановлен, а его volume сохранён как
`open-webui-data.tgz`. Архив содержит исходные вопросы, восстановленные ответы и account data;
применяйте шифрование, ограничение доступа и согласованный retention.

## Restore drill

Автоматическое восстановление намеренно не добавлено: оно перезаписывает live state и требует
явного change/incident decision. Проверка восстановления выполняется на отдельном хосте или в
изолированном Compose project:

1. checkout Git commit из `snapshot.json`;
2. создать `.env` и проверить pinned model revisions;
3. восстановить `documents.sqlite` в `runtime/data/manifest/`;
4. восстановить `chat-state.sqlite` в `runtime/data/sessions/`, если он есть в snapshot;
5. загрузить Qdrant snapshot в collection с тем же именем;
6. при наличии распаковать Open WebUI archive в новый пустой volume;
7. выполнить `doctor`, проверить counts/signature, identity и multi-turn retrieval;
8. только после проверки переключать пользователей на восстановленный экземпляр.

Manifest и Qdrant snapshot должны восстанавливаться как одна логическая версия. Смешивание
SQLite из одного backup с vectors из другого может вернуть устаревшие offsets или неполный
набор документов.

## Retention

Минимальный стартовый режим:

- snapshot перед каждым deployment;
- ежедневный state snapshot;
- 7 daily + 4 weekly + 3 monthly copies;
- минимум одна encrypted off-host copy;
- ежемесячный restore drill с зафиксированным временем восстановления.

Точные RPO/RTO, encryption keys и место хранения согласуются с владельцами данных и ИБ.
