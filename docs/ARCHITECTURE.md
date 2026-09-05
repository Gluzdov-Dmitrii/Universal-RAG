# Архитектура Secure RAG

Этот файл — единый источник истины о границах модулей, потоках данных и security-инвариантах.
Команды запуска находятся в корневом README, эксплуатация Qdrant — в `deploy/README.md`.

## Границы пакета

- `api`: предоставляет CLI и OpenAI-совместимый HTTP transport; собирает зависимости через
  `composition`.
- `orchestration`: координирует use case, но не знает деталей Open WebUI/FastAPI/CLI.
- `ingestion`: офлайн-инвентаризация, extraction, chunking, manifest и синхронизация индекса.
- `retrieval`: локальный query embedding, Qdrant search, перечитывание исходника и attachments.
- `sanitization`: detectors, overlap resolution, canonical markers и strict demarking.
- `security`: явные fail-closed политики доступа и обработки информации.
- `llm`: заменяемая генерация; contracts/factory не зависят от SDK, конкретные интеграции
  изолированы в `llm/adapters`, а файловый bridge образует внешний privacy boundary.
- `domain`: структуры данных без зависимостей от инфраструктуры.
- `infrastructure`: узкие политики интеграции со сторонними runtime-библиотеками.
- `config.py`: типизированная конфигурация; `composition.py`: создание concrete adapters.

Зависимости направлены от delivery к use cases и далее к специализированным слоям. Domain не
импортирует другие слои. Нельзя переносить FastAPI, argparse, OpenAI SDK или Qdrant client в
`domain`/`orchestration`. Новые внешние системы получают отдельный adapter; каталоги
`connectors` и `tools` появятся только вместе с первой реальной реализацией.

Pipeline зависит от `Provider` protocol и получает concrete provider через `ProviderFactory`.
Responses API, local Codex и будущая локальная модель взаимозаменяемы на этой границе. SDK,
модельные настройки и особенности транспорта не должны просачиваться в orchestration.

## Runtime layout

```text
runtime/
├── data/          durable SQLite manifest; embedded vectors только по явной конфигурации
├── state/         private request workspaces и Marker Vault
├── cache/         rebuildable model/extraction artifacts
├── diagnostics/   logs, pipeline events и safe reports
├── run/           process IDs и interprocess locks
└── tmp/           disposable isolated workspaces
```

Feature-код использует только path properties `AppConfig`: литеральные runtime-подкаталоги
допустимы в config/migration/launcher, но не распределяются по pipeline. Qdrant server хранит
основной vector index в Docker named volume. `data/qdrant` не является штатным каталогом
server-mode и создаётся embedded-адаптером только при явном выборе этого режима.
`data` и `state` нельзя очищать как cache; `run` и `tmp` не являются backup-данными.

## Indexing flow

```text
read-only source
  → safe inventory + stable document_id + SHA-256 revision
  → format-specific extractor
  → deterministic character chunks
  → local passage embeddings
  → Qdrant upsert
  → SQLite manifest commit
```

Manifest хранит локальный path и состояние целого документа. Qdrant хранит vector,
document/revision/chunk IDs, offsets, build/signature и policy fields, но не raw text.
Point IDs детерминированы. Повторяемые write-операции идемпотентны; временные ошибки имеют
ограниченный retry, после чего manifest получает только безопасный error code.

Indexer защищён межпроцессным single-writer lock. Fast skip допускается только при совпадении
revision, index signature, manifest chunks и Qdrant points. Удаление unseen документов требует
явного `--prune-missing` после полного authoritative scan.

Persistent extracted-text cache выключен по умолчанию: его включение создаёт вторую копию
чувствительного текста и требует retention/encryption policy. API использует отдельный
ограниченный process-memory cache и всё равно проверяет SHA-256 исходника.

## Online flow

```text
raw local question
  → local query embedding
  → Qdrant filter: access_group AND goz=false AND is_final=true
  → opaque IDs + offsets
  → local source rehydration
  → regex + NER + overlap resolution
  → Marker Vault + sanitized payload
  → no-tools Responses API / manual bridge / local stub
  → final answer OR validated retrieval_request
       → local marker restore for query only
       → semantic anchor check + multi-query retrieval / adjacent citation expansion
       → re-sanitize all accumulated context in the same marker namespace
       → repeat up to max_iterations
  → strict marker and leak validation
  → durable sanitized response staging
  → local demarking
  → local citation-to-path mapping for Open WebUI/CLI
```

Query не маркируется до retrieval: embedding и поиск локальны, а ранняя замена сущностей
ухудшает релевантность. Privacy boundary начинается до создания `codex_input.txt`.

Provider не получает настоящий Qdrant/SQLite tool. Он может вернуть только точный
`retrieval_request` envelope с ограниченным списком queries и `expand_citations`. Любое
отклонение от схемы блокируется. Markers из rewrite восстанавливаются локально, неизвестные
markers запрещены, а cosine similarity не позволяет rewrite слишком далеко уйти от исходного
вопроса. Расширять соседние chunks можно только для citation, уже прошедшего policy filter.
В persistent `codex-local` режиме один request соответствует одному Codex thread, и все его
retrieval-итерации продолжают этот thread. Thread запускается с точным `cwd` отдельного
`llm.agent_workspace_root`;
разные requests не разделяют историю и marker namespace.

Open WebUI обращается к backend через Bearer-authenticated OpenAI-compatible API. Он хранит
пользователей, группы, model visibility и историю в собственном persistent volume. Open WebUI
не получает Qdrant/SQLite/Nextcloud credentials. Для каждого model request Open WebUI передаёт
HS256-подписанный user identity и `X-OpenWebUI-Chat-Id`; backend проверяет подпись и хранит
зеркало transcript плюс retrieval runs в `runtime/data/sessions/chat-state.sqlite`. Ключ
изоляции — `(user_id, chat_id)`, поэтому смена модели/агента внутри чата не теряет контекст, а
совпавший chat ID другого пользователя не открывает его состояние. Перед использованием
истории как model context backend удаляет локальный footer источников, ограничивает окно и
повторно выполняет request-scoped sanitization всего диалога.

Identity пока не участвует в document retrieval policy: первая серверная конфигурация
использует общий access group `employees`. Дифференцированный доступ нельзя включать до
реализации и тестирования identity-to-ACL mapping.

Внешняя LLM видит `file_type` и opaque `Rxxx`, но не filename/path. Для `xls/xlsx/csv` prompt
разрешает запросить соседние chunks, если не хватает заголовков или строк. Реальные абсолютные
пути собираются из проверенного `DocumentRecord.source_path`, сохраняются отдельно в локальном
`sources.json` и показываются пользователю после выполнения. Во время extraction PDF pages,
PPTX slides и XLSX sheets получают диапазоны в нормализованном тексте. Chunk наследует
пересекающуюся локацию (включая диапазон при переходе через границу), а `sources.json`
связывает её с конкретным `Rxxx`. Для DOCX extractor читает сохранённый Word page count и
строит пропорциональную оценку по char offsets; наружу она маркируется как `approx_page`, а UI
показывает `примерно стр.`. Это навигационная подсказка, не точная пагинация: достоверная
страница потребует одинакового с Word layout/render engine.
Indexer сохраняет chunk location и в SQLite manifest, и в Qdrant payload; retrieval повторно
вычисляет её по char offsets только как совместимый fallback для ранее построенных points.
Бинарные `.doc`/`.xls` не входят в поддерживаемый ingestion contract.

## Security invariants

1. Retrieval и embeddings выполняются локально по исходному запросу.
2. Access filter применяется внутри vector search, а не после получения кандидатов.
3. В outbound не входят raw path, Marker Vault и исходный filename.
4. Query и все chunks каждой итерации используют один request marker namespace.
5. Ошибка sanitizer завершает запрос; fallback на raw payload запрещён.
6. Неизвестный marker или известное немаркированное значение блокирует restoration/outbound.
7. Текст документов и provider output считаются недоверенными данными и не исполняются.
8. HF-модели обычно открываются только из локального cache с pinned revision.
9. Provider output сначала атомарно сохраняется в sanitized staging и лишь затем demark-ится.
10. Технические события и ошибки не содержат вопрос, raw path или значения сущностей.
11. Provider-controlled retrieval ограничен ACL исходного запроса, schema/size limits,
    semantic anchor и max iterations/contexts.
12. Open WebUI backend key не выдаётся пользователям; прямой доступ к API/Qdrant блокируется
    сетевой политикой сервера.

Baseline NER не доказывает отсутствие false negatives. До коммерческих данных нужен
размеченный security set с canary в query, chunk, filename, metadata и errors, измерение recall
критичных классов и согласованный human-review/fail-closed outbound gate.

## Основные контракты

- `DocumentRecord`: stable ID, local path, revision, size/mtime и status.
- `ChunkRecord`: document/revision, ordinal, char offsets и source location; raw text только в
  памяти процесса.
- `RetrievalHit`: локальный текст, verified local path, file type, ordinal, offsets, source
  location и score.
- `EntitySpan`: offsets, normalized label, detector, score и priority.
- `MarkerState`: marker map и aliases; существует только в request vault/process.
- `DocumentSource`: локальная карта citations к проверенному пути, file type и доступной
  странице/слайду/листу.
- `BridgeResult`: request artifacts, sources, iteration/counter metadata.

## Версии и совместимость индекса

Совместимость определяется не только Git-кодом. Index signature включает extraction,
chunking, embedding model revision и релевантную конфигурацию. Изменение семантики этих
компонентов требует version bump/signature change.

Текущий dense baseline — `intfloat/multilingual-e5-small`, 384 dimensions, окно до 512
токенов и обязательные `query:`/`passage:` prefixes. Legal NER и Collection3 должны пройти
внутреннюю валидацию. Обучение доменного NER развивается отдельно
в `../ft-bert`; этот репозиторий потребляет только одобренный artifact.

## Что ещё не реализовано из целевой схемы

- trusted ACL/ГОЗ/final-version connectors и identity-to-ACL mapping;
- Nextcloud/WebDAV change-feed adapter вместо scan синхронизированной папки;
- Bitrix/1C adapters;
- encrypted Marker Vault и per-user workspaces;
- hybrid BM25 + dense + RRF + reranker;
- LangGraph workflow с типизированными retrieval tools вместо текущего prompt protocol;
- allowlisted local tool gateway;
- versioned Qdrant collections с atomic alias switch;
- monitoring экономического эффекта и проверенный production rollout.

Не добавляйте заглушечные packages под эти элементы: новый каталог должен иметь владельца,
контракт, тест и реальный вызывающий поток.

## Codex client workspace

User-facing Codex agent должен запускаться из отдельного пустого client workspace, который не
содержит этот repository, runtime БД или source documents. Его rules/skills описывают только
формат ответа и `retrieval_request` protocol. Project root сам по себе не ограничивает чтение.
Production-safe путь: backend загружает только перечисленные в `llm.instruction_files` файлы,
проверяет, что workspace не пересекается с repo/source/runtime, и передаёт их no-tools provider
как trusted instructions. Responses provider видит только эти инструкции и sanitized payload,
сформированный `llm/bridge.py`. Небезопасный `codex-local` использует workspace как `cwd` и
может читать его файлы; поэтому там допустимы только управляемые инструкции. Обычная Codex
project task не считается privacy boundary.

Каноническая версия компонента хранится в `llm-workspaces/universal-rag-agent/`. Каталог содержит только
переносимые `AGENTS.md`, project agent, rule, skill и manifest управляемых файлов. Скрипт
`scripts/sync-rag-agent-workspace.ps1` разворачивает их в отдельный Codex project и умеет
fail-fast проверять drift. На inference-машине target path задаётся независимо от repository;
backend и LLM workspace могут находиться на разных виртуальных или физических машинах и
связываться через будущий узкий transport/tool gateway.

Codex workspace остаётся отдельным управляемым LLM-компонентом и не используется как
пользовательский интерфейс: эту роль выполняет Open WebUI.
