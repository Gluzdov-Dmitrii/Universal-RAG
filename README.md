# Secure RAG

Локальный учебный privacy-прототип: документы индексируются локально, retrieval возвращает
только разрешённые фрагменты, чувствительные значения маркируются перед внешней LLM, а
восстановление ответа выполняется локально. Automatic provider может запросить уточняющий
multi-query retrieval или соседние chunks; все такие операции исполняются локальным pipeline.

Это не готовая DLP-система. Для production нужны доверенные ACL и признаки финальной версии,
проверенный NER на размеченном security-наборе, защищённый Marker Vault, аудит cloud boundary,
backup/restore и изоляция пользователей.

## Быстрый запуск

Текущее окружение находится в `.venv`. После перезагрузки ПК:

```powershell
.\scripts\start-local.ps1
.\scripts\status-local.ps1
.\scripts\stop-local.ps1
```

Launcher поднимает Qdrant в Docker и Streamlit как локальный Windows-процесс. Интерфейс:
<http://127.0.0.1:8501>. Подробности управления — в [deploy/README.md](deploy/README.md).

Проверка конфигурации и основные команды:

```powershell
.\.venv\Scripts\python.exe -m secure_rag.api.cli doctor
.\.venv\Scripts\python.exe -m secure_rag.api.cli index --max-files 10
.\.venv\Scripts\python.exe -m secure_rag.api.cli ask --provider stub --ner regex "Тест"
.\.venv\Scripts\python.exe -m secure_rag.api.cli serve
```

После `pip install -e ".[web,dev]"` те же операции доступны через `secure-rag`.
`index --max-files 0` выполняет полный проход. `--prune-missing` используйте только после
полного authoritative scan источника.

## Новая машина: установка и первичная индексация

Нужны Windows, Python 3.12 x64 и Docker Desktop. Команды выполняются из корня полученного
репозитория; LLM для установки и индексации не требуется.

1. Создайте Python-окружение и установите приложение:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[web,dev]"
```

2. В `config/pilot.yaml` задайте `paths.source_root` — абсолютный путь к каталогу документов.
   `paths.runtime_root: "runtime"` можно оставить без изменения. Вместо правки YAML для
   текущего PowerShell-сеанса допустимо задать
   `$env:SECURE_RAG_SOURCE_ROOT = "D:\Documents"`.

3. Запустите Qdrant и проверьте конфигурацию:

```powershell
docker compose -f deploy/docker-compose.yml up -d qdrant
.\.venv\Scripts\python.exe -m secure_rag.api.cli doctor
```

4. Один раз загрузите pinned embedding- и NER-модели, затем верните обычный offline-режим:

```powershell
$env:SECURE_RAG_HF_LOCAL_ONLY = "0"
.\.venv\Scripts\python.exe -m secure_rag.api.cli prepare-models --ner all
$env:SECURE_RAG_HF_LOCAL_ONLY = "1"
```

Модели сохраняются в `runtime/cache/models`. В последующих запусках сеть Hugging Face не
нужна. На машине без GPU автоматически используется CPU; CUDA-сборку PyTorch при
необходимости устанавливают отдельно до `pip install -e`.

5. Постройте полный индекс и запустите интерфейс:

```powershell
.\.venv\Scripts\python.exe -m secure_rag.api.cli index --max-files 0
.\scripts\start-local.ps1
```

Индексатор печатает безопасный JSONL-прогресс без имён и текста документов. Успешный полный
проход заканчивается отчётом с `failed: 0`; состояние также проверяется командой `doctor`.
Streamlit будет доступен на <http://127.0.0.1:8501>.

### Повторная индексация и новые документы

Для добавленных или изменённых файлов запускается та же команда:

```powershell
.\.venv\Scripts\python.exe -m secure_rag.api.cli index --max-files 0
```

Это инкрементальный проход: manifest сопоставляет path, SHA-256 revision и index signature;
совместимые неизменившиеся документы пропускаются. `--max-files 10` полезен только для
короткой проверки первых десяти обнаруженных файлов, а не для штатного обновления всего
корпуса. `--prune-missing` удаляет из индекса отсутствующие документы и поэтому разрешён
только после полного, заведомо доступного scan источника.

При индексации каждого нового или изменённого документа source location записывается для
каждого chunk одновременно в SQLite manifest и Qdrant payload:

- PDF (`.pdf`) — физическая страница или диапазон страниц;
- PowerPoint (`.pptx`) — слайд или диапазон слайдов;
- Excel (`.xlsx`) — лист или диапазон листов;
- Word (`.docx`) — приблизительная страница по сохранённому Word page count и позиции текста;
  UI специально маркирует её как `примерно стр.`. Оценка зависит от последнего сохранения,
  шрифтов, изображений, принтера и layout engine и не считается точной пагинацией;
- остальные поддерживаемые форматы — без выдуманного номера страницы;
- старые бинарные `.doc` и `.xls` сейчас не поддерживаются; перед индексацией их нужно
  конвертировать в `.docx`/`.xlsx` либо позднее подключить отдельный контролируемый
  LibreOffice-конвертер.

Динамическое вычисление location при retrieval остаётся fallback для chunks старого индекса.

## Структура

```text
src/secure_rag/
├── api/              # CLI и Streamlit, без бизнес-логики
├── domain/           # общие типизированные контракты
├── infrastructure/   # политики сторонних runtime-библиотек
├── ingestion/        # inventory, extract, chunk, manifest, index sync
├── llm/              # заменяемая генерация: contracts, factory, adapters, bridge
├── orchestration/    # online pipeline и безопасные технические события
├── retrieval/        # embeddings, Qdrant, rehydrate source, attachments
├── sanitization/     # regex/NER, markers, normalization, demarking
├── security/         # fail-closed retrieval policy
├── composition.py    # единственное место сборки конкретных компонентов
└── config.py         # конфигурационные контракты и загрузка YAML/env
```

`llm-workspaces/rag-test/` — канонический переносимый шаблон пользовательского LLM-компонента:
agent, skill, rules и deployment manifest. Рабочий `C:\Dev\LLM\Codex_RAG_Test` является его
развёрнутой копией, а не отдельным источником истины.

Каталоги будущих `connectors/` и `tools/` не создаются до появления реальных адаптеров.
Подробные зависимости, потоки данных и security-инварианты собраны в
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Это основной технический документ для агентов.

## Локальные и внешние данные

```text
runtime/
├── data/          # постоянные БД и локальные индексы
├── state/         # приватное состояние запросов и Marker Vault
├── cache/         # пересоздаваемые модели и extracted-text cache
├── diagnostics/   # безопасные логи, события и отчёты
├── run/           # PID и lock-файлы живых процессов
└── tmp/           # одноразовые изолированные workspace
```

- `runtime/data/manifest/documents.sqlite` — локальные пути, revisions, char offsets,
  страницы/слайды/листы chunks и статусы.
- Qdrant server хранит vectors в Docker volume `secure_rag_qdrant_storage`; внутри
  `runtime/data` остаётся SQLite manifest. Каталог `runtime/data/qdrant` создаётся лишь при
  явном переключении конфигурации в embedded-режим.
- `runtime/state/marker-vault/` — обратная таблица маркеров; никогда не отправляется наружу.
- `runtime/state/requests/<id>/` — outbound payload, sanitized response, restored answer и
  локальная карта citation → абсолютный путь.
- `runtime/cache/extracted-text/` появляется только при явном opt-in и содержит raw text.

`runtime/` исключён из Git. Process-memory cache Streamlit рассчитан только на
однопользовательский pilot.

## Provider и модели

`SECURE_RAG_PROVIDER=auto` использует Responses API только при наличии `OPENAI_API_KEY`,
иначе выбирает локальный `stub`. API вызывается с `store=False` и без tools. Iterative RAG
использует строгий текстовый control envelope: приложение валидирует rewrite, восстанавливает
markers только для локального поиска и заново sanitizes найденный контекст. Режим
`codex-local` не является privacy boundary и разрешается только отдельным unsafe-флагом.
При `SECURE_RAG_CODEX_PERSIST_THREADS=1` каждый запрос создаёт одну видимую Codex-задачу:
её `cwd` совпадает с `llm.agent_workspace_root`, поэтому задача относится к проекту `RAG Test`,
а все уточняющие retrieval-итерации продолжаются внутри этой же задачи.

Основные лимиты находятся в `config/pilot.yaml`: `max_iterations`,
`max_queries_per_iteration`, `max_contexts`, `rewrite_min_similarity` и
`adjacent_chunk_radius`. Реальные пути никогда не входят в provider payload: UI/CLI получают
их отдельно из локального `sources.json`; рядом с каждым citation возвращается доступная
локация (`стр.`, `слайд` или `лист`). LLM не получает filename/path и работает с file type и
opaque citation.

Hugging Face модели по умолчанию загружаются только из локального cache. Сетевой prefetch
включается отдельным setup-шагом через `SECURE_RAG_HF_LOCAL_ONLY=0`, после чего offline-режим
нужно вернуть.

### Заменяемый LLM-слой

Оркестрация зависит только от `llm/contracts.py`. Реализации Responses API, локального Codex,
stub и будущей локальной модели находятся в `llm/adapters/`, а выбор выполняет
`llm/factory.py`. Новый интерфейс LLM не должен импортироваться напрямую в pipeline.

User-facing Codex agent может работать из отдельного client workspace, не содержащего исходный
код, runtime БД и документы. Сам project root задаёт контекст, но не является файловой
security boundary. Поэтому backend читает из него только allowlisted rule/skill файлы и
добавляет их к инструкциям no-tools provider. Responses provider не получает доступ к
workspace; явно небезопасный `codex-local` использует его как `cwd` для привязки к проекту и
технически может читать находящиеся там файлы, поэтому workspace должен оставаться пустым за
пределами управляемых инструкций.
Обычные Codex-задачи в этом проекте предназначены для редактирования и тестирования инструкций,
а не для обработки конфиденциальных запросов.

Развернуть или проверить компонент:

```powershell
.\scripts\sync-rag-agent-workspace.ps1 -Mode Push `
    -TargetRoot C:\Dev\LLM\Codex_RAG_Test
.\scripts\sync-rag-agent-workspace.ps1 -Mode Check `
    -TargetRoot C:\Dev\LLM\Codex_RAG_Test
```

`Push` перезаписывает только файлы из `.rag-workspace-manifest.json` и не удаляет остальные
файлы target workspace. На другой машине передайте новый `-TargetRoot` и при необходимости
задайте backend-путь через `SECURE_RAG_AGENT_WORKSPACE_ROOT`.

## Разработка

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest
```

При изменении путей модулей обновляйте одновременно entry point в `pyproject.toml`,
PowerShell launcher и архитектурную карту. При изменении extraction, embedding или
chunking semantics обновляйте соответствующую version/signature, чтобы старый индекс не
считался совместимым.
