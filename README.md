# Secure RAG — локальный учебный прототип

Этот репозиторий реализует один полный вертикальный срез:

    D:\RAG TEST
      → extract + deterministic chunks
      → multilingual-e5-small
      → Qdrant server on localhost
      → filtered retrieval
      → regex + Legal NER + Collection3
      → reversible markers
      → codex_input.txt
      → Responses API без tools / ручной bridge / local mock
      → local demarker
      → restored_answer.txt

Это учебный privacy-прототип, а не готовая DLP-система. Перед рабочим пилотом нужны
проверенный NER, ACL из доверенного источника, отдельный Qdrant server, защищённый
Marker Vault, журналирование без исходного текста и формальное согласование cloud boundary.

## Что уже работает

- TXT, MD, CSV, JSON, XML, HTML/HTM, DOCX, XLSX, PPTX и читаемые PDF.
- Инкрементальный manifest в SQLite: неизменившийся файл повторно не индексируется.
- Локальные embeddings `multilingual-e5-small` с зафиксированной ревизией,
  384 измерениями, окном до 512 токенов и обязательными `query:`/`passage:` prefixes.
- Qdrant server в Docker с обязательным фильтром access_group=pilot, goz=false,
  is_final=true.
- В Qdrant нет исходного текста и реальных путей: только vector, IDs, offsets и policy fields.
- Исходный фрагмент повторно читается из read-only source после retrieval.
- Persistent cache извлечённого текста по умолчанию выключен: базовый режим не создаёт
  вторую копию корпуса на SSD. Его можно включить только после согласования retention
  через `SECURE_RAG_PERSIST_EXTRACTED_CACHE=true`; cache содержит чувствительный raw text.
- Streamlit держит отдельный process-memory LRU только для повторных запросов: до 16
  документов, 20 млн символов суммарно и 5 млн на документ. Он очищается при остановке
  процесса, не создаёт `runtime/extracted-text-cache` и не отключает проверку SHA-256
  исходника перед каждым использованием.
- Regex для email, телефона, ИНН, СНИЛС, паспорта, банковского счёта и суммы в рублях.
- LLAIMlegal/ru-legal-ner и Collection3 через единый span contract.
- Маркировка question, context и filename в одном namespace.
- Локальный Marker Vault, проверка известных утечек и строгий demarker.
- Web-чат на Streamlit и CLI.
- Streamlit telemetry отключена в .streamlit/config.toml.

GLiNER намеренно не включён: он тяжелее, требует отдельную библиотеку и пока не доказал
прирост recall на ваших документах. Его место — отдельный эксперимент после baseline.

## Быстрый запуск на этом ПК

После перезагрузки из корня репозитория достаточно одной команды:

    .\scripts\start-local.ps1

Она при необходимости поднимает Docker Desktop, запускает/проверяет Qdrant, затем
запускает Streamlit скрытым Windows-процессом и ждёт health endpoint. Управление:

    .\scripts\status-local.ps1
    .\scripts\stop-local.ps1
    .\scripts\stop-local.ps1 -StopQdrant

Индекс Qdrant сохраняется в Docker volume при обычной остановке. Автозапуск Windows
этот прототип сам не устанавливает; подробности и пути логов — в `deploy/README.md`.

Окружение уже создано в .venv. Проверка без загрузки моделей:

    .\.venv\Scripts\python.exe -m secure_rag.cli doctor

На чистом Python 3.12 окружение создаётся так:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e ".[web,dev]"

HF-загрузчики по умолчанию работают с `local_files_only=True`: обычный запуск не обращается
к model hub. На новом стенде модели загружаются отдельным, явно сетевым setup-шагом, после
чего флаг сразу возвращается в offline:

    $env:SECURE_RAG_HF_LOCAL_ONLY = "0"
    .\.venv\Scripts\python.exe -m secure_rag.cli index --max-files 1
    .\.venv\Scripts\python.exe -m secure_rag.cli ask --provider stub --ner all "Тест"
    $env:SECURE_RAG_HF_LOCAL_ONLY = "1"

Для NVIDIA заранее выберите подходящую сборку PyTorch по официальной инструкции.
На текущем ПК используется Torch 2.6.0 + CUDA 12.4 из общего подготовленного ML-окружения.

Первый небольшой индекс:

    docker compose -f deploy/docker-compose.yml up -d
    .\.venv\Scripts\python.exe -m secure_rag.cli index --max-files 10

Полный проход выполняется с `--max-files 0`. Индексатор защищён single-writer lock:
второй процесс сразу откажется запускаться. Удаление документов, которых не увидел scan,
по умолчанию отключено; `--prune-missing` допустим только для заведомо полного и
authoritative local scan, а не для частично доступного сетевого каталога.

Операции замены points используют детерминированные IDs, timeout 60 секунд и до трёх
попыток с коротким exponential backoff. Поэтому неоднозначный ответ после сетевого
timeout можно безопасно повторить. Параметры находятся в секции `qdrant` файла конфигурации
и переопределяются через `SECURE_RAG_QDRANT_TIMEOUT_SECONDS`,
`SECURE_RAG_QDRANT_WRITE_MAX_ATTEMPTS` и
`SECURE_RAG_QDRANT_RETRY_BACKOFF_SECONDS`. После исчерпания попыток manifest получает
только безопасный код `qdrant_timeout` или `qdrant_transport`; адреса, пути и содержимое
исключений не сохраняются. Эти транспортные настройки не меняют index signature.

Модель лежит в runtime/model-cache. Индексируются только поддержанные
форматы. Имена и содержимое документов CLI не печатает.

Быстрый synthetic flow только с regex:

    .\.venv\Scripts\python.exe -m secure_rag.cli ask --provider stub --ner regex "Какой документ описывает испытания?"

Полный baseline с двумя NER-моделями:

    .\.venv\Scripts\python.exe -m secure_rag.cli ask --provider manual --ner all "Ваш тестовый вопрос"

Для реального чувствительного вопроса лучше web-интерфейс: значение не попадёт в историю
PowerShell.

    .\.venv\Scripts\python.exe -m secure_rag.cli serve

Открыть: http://127.0.0.1:8501

В web-интерфейсе одно действие: заполните вопрос, при необходимости укажите путь к
поддерживаемому файлу внутри `source_root` и нажмите «Отправить». Поиск, sanitizer,
provider и demarker выполняются одним потоком. В раскрываемом журнале показывается время
каждой стадии без текста запроса, исходного пути и значений сущностей. Такой же безопасный
технический журнал сохраняется в `runtime/reports/pipeline-events/*.jsonl`.

RAM-кэш Streamlit рассчитан на однопользовательский pilot. `st.cache_resource` разделяет
его между web-сессиями одного процесса, поэтому для нескольких пользователей с разными
ACL нужны отдельные процессы/кэши на principal либо отдельная авторизационная модель.
Приложение само не сохраняет raw cache на SSD; pagefile и crash dump Windows относятся к
отдельным настройкам хоста и для production также должны быть защищены.

По умолчанию web использует `SECURE_RAG_PROVIDER=auto`:

- если в окружении есть `OPENAI_API_KEY`, маркированный payload автоматически уходит через
  официальный Responses API с `store=False`; параметр `tools` вызову вообще не передаётся,
  а скрытые SDK retries отключены (`max_retries=0`);
- если ключа нет, весь round-trip автоматически выполняет локальный `stub`, без сети.

Для настоящего автоматического ответа задайте ключ и модель в окружении процесса перед
запуском:

    $env:OPENAI_API_KEY = "<ваш API key>"
    $env:SECURE_RAG_OPENAI_MODEL = "gpt-5-mini"
    $env:SECURE_RAG_PROVIDER = "auto"

Подписка/авторизация Codex и `OPENAI_API_KEY` — разные механизмы. Python Codex SDK может
переиспользовать вход в Codex, но этот вход нельзя передать официальному Responses API
вместо API key.

Режим `codex-local` сохранён только для явно небезопасного лабораторного опыта. Нужно
одновременно задать `SECURE_RAG_PROVIDER=codex-local` и
`SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL=1`. Его рабочий каталог создаётся вне репозитория в
`%LOCALAPPDATA%\SecureRagCodexSandbox`, однако это не security boundary: `read_only`
ограничивает запись, а не область чтения. На этом ПК проверка показала, что local Codex SDK
может читать доступные файлы на `D:`, включая `D:\RAG TEST`. Не используйте этот режим
для конфиденциального корпуса.

## Ручной Codex bridge

1. Подготовьте запрос через CLI с `--provider manual` либо временно задайте
   `SECURE_RAG_PROVIDER=manual` для web и задайте вопрос.
2. Проверьте созданный runtime/requests/REQUEST_ID/codex_input.txt как обычный текст.
3. Передайте в отдельный cloud-чат только этот файл.
4. Не передавайте runtime/marker-vault, manifest SQLite, каталог Qdrant или весь репозиторий.
5. Сохраните ответ как `codex_output.txt` в каталоге запроса и выполните:

       .\.venv\Scripts\python.exe -m secure_rag.cli demark REQUEST_ID

6. Локальный результат появится в restored_answer.txt.

Web-форма сейчас не импортирует ручной ответ: для `manual` используется этот CLI-шаг.
В автоматическом режиме маркированный ответ сначала атомарно сохраняется в
`codex_output.txt`, и только затем запускается demarker. После сбоя локальное восстановление
можно повторить без нового обращения к provider.

Если Codex запущен как локальный coding agent с правом чтения всего диска, файловый bridge
сам по себе не создаёт security boundary. Для честной проверки используйте отдельный
облачный чат, которому доступен только codex_input.txt. Автоматический вариант использует
Responses API без tools; local coding agent не является равноценной границей.

## Где лежат данные

- runtime/manifest/documents.sqlite — пути, hash/revision и offsets; локально.
- Docker volume secure_rag_qdrant_storage — vectors и непрозрачный payload; локально.
- runtime/qdrant — прежний embedded-индекс; после проверки server-индекса его можно
  отдельно архивировать или удалить.
- runtime/model-cache — зафиксированные модели; локально.
- runtime/extracted-text-cache — появляется только при явном opt-in и содержит
  source-derived raw text; не отправлять и не считать частью безопасного baseline.
- runtime/requests/REQUEST_ID/codex_input.txt — единственный outbound-файл; plain text
  не позволяет локальному preview выполнить ссылку или HTML из недоверенного документа.
- runtime/marker-vault/REQUEST_ID.json — исходные значения; никогда не отправлять.
- runtime/requests/REQUEST_ID/restored_answer.txt — приватный финальный результат;
  plain text выбран намеренно, чтобы preview Markdown/HTML не отправил восстановленное
  значение через provider-controlled URL.

Весь runtime исключён из Git.

## Ограничения текущего среза

- Qdrant server запускается через deploy/docker-compose.yml и доступен только через
  127.0.0.1:6333. Команды запуска и остановки описаны в deploy/README.md.
- PDF без текстового слоя и legacy DOC пока пропускаются/завершаются safe error. Для
  XLSX извлекаются значения ячеек и названия листов без исполнения формул; для PPTX —
  текстовые блоки, таблицы и заметки. OOXML-контейнеры проходят bounds/zip-bomb проверку.
- ACL, ГОЗ и final-version сейчас являются консервативными pilot-заглушками.
- Для production путь вложения должен читаться сервисным аккаунтом из каталога, который
  пользователи не могут подменять symlink/junction во время запроса; одна path-containment
  проверка Python не является полной защитой от такой локальной гонки.
- Web хранит текущий ответ только в session state; после перезапуска артефакты остаются на
  диске, но UI пока не содержит списка и resume-кнопки для старых request ID.
  Поэтому весь D:\RAG TEST в этом стенде считается одной заранее разрешённой
  non-GOZ коллекцией. Прикладывать outbound-файл к cloud нельзя, пока это допущение
  не подтверждено владельцем данных.
- При аварии обновления одного документа возможен временный пропуск его chunks; смешения
  старой и новой ревизии retrieval не допускает.
- NER может пропустить неизвестную коммерческую сущность. Автоматическая отправка — только
  допущение этого стенда; до рабочего пилота нужны canary/golden тесты и согласованный
  human-review/fail-closed policy.
- Marker Vault пока обычный локальный JSON под NTFS/BitLocker, а не зашифрованная БД.
- Mock-провайдер не отвечает по смыслу: он показывает именно payload и round-trip.
- Pilot использует `multilingual-e5-small` с окном 512 токенов и chunks около
  450 символов. Предыдущий MiniLM с окном 128 оставлен только в старом model cache;
  новый индекс хранится в отдельной collection `secure_rag_pilot_e5_v1`.
- Быстрый skip всё равно вычисляет SHA-256 каждого выбранного файла. Для десятков ТБ это
  заменяется version/change feed исходной DMS, а не ежедневным полным hashing.
- Во время build прототип обновляет одну collection по документам. Production-вариант
  должен строить versioned collection и переключать alias атомарно; UI во время текущего
  build может видеть только уже обработанную часть корпуса.

Архитектура подробнее: docs/ARCHITECTURE.md. Практический сценарий: docs/WALKTHROUGH.md.
Fine-tuning развивается отдельно в соседнем репозитории ../ft-bert.
