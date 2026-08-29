# Secure RAG — локальный учебный прототип

Этот репозиторий реализует один полный вертикальный срез:

    D:\RAG TEST
      → extract + deterministic chunks
      → multilingual-e5-small
      → Qdrant local
      → filtered retrieval
      → regex + Legal NER + Collection3
      → reversible markers
      → codex_input.md
      → ручной Codex или local mock
      → local demarker
      → restored_answer.md

Это учебный privacy-прототип, а не готовая DLP-система. Перед рабочим пилотом нужны
проверенный NER, ACL из доверенного источника, отдельный Qdrant server, защищённый
Marker Vault, журналирование без исходного текста и формальное согласование cloud boundary.

## Что уже работает

- TXT, MD, CSV, JSON, XML, HTML/HTM, DOCX и читаемые PDF.
- Инкрементальный manifest в SQLite: неизменившийся файл повторно не индексируется.
- Локальные embeddings с точной ревизией multilingual MiniLM, уже находившейся в
  model cache этого ПК.
- Embedded Qdrant с обязательным фильтром access_group=pilot, goz=false, is_final=true.
- В Qdrant нет исходного текста и реальных путей: только vector, IDs, offsets и policy fields.
- Исходный фрагмент повторно читается из read-only source после retrieval.
- Regex для email, телефона, ИНН, СНИЛС, паспорта, банковского счёта и суммы в рублях.
- LLAIMlegal/ru-legal-ner и Collection3 через единый span contract.
- Маркировка question, context и filename в одном namespace.
- Локальный request_map.json, проверка известных утечек и строгий demarker.
- Web-чат на Streamlit и CLI.
- Streamlit telemetry отключена в .streamlit/config.toml.

GLiNER намеренно не включён: он тяжелее, требует отдельную библиотеку и пока не доказал
прирост recall на ваших документах. Его место — отдельный эксперимент после baseline.

## Быстрый запуск на этом ПК

Окружение уже создано в .venv. Проверка без загрузки моделей:

    .\.venv\Scripts\python.exe -m secure_rag.cli doctor

На чистом Python 3.12 окружение создаётся так:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e ".[web,dev]"

Для NVIDIA заранее выберите подходящую сборку PyTorch по официальной инструкции.
На текущем ПК используется Torch 2.6.0 + CUDA 12.4 из общего подготовленного ML-окружения.

Первый небольшой индекс:

    .\.venv\Scripts\python.exe -m secure_rag.cli index --max-files 10

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

## Ручной Codex bridge

1. В web-чате выберите «Файл для ручного Codex» и задайте вопрос.
2. Проверьте созданный runtime/requests/REQUEST_ID/codex_input.md.
3. Передайте в отдельный cloud-чат только этот файл.
4. Не передавайте runtime/marker-vault, manifest SQLite, каталог Qdrant или весь репозиторий.
5. Вставьте ответ Codex в секцию «Импорт ответа Codex» либо сохраните его как
   codex_output.md и выполните:

       .\.venv\Scripts\python.exe -m secure_rag.cli demark REQUEST_ID

6. Локальный результат появится в restored_answer.md.

Если Codex запущен как локальный coding agent с правом чтения всего диска, файловый bridge
сам по себе не создаёт security boundary. Для честной проверки используйте отдельный
облачный чат, которому доступен только codex_input.md. Позже ручной мост заменяется одним
provider adapter с жёстким outbound gateway.

## Где лежат данные

- runtime/manifest/documents.sqlite — пути, hash/revision и offsets; локально.
- runtime/qdrant — vectors и непрозрачный payload; локально.
- runtime/model-cache — зафиксированные модели; локально.
- runtime/requests/REQUEST_ID/codex_input.md — единственный outbound-файл.
- runtime/marker-vault/REQUEST_ID.json — исходные значения; никогда не отправлять.
- runtime/requests/REQUEST_ID/restored_answer.md — приватный финальный результат.

Весь runtime исключён из Git.

## Ограничения текущего среза

- Embedded Qdrant использует локальный brute-force режим и предназначен для небольшого
  стенда. Примерно после 20 000 points переходите на Qdrant server.
- PDF без текстового слоя и legacy DOC пока пропускаются/завершаются safe error.
- ACL, ГОЗ и final-version сейчас являются консервативными pilot-заглушками.
  Поэтому весь D:\RAG TEST в этом стенде считается одной заранее разрешённой
  non-GOZ коллекцией. Прикладывать outbound-файл к cloud нельзя, пока это допущение
  не подтверждено владельцем данных.
- При аварии обновления одного документа возможен временный пропуск его chunks; смешения
  старой и новой ревизии retrieval не допускает.
- NER может пропустить неизвестную коммерческую сущность. Перед outbound нужен human review;
  реальные canary/golden тесты появятся до пилота.
- Marker Vault пока обычный локальный JSON под NTFS/BitLocker, а не зашифрованная БД.
- Mock-провайдер не отвечает по смыслу: он показывает именно payload и round-trip.
- Pilot использует multilingual MiniLM с окном 128 токенов и chunks около 450 символов.
  Для следующего quality experiment подготовлена замена на multilingual-e5-small с
  query/passage prefixes и более длинными chunks; менять модель внутри старой collection
  нельзя.
- Быстрый skip всё равно вычисляет SHA-256 каждого выбранного файла. Для десятков ТБ это
  заменяется version/change feed исходной DMS, а не ежедневным полным hashing.

Архитектура подробнее: docs/ARCHITECTURE.md. Практический сценарий: docs/WALKTHROUGH.md.
Fine-tuning развивается отдельно в соседнем репозитории ../ft-bert.
