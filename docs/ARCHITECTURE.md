# Архитектура локального среза

## Indexing

    Read-only source
      → safe inventory
      → document_id + SHA-256 revision
      → extractor
      → character windows with overlap
      → local sentence embedding
      → Qdrant upsert

Manifest хранит path локально. Qdrant хранит document_id, revision, character offsets,
build_id и policy fields. Исходный chunk в payload не копируется.

`delete_document` и каждый bounded upsert batch идемпотентны: point IDs определяются
из document/revision/ordinal. Qdrant client и отдельная write-операция имеют timeout
60 секунд; временные timeout/transport ошибки повторяются не более трёх попыток с
коротким backoff. Если попытки исчерпаны, индексатор best-effort удаляет частично
записанный документ и сохраняет только безопасную категорию ошибки. Следующий проход
проверяет точное равенство manifest chunks и Qdrant points перед быстрым skip.

Persistent extracted-text cache в безопасном baseline выключен. Его явный opt-in
создаёт локальную копию source-derived raw text и требует отдельной retention/encryption
policy. Индексатор имеет межпроцессный single-writer lock; удаление unseen документов
выполняется только явным `--prune-missing` после authoritative scan.

Web-процесс использует отдельный bounded LRU в RAM (revision + extraction-policy key).
Он не участвует в CLI/indexing, не пишет raw text в runtime и не отменяет полный SHA-256
исходника перед каждым cache lookup. `st.cache_resource` общий для web-сессий процесса,
поэтому это только single-user pilot; multi-user deployment должен разделять cache по
security principal или процессам. OS pagefile/crash dumps защищаются политикой хоста.

## Online flow

    raw local question
      → local query embedding
      → Qdrant filter inside search
      → IDs + offsets
      → reread original locally
      → regex + NER spans
      → overlap resolver
      → Marker Vault + sanitized Markdown
      → no-tools Responses API / manual bridge / local mock
      → strict marker validation
      → durable sanitized response staging
      → local restoration

Raw query не надо маркировать до retrieval: embedding и поиск локальны. Privacy boundary
начинается перед созданием codex_input.txt.

## Контракты

- DocumentRecord: stable ID, local path, revision, size/mtime and status.
- ChunkRecord: ID, document/revision, ordinal, offsets, raw text only in process memory.
- RetrievalHit: local text plus opaque source identity and score.
- EntitySpan: start, end, normalized label, score, detector and priority.
- MarkerState: marker-to-value map; exists only in the request vault.
- BridgeResult: paths and safe counters, without raw values.

## Выбор моделей

- intfloat/multilingual-e5-small, revision
  614241f622f53c4eeff9890bdc4f31cfecc418b3: текущий dense baseline,
  384 dimensions, окно 512 токенов, Russian, query/passage prefixes, MIT.
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, revision
  e8f8c211226b894fcb81acc59f3b34ba3efd5f42: предыдущий лёгкий baseline,
  384 dimensions, 50 languages, Apache-2.0, max sequence length 128.
- LLAIMlegal/ru-legal-ner, revision
  c0313a42ac147ccc38f5c3b0b3e77cab53208683: small legal-domain candidate; MIT;
  requires internal validation.
- viktor-shcherb/sberbank-rubert-base-collection3, revision
  65ccbf8d2d25229c3858631bb24bb794c0c6907c: PER/ORG/LOC; Apache-2.0 weights,
  but dataset provenance should be reviewed.

All models load with trust_remote_code disabled and safetensors where applicable.

Primary references:

- https://huggingface.co/intfloat/multilingual-e5-small
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/LLAIMlegal/ru-legal-ner
- https://huggingface.co/viktor-shcherb/sberbank-rubert-base-collection3
- https://github.com/qdrant/qdrant-client#local-mode

## Следующая замена заглушек

1. Локальный Qdrant server v1.19.0 → production server с backup и API key.
2. Pilot policy fields → trusted ACL/ГОЗ/final-version connector.
3. Plain JSON Marker Vault → encrypted per-user/session storage.
4. Provider policy → production API credentials, approved model and outbound audit.
5. Baseline NER → immutable approved artifact from ft-bert/MLflow.
6. Only then Bitrix, tool gateway, hybrid retrieval and reranker.

Server mode уже включён: `qdrant.mode=server`, URL можно переопределить через
`SECURE_RAG_QDRANT_URL`, а ключ передаётся только через `SECURE_RAG_QDRANT_API_KEY`.

## Provider boundary

`auto` выбирает официальный Responses API, только если процесс получил
`OPENAI_API_KEY`; иначе выбирается локальный `stub`. В API-вызове не передаётся
параметр `tools`, а `store=False`. Авторизация подписки Codex не заменяет API key.

`codex-local` — отдельный явно небезопасный режим. Его `read_only` sandbox ограничивает
запись, но не список читаемых файлов: на этом ПК SDK смог прочитать доступный файл на
`D:\RAG TEST`. Поэтому local coding agent не находится за privacy boundary, даже если
его cwd вынесен из репозитория.
