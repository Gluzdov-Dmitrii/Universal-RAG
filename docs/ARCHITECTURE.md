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

## Online flow

    raw local question
      → local query embedding
      → Qdrant filter inside search
      → IDs + offsets
      → reread original locally
      → regex + NER spans
      → overlap resolver
      → Marker Vault + sanitized Markdown
      → manual Codex bridge / mock
      → strict marker validation
      → local restoration

Raw query не надо маркировать до retrieval: embedding и поиск локальны. Privacy boundary
начинается перед созданием codex_input.md.

## Контракты

- DocumentRecord: stable ID, local path, revision, size/mtime and status.
- ChunkRecord: ID, document/revision, ordinal, offsets, raw text only in process memory.
- RetrievalHit: local text plus opaque source identity and score.
- EntitySpan: start, end, normalized label, score, detector and priority.
- MarkerState: marker-to-value map; exists only in the request vault.
- BridgeResult: paths and safe counters, without raw values.

## Выбор моделей

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, revision
  e8f8c211226b894fcb81acc59f3b34ba3efd5f42: текущий доступный на ПК baseline,
  384 dimensions, 50 languages, Apache-2.0, max sequence length 128.
- intfloat/multilingual-e5-small, revision
  614241f622f53c4eeff9890bdc4f31cfecc418b3: следующий quality experiment,
  384 dimensions, Russian, query/passage prefixes, MIT.
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

1. Embedded Qdrant → Qdrant server v1.19.0.
2. Pilot policy fields → trusted ACL/ГОЗ/final-version connector.
3. Plain JSON Marker Vault → encrypted per-user/session storage.
4. Manual bridge → one cloud provider adapter behind the same gateway.
5. Baseline NER → immutable approved artifact from ft-bert/MLflow.
6. Only then Bitrix, tool gateway, hybrid retrieval and reranker.

Переключение на server mode уже предусмотрено: qdrant.mode=server, qdrant.url либо
SECURE_RAG_QDRANT_URL, а ключ передаётся только через SECURE_RAG_QDRANT_API_KEY.
