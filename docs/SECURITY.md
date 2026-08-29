# Security invariants прототипа

1. Retrieval и embeddings выполняются локально по исходному запросу.
2. Qdrant filter содержит access_group, goz=false и is_final=true.
3. В outbound не входят raw path, Marker Vault и исходный filename.
4. Query, chunks и source name маркируются в одном request namespace.
5. marker-vault отделён от outbox, исключён из Git и не прикладывается к Codex.
6. Неизвестный marker в cloud-ответе блокирует restoration.
7. Документный текст и provider output считаются недоверенными данными.
8. Ошибка sanitizer завершает запрос, а не переключает его на raw mode.

Эти меры снижают риск, но baseline NER не доказывает отсутствие пропусков. До реальных
коммерческих данных нужен размеченный security set с canary в query, chunk, filename,
metadata и error, измерение recall каждого критичного класса и ручной outbound gate.
