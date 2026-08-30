# Практический walkthrough

## Урок 1 — увидеть retrieval без NER

1. Выполните doctor.
2. Проиндексируйте 10 файлов.
3. Задайте безопасный тестовый вопрос в режиме regex + mock.
4. Откройте manifest.json: там версии и counters, но нет исходного текста.
5. Откройте codex_input.txt и restored_answer.txt рядом как plain text.

Цель: увидеть разницу между local raw retrieval и outbound sanitized context.

## Урок 2 — проверить маркеры

Создайте безопасный тестовый TXT с вымышленными PER, ORG, email, phone, ИНН, СНИЛС,
паспортом, счётом и суммой. Перестройте индекс. Задайте вопрос, содержащий те же сущности.

Проверьте:

- одинаковое точное значение получает одинаковый marker в query и context;
- filename скрыт целиком;
- runtime/marker-vault/REQUEST_ID.json содержит обратную таблицу и остаётся локально;
- изменение marker на неизвестный блокирует demarker;
- local mock возвращает исходные значения только в restored_answer.txt.

## Урок 3 — сравнить NER

Повторите один вопрос четырежды:

- regex;
- legal;
- collection3;
- all.

Сравните marker_count и сам sanitized payload. Не сравнивайте только confidence:
важнее false negatives критичных классов.

## Урок 4 — ручной Codex

1. Выберите all + manual.
2. Лично просмотрите codex_input.txt как обычный текст, без Markdown preview.
3. Передайте только этот файл в изолированный cloud-чат.
4. Попросите вернуть обычный текст и сохранить markers.
5. Сохраните ответ как `runtime/requests/REQUEST_ID/codex_output.txt` и выполните
   `secure-rag demark REQUEST_ID`.
6. Сравните cloud-ответ и local restored_answer.txt.

## Урок 4.1 — автоматический provider

Сначала оставьте `SECURE_RAG_PROVIDER=auto` без `OPENAI_API_KEY`: должен отработать
локальный stub. Затем задайте API key только в окружении процесса и повторите тест на
вымышленных данных. В manifest проверьте `provider=responses` и
`provider_boundary=no-tools-api`. Не используйте `codex-local` как privacy boundary.

## Урок 5 — первый retrieval eval

Подготовьте 10 безопасных вопросов и ожидаемый document_id. Зафиксируйте top-k,
Recall@k, неправильные документы и время. Лишь после этого решайте, нужен ли BM25.

Dense baseline сначала отвечает на вопрос «достаточно ли семантики?». BM25 добавляется,
если теряются точные обозначения, номера, коды и редкие термины. Для промышленного
варианта разумен hybrid retrieval: BM25 + dense candidates → reranker.
