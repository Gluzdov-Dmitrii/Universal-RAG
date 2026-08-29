# Практический walkthrough

## Урок 1 — увидеть retrieval без NER

1. Выполните doctor.
2. Проиндексируйте 10 файлов.
3. Задайте безопасный тестовый вопрос в режиме regex + mock.
4. Откройте manifest.json: там версии и counters, но нет исходного текста.
5. Откройте codex_input.md и restored_answer.md рядом.

Цель: увидеть разницу между local raw retrieval и outbound sanitized context.

## Урок 2 — проверить маркеры

Создайте безопасный тестовый TXT с вымышленными PER, ORG, email, phone, ИНН, СНИЛС,
паспортом, счётом и суммой. Перестройте индекс. Задайте вопрос, содержащий те же сущности.

Проверьте:

- одинаковое точное значение получает одинаковый marker в query и context;
- filename скрыт целиком;
- runtime/marker-vault/REQUEST_ID.json содержит обратную таблицу и остаётся локально;
- изменение marker на неизвестный блокирует demarker;
- local mock возвращает исходные значения только в restored_answer.md.

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
2. Лично просмотрите codex_input.md.
3. Передайте только этот файл в изолированный cloud-чат.
4. Попросите вернуть Markdown и сохранить markers.
5. Вставьте ответ в web demarker.
6. Сравните cloud-ответ и local restored_answer.md.

## Урок 5 — первый retrieval eval

Подготовьте 10 безопасных вопросов и ожидаемый document_id. Зафиксируйте top-k,
Recall@k, неправильные документы и время. Лишь после этого решайте, нужен ли BM25.

Dense baseline сначала отвечает на вопрос «достаточно ли семантики?». BM25 добавляется,
если теряются точные обозначения, номера, коды и редкие термины. Для промышленного
варианта разумен hybrid retrieval: BM25 + dense candidates → reranker.
