# Развёртывание

## Состав одного сервера

| Компонент | Runtime | Порт | Данные |
|---|---|---:|---|
| Open WebUI `v0.11.1` | Docker | `0.0.0.0:3000` | volume `universal_rag_open_webui_data` |
| Universal RAG API | Windows/Python | `0.0.0.0:8000` | `runtime/data/sessions`, `runtime/state`, `runtime/diagnostics` |
| Qdrant `v1.19.0` | Docker | `127.0.0.1:6333` | volume `secure_rag_qdrant_storage` |
| Manifest | SQLite | без порта | `runtime/data/manifest/documents.sqlite` |
| Nextcloud corpus | Windows folder | без порта | путь из `SECURE_RAG_SOURCE_ROOT` |

В LAN публикуется только Open WebUI. API слушает host-интерфейс, чтобы Docker Desktop мог
обращаться к `host.docker.internal:8000`, но защищён отдельным Bearer key. На сервере следует
добавить Windows Firewall rule, разрешающую `3000/tcp` только корпоративным подсетям и
запрещающую прямой клиентский доступ к `8000/tcp` и `6333/tcp`. Для постоянной эксплуатации
перед Open WebUI нужен внутренний reverse proxy с HTTPS и WebSocket support.

## Lifecycle

```powershell
.\scripts\start-local.ps1
.\scripts\status-local.ps1
.\scripts\stop-local.ps1
.\scripts\stop-local.ps1 -StopQdrant
```

`stop-local.ps1` сохраняет оба Docker volume и весь `runtime`. Никогда не используйте
`docker compose down --volumes` в обычной эксплуатации.

Open WebUI получает одну OpenAI-compatible connection:

```text
http://host.docker.internal:8000/v1
```

Её key берётся из `SECURE_RAG_API_KEY`. `WEBUI_SECRET_KEY` должен оставаться стабильным:
его смена инвалидирует сессии и данные, зашифрованные Open WebUI.

Compose включает `ENABLE_FORWARD_USER_INFO_HEADERS` и подписывает identity JWT тем же
закрытым host-to-host key. Chat completion без корректной подписи и
`X-OpenWebUI-Chat-Id` отклоняется. Backend хранит ограниченную копию истории и retrieval state
в `runtime/data/sessions/chat-state.sqlite`; Open WebUI остаётся источником полной истории.

Встроенный retrieval Open WebUI отключён: документы индексирует Universal RAG, поэтому при
первом запуске UI не загружает вторую embedding-модель и не создаёт параллельный индекс.
Загрузка файлов через стандартный uploader Open WebUI в эту схему не входит.

## Первый запуск

1. Создайте постоянный clone, например `C:\Services\Universal-RAG`.
2. Установите Python 3.12, Docker Desktop и зависимости `.[web]`.
3. Скопируйте `.env.example` в `.env`, задайте source root, URL и два разных random secret.
4. Выполните `prepare-models`, полный `index`, затем `start-local.ps1`.
5. Откройте `http://<server-ip>:3000` и создайте первый admin account.
6. Создавайте пользователей/группы через admin panel; не включайте публичную регистрацию.
7. Настройте backup вне Git и выполните тестовое восстановление.

### Переход с версии 0.1

`config/app.yaml` заменяет `config/pilot.yaml`, а новая collection называется
`universal_rag_e5_v1` и использует access group `employees`. Локальный `.env` нужно перевести
на `SECURE_RAG_CONFIG=config\app.yaml`, после чего выполнить полный `index --max-files 0`.
Старая collection остаётся в Qdrant volume и не удаляется автоматически; удалять её можно
только после snapshot и проверки нового индекса.

Перед переиндексацией старое согласованное состояние можно сохранить без возврата конфигурации:

```powershell
.\scripts\backup-state.ps1 `
    -DestinationRoot D:\Universal-RAG-backups `
    -QdrantCollection secure_rag_pilot_e5_v1
```

## CI/CD

`.github/workflows/ci.yml` выполняет проверки на GitHub-hosted Windows runner. Deployment после
push в `main` выполняется только runner-ом с labels:

```text
self-hosted, Windows, X64, universal-rag
```

На Windows service account runner-а задайте системную переменную:

```powershell
[Environment]::SetEnvironmentVariable(
    "UNIVERSAL_RAG_DEPLOY_ROOT",
    "C:\Services\Universal-RAG",
    "Machine"
)
```

Перезапустите runner service после изменения переменной. У service account должны быть права
на clone, `.venv`, Docker Desktop/Engine и runtime, но только read-доступ к разрешённой
Nextcloud-копии. Развёртывание делает `git fetch` + `merge --ff-only`, устанавливает Python
dependencies, обновляет закреплённые container images и перезапускает UI/API. Dirty working
tree или non-fast-forward останавливают deployment.

После ручной проверки runner-а создайте GitHub repository variable
`ENABLE_INTERNAL_DEPLOY=true`. Пока переменной нет, job `deploy` пропускается и push запускает
только CI.

GitHub Environment `internal-production` рекомендуется защитить required reviewer-ом, пока
нет автоматического smoke test и rollback. Первый deployment выполняется вручную, чтобы
создать `.env`, runtime и зарегистрировать runner; после этого работает pull-based CD.

## Обновление и rollback

Перед обновлением создайте snapshot состояния. После CI:

```powershell
.\scripts\deploy-host.ps1
```

Rollback к старому коду допускается только если release notes подтверждают совместимость
SQLite schema, Qdrant index signature и Open WebUI database. Для Open WebUI нельзя подключать
pre-release и stable image к одному volume. При несовместимой миграции сначала восстановите
соответствующий state snapshot, затем переключайте Git revision.

## Следующие production-шаги

- внутренний DNS + TLS certificate + reverse proxy;
- SSO/OIDC и сопоставление Open WebUI user/group с document ACL;
- service account вместо интерактивного Windows-пользователя;
- scheduled Nextcloud sync/index job с single-writer lock;
- централизованные метрики и backup retention;
- отдельный Label Studio stack с PostgreSQL и object storage.

Официальные ориентиры: [Open WebUI Docker deployment](https://docs.openwebui.com/getting-started/quick-start/),
[hardening](https://docs.openwebui.com/getting-started/advanced-topics/hardening/) и
[monitoring](https://docs.openwebui.com/reference/monitoring/).
