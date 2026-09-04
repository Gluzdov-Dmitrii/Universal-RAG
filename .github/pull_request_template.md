## Что изменено


## Как проверено

- [ ] `python -m ruff check src tests`
- [ ] `python -m pytest`
- [ ] Compose validation, если менялся deployment

## Данные и безопасность

- [ ] В diff нет credentials, реальных документов, путей или runtime-артефактов
- [ ] Privacy/ACL boundary не изменён либо изменение описано в `docs/ARCHITECTURE.md`
- [ ] Для несовместимого index/schema change описаны migration и rollback
