# Repository workflow

- When a task changes tracked files and the implementation is complete, inspect `git status` and the diff, run the focused checks appropriate for the change, and create a local Git commit before handing the task back.
- Stage only files related to the current task. Never stage secrets, `.env` files, tokens, runtime state, caches, generated artifacts, or unrelated user changes unless the user explicitly asks for them.
- Use a short imperative commit message that describes the result.
- Do not push, amend, rebase, reset, or rewrite history unless the user explicitly asks.
- If checks fail, report the failure and fix it when possible before committing. Do not claim that a change is complete while leaving known task-caused failures unexplained.
- For read-only questions or tasks that make no file changes, do not create an empty commit.
