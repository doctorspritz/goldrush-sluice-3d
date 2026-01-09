---
description: Mandatory pre-flight check for every agent session
---

// turbo-all
1. Verify worktree status:
   ```bash
   git worktree list
   ```
2. Check current working directory:
   ```bash
   pwd
   ```
3. Check current branch:
   ```bash
   git branch --show-current
   ```
4. **DECISION**:
   - If in root repo on `master`: **DO NOT PROCEED**.
   - If a relevant worktree exists: `cd` into it.
   - If no relevant worktree exists: Ask user to `git worktree add .worktrees/<task> -b <branch>`.
