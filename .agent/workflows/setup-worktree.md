---
description: Create a new worktree for a feature or bugfix
---

// turbo-all
1. Ask the user for the branch name or task name if not provided.
2. Create the worktree:
   ```bash
   git worktree add .worktrees/<name> -b feature/<name>
   ```
3. Inform the user and switch context to the new directory.
4. **WARNING**: Ensure you are in the root directory before running this.
