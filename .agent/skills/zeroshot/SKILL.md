# Zeroshot Multi-Agent Task Runner

## Overview

Zeroshot is a multi-agent coordination framework that runs isolated agents with validators that check each other's work. Unlike single-agent execution, validators didn't write the code so they can't lie about tests.

## When to Use Zeroshot

**Use zeroshot when:**
- Task has clear acceptance criteria
- "Done" is measurable (tests pass, builds compile)
- Well-defined requirements exist
- Long-running batch tasks (overnight runs)

**Don't use zeroshot when:**
- Exploratory work (unknown unknowns)
- No clear completion criteria
- Need interactive back-and-forth

| Scenario | Use? | Why |
|----------|------|-----|
| Add rate limiting with specific behavior | ✅ | Clear requirements |
| Implement feature from detailed plan | ✅ | Defined end state |
| Fix specific bug | ✅ | Success is measurable |
| Fix N lint violations | ✅ | Clear completion |
| "Make the app faster" | ❌ | Needs exploration first |
| "Improve the codebase" | ❌ | No acceptance criteria |

## Commands

### Running Tasks

```bash
# From GitHub issue
zeroshot run 123

# From description
zeroshot run "Add dark mode with system preference detection"

# With isolation levels (cascading: --ship → --pr → --worktree)
zeroshot run 123 --worktree    # Git worktree isolation
zeroshot run 123 --pr          # Worktree + creates PR
zeroshot run 123 --ship        # Worktree + PR + auto-merge

# Background/daemon mode (recommended for long tasks)
zeroshot run 123 -d
zeroshot run 123 --worktree -d

# Full automation, background
zeroshot run 123 --ship -d
```

### Monitoring

```bash
zeroshot list                  # See all running clusters
zeroshot status <id>           # Cluster status
zeroshot logs <id> -f          # Follow output live
zeroshot watch                 # TUI dashboard
```

### Control

```bash
zeroshot resume <id>           # Continue after crash
zeroshot kill <id>             # Stop cluster
zeroshot clean                 # Remove old records
zeroshot purge                 # NUCLEAR: kill all + delete all
```

## How It Works

Zeroshot classifies tasks by complexity and spawns appropriate agents:

| Complexity | Planner | Worker | Validators |
|------------|---------|--------|------------|
| TRIVIAL | - | haiku | 0 |
| SIMPLE | - | sonnet | 1 (generic) |
| STANDARD | sonnet | sonnet | 2 (requirements, code) |
| CRITICAL | opus | sonnet | 5 (req, code, security, tester, adversarial) |

### Agent Flow

```
TASK → CONDUCTOR (classifies) → PLANNER → WORKER → VALIDATORS
                                              ↑         │
                                              └─ REJECT ─┘
                                                   │
                                              ALL OK → COMPLETE
```

Validators actually run tests and check work. If they reject, the worker fixes and retries.

## Best Practices

### Writing Good Task Descriptions

**Good:**
```bash
zeroshot run "Add rate limiting: sliding window algorithm, per-IP tracking,
return 429 with Retry-After header, configurable limits per endpoint"
```

**Bad:**
```bash
zeroshot run "Add rate limiting"  # Too vague
```

### Reference Existing Plans

If you have a detailed plan file:
```bash
zeroshot run "Implement the feature as specified in plans/my-feature.md.
Acceptance: all tests pass, cargo check compiles."
```

### Long-Running Tasks

Use daemon mode (`-d`) for tasks that take time:
```bash
zeroshot run "Fix all 500 type errors" --worktree -d
```

Check back with:
```bash
zeroshot status <id>
zeroshot logs <id> | tail -100
```

### Crash Recovery

If a cluster crashes or your machine restarts:
```bash
zeroshot resume <id>
```

## Integration with Claude Code Workflow

1. **Explore** with Claude Code interactively to understand the problem
2. **Plan** and write detailed requirements to a plan file
3. **Delegate** to zeroshot with the plan reference
4. **Review** the PR or merged changes

Example workflow:
```bash
# 1. Interactive exploration (Claude Code)
claude "help me understand how auth works in this codebase"

# 2. Write plan
claude "write a plan for JWT auth migration to plans/jwt-migration.md"

# 3. Delegate to zeroshot
zeroshot run "Implement plans/jwt-migration.md. Acceptance: all auth tests pass." --pr -d

# 4. Review the PR when done
gh pr view
```

## Troubleshooting

### Task Not Progressing

Check logs for errors:
```bash
zeroshot logs <id> | grep -i error
```

### Validators Keep Rejecting

The task may be too vague. Kill and restart with clearer acceptance criteria:
```bash
zeroshot kill <id>
zeroshot run "..." --worktree -d  # With better description
```

### Worktree Conflicts

If worktree has conflicts with main:
```bash
zeroshot kill <id>
# Manually clean up worktree if needed
git worktree list
git worktree remove <path>
```
