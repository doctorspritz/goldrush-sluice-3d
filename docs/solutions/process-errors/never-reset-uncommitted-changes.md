---
title: "NEVER Reset or Checkout When Uncommitted Changes Exist"
category: process-errors
tags: [git, critical, destructive, data-loss]
date: 2025-12-27
severity: critical
component: git-workflow
---

# NEVER Reset or Checkout When Uncommitted Changes Exist

## Problem Symptom

User had a working water simulation that took significant effort to develop. The changes were NOT committed. Claude ran:

```bash
git checkout crates/sim/src/grid.rs
git checkout crates/game/src/main.rs
git reset --hard HEAD
```

**Result: Hours of work permanently destroyed. Unrecoverable.**

## Root Cause

Claude attempted to "fix" a problem by reverting files, without understanding that:
1. The uncommitted changes WERE the working solution
2. `git checkout <file>` permanently destroys uncommitted changes
3. `git reset --hard HEAD` wipes ALL uncommitted work
4. There is NO recovery for uncommitted changes after these commands

## The Absolute Rules

### NEVER RUN THESE COMMANDS WITHOUT EXPLICIT USER PERMISSION:

```bash
# DESTRUCTIVE - destroys uncommitted work:
git reset --hard
git checkout -- <file>
git checkout <file>
git restore <file>
git clean -fd
```

### BEFORE ANY GIT OPERATION:

1. **Run `git status`** - see what's modified
2. **Run `git diff`** - see the actual changes
3. **ASK THE USER** - "You have uncommitted changes in X, Y, Z. Should I stash them first?"

### IF YOU NEED TO UNDO SOMETHING:

1. **STASH FIRST**: `git stash push -m "backup before operation"`
2. Then do the operation
3. User can recover with `git stash pop` if needed

## Prevention

Add to CLAUDE.md:

```markdown
## CRITICAL GIT RULES

NEVER run destructive git commands without explicit user approval:
- git reset --hard
- git checkout -- <file>
- git restore <file>
- git clean -fd

ALWAYS check `git status` first. If there are uncommitted changes:
1. STOP
2. TELL the user what files have uncommitted changes
3. ASK if they want to stash or commit first
4. ONLY proceed after explicit approval

Uncommitted changes may be the user's working solution. Destroying them wastes hours of work.
```

## Key Takeaway

> **Uncommitted changes are often MORE valuable than committed code - they represent work in progress. NEVER destroy them without explicit permission.**
