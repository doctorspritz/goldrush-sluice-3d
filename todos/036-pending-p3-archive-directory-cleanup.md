---
status: pending
priority: p3
issue_id: "036"
tags: [code-review, cleanup, dead-code]
dependencies: []
---

# Archive Directory Contains 10K+ Lines of Dead Examples

## Problem Statement

The `crates/game/examples/archive/` directory contains 30+ archived example files (~10,000+ lines) that are no longer used but still tracked in git.

## Findings

**Location:** `crates/game/examples/archive/`

**Files include:**
- box_3d_test.rs
- dam_break_3d.rs
- sdf_erosion.rs
- And 27 more...

These serve no runtime purpose but add to repo size and maintenance burden.

## Proposed Solutions

### Option A: Delete archive directory (Recommended)
**Pros:** Cleaner repo, reduced noise
**Cons:** Lose quick reference (but git history preserves)
**Effort:** Trivial
**Risk:** None

### Option B: Move to separate documentation branch
**Pros:** Keeps reference accessible
**Cons:** More complex
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] Archive directory removed
- [ ] Git history preserved (naturally)
- [ ] No broken imports

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by code-simplicity-reviewer agent |
