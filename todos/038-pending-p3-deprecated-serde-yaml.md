---
status: pending
priority: p3
issue_id: "038"
tags: [code-review, dependency, maintenance]
dependencies: []
---

# Deprecated serde_yaml Dependency

## Problem Statement

The `serde_yaml` crate version 0.9.34 is marked as deprecated. Deprecated crates may not receive security updates.

## Findings

**Location:** `crates/game/Cargo.toml`, line 24

```toml
serde_yaml = "0.9"
```

**Status:** Crate maintainers recommend migrating to alternatives.

## Proposed Solutions

### Option A: Remove YAML support (if unused)
**Pros:** Removes deprecated dependency entirely
**Cons:** Loses YAML config capability
**Effort:** Small
**Risk:** Low

### Option B: Migrate to serde_yaml2 or alternative
**Pros:** Maintains YAML support with maintained crate
**Cons:** Potential API changes
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] No deprecated dependencies
- [ ] If YAML needed, use maintained crate

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by security-sentinel agent |
