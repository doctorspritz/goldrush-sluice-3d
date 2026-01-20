---
status: pending
priority: p2
issue_id: "032"
tags: [code-review, security, input-validation]
dependencies: []
---

# Missing Bounds Validation on Deserialized Data

## Problem Statement

The application deserializes JSON/YAML files without bounds validation on numeric fields. A malformed scenario file could specify extreme values causing memory exhaustion or panics.

## Findings

**Locations:**
- `crates/game/src/scenario.rs` (lines 71-83)
- `crates/game/src/editor.rs` (lines 799-811)
- `crates/game/src/washplant/config.rs` (lines 296-322)

```rust
pub fn load_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let scenario = serde_json::from_str(&json)?;  // No validation
    Ok(scenario)
}
```

**Potential issues:**
- Extreme grid dimensions causing memory exhaustion
- Negative dimensions causing panics
- Extreme particle counts

## Proposed Solutions

### Option A: Post-deserialization validation (Recommended)
**Pros:** Simple, catches issues early
**Cons:** Manual validation code
**Effort:** Small (1-2 hours)
**Risk:** Low

```rust
impl Scenario {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.grid_width > 1000 { return Err(...); }
        if self.max_particles > 10_000_000 { return Err(...); }
        // ...
    }
}
```

## Acceptance Criteria

- [ ] Grid dimensions validated (positive, reasonable bounds)
- [ ] Particle counts validated
- [ ] Helpful error messages for invalid configs

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by security-sentinel agent |
