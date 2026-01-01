---
title: "Always Confirm Approach Before Major Implementation"
type: workflow-principle
category: development-workflow
severity: medium
status: resolved
date: 2025-12-19

problem:
  type: premature_implementation
  core_issue: "Starting major architectural implementation without explicit user confirmation"
  symptoms:
    - "Documentation phase flows directly into implementation without checkpoint"
    - "User loses control over implementation timing"
    - "No opportunity to review, adjust, or defer before coding starts"
    - "User must interrupt to regain control"

root_cause: "Treating completed documentation as implicit authorization to implement"

tags:
  - workflow
  - communication
  - user-confirmation
  - implementation-control
  - planning-vs-execution
  - consent-driven-development

related_files:
  - docs/solutions/architecture-decisions/foundational-physics-over-workarounds.md
  - plans/pressure-based-particle-physics-engine.md
  - plans/fix-water-leveling.md
---

# Always Confirm Approach Before Major Implementation

## The Insight

> "Don't implement without asking first."

Documentation and implementation are distinct phases. Completing one doesn't automatically authorize starting the next.

## The Problem

After documenting a design philosophy about foundational physics systems, implementation began immediately:
- Todo list created for implementation steps
- Files read to understand current system
- Code changes about to start

The user had to interrupt because they wanted to **confirm the approach first**. The documentation was complete, but that didn't mean "now implement."

## Root Cause

**Eagerness without checkpoint.** When a clear architectural direction is documented, there's natural momentum to start implementing. However:

- **Documentation = What should be done**
- **Implementation = Actually doing it**

These are separate decisions. Completing one doesn't grant permission for the other.

## The Solution

### Always Pause After Design Phase

After documenting or discussing architectural changes, explicitly ask:

```
"I've documented the approach. Should I proceed with implementation,
or would you like to review the plan first?"
```

This creates a natural checkpoint, giving the user control over when work begins.

### Good Pattern

```
User: "Here's the design for the new physics system"
Claude: [Documents the design]
Claude: "I've documented the foundational physics approach.
        Should I proceed with implementing this?"
User: "Yes, start with the velocity field"
Claude: [Begins implementation]
```

### Bad Pattern

```
User: "Here's the design for the new physics system"
Claude: [Documents the design]
Claude: [Creates todos, reads files, starts implementing]
User: "Wait, I didn't say to implement yet"
```

## When This Applies

Use confirmation checkpoint for:

| Situation | Why |
|-----------|-----|
| Architectural changes | Core systems affect everything |
| Multi-file refactors | Scope is significant |
| Foundational rewrites | High risk, hard to reverse |
| Breaking changes | Could affect existing functionality |
| Design philosophy implementations | Translating principles into code |

## When This Doesn't Apply

Proceed directly without confirmation for:

| Situation | Why |
|-----------|-----|
| Small bug fixes | Clear scope, low risk |
| Explicit requests | "Implement the velocity system" |
| Single-file changes | Isolated, obvious intent |
| Trivial updates | Typos, formatting |
| Direct commands | User already said what to do |

## Key Principle

**Documentation is not authorization.**

Writing down what should be done is different from permission to do it. When architectural work is involved, always ask before executing.

## Related Documentation

- [Foundational Physics Over Workarounds](../architecture-decisions/foundational-physics-over-workarounds.md) - The document that triggered this insight
- [Pressure-Based Physics Engine Plan](../../../plans/pressure-based-particle-physics-engine.md) - Shows decision framework with options
- [Water Leveling Fix Attempts](../../../plans/fix-water-leveling.md) - Documents cost of implementing without confirmed approach (4 failed attempts)

## Prevention

Add explicit checkpoints at phase transitions:

1. **Research complete** → "Ready to document approach?"
2. **Documentation complete** → "Ready to implement?"
3. **Implementation complete** → "Ready to test/deploy?"

Each transition should be user-driven, not assumed.
