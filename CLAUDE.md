# CLAUDE.md - Project Instructions

## CRITICAL RULES - READ FIRST

### GIT SAFETY - ABSOLUTELY NEVER VIOLATE

**MANDATORY: Run `/boot` workflow as your very first action in EVERY session.**
Check `.agent/workflows/boot.md` for pre-flight requirements.

**NEVER run destructive git commands without EXPLICIT user approval:**

```bash
# FORBIDDEN without explicit permission:
git reset --hard
git checkout -- <file>
git checkout <file>
git restore <file>
git clean -fd
```

**BEFORE any git operation:**
1. Run `git status` first
2. If uncommitted changes exist: STOP
3. Tell the user what files have uncommitted changes
4. ASK if they want to stash or commit first
5. ONLY proceed after explicit approval

**Uncommitted changes may be the user's working solution. Destroying them wastes hours of work.**

### WORKTREE SAFETY - READ FIRST

**STEP ZERO for EVERY session:**
1. Run `git worktree list` immediately.
2. Check if your current directory is the root repo (`.../goldrush-sluice-3d`) on `master`.
3. **If yes: STOP.** You are in the "Read-Only" master root.
4. Search for a relevant worktree or ASK the user to create/assign one.
5. **NEVER** write implementation code or modify project files (except docs/plans) in the root `master` branch.

**WORKTREE-FIRST COMMANDS:**
- Start new feature: `git worktree add .worktrees/<name> -b feature/<name>`
- List active zones: `git worktree list`
- Remove zone: `git worktree remove .worktrees/<name>` (exit directory first!)

**NEVER remove a worktree while your shell is inside it.**

### NO PATCH FIXING - DEBUG NEW CODE

When new code breaks something:

```
Working System + New Code = Problem
                    ↓
            Problem is in NEW CODE
                    ↓
            Debug the NEW code ONLY
                    ↓
            DO NOT touch working systems
```

**FORBIDDEN responses to bugs in new code:**
- ❌ Changing unrelated constants (damping, gravity, etc.)
- ❌ Adding velocity boosts or hacks
- ❌ Looking at unrelated todos
- ❌ Modifying existing working systems
- ❌ Adding new mechanisms without checking if they exist

**REQUIRED response:**
1. Acknowledge the NEW code is the problem
2. Read and debug the NEW code specifically
3. Write tests for the NEW code
4. Fix only what's broken

### INCREMENTAL CHANGES ONLY

- Make ONE small change at a time
- Test after each change
- Commit working states frequently
- Never make large refactors without user approval

### PERFORMANCE TESTING - USE REALISTIC SCALES

**Short tests with few particles tell us NOTHING.**

When testing performance optimizations:
- Run simulation for **at least 60 seconds** (preferably 2+ minutes)
- Use **100k+ particles** minimum, ideally 500k-1M
- Wait for steady state (particle count stabilizes)
- Measure after warmup, not during initial ramp-up

**BAD test:** 5 seconds, 5000 particles → "looks faster!"
**GOOD test:** 2 minutes, 500k particles → actual FPS comparison

The goal is 1 million particles. Optimize for that, not for toy demos.

## Project: Goldrush Fluid Miner

FLIP/APIC fluid simulation with sediment transport.

### Key Files
- `crates/sim/src/flip.rs` - Main simulation
- `crates/sim/src/grid.rs` - MAC grid, pressure solver
- `crates/sim/src/particle.rs` - Particle types
- `crates/game/src/main.rs` - Visual demo

### Running
```bash
cargo run --bin game --release        # Main visual sim
cargo run --example <name> --release  # Examples
cargo test -p sim                     # Tests
```

## Documentation

See `docs/solutions/` for documented problems and solutions.

## CONVERSATION ARCHIVING

**This project's development history is valuable for retrospective documentation.**

### Before Major Git Operations

Before `git push`, `git merge`, or ending a long session, **remind the user:**

> "📝 Consider exporting this conversation to `archive/conversations/` before proceeding.  
> Suggested filename: `YYYY-MM-DD-<brief-description>.md`"

### Archive Location
```
archive/conversations/
├── 2025-12-20-initial-gpu-rendering.md
├── 2025-12-21-sediment-physics-journey.md
├── 2026-01-08-3d-flip-integration.md
└── ...
```

### Why This Matters
The user is building a physics simulation iteratively with LLM assistance. The conversation logs document:
- Design decisions and alternatives considered
- Debugging journeys and dead ends (valuable lessons)
- Evolution from 2D → 3D, CPU → GPU
- Material for a future blog post about LLM-assisted development

**Note:** The AI cannot programmatically export conversations. This is a manual reminder for the user to use their UI's export function.
