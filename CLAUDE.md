# CLAUDE.md - Project Instructions

## CRITICAL RULES - READ FIRST

### GIT SAFETY - ABSOLUTELY NEVER VIOLATE

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

---

## CURRENT WORK: Two-State DEM Implementation

### Master Plan Location
**`/Users/simonheikkila/.claude/plans/dynamic-crunching-cloud.md`**

Read this plan BEFORE making any changes. It contains the full physics model and implementation levels.

### The Core Problem
Current DEM uses `my_fraction = mass_i / (mass_i + mass_j)` which is NEVER ZERO.
This means sleeping particles STILL GET PUSHED. This is fundamentally wrong.

**The fix:** Two-state model where STATIC particles skip physics entirely until force exceeds threshold.

### Implementation Levels (SEQUENTIAL - NO SKIPPING)

| Level | Description | Exit Gate |
|-------|-------------|-----------|
| 0 | Clean slate (DONE) | git clean, build passes, existing tests pass |
| 1 | Add static_state buffer | Buffer exists, bound to shader, no regression |
| 2 | Static particle freeze | Static particles have ZERO position change |
| 3 | Dynamic→Static transition | Slow+supported particles become static |
| 4 | Force-threshold wake | Gold on sand stays on top, bulldozer wakes pile |
| 5 | Coulomb friction | Angle of repose = arctan(μ) ± 5° |
| 6 | Wake propagation | Support removal causes cascade wake |

### BULLETPROOF PROTOCOL - MANDATORY

1. **ONE CHANGE AT A TIME** - Single logical change, test, commit if pass, fix if fail
2. **NO QUICK FIXES** - If something breaks, the NEW code is wrong. Debug it.
3. **TESTS ARE LAW** - Failing test = cannot proceed. Never modify passing tests.
4. **REGRESSION = ROLLBACK** - If previously passing test fails: `git checkout -- <file>`
5. **EACH LEVEL COMPLETE** - All exit gate tests must pass before next level

### DOOM LOOP PREVENTION

STOP and ask user if:
- More than 3 attempts to fix a failing test
- Touching code not in current level
- Adding mechanisms not in plan
- Modifying a passing test
- "Quick fix" or "temporary workaround" being considered

### Key Files for This Work

| File | Purpose |
|------|---------|
| `crates/game/src/gpu/dem.rs` | Add static_state buffer, static_frames buffer |
| `crates/game/src/gpu/shaders/dem_forces.wgsl` | Two-state logic, force-threshold wake, Coulomb friction |
| `crates/game/tests/gpu_dem_settling.rs` | Test suite for each level |
| `todos/dem-implementation.md` | Structured progress tracking |

### Test Commands

```bash
# Run settling tests (required for each level)
cargo test -p game --release --test gpu_dem_settling

# Run specific test
cargo test -p game --release --test gpu_dem_settling test_name

# Visual verification
cargo run --example sediment_stages_visual -p game --release
```

### Physics Constants (from plan)

```
MU_STATIC = 0.6 (sand), 0.7 (gravel)
Angle of repose = arctan(μ) = 31° (sand), 35° (gravel)
STATIC_THRESHOLD = 1.0 px/s (speed below which to start static transition)
STATIC_DELAY = 30 frames (how long slow+supported before becoming static)
Wake threshold = μ × weight (force needed to wake static particle)
```
