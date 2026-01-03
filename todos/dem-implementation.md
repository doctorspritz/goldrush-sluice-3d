# DEM Two-State Implementation - Level Gates

**Master Plan:** `/Users/simonheikkila/.claude/plans/dynamic-crunching-cloud.md`

---

## HOW THIS WORKS

Each level has:
1. **Prerequisites** - What must be done before starting
2. **Unit Tests** - Programmatic tests that run headless
3. **Visual Test** - What to run visually
4. **Acceptance Criteria** - EXACTLY what you will observe
5. **Gate** - User confirms "PASS" before next level

**NO LEVEL PROCEEDS WITHOUT USER CONFIRMATION.**

---

## Level 0: Clean Slate
**Status:** COMPLETE

### Prerequisites
- In `.worktrees/fix-dem-settling` directory

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling
```
Expected: 4 tests pass (baseline tests)

### Visual Test
```bash
cargo run --example sediment_stages_visual -p game --release
```

### Acceptance Criteria
- Window opens
- Press 3 for gold stream
- Gold particles fall and pile up
- Pile has some jitter/vibration (this is EXPECTED - no freeze yet)

### Gate
- [ ] User confirms: "Level 0 PASS"

---

## Level 1: Static State Buffer
**Status:** COMPLETE (commit 80e6a46)

### Prerequisites
- Level 0 passed

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling
```
Expected: 4 tests pass (no regression)

### Visual Test
Same as Level 0 - no visible change expected.

### Acceptance Criteria
- Same as Level 0
- No crashes
- Behavior identical to before

### Gate
- [ ] User confirms: "Level 1 PASS"

---

## Level 2: Static Particle Freeze
**Status:** COMPLETE (commit b8a8b90)

### Prerequisites
- Level 1 passed

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling
```
Expected: 5 tests pass, including `test_static_particles_frozen`

### Visual Test
No visible change in normal operation (particles start DYNAMIC).
The freeze logic exists but nothing triggers it yet.

### Acceptance Criteria
- 5 tests pass
- Visual behavior same as Level 1
- No crashes

### Gate
- [ ] User confirms: "Level 2 PASS"

---

## Level 3: Dynamic → Static Transition
**Status:** COMPLETE (commit 226fd5c)

### Prerequisites
- Level 2 passed
- User confirmed Level 2

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling test_settling_becomes_static
```

**Test logic:**
1. Drop 50 particles onto floor
2. Run 300 frames (5 seconds)
3. Download static states from GPU
4. Assert: >80% of particles have `is_static = 1`
5. Assert: Static particles have velocity < 0.1

### Visual Test
```bash
cargo run --example sediment_stages_visual -p game --release
# Press 3 for gold stream, then F to stop flow
```

### Acceptance Criteria
**EXACT OBSERVATIONS:**
1. Start gold stream (key 3)
2. Let particles pile up for 3 seconds
3. Press F to stop new particles
4. Wait 2 more seconds
5. **OBSERVE:** Pile becomes COMPLETELY STILL
   - No jitter
   - No micro-movements
   - Particles frozen in place
6. Press F again to resume flow
7. **OBSERVE:** New particles fall onto frozen pile
   - Frozen particles don't move
   - New particles settle and eventually freeze too

**FAIL if:**
- Pile keeps jittering after flow stops
- Particles slowly drift
- Any movement in settled pile

### Gate
- [ ] User confirms: "Level 3 PASS - pile freezes completely"

---

## Level 4: Force-Threshold Wake
**Status:** COMPLETE (commit 92d10be)

### Prerequisites
- Level 3 passed

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling test_gold_on_sand_stays_on_top
cargo test -p game --release --test gpu_dem_settling test_impact_wakes_pile
```

**Test 1 logic (gold on sand):**
1. Create sand pile, let it settle and become static
2. Drop single gold particle on top
3. Run 100 frames
4. Assert: Gold particle y-position is ABOVE sand pile surface
5. Assert: Sand particles remain static

**Test 2 logic (impact wakes):**
1. Create settled static pile
2. Launch fast particle into pile (vel = 500)
3. Assert: Particles near impact become dynamic
4. Run 200 more frames
5. Assert: Pile resettles, particles become static again

### Visual Test
```bash
cargo run --example sediment_stages_visual -p game --release
# Press 4 for mixed stream (sand + gold)
```

### Acceptance Criteria
**EXACT OBSERVATIONS:**
1. Press 4 for mixed sand/gold stream
2. Let pile build for 5 seconds
3. **OBSERVE:** Gold particles (yellow) visible ON SURFACE
   - Gold is denser but sits on top
   - Gold does NOT sink through sand
4. Press F to stop flow, wait for settling
5. **OBSERVE:** Gold particles remain on surface of frozen pile

**FAIL if:**
- Gold sinks into sand pile
- Gold ends up at bottom
- Gold slowly descends through pile

### Gate
- [ ] User confirms: "Level 4 PASS - gold sits on top of sand"

---

## Level 5: Proper Coulomb Friction
**Status:** PENDING

### Prerequisites
- Level 4 passed

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling test_angle_of_repose
```

**Test logic:**
1. Pour sand from narrow point
2. Let pile form and settle
3. Measure pile dimensions (width, height)
4. Calculate angle: arctan(height / half_width)
5. Assert: Angle is 28-34° (for μ=0.6)

### Visual Test
```bash
cargo run --example sediment_stages_visual -p game --release
# Press 2 for sand stream
```

### Acceptance Criteria
**EXACT OBSERVATIONS:**
1. Press 2 for sand stream
2. Let pile build for 10 seconds
3. Press F to stop flow
4. **OBSERVE:** Pile forms consistent slope
   - Not a vertical tower
   - Not a flat pancake
   - Slope approximately 30° from horizontal
5. **MEASURE:** Visually estimate slope
   - If pile is 100 pixels high, base should extend ~170 pixels from center
   - Rise:Run ratio approximately 1:1.7

**FAIL if:**
- Pile is vertical (stacking)
- Pile is nearly flat
- Slope varies randomly across pile
- Different runs produce different angles

### Gate
- [ ] User confirms: "Level 5 PASS - pile has consistent ~30° slope"

---

## Level 6: Support-Based Wake Propagation
**Status:** PENDING

### Prerequisites
- Level 5 passed

### Unit Tests
```bash
cargo test -p game --release --test gpu_dem_settling test_support_removal_cascade
```

**Test logic:**
1. Create pile on elevated platform (solid cells)
2. Let pile settle and become static
3. Remove platform (set those cells to fluid)
4. Assert: All particles wake (become dynamic)
5. Run 200 frames
6. Assert: Particles fall and resettle on floor

### Visual Test
Need new stage: `stage_platform_collapse`

### Acceptance Criteria
**EXACT OBSERVATIONS:**
1. Run platform collapse stage
2. See pile resting on platform
3. Platform removed (key trigger)
4. **OBSERVE:** Entire pile falls
   - Not just bottom particles
   - All particles become dynamic
   - Pile restructures on new floor

**FAIL if:**
- Only bottom particles fall
- Upper particles float in air
- Cascade is slow/gradual instead of immediate

### Gate
- [ ] User confirms: "Level 6 PASS - pile collapses when support removed"

---

## CURRENT POSITION

- Level 0: COMPLETE
- Level 1: COMPLETE (commit 80e6a46)
- Level 2: COMPLETE (commit b8a8b90)
- Level 3: COMPLETE (commit 226fd5c) - User confirmed PASS
- Level 4: COMPLETE (commit 92d10be) - Atomic-based wake detection

**NEXT ACTION:**
Proceed to Level 5 (proper Coulomb friction) when ready.

---

## DOOM LOOP TRACKING

If any level takes >3 attempts, STOP and discuss with user.

| Level | Attempts | Notes |
|-------|----------|-------|
| 0 | 1 | Clean |
| 1 | 1 | Clean |
| 2 | 1 | Clean |
| 3 | 1 | Clean |
| 4 | 3 | Solved GPU race condition with atomic operations |
| 5 | 0 | Not started |
| 6 | 0 | Not started |
