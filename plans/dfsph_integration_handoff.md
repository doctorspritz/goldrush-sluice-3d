# Handoff Status: DFSPH Integration

## 1. Objective
Replace the unstable FLIP/APIC simulation with a robust Divergence-Free SPH (DFSPH) implementation to fix stability issues and improve water behavior.

## 2. Physics Status: **SOLVED**
*   **Crate:** `crates/dfsph`
*   **Stability:** The core instability (NaN explosions) was identified and fixed.
    *   **Root Cause:** A sign error in the `alpha` calculation in `simulation.rs`. It was `alpha = -1.0 / denom`, which created *attractive* forces when fluid was compressed.
    *   **Fix:** Changed to `alpha = 1.0 / denom`. This ensures repulsive pressure forces.
*   **Tests:** `test_flow` now passes without NaNs. Water flows downhill.

## 3. Integration Status: **BROKEN BUILD**
We are integrating the new `dfsph` crate into the `game` crate for visualization.

*   **`crates/game/src/main.rs`:** Successfully rewritten to use `DfsphSimulation`, `Particle::new_wall`, etc.
*   **`crates/game/src/render.rs`:** **CORRUPTED.**
    *   During a refactor to change `&Particles` to `&[Particle]`, the function signature for `MetaballRenderer::draw` was accidentally deleted.
    *   **Location:** Around line 408-412.
    *   **Current State:**
        ```rust
        // ... comments ...
        // MISSING LINE HERE: pub fn draw(&self, particles: &[Particle], screen_scale: f32) {
            let base_size = self.particle_scale * screen_scale;
            // ... body ...
        ```
    *   **Error:** `unexpected closing delimiter: }` (because the function body braces are mismatched with the impl block).

## 4. Next Steps for Next Agent
1.  **Fix `crates/game/src/render.rs`:**
    *   Open the file.
    *   Locate the `MetaballRenderer` implementation.
    *   Insert the missing function signature: `pub fn draw(&self, particles: &[Particle], screen_scale: f32) {` before the body starts (approx line 411).
    *   Also scan for any other missing signatures in that file (the `multi_replace` tool might have missed others).
2.  **Compile & Run:**
    *   Run `cargo run -p game`.
    *   The simulation should run, showing water spawning on a sloped sluice and flowing stably.
3.  **Tuning:**
    *   Once running, you may tune `Particle` spacing, `H` (smoothing length), or viscosity in `dfsph/src/simulation.rs` or `main.rs` for visual quality.

## 5. File References
*   `crates/dfsph/src/simulation.rs`: Core physics logic (Fixed).
*   `crates/game/src/main.rs`: Demo logic (Updated).
*   `crates/game/src/render.rs`: Renderer (Needs Syntax Fix).
