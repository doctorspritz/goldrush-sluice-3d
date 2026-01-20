# Visual Regression Testing Implementation

## Overview

Screenshot-based regression tests have been implemented for the FLIP fluid simulation renderer to ensure visual consistency across code changes.

## Implementation

**File:** `/crates/game/tests/visual_regression_tests.rs`

### Test Infrastructure

1. **Headless Rendering**
   - Uses existing `headless_harness::init_device_queue()` for GPU context
   - Renders to offscreen RGBA8 texture (800x600)
   - Copies rendered frames to staging buffer for CPU readback

2. **Image Comparison**
   - Saves/loads reference images as PNG using `png` crate
   - Pixel-by-pixel comparison with tolerance
   - Allows 1% pixel difference threshold
   - Per-channel tolerance of ±1 bit for compression artifacts

3. **Reference Management**
   - References stored in `crates/game/tests/visual_references/`
   - First run generates new references
   - Subsequent runs compare against references
   - Failed tests save `*_FAILED.png` for debugging

### Test Cases

#### 1. `test_render_empty_scene`
- **Purpose:** Verify background/grid renders correctly without particles
- **Setup:** Empty particle array
- **Camera:** Eye at (3.2, 2.4, 3.2), looking at (3.2, 3.2, 3.2)
- **Reference:** `empty_scene.png`

#### 2. `test_render_particles_grid`
- **Purpose:** Test particle rendering with known positions
- **Setup:** 4×4×4 grid of particles (64 total)
- **Spacing:** 0.2 units between particles
- **Camera:** Eye at (2.5, 3.2, 4.5), looking at (3.0, 3.0, 3.0)
- **Reference:** `particles_grid.png`

#### 3. `test_render_settled_box`
- **Purpose:** Test realistic settled fluid rendering
- **Setup:** 8×4×8 cell box (2048 particles) at container bottom
- **Particle Density:** 8 particles per cell (2×2×2 sub-cell sampling)
- **Camera:** Eye at (2.0, 2.0, 5.0), looking at (3.2, 1.5, 3.2)
- **Reference:** `settled_box.png`

## Running Tests

```bash
# Run all visual tests
cargo test -p game --test visual_regression_tests test_render -- --nocapture

# Run specific test
cargo test -p game --test visual_regression_tests test_render_empty_scene -- --nocapture

# List available tests
cargo test -p game --test visual_regression_tests -- --list
```

## Key Features

### Headless Context Reuse
Tests leverage the existing headless testing infrastructure, no new window/surface creation needed.

### Deterministic Rendering
- Fixed camera positions and parameters
- Known particle configurations
- Consistent grid setup (64×64×64 cells, 0.1m cell size)

### Failure Handling
When a test fails:
1. Saves actual output to `*_FAILED.png`
2. Reports difference percentage and max channel diff
3. Points to failed output file for manual inspection

### Platform Tolerance
The 1% pixel difference threshold accounts for:
- GPU driver variations
- Floating-point precision differences
- PNG compression artifacts
- Platform-specific rendering quirks

## Future Enhancements

Potential additions:
- Tests for different particle counts (stress testing)
- Animation sequence tests (multiple frames)
- Shader modification detection
- Color scheme verification
- Performance benchmarks alongside visual tests

## Integration

These tests run as part of the standard test suite but are filtered by name (`test_render*`) for easy isolation from physics tests.

They require:
- GPU with compute shader support
- 16+ storage buffers per shader stage
- wgpu backend initialization

If GPU is unavailable, tests are skipped with appropriate message.
