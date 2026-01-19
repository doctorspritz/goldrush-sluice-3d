# Visual Regression Test References

This directory contains reference images for visual regression tests of the FLIP fluid simulation renderer.

## Test Images

- **empty_scene.png** - Empty scene with just the background (no particles)
- **particles_grid.png** - 4x4x4 grid of particles in a regular pattern
- **settled_box.png** - 8x4x8 cell box of particles settled at the bottom of the container

## Running Tests

```bash
# Run all visual regression tests
cargo test -p game --test visual_regression_tests test_render

# Run a specific test
cargo test -p game --test visual_regression_tests test_render_empty_scene -- --nocapture
```

## Regenerating References

If you need to regenerate the reference images (e.g., after intentional rendering changes):

1. Delete the reference image you want to regenerate:
   ```bash
   rm tests/visual_references/empty_scene.png
   ```

2. Run the test again - it will save a new reference:
   ```bash
   cargo test -p game --test visual_regression_tests test_render_empty_scene
   ```

3. Verify the new reference looks correct by opening the PNG file

## Tolerance

The tests use a 1% pixel difference tolerance to account for:
- Minor compression artifacts
- GPU driver differences
- Platform-specific rendering variations

Each pixel channel allows ±1 bit difference before counting as a failed pixel.

## Failed Test Output

When a test fails, it saves a `*_FAILED.png` file showing the actual output for debugging.

## Test Implementation

See `/crates/game/tests/visual_regression_tests.rs` for implementation details.
