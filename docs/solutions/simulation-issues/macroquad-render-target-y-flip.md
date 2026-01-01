---
title: "Macroquad Render Target Y-Coordinate Inversion"
category: simulation-issues
component: game/render
severity: major
symptoms:
  - Metaballs appear upside down relative to terrain
  - Render target content is Y-inverted when drawn to screen
  - Camera zoom adjustments don't fix the orientation
tags: [macroquad, render-target, camera, shader, metaball]
date: 2025-12-20
---

# Macroquad Render Target Y-Coordinate Inversion

## Symptom

When rendering particles to a render target (for metaball density accumulation), then drawing that render target to screen, the content appeared upside down relative to the main scene geometry.

## Failed Approaches

1. **flip_y parameter**: `Camera2D { flip_y: true/false, ... }` - No effect because macroquad's internal logic overrides this for render targets

2. **Negative Y zoom**: `zoom: vec2(2.0/w, -2.0/h)` - This is what the default camera uses, but it still resulted in inverted output

3. **Camera2D::from_display_rect()**: Produced similar incorrect results

4. **Default camera settings**: Using `..Default::default()` for Camera2D gave zoom (1,1) which mapped particles far outside the clip space, making them invisible

## Root Cause

Macroquad's `Camera2D::matrix()` function (in `src/camera.rs`) contains special logic:

```rust
fn matrix(&self) -> Mat4 {
    let invert_y: f32 = if self.render_target.is_some() { 1.0 } else { -1.0 };
    // ...
    let mat = Mat4::from_cols(
        vec4(self.zoom.x, 0.0, 0.0, 0.0),
        vec4(0.0, self.zoom.y * invert_y, 0.0, 0.0),  // <-- Y is NOT inverted for render targets
        // ...
    );
}
```

**Key insight**:
- For **screen rendering**: `invert_y = -1.0`, so Y-axis is flipped (screen coords: Y=0 at top)
- For **render targets**: `invert_y = 1.0`, so Y-axis is NOT flipped (OpenGL coords: Y=0 at bottom)

When you draw a render target texture to the screen, the texture's origin is at bottom-left (OpenGL convention), but screen draws expect origin at top-left. This causes the apparent Y-inversion.

This is documented in [macroquad GitHub issue #171](https://github.com/not-fl3/macroquad/issues/171).

## Solution

Flip Y coordinates directly in the shader when sampling the render target texture:

```glsl
// In the threshold pass fragment shader
void main() {
    // Flip Y when sampling to correct for render target coordinate mismatch
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 texSample = texture2D(Texture, uv);
    // ... rest of shader
}
```

**File:** `crates/game/src/render.rs` line 92

## Why This Works

The shader-based fix is simpler and more reliable than camera manipulation because:

1. It operates at the final sampling stage where the coordinate systems must reconcile
2. It bypasses all of macroquad's internal camera logic
3. It's explicit about what it's doing (no hidden behavior)
4. It doesn't affect the density accumulation pass (which renders correctly to the render target)

## Camera Setup for Render Target

For the density pass, use this camera configuration:

```rust
set_camera(&Camera2D {
    zoom: vec2(2.0 / rt_w, -2.0 / rt_h),  // Negative Y for proper orientation
    target: vec2(rt_w / 2.0, rt_h / 2.0),
    render_target: Some(render_target.clone()),
    ..Default::default()
});
```

With negative Y zoom:
- World (0, 0) → clip (-1, 1) → texture top-left
- World (w, h) → clip (1, -1) → texture bottom-right

Then the shader Y-flip when sampling corrects the final output.

## Prevention

When working with render targets in macroquad:

1. **Expect Y-inversion**: Always anticipate that render target content will appear flipped
2. **Prefer shader-based fixes**: Camera parameter manipulation is unreliable due to internal `invert_y` logic
3. **Test incrementally**: Render solid colors first to verify orientation before complex shaders
4. **Document coordinate systems**: Comment which coordinate system each pass uses

## Related

- macroquad Camera2D source: `src/camera.rs` in macroquad repository
- GitHub issue #171: documents render target orientation behavior
- The particle renderer (non-metaball) doesn't have this issue because it renders directly to screen
