# Rendering Improvements Plan

## Status: Phase 1 Complete

**FastMetaballRenderer implemented with:**
- Half-resolution (640x320) density buffer
- Mesh batching for water (8K particles/batch)
- Separable 9-tap Gaussian blur (H + V passes)
- Surface shader with gradient-based normals + specular
- Mesh batching for sediment

**Performance:** ~43-48 FPS at 17K particles (vs ~35 FPS old Metaball mode)

---

## Current State

7 render modes exist (FastMetaball is new default):
- **Mesh** (default) - Batched quads, ~8000/draw call, fastest but squares
- **FastRect** - `draw_rectangle` per particle
- **FastCircle** - `draw_circle` per particle
- **Shader** - Per-particle shader with smooth edges
- **Metaball** - Two-pass density/threshold for blobby water
- **Hybrid** - Water metaballs + sharp solids

**Issues:**
1. Water looks like discrete particles, not fluid
2. No depth/translucency - everything opaque
3. Metaball mode too slow (per-particle draw calls in density pass)
4. Sediment/gold particles lack visual distinction

## Proposed Improvements

### Phase 1: Optimized Water Rendering (Priority)

**Screen-Space Fluid Rendering (2D adaptation):**
1. Render particles to density texture (half resolution)
2. Gaussian blur the density (separable, 2 passes)
3. Threshold + smooth edge detection
4. Add simple lighting (normal from density gradient)
5. Translucency based on depth/thickness

**Performance target:** Same FPS as Mesh mode with metaball-quality water

### Phase 2: Particle Visual Improvements

**Sediment distinction:**
- Sand: Rough texture, slightly larger
- Magnetite: Darker, slight metallic sheen
- Gold: Bright, sparkle effect (random highlight)

**Techniques:**
- Single texture atlas with all particle types
- UV offset per material type
- Optional: velocity-based stretching for motion blur feel

### Phase 3: Polish Effects

- Surface foam/spray at high velocity
- Depth-based water darkening
- Simple reflection on water surface (environment color blend)
- Particle trails for fast-moving gold

## Technical Approach

### Shader Architecture

```
[Density Pass] → [Blur Pass H] → [Blur Pass V] → [Surface Pass]
     ↓                                                ↓
  Low-res RT                                    Final composite
```

### Key Optimizations

1. **Half-res density buffer** - Metaballs at 640x320 instead of 1280x640
2. **Separable blur** - O(2n) instead of O(n²) per pixel
3. **Single mesh for all particles** - Already have this
4. **Batch by material** - Minimize uniform changes

### Macroquad Constraints

- GLSL 100 only (no compute shaders)
- Limited blend modes
- RenderTarget for multi-pass

## Implementation Order

1. [ ] Half-res metaball optimization
2. [ ] Separable Gaussian blur shader
3. [ ] Water surface shader with gradient normals
4. [ ] Translucency/depth effect
5. [ ] Particle texture atlas
6. [ ] Gold sparkle effect
7. [ ] Optional: velocity-based particle stretching

## Visual Reference Goals

- Water: Cohesive fluid surface like World of Goo or PixelJunk Shooter
- Sediment: Distinct, material-appropriate appearance
- Gold: Unmistakable, satisfying to see captured

## Files to Modify

- `crates/game/src/render.rs` - New shaders and renderers
- `crates/game/src/main.rs` - Hook up new render modes
- New: `crates/game/assets/particles.png` - Texture atlas (if needed)
