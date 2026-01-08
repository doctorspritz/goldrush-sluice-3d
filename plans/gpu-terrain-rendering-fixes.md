# GPU Terrain Rendering & Visual Stability Fixes

## Overview
This plan covers the integration of GPU-accelerated heightfield rendering and the subsequent fixes for water stability and terrain aesthetics.

## Changes

### 1. GPU Integration
- Integrated `GpuHeightfield` into `world_test.rs`.
- Moved from CPU-side mesh generation to GPU storage buffers and a static vertex grid.

### 2. Water Level & Sync Fixes
- Resolved "high water plane" issue by refining shelf snapping logic to ignore dry cells.
- Partitioned compute passes to ensure memory visibility between simulation steps.
- Synchronized initial water surface to prevent startup flashes.

### 3. Visual Stability (Flickering & Z-Fighting)
- **Increased Snapping Threshold**: Raised to 0.05m to prevent shoreline oscillations.
- **Depth Bias**: Added 0.002m vertical lift to water to prevent Z-fighting with terrain.
- **Smooth Terrain Colors**: Replaced hard material steps with linear blending (`smoothstep`) to eliminate color flickering.

### 4. Shoreline Polishing
- **Soft Transparency**: Implemented smooth alpha fade at edges.
- **Organic Foam**: Reduced foam reach and added dynamic noise to break up artificial lines.

## Verification
- Verified stable 60+ FPS rendering with no visual artifacts in `world_test` example.
