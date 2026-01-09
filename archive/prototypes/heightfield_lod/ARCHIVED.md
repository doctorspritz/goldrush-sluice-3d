# ARCHIVED: Heightfield LOD Prototype

**Archived:** 2026-01-08
**Reason:** Focus shifted from 2.5D LOD optimization to 2.5D-3D LOD interaction

## What This Was
Standalone prototype for multi-LOD 2.5D heightfield simulation designed for a 2km×2km world.

## Why Archived
We did build a working 2.5D LOD multigrid, but realized we were fixating on LOD for the low-compute 2.5D part instead of tackling the more important problem: **2.5D-3D LOD interaction**.

The real challenge is seamless integration between the broad world (2.5D heightfield) and detail zones (3D FLIP/DEM), not just optimizing the 2.5D layer in isolation.

## Reference Value
- Design doc still valuable for future large-world streaming
- LOD tile architecture concepts may be useful later
