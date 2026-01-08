# Architecture: Hybrid LOD Simulation & Rendering

## Objective
Scaling the world to high resolution (1cm - 10cm) across large areas (1km+) while integrating high-fidelity 3D particle physics (PIC/FLIP/DEM) for dynamic machinery and local phenomena.

---

## 1. Resolution & Performance (The LOD Problem)
A uniform 10cm grid for a 1km world is $10,000 \times 10,000$ cells ($10^8$). This is too heavy for real-time simulation on most consumer GPUs.

### Solution: GPU Clipmaps / Nested Grids
Instead of one big grid, we use a stack of progressively larger, coarser grids centered on the camera.
- **Level 0 (Inner)**: 128x128 @ 2cm (Player immediate area).
- **Level 1**: 256x256 @ 10cm (Work area).
- **Level 2**: 512x512 @ 50cm (Local valley).
- **Level 3 (Outer)**: 1024x1024 @ 2m (Distant hills).

**Impact**: 
- We only simulate high-frequency changes where they are visible. 
- Memory usage drops from Gigabytes to Megabytes.
- Rendering uses "Stitching" in the vertex shader to avoid gaps between resolution levels.

---

## 2. PIC/FLIP & DEM Integration (The "Slurry" Hybrid)
3D particles are $10\times$ more expensive than 2.5D heightfields. We cannot simulate the whole world as particles.

### Solution: Volume of Interest (VOI) Coupling
A hybrid approach where physics transitions based on complexity needs.

#### A. The Global Layer (2.5D Heightfield)
- **Base**: SWE (Shallow Water Equations) for mass movement across the map.
- **State**: World terrain, ponds, and river flows.
- **Role**: Serves as the boundary and "reservoir" for the detailed sims.

#### B. The Detail Layer (3D Particles)
- **Active Zones**: Bounding boxes around wash plants, excavator buckets, or water jets.
- **Physics**: PIC/FLIP for fluid/slurry, DEM for rock-to-rock collision.
- **Transition Logic**: 
  - **Ejection**: When an excavator digs, it "lifts" mass from the 2.5D heightfield and spawns 3D particles.
  - **Deposition**: When particles slow down and settle (or leave the Active Zone), they are "absorbed" back into the heightfield.

#### C. Two-Way Force Coupling
- **3D -> 2.5D**: Particle weight and momentum apply pressure to the underlying heightfield water/sediment.
- **2.5D -> 3D**: The heightfield ground and water surface act as SDF (Signed Distance Field) boundaries for the 3D particles.

---

## 3. Implementation Roadmap

### Phase 1: Rendering LOD (Clipmaps)
- Refactor `HeightfieldRender` to support multi-layer nested grids.
- Implement morphing/stitching in `heightfield_render.wgsl`.

### Phase 2: Sparse/Tiled Compute
- Partition the Sweet simulation into tiles.
- Only dispatch compute workgroups for tiles "marked" as dirty or containing water.

### Phase 3: High-Fidelity "Active Zones"
- Create a `ParticleSimulationZone` struct that defines a 3D box in the world.
- Implement mass transfer logic: `SWE mass <-> FLIP particles`.
- Example: A wash plant emitter box that spawns tailings into the world heightfield.

---

## Technical Challenges
- **Mass Conservation**: Ensuring no volume is lost when "melting" particles into the heightfield.
- **Visual Continuity**: Making the transition between a 3D splash and a 2.5D ripple look seamless (requires matched shading).
- **GPU Memory**: Efficiently managing many dynamic storage buffers using a unified heap or sparse residence.
