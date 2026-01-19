# Refactoring Plan: flip_3d.rs and world.rs

## Executive Summary

Two "god files" need decomposition:
- **flip_3d.rs** (5,216 lines) - GPU FLIP solver
- **world.rs** (3,386 lines) - 2.5D terrain/water simulation

Both share similar problems: massive initialization, duplicated patterns, poor separation of concerns.

---

## flip_3d.rs Analysis (5,216 lines)

### Current Structure

| Section | Lines | % | Notes |
|---------|-------|---|-------|
| Parameter structs | 191 | 4% | 15 GPU uniform structs |
| ReadbackSlot | 253 | 5% | Async GPU→CPU transfer |
| GpuFlip3D struct | 192 | 4% | 65 fields (!!) |
| `new()` initialization | 2,582 | **50%** | Creates 29 pipelines |
| `run_gpu_passes()` | 993 | **19%** | Main simulation loop |
| Diagnostics | 459 | 9% | Debug utilities |

### Key Problems

1. **49.5% of file is initialization** - `new()` is 2,582 lines
2. **29 GPU pipelines** all built with identical boilerplate
3. **65 struct fields** - state is unmanageable
4. **~2000 lines of pure boilerplate** for pipeline/bind group creation

### Recommended Module Structure

```
gpu/
├── flip_3d.rs              (~500 lines - public API only)
└── flip_3d/
    ├── mod.rs              (re-exports)
    ├── params.rs           (~200 lines - 15 parameter structs)
    ├── readback.rs         (~250 lines - ReadbackSlot)
    ├── pipeline_builder.rs (~300 lines - reusable factories)
    ├── initialization.rs   (~1200 lines - builders for new())
    ├── simulation.rs       (~800 lines - run_gpu_passes phases)
    └── diagnostics.rs      (~450 lines - debug utilities)
```

### Extraction Order

| Phase | Module | Effort | Risk | Reduction |
|-------|--------|--------|------|-----------|
| 1 | params.rs | 30 min | None | 200 lines |
| 1 | readback.rs | 1-2 hrs | Low | 250 lines |
| 2 | pipeline_builder.rs | 2-3 hrs | Low | 40% of new() |
| 2 | diagnostics.rs | 1-2 hrs | Low | 450 lines |
| 3 | initialization.rs | 3-4 hrs | Medium | 2000+ lines |
| 3 | simulation.rs | 3-4 hrs | Medium | 800 lines |

**Total: 12-18 hours**

### Pipeline Builder Pattern

Current (repeated 29 times):
```rust
let shader = device.create_shader_module(...);
let layout = device.create_bind_group_layout(...);
let pipeline_layout = device.create_pipeline_layout(...);
let pipeline = device.create_compute_pipeline(...);
```

After:
```rust
let pipeline = PipelineBuilder::new(device)
    .shader("shaders/gravity_3d.wgsl")
    .storage_buffers(&[grid_u, grid_v, grid_w])
    .uniform_buffer::<GravityParams3D>()
    .build();
```

---

## world.rs Analysis (3,386 lines)

### Current Structure

| Section | Lines | % | Notes |
|---------|-------|---|-------|
| WorldParams | 125 | 4% | Configuration struct |
| FineRegion | 482 | 14% | Adaptive LOD region |
| World struct | 39 | 1% | Core state |
| Physics calculations | 139 | 4% | Shields, settling, etc. |
| Water flow | 450 | 13% | SWE solver |
| Sediment transport | 400 | 12% | Advection + settling |
| Erosion | 450 | 13% | **Duplicated 70%** |
| Terrain manipulation | 185 | 5% | Excavate, collapse |
| Fine region mgmt | 200 | 6% | LOD transitions |
| Tests | 564 | 17% | 15 test functions |

### Key Problems

1. **70% code duplication** in erosion between World and FineRegion
2. **60% duplication** in terrain collapse logic
3. **50% duplication** in water flow solver
4. Large methods: `update_water_flow()` (269), `update_sediment_advection()` (311), `update_erosion()` (233)

### Recommended Module Structure

```
sim3d/src/
├── world.rs                (~600 lines - orchestration only)
├── world/
│   ├── mod.rs              (re-exports)
│   ├── physics.rs          (~180 lines - pure calculations)
│   ├── water_flow.rs       (~400 lines - SWE solver)
│   ├── sediment.rs         (~380 lines - transport)
│   ├── erosion.rs          (~300 lines - DEDUPLICATED)
│   ├── terrain.rs          (~160 lines - manipulation)
│   ├── fine_region.rs      (~200 lines - LOD management)
│   └── geometry.rs         (~80 lines - visualization)
```

### Extraction Order

| Phase | Module | Effort | Risk | Reduction |
|-------|--------|--------|------|-----------|
| 1 | geometry.rs | 1-2 hrs | None | 80 lines |
| 1 | physics.rs | 2-3 hrs | Low | 180 lines |
| 2 | terrain.rs | 4-6 hrs | Medium | 160 lines + dedup |
| 3 | water_flow.rs | 8-12 hrs | High | 400 lines |
| 4 | erosion.rs | 12-18 hrs | High | 300 lines + 70% dedup |
| 5 | sediment.rs | 6-10 hrs | Medium | 380 lines |
| 5 | fine_region.rs | 3-4 hrs | Low | 200 lines |

**Total: 36-55 hours**

### Erosion Deduplication Strategy

Current: Nearly identical code in `World::update_erosion()` and `FineRegion::update_erosion()`

After:
```rust
// In erosion.rs
pub fn erode_cell<G: GridAccess>(
    grid: &mut G,
    params: &ErosionParams,
    x: usize, z: usize,
) -> ErosionResult {
    let shear = physics::shear_stress(grid.water_velocity(x, z), ...);
    let shields = physics::shields_stress(shear, ...);
    // ... shared algorithm
}

// World and FineRegion both implement GridAccess trait
```

---

## Combined Timeline

### Week 1: Low-Risk Foundations
- [ ] flip_3d: Extract params.rs, readback.rs
- [ ] world: Extract geometry.rs, physics.rs
- **Deliverable**: 4 new modules, ~700 lines moved

### Week 2: Utility Patterns
- [ ] flip_3d: Create pipeline_builder.rs
- [ ] flip_3d: Extract diagnostics.rs
- [ ] world: Extract terrain.rs with dedup
- **Deliverable**: 3 more modules, ~40% new() reduction

### Week 3-4: Core Decomposition
- [ ] flip_3d: Extract initialization.rs (builder pattern)
- [ ] flip_3d: Extract simulation.rs (phase functions)
- [ ] world: Extract water_flow.rs
- **Deliverable**: Main files reduced to <1000 lines

### Week 5-6: High-Value Deduplication
- [ ] world: Extract erosion.rs with trait-based dedup
- [ ] world: Extract sediment.rs, fine_region.rs
- **Deliverable**: 25-30% code reduction via deduplication

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| flip_3d.rs lines | 5,216 | ~500 |
| world.rs lines | 3,386 | ~600 |
| Total modules | 2 | 14 |
| Code duplication | ~25% | <5% |
| Max function size | 2,582 lines | <200 lines |
| Struct field count | 65 | <20 per struct |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU pipeline regressions | Shader validation tests before/after |
| Water flow perf regression | Benchmark 1M particles before extraction |
| Erosion physics drift | Property-based tests for conservation |
| Breaking public API | Keep flip_3d.rs and world.rs as facades |

---

## Next Steps

1. Create beads issues for each extraction phase
2. Start with Phase 1 (lowest risk, highest clarity gain)
3. Run full test suite after each module extraction
4. Benchmark GPU performance after flip_3d changes
