<!-- TODO: Review for 2.5D map usage - consider combining these two docs -->

# Implementation Plan: Drucker-Prager Yield for Sediment

This plan is for Codex to implement. Follow each phase in order. Test after each phase before proceeding.

**Base branch**: `plan/gpu3d-slurry`
**Target**: Add Drucker-Prager yield criterion to sediment particles for realistic clogging and angle of repose.

---

## Phase 1: Sediment Pressure Buffer

### 1.1 Create new shader file

**Create**: `crates/game/src/gpu/shaders/sediment_pressure_3d.wgsl`

```wgsl
// Sediment Pressure Shader (3D) - Compute pressure from sediment column weight
//
// For each (x,z) column, scan from top to bottom accumulating sediment mass.
// Pressure at cell (i,j,k) = weight of all sediment above it.
// This is used for Drucker-Prager yield criterion.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
    cell_size: f32,
    particle_mass: f32,
    gravity: f32,
    buoyancy_factor: f32,  // 1 - rho_water/rho_sediment, ~0.62 for sand
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sediment_count: array<i32>;
@group(0) @binding(2) var<storage, read_write> sediment_pressure: array<f32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 1, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let k = id.z;

    if (i >= params.width || k >= params.depth) {
        return;
    }

    let cell_area = params.cell_size * params.cell_size;

    // Scan from top to bottom, accumulating pressure
    var accumulated_pressure: f32 = 0.0;

    for (var j: i32 = i32(params.height) - 1; j >= 0; j--) {
        let idx = cell_index(i, u32(j), k);
        let count = f32(sediment_count[idx]);

        // Mass of sediment in this cell
        let cell_mass = count * params.particle_mass;

        // Weight (buoyancy-corrected)
        let effective_weight = cell_mass * params.gravity * params.buoyancy_factor;

        // Add to accumulated pressure (pressure = force/area)
        accumulated_pressure += effective_weight / cell_area;

        // Store pressure at this cell (pressure from sediment ABOVE, not including this cell)
        // Actually, include this cell's contribution for particles sitting in it
        sediment_pressure[idx] = accumulated_pressure;
    }
}
```

### 1.2 Add buffer and pipeline to GpuFlip3D

**Modify**: `crates/game/src/gpu/flip_3d.rs`

Add to struct fields (near other buffers):

```rust
// Sediment pressure (Drucker-Prager)
sediment_pressure_buffer: wgpu::Buffer,
sediment_pressure_pipeline: wgpu::ComputePipeline,
sediment_pressure_bind_group: wgpu::BindGroup,
sediment_pressure_params_buffer: wgpu::Buffer,
```

Add params struct (near other param structs):

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SedimentPressureParams {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
    cell_size: f32,
    particle_mass: f32,
    gravity: f32,
    buoyancy_factor: f32,
}
```

In `new()` function, create the buffer:

```rust
let cell_count = (width * height * depth) as usize;
let sediment_pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Sediment Pressure Buffer"),
    size: (cell_count * std::mem::size_of::<f32>()) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

Create params buffer:

```rust
let sediment_pressure_params = SedimentPressureParams {
    width,
    height,
    depth,
    _pad0: 0,
    cell_size,
    particle_mass: 1.0,  // Will be updated per-frame if needed
    gravity: 9.81,
    buoyancy_factor: 0.62,  // 1 - 1000/2650 for sand in water
};

let sediment_pressure_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Sediment Pressure Params"),
    contents: bytemuck::cast_slice(&[sediment_pressure_params]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

Create shader module and pipeline:

```rust
let sediment_pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Sediment Pressure Shader"),
    source: wgpu::ShaderSource::Wgsl(
        include_str!("shaders/sediment_pressure_3d.wgsl").into()
    ),
});

let sediment_pressure_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Sediment Pressure Bind Group Layout"),
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
});

let sediment_pressure_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("Sediment Pressure Pipeline Layout"),
    bind_group_layouts: &[&sediment_pressure_bind_group_layout],
    push_constant_ranges: &[],
});

let sediment_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Sediment Pressure Pipeline"),
    layout: Some(&sediment_pressure_pipeline_layout),
    module: &sediment_pressure_shader,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: None,
});

// sediment_count buffer is from p2g: self.p2g.sediment_count_buffer
let sediment_pressure_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("Sediment Pressure Bind Group"),
    layout: &sediment_pressure_bind_group_layout,
    entries: &[
        wgpu::BindGroupEntry {
            binding: 0,
            resource: sediment_pressure_params_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: self.p2g.sediment_count_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: sediment_pressure_buffer.as_entire_binding(),
        },
    ],
});
```

### 1.3 Dispatch the pass

In `run_gpu_passes()`, add dispatch after `sediment_fraction` pass:

```rust
// Sediment pressure pass (for Drucker-Prager)
{
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Sediment Pressure Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sediment Pressure Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sediment_pressure_pipeline);
        pass.set_bind_group(0, &self.sediment_pressure_bind_group, &[]);
        // Dispatch one workgroup per 8x8 column
        let workgroups_x = (self.width + 7) / 8;
        let workgroups_z = (self.depth + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, 1, workgroups_z);
    }
    queue.submit(std::iter::once(encoder.finish()));
}
```

### 1.4 Test Phase 1

Run `cargo build -p game --release` to verify compilation.
Run `cargo run --example industrial_sluice --release` to verify no crashes.

---

## Phase 2: Drucker-Prager Parameters

### 2.1 Add DP params struct

**Modify**: `crates/game/src/gpu/g2p_3d.rs`

Add new params struct:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DruckerPragerParams {
    pub friction_coeff: f32,     // tan(friction_angle), ~0.62 for 32°
    pub cohesion: f32,           // Pa, 0 for sand
    pub buoyancy_factor: f32,    // 1 - rho_w/rho_s, ~0.62
    pub viscosity: f32,          // Pa·s, for yielded flow, ~1.0
    pub jammed_drag: f32,        // drag coefficient when jammed, ~50
    pub min_pressure: f32,       // minimum pressure to avoid div/0, ~0.1
    pub yield_smoothing: f32,    // smoothing around yield surface, ~0.1
    pub _pad0: f32,
}

impl Default for DruckerPragerParams {
    fn default() -> Self {
        Self {
            friction_coeff: 0.62,      // tan(32°)
            cohesion: 0.0,
            buoyancy_factor: 0.62,
            viscosity: 1.0,
            jammed_drag: 50.0,
            min_pressure: 0.1,
            yield_smoothing: 0.1,
            _pad0: 0.0,
        }
    }
}
```

### 2.2 Add buffer to GpuG2p3D

In `GpuG2p3D` struct, add:

```rust
dp_params_buffer: wgpu::Buffer,
```

In `new()`, create it:

```rust
let dp_params = DruckerPragerParams::default();
let dp_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Drucker-Prager Params"),
    contents: bytemuck::cast_slice(&[dp_params]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

Add to bind group layout (new binding after existing ones):

```rust
// Binding 15: Drucker-Prager params
wgpu::BindGroupLayoutEntry {
    binding: 15,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
},
// Binding 16: Sediment pressure buffer (read-only)
wgpu::BindGroupLayoutEntry {
    binding: 16,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
},
```

Add to bind group entries:

```rust
wgpu::BindGroupEntry {
    binding: 15,
    resource: dp_params_buffer.as_entire_binding(),
},
wgpu::BindGroupEntry {
    binding: 16,
    resource: sediment_pressure_buffer.as_entire_binding(),  // passed from GpuFlip3D
},
```

Add method to update params:

```rust
pub fn set_drucker_prager_params(&self, queue: &wgpu::Queue, params: DruckerPragerParams) {
    queue.write_buffer(&self.dp_params_buffer, 0, bytemuck::cast_slice(&[params]));
}
```

---

## Phase 3: Modify G2P Shader

### 3.1 Add bindings and structs to shader

**Modify**: `crates/game/src/gpu/shaders/g2p_3d.wgsl`

Add after existing structs:

```wgsl
struct DruckerPragerParams {
    friction_coeff: f32,
    cohesion: f32,
    buoyancy_factor: f32,
    viscosity: f32,
    jammed_drag: f32,
    min_pressure: f32,
    yield_smoothing: f32,
    _pad0: f32,
}

@group(0) @binding(15) var<uniform> dp_params: DruckerPragerParams;
@group(0) @binding(16) var<storage, read> sediment_pressure: array<f32>;
```

### 3.2 Add shear rate helper function

Add before the main `g2p` function:

```wgsl
// Sample velocity at arbitrary position using trilinear interpolation
fn sample_velocity_at(sample_pos: vec3<f32>) -> vec3<f32> {
    let cell_size = params.cell_size;
    let width = i32(params.width);
    let height = i32(params.height);
    let depth = i32(params.depth);

    // Sample U component
    let u_pos = sample_pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
    let base_u = vec3<i32>(floor(u_pos));
    let frac_u = u_pos - vec3<f32>(base_u);

    var u_val: f32 = 0.0;
    var u_weight: f32 = 0.0;
    for (var dk: i32 = 0; dk <= 1; dk++) {
        for (var dj: i32 = 0; dj <= 1; dj++) {
            for (var di: i32 = 0; di <= 1; di++) {
                let ni = base_u.x + di;
                let nj = base_u.y + dj;
                let nk = base_u.z + dk;
                if (ni >= 0 && ni <= width && nj >= 0 && nj < height && nk >= 0 && nk < depth) {
                    let w = (1.0 - abs(frac_u.x - f32(di))) *
                            (1.0 - abs(frac_u.y - f32(dj))) *
                            (1.0 - abs(frac_u.z - f32(dk)));
                    u_val += w * grid_u[u_index(ni, nj, nk)];
                    u_weight += w;
                }
            }
        }
    }
    if (u_weight > 0.0) { u_val /= u_weight; }

    // Sample V component
    let v_pos = sample_pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
    let base_v = vec3<i32>(floor(v_pos));
    let frac_v = v_pos - vec3<f32>(base_v);

    var v_val: f32 = 0.0;
    var v_weight: f32 = 0.0;
    for (var dk: i32 = 0; dk <= 1; dk++) {
        for (var dj: i32 = 0; dj <= 1; dj++) {
            for (var di: i32 = 0; di <= 1; di++) {
                let ni = base_v.x + di;
                let nj = base_v.y + dj;
                let nk = base_v.z + dk;
                if (ni >= 0 && ni < width && nj >= 0 && nj <= height && nk >= 0 && nk < depth) {
                    let w = (1.0 - abs(frac_v.x - f32(di))) *
                            (1.0 - abs(frac_v.y - f32(dj))) *
                            (1.0 - abs(frac_v.z - f32(dk)));
                    v_val += w * grid_v[v_index(ni, nj, nk)];
                    v_weight += w;
                }
            }
        }
    }
    if (v_weight > 0.0) { v_val /= v_weight; }

    // Sample W component
    let w_pos = sample_pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
    let base_w = vec3<i32>(floor(w_pos));
    let frac_w = w_pos - vec3<f32>(base_w);

    var w_val: f32 = 0.0;
    var w_weight_sum: f32 = 0.0;
    for (var dk: i32 = 0; dk <= 1; dk++) {
        for (var dj: i32 = 0; dj <= 1; dj++) {
            for (var di: i32 = 0; di <= 1; di++) {
                let ni = base_w.x + di;
                let nj = base_w.y + dj;
                let nk = base_w.z + dk;
                if (ni >= 0 && ni < width && nj >= 0 && nj < height && nk >= 0 && nk <= depth) {
                    let w = (1.0 - abs(frac_w.x - f32(di))) *
                            (1.0 - abs(frac_w.y - f32(dj))) *
                            (1.0 - abs(frac_w.z - f32(dk)));
                    w_val += w * grid_w[w_index(ni, nj, nk)];
                    w_weight_sum += w;
                }
            }
        }
    }
    if (w_weight_sum > 0.0) { w_val /= w_weight_sum; }

    return vec3<f32>(u_val, v_val, w_val);
}

// Compute shear rate magnitude from velocity gradient
fn compute_shear_rate(pos: vec3<f32>) -> f32 {
    let h = params.cell_size * 0.5;

    // Finite difference velocity gradient
    let vel_px = sample_velocity_at(pos + vec3<f32>(h, 0.0, 0.0));
    let vel_mx = sample_velocity_at(pos - vec3<f32>(h, 0.0, 0.0));
    let vel_py = sample_velocity_at(pos + vec3<f32>(0.0, h, 0.0));
    let vel_my = sample_velocity_at(pos - vec3<f32>(0.0, h, 0.0));
    let vel_pz = sample_velocity_at(pos + vec3<f32>(0.0, 0.0, h));
    let vel_mz = sample_velocity_at(pos - vec3<f32>(0.0, 0.0, h));

    let inv_2h = 1.0 / (2.0 * h);

    // Velocity gradient components
    let dudx = (vel_px.x - vel_mx.x) * inv_2h;
    let dudy = (vel_py.x - vel_my.x) * inv_2h;
    let dudz = (vel_pz.x - vel_mz.x) * inv_2h;
    let dvdx = (vel_px.y - vel_mx.y) * inv_2h;
    let dvdy = (vel_py.y - vel_my.y) * inv_2h;
    let dvdz = (vel_pz.y - vel_mz.y) * inv_2h;
    let dwdx = (vel_px.z - vel_mx.z) * inv_2h;
    let dwdy = (vel_py.z - vel_my.z) * inv_2h;
    let dwdz = (vel_pz.z - vel_mz.z) * inv_2h;

    // Strain rate tensor (symmetric part): D = 0.5*(grad_v + grad_v^T)
    let D_xx = dudx;
    let D_yy = dvdy;
    let D_zz = dwdz;
    let D_xy = 0.5 * (dudy + dvdx);
    let D_xz = 0.5 * (dudz + dwdx);
    let D_yz = 0.5 * (dvdz + dwdy);

    // Second invariant: |D| = sqrt(0.5 * D:D)
    // D:D = D_xx² + D_yy² + D_zz² + 2*(D_xy² + D_xz² + D_yz²)
    let D_sq = D_xx*D_xx + D_yy*D_yy + D_zz*D_zz +
               2.0*(D_xy*D_xy + D_xz*D_xz + D_yz*D_yz);

    return sqrt(0.5 * D_sq);
}
```

### 3.3 Replace sediment branch in main function

Find the existing sediment branch (starts with `if (density > 1.0) {`) and replace it entirely:

```wgsl
    let density = densities[id.x];
    if (density > 1.0) {
        // ========== SEDIMENT: Drucker-Prager yield model ==========

        // Get cell indices
        let cell_i = clamp(i32(pos.x / cell_size), 0, width - 1);
        let cell_j = clamp(i32(pos.y / cell_size), 0, height - 1);
        let cell_k = clamp(i32(pos.z / cell_size), 0, depth - 1);
        let cell_idx = cell_index(cell_i, cell_j, cell_k);

        // Sample sediment pressure (from column weight)
        let pressure = max(sediment_pressure[cell_idx], dp_params.min_pressure);

        // Compute shear rate
        let shear_rate = compute_shear_rate(pos);

        // Drucker-Prager yield criterion
        // yield_stress = cohesion + pressure * friction_coeff
        let yield_stress = dp_params.cohesion + pressure * dp_params.friction_coeff;

        // Shear stress estimate: τ = η * γ̇
        let shear_stress = dp_params.viscosity * shear_rate;

        // Yield ratio: how much we exceed yield (0 = jammed, 1 = fully yielding)
        let stress_diff = shear_stress - yield_stress;
        let yield_ratio = clamp(
            stress_diff / (dp_params.yield_smoothing * yield_stress + 0.001),
            0.0,
            1.0
        );

        var final_velocity: vec3<f32>;

        if (yield_ratio > 0.01) {
            // YIELDING - viscoplastic flow
            // Blend toward grid velocity based on yield ratio
            let effective_drag = yield_ratio * sediment_params.drag_rate;
            let drag_factor = 1.0 - exp(-effective_drag * params.dt);
            final_velocity = mix(old_particle_vel, new_velocity, drag_factor);

            // Reduced settling when yielding (material is flowing)
            let settling_reduction = 1.0 - yield_ratio * 0.7;
            final_velocity.y -= sediment_params.settling_velocity * params.dt * settling_reduction;
        } else {
            // JAMMED - move with the pack (high drag)
            let jam_drag = 1.0 - exp(-dp_params.jammed_drag * params.dt);
            final_velocity = mix(old_particle_vel, new_velocity, jam_drag);

            // Full settling when jammed
            final_velocity.y -= sediment_params.settling_velocity * params.dt;
        }

        // Vorticity lift (suspension in turbulent flow)
        let vort = vorticity_mag[cell_idx];
        let vort_excess = max(vort - sediment_params.vorticity_threshold, 0.0);
        let lift_factor = clamp(sediment_params.vorticity_lift * vort_excess, 0.0, 0.9);
        final_velocity.y += sediment_params.settling_velocity * params.dt * lift_factor;

        // Safety clamp
        let speed = length(final_velocity);
        if (speed > params.max_velocity) {
            final_velocity *= params.max_velocity / speed;
        }

        velocities[id.x] = final_velocity;

        // Zero C matrix for sediment (no APIC affine transfer)
        c_col0[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        c_col1[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        c_col2[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }
```

---

## Phase 4: Wire Up Sediment Pressure Buffer

### 4.1 Pass buffer reference to G2P

The `sediment_pressure_buffer` is created in `GpuFlip3D` but needs to be accessible to `GpuG2p3D`.

**Option A**: Pass buffer reference when creating G2P bind group.

In `GpuFlip3D::new()`, after creating `sediment_pressure_buffer`, pass it to G2P:

```rust
// Create G2P with sediment pressure buffer reference
let g2p = GpuG2p3D::new(
    device,
    // ... existing params ...
    &sediment_pressure_buffer,  // Add this parameter
);
```

Update `GpuG2p3D::new()` signature to accept the buffer:

```rust
pub fn new(
    // ... existing params ...
    sediment_pressure_buffer: &wgpu::Buffer,
) -> Self {
```

And use it when creating the bind group.

### 4.2 Alternative: Store buffer in GpuFlip3D and create bind group later

If the G2P is created before the pressure buffer, restructure so the bind group is created after both exist.

---

## Phase 5: Expose Parameters in Industrial Sluice

### 5.1 Add UI controls

**Modify**: `crates/game/examples/industrial_sluice.rs`

Add to config struct:

```rust
// Drucker-Prager parameters
dp_friction_angle_deg: f32,  // degrees, default 32.0
dp_cohesion: f32,            // Pa, default 0.0
dp_viscosity: f32,           // Pa·s, default 1.0
dp_jammed_drag: f32,         // default 50.0
```

Add UI sliders in egui panel:

```rust
ui.add(egui::Slider::new(&mut config.dp_friction_angle_deg, 20.0..=45.0)
    .text("Friction angle (°)"));
ui.add(egui::Slider::new(&mut config.dp_cohesion, 0.0..=100.0)
    .text("Cohesion (Pa)"));
ui.add(egui::Slider::new(&mut config.dp_viscosity, 0.1..=10.0)
    .logarithmic(true)
    .text("Viscosity (Pa·s)"));
ui.add(egui::Slider::new(&mut config.dp_jammed_drag, 10.0..=100.0)
    .text("Jammed drag"));
```

Update params each frame:

```rust
let dp_params = DruckerPragerParams {
    friction_coeff: (config.dp_friction_angle_deg.to_radians()).tan(),
    cohesion: config.dp_cohesion,
    buoyancy_factor: 0.62,
    viscosity: config.dp_viscosity,
    jammed_drag: config.dp_jammed_drag,
    min_pressure: 0.1,
    yield_smoothing: 0.1,
    _pad0: 0.0,
};
flip.g2p.set_drucker_prager_params(&queue, dp_params);
```

---

## Phase 6: Testing

### 6.1 Compilation test

```bash
cargo build -p game --release
```

### 6.2 Runtime test

```bash
cargo run --example industrial_sluice --release
```

Verify:
- No crashes
- Sediment still settles
- Adjusting friction angle changes pile slope
- High flow breaks up accumulated sediment

### 6.3 Angle of repose test (manual)

1. Reduce inlet flow to let sediment accumulate
2. Observe pile forming behind riffles
3. Pile should stabilize at roughly 30-35° slope
4. Increasing friction angle should make steeper piles

---

## File Summary

| File | Action |
|------|--------|
| `crates/game/src/gpu/shaders/sediment_pressure_3d.wgsl` | CREATE |
| `crates/game/src/gpu/shaders/g2p_3d.wgsl` | MODIFY (add DP logic) |
| `crates/game/src/gpu/flip_3d.rs` | MODIFY (add pressure buffer + pipeline) |
| `crates/game/src/gpu/g2p_3d.rs` | MODIFY (add DP params buffer) |
| `crates/game/examples/industrial_sluice.rs` | MODIFY (add UI controls) |

---

## Checklist

- [ ] Phase 1: Create `sediment_pressure_3d.wgsl`
- [ ] Phase 1: Add pressure buffer to `flip_3d.rs`
- [ ] Phase 1: Add pressure pipeline and dispatch
- [ ] Phase 2: Add `DruckerPragerParams` struct
- [ ] Phase 2: Add DP params buffer to `g2p_3d.rs`
- [ ] Phase 3: Add `sample_velocity_at` function to shader
- [ ] Phase 3: Add `compute_shear_rate` function to shader
- [ ] Phase 3: Replace sediment branch with DP yield logic
- [ ] Phase 3: Add bindings 15 and 16 to shader
- [ ] Phase 4: Wire sediment_pressure_buffer to G2P bind group
- [ ] Phase 5: Add UI controls to industrial_sluice
- [ ] Phase 6: Test compilation
- [ ] Phase 6: Test runtime behavior
