# Unified App Framework

## Overview

Create a shared framework module (`crates/game/src/app.rs`) that eliminates boilerplate across examples. Target: reduce example files by 300-500 lines each.

## Architecture

```
crates/game/src/app.rs          # New framework module
crates/game/src/app/
  ├── context.rs                # GpuContext - device, queue, surface
  ├── camera.rs                 # FlyCamera - FPS-style controls
  ├── uniforms.rs               # ViewUniforms + bind group
  ├── pipeline.rs               # Pipeline presets
  ├── vertex.rs                 # Common vertex types
  └── runner.rs                 # App trait + event loop
```

## Components

### 1. GpuContext (`context.rs`)

```rust
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    pub view_bind_group_layout: wgpu::BindGroupLayout,
    pub view_bind_group: wgpu::BindGroup,
    pub view_uniform_buffer: wgpu::Buffer,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Self;
    pub fn resize(&mut self, width: u32, height: u32);
    pub fn update_view_uniforms(&self, uniforms: &ViewUniforms);
    pub fn surface_format(&self) -> wgpu::TextureFormat;
    pub fn depth_format(&self) -> wgpu::TextureFormat; // Always Depth32Float
}
```

**Requirements:**
- Default window: 1920×1080
- GPU limits: `max_storage_buffers_per_shader_stage = 16`, `max_storage_buffer_binding_size = 256MB`
- Depth format: `Depth32Float`
- Surface format: `Bgra8UnormSrgb` (or adapter preferred)

### 2. FlyCamera (`camera.rs`)

```rust
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,        // Radians, 0 = looking along +X
    pub pitch: f32,      // Radians, clamped to ±89°
    pub speed: f32,      // Units per second
    pub sensitivity: f32, // Mouse sensitivity
    pub fov: f32,        // Field of view in radians
    pub near: f32,
    pub far: f32,
}

impl FlyCamera {
    pub fn new() -> Self;
    pub fn with_position(self, pos: Vec3) -> Self;
    pub fn with_target(self, target: Vec3) -> Self; // Sets yaw/pitch to look at target

    pub fn forward(&self) -> Vec3;  // Unit vector
    pub fn right(&self) -> Vec3;    // Unit vector
    pub fn up(&self) -> Vec3;       // Always world up (0,1,0)

    pub fn view_matrix(&self) -> Mat4;
    pub fn projection_matrix(&self, aspect: f32) -> Mat4;
    pub fn view_projection(&self, aspect: f32) -> Mat4;

    // Input handling
    pub fn on_mouse_move(&mut self, delta_x: f32, delta_y: f32);
    pub fn on_scroll(&mut self, delta: f32);
    pub fn update(&mut self, input: &InputState, dt: f32); // WASD + Space/Shift
}

pub struct InputState {
    pub forward: bool,   // W
    pub back: bool,      // S
    pub left: bool,      // A
    pub right: bool,     // D
    pub up: bool,        // Space
    pub down: bool,      // Shift
}
```

**Controls:**
- Mouse drag: rotate (yaw/pitch)
- Scroll wheel: move forward/back along look direction
- W/S: forward/back
- A/D: strafe left/right
- Space: move up
- Shift: move down

**Defaults:**
- `speed`: 5.0
- `sensitivity`: 0.003
- `fov`: 60° (1.047 rad)
- `near`: 0.01
- `far`: 100.0

### 3. ViewUniforms (`uniforms.rs`)

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ViewUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad: f32,
}

impl ViewUniforms {
    pub fn from_camera(camera: &FlyCamera, aspect: f32) -> Self;
}
```

**Bind group layout (group 0):**
```wgsl
@group(0) @binding(0) var<uniform> view: ViewUniforms;
```

### 4. Pipeline Presets (`pipeline.rs`)

```rust
pub enum PipelinePreset {
    OpaqueMesh,      // Triangles, depth test, backface cull
    OpaqueInstanced, // Same but with instance buffer
    Transparent,     // Alpha blend, no depth write
    Lines,           // Line list, depth test
}

impl GpuContext {
    pub fn create_pipeline(
        &self,
        preset: PipelinePreset,
        shader_source: &str,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        additional_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::RenderPipeline;
}
```

**Preset details:**

| Preset | Topology | Depth | Cull | Blend |
|--------|----------|-------|------|-------|
| OpaqueMesh | TriangleList | Less, write | Back | None |
| OpaqueInstanced | TriangleList | Less, write | Back | None |
| Transparent | TriangleList | Less, no write | None | Alpha |
| Lines | LineList | Less, write | None | None |

### 5. Common Vertex Types (`vertex.rs`)

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColoredVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InstanceTransform {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl MeshVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static>;
}
// etc for each type
```

### 6. App Runner (`runner.rs`)

```rust
pub trait App: 'static {
    fn init(ctx: &GpuContext) -> Self;
    fn update(&mut self, ctx: &GpuContext, dt: f32);
    fn render(&mut self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView);

    // Optional overrides
    fn on_key(&mut self, _key: KeyCode, _pressed: bool) {}
    fn on_resize(&mut self, _ctx: &GpuContext) {}
    fn camera(&self) -> &FlyCamera;
    fn camera_mut(&mut self) -> &mut FlyCamera;
    fn title() -> &'static str { "App" }
}

pub fn run<A: App>() -> !;
```

**Event loop behavior:**
1. Create window (1920×1080, title from `A::title()`)
2. Create `GpuContext`
3. Call `A::init(&ctx)`
4. Loop:
   - Handle input events → update `InputState`, call `on_key`
   - Call `app.camera_mut().update(&input, dt)`
   - Call `app.update(&ctx, dt)`
   - Update view uniforms from camera
   - Begin render pass (clear color to dark gray, clear depth)
   - Call `app.render(&ctx, &mut encoder, &view)`
   - Submit + present

---

## Parallel Work Packages

### Package A: Core Context (`context.rs` + `uniforms.rs`)

**Files to create:**
- `crates/game/src/app/mod.rs`
- `crates/game/src/app/context.rs`
- `crates/game/src/app/uniforms.rs`

**Acceptance criteria:**
1. `GpuContext::new()` creates device with required limits
2. `GpuContext::resize()` recreates depth texture
3. View bind group layout at group(0), binding(0)
4. `update_view_uniforms()` writes to GPU buffer

**Test:** Create standalone test that initializes context headlessly (no window).

```rust
#[test]
fn test_gpu_context_headless() {
    // Use wgpu::Instance with no surface to verify device creation
    let instance = wgpu::Instance::new(Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: GpuContext::required_limits(),
        ..Default::default()
    }, None)).unwrap();
    assert!(device.limits().max_storage_buffers_per_shader_stage >= 16);
}
```

---

### Package B: Camera (`camera.rs`)

**Files to create:**
- `crates/game/src/app/camera.rs`

**Acceptance criteria:**
1. `FlyCamera::forward()` returns correct direction based on yaw/pitch
2. `FlyCamera::view_matrix()` produces valid look-at matrix
3. `on_mouse_move()` updates yaw/pitch correctly
4. Pitch is clamped to ±89° (±1.553 rad)
5. `update()` moves position based on WASD + Space/Shift

**Tests:**

```rust
#[test]
fn test_camera_forward_direction() {
    let mut cam = FlyCamera::new();
    cam.yaw = 0.0;
    cam.pitch = 0.0;
    let fwd = cam.forward();
    assert!((fwd.x - 1.0).abs() < 0.001); // Looking along +X
    assert!(fwd.y.abs() < 0.001);
    assert!(fwd.z.abs() < 0.001);
}

#[test]
fn test_camera_pitch_clamp() {
    let mut cam = FlyCamera::new();
    cam.on_mouse_move(0.0, 10000.0); // Huge downward motion
    assert!(cam.pitch <= 1.554); // ~89°
    assert!(cam.pitch >= -1.554);
}

#[test]
fn test_camera_wasd_movement() {
    let mut cam = FlyCamera::new();
    cam.position = Vec3::ZERO;
    cam.yaw = 0.0; // Looking +X
    cam.speed = 10.0;

    let input = InputState { forward: true, ..Default::default() };
    cam.update(&input, 1.0);

    assert!(cam.position.x > 9.0); // Moved ~10 units forward
}
```

---

### Package C: Pipeline Presets (`pipeline.rs`)

**Files to create:**
- `crates/game/src/app/pipeline.rs`

**Acceptance criteria:**
1. `OpaqueMesh` preset: depth test less, depth write, backface cull
2. `Transparent` preset: alpha blend, depth test, no depth write
3. All presets include view bind group layout at group(0)
4. Vertex layouts are correctly passed through
5. Shader compilation errors surface clearly

**Test:**

```rust
#[test]
fn test_pipeline_preset_compiles() {
    // Minimal shader that uses ViewUniforms
    let shader = r#"
        struct ViewUniforms { view_proj: mat4x4<f32>, camera_pos: vec3<f32>, _pad: f32 }
        @group(0) @binding(0) var<uniform> view: ViewUniforms;

        struct VertexOutput { @builtin(position) pos: vec4<f32> }

        @vertex fn vs_main(@location(0) pos: vec3<f32>) -> VertexOutput {
            var out: VertexOutput;
            out.pos = view.view_proj * vec4(pos, 1.0);
            return out;
        }

        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4(1.0);
        }
    "#;

    // Should not panic
    let pipeline = ctx.create_pipeline(
        PipelinePreset::OpaqueMesh,
        shader,
        &[MeshVertex::desc()],
        &[],
    );
}
```

---

### Package D: Vertex Types (`vertex.rs`)

**Files to create:**
- `crates/game/src/app/vertex.rs`

**Acceptance criteria:**
1. `MeshVertex` has position (location 0) and normal (location 1)
2. `ColoredVertex` has position (location 0) and color (location 1)
3. `InstanceTransform` has model matrix (locations 0-3) and color (location 4)
4. All implement `Pod`, `Zeroable`
5. `desc()` returns correct `VertexBufferLayout`

**Tests:**

```rust
#[test]
fn test_mesh_vertex_layout() {
    let layout = MeshVertex::desc();
    assert_eq!(layout.array_stride, 24); // 3*f32 + 3*f32
    assert_eq!(layout.attributes.len(), 2);
}

#[test]
fn test_instance_transform_layout() {
    let layout = InstanceTransform::desc();
    assert_eq!(layout.array_stride, 80); // 16*f32 + 4*f32
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
}
```

---

### Package E: App Runner (`runner.rs`)

**Files to create:**
- `crates/game/src/app/runner.rs`

**Depends on:** A, B, C, D (run last)

**Acceptance criteria:**
1. Window created at 1920×1080
2. Event loop handles: resize, keyboard, mouse motion, scroll
3. `InputState` correctly tracks WASD + Space + Shift
4. Camera updates each frame from input
5. View uniforms uploaded before render
6. Render pass clears to dark gray `[0.1, 0.1, 0.1, 1.0]`
7. Frame timing provides reasonable dt (capped at 0.1s)

**Test:** Create minimal example that compiles and runs:

```rust
// examples/framework_test.rs
use game::app::{App, FlyCamera, GpuContext, run};

struct MinimalApp {
    camera: FlyCamera,
}

impl App for MinimalApp {
    fn init(_ctx: &GpuContext) -> Self {
        Self { camera: FlyCamera::new().with_position(Vec3::new(0.0, 1.0, 5.0)) }
    }

    fn update(&mut self, _ctx: &GpuContext, _dt: f32) {}

    fn render(&mut self, _ctx: &GpuContext, _encoder: &mut wgpu::CommandEncoder, _view: &wgpu::TextureView) {}

    fn camera(&self) -> &FlyCamera { &self.camera }
    fn camera_mut(&mut self) -> &mut FlyCamera { &mut self.camera }
    fn title() -> &'static str { "Framework Test" }
}

fn main() {
    run::<MinimalApp>();
}
```

**Manual verification:** Run example, verify WASD/mouse controls work.

---

### Package F: Port `friction_sluice` Example

**Depends on:** E

**Files to modify:**
- `crates/game/examples/friction_sluice.rs`

**Acceptance criteria:**
1. Example compiles and runs identically to before
2. Lines of code reduced by at least 250
3. Uses `App` trait, `FlyCamera`, `GpuContext`
4. Uses `PipelinePreset::OpaqueMesh` for sluice geometry
5. Camera controls match new FPS-style (not orbit)

**Test:** Visual comparison - simulation behavior unchanged.

---

## Execution Order

```
     ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
     │ Pkg A   │  │ Pkg B   │  │ Pkg C   │  │ Pkg D   │
     │ Context │  │ Camera  │  │Pipeline │  │ Vertex  │
     └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
          │            │            │            │
          └────────────┴─────┬──────┴────────────┘
                             │
                       ┌─────▼─────┐
                       │  Pkg E    │
                       │  Runner   │
                       └─────┬─────┘
                             │
                       ┌─────▼─────┐
                       │  Pkg F    │
                       │  Port Ex  │
                       └───────────┘
```

**Packages A, B, C, D** can run in parallel.
**Package E** depends on A, B, C, D.
**Package F** depends on E.

---

## File Structure After Completion

```
crates/game/src/
├── app.rs              # pub mod + re-exports
├── app/
│   ├── mod.rs
│   ├── context.rs      # GpuContext
│   ├── camera.rs       # FlyCamera, InputState
│   ├── uniforms.rs     # ViewUniforms
│   ├── pipeline.rs     # PipelinePreset, create_pipeline
│   ├── vertex.rs       # MeshVertex, ColoredVertex, InstanceTransform
│   └── runner.rs       # App trait, run()
├── lib.rs              # Add: pub mod app;
└── ...
```

---

## Success Metrics

1. All tests pass: `cargo test -p game`
2. `framework_test` example runs with FPS camera controls
3. `friction_sluice` ported and working
4. Code reduction: friction_sluice.rs < 1300 lines (from 1778)
