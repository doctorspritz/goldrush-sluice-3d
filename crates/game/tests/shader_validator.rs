use std::fs;
use std::path::Path;

/// Standard GPU limits required by this project.
/// All examples and code MUST request these limits.
pub const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 16;
pub const REQUIRED_STORAGE_BUFFER_BINDING_SIZE: u32 = 256 * 1024 * 1024;

/// Creates a headless GPU device with our standard limits.
/// Use this in examples to avoid the "Too many bindings" error.
fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = REQUIRED_STORAGE_BUFFERS_PER_STAGE;
        limits.max_storage_buffer_binding_size = REQUIRED_STORAGE_BUFFER_BINDING_SIZE;

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create GPU device")
    })
}

/// Test that GpuHeightfield can be created with our standard limits.
/// This catches the "Too many bindings of type StorageBuffers" error at test time.
#[test]
fn gpu_heightfield_respects_storage_limits() {
    let (device, _queue) = create_test_device();

    // This will panic if GpuHeightfield needs more storage buffers than our limits allow
    let _heightfield = game::gpu::heightfield::GpuHeightfield::new(
        &device,
        64,  // small test size
        64,
        1.0,
        10.0,
        wgpu::TextureFormat::Bgra8Unorm,
    );
}

/// Test that GpuFlip3D can be created with our standard limits.
#[test]
fn gpu_flip3d_respects_storage_limits() {
    let (device, _queue) = create_test_device();

    let _flip = game::gpu::flip_3d::GpuFlip3D::new(
        &device,
        16, 16, 16,  // small test grid
        0.1,         // cell size
        1000,        // max particles
    );
}

#[test]
fn validate_all_shaders() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders");
    let mut errors = Vec::new();

    if !shader_dir.exists() {
        panic!("Shader directory not found: {:?}", shader_dir);
    }

    validate_dir(&shader_dir, &mut errors);

    if !errors.is_empty() {
        panic!("Shader validation failed:\n{}", errors.join("\n"));
    }
}

fn validate_dir(dir: &Path, errors: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            validate_dir(&path, errors);
        } else if path.extension().map_or(false, |ext| ext == "wgsl") {
            validate_shader(&path, errors);
        }
    }
}

fn validate_shader(path: &Path, errors: &mut Vec<String>) {
    let source = fs::read_to_string(path).unwrap();
    let module = match naga::front::wgsl::parse_str(&source) {
        Ok(module) => module,
        Err(e) => {
            errors.push(format!("Failed to parse {:?}:\n{}", path.file_name().unwrap(), e.emit_to_string(&source)));
            return;
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(e) = validator.validate(&module) {
        errors.push(format!("Failed to validate {:?}:\n{:?}", path.file_name().unwrap(), e));
    }
}
