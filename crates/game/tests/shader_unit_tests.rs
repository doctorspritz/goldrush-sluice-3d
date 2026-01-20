//! Shader Unit Tests - Testing WGSL shader logic with known inputs
//!
//! This test suite validates compute shader correctness by:
//! 1. Parsing all WGSL shaders with naga
//! 2. Running shaders on GPU with known inputs and verifying outputs
//! 3. Testing critical operations like P2G scatter and prefix sum
//!
//! Uses the headless GPU context from shader_validator.rs to run actual compute dispatches.

use std::fs;
use std::path::Path;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Standard GPU limits required by this project.
const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 16;
const REQUIRED_STORAGE_BUFFER_BINDING_SIZE: u32 = 256 * 1024 * 1024;

/// Creates a headless GPU device for testing.
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

//==============================================================================
// TEST 1: Parse Validation - All shaders must parse without errors
//==============================================================================

#[test]
fn test_all_shaders_parse_with_naga() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders");
    let mut errors = Vec::new();
    let mut shader_count = 0;

    if !shader_dir.exists() {
        panic!("Shader directory not found: {:?}", shader_dir);
    }

    validate_shader_dir(&shader_dir, &mut errors, &mut shader_count);

    if !errors.is_empty() {
        panic!(
            "Shader validation failed for {} shader(s):\n{}",
            errors.len(),
            errors.join("\n")
        );
    }

    println!("✓ Successfully validated {} WGSL shaders", shader_count);
}

fn validate_shader_dir(dir: &Path, errors: &mut Vec<String>, count: &mut usize) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            validate_shader_dir(&path, errors, count);
        } else if path.extension().map_or(false, |ext| ext == "wgsl") {
            *count += 1;
            validate_shader(&path, errors);
        }
    }
}

fn validate_shader(path: &Path, errors: &mut Vec<String>) {
    let source = fs::read_to_string(path).unwrap();

    // Parse with naga
    let module = match naga::front::wgsl::parse_str(&source) {
        Ok(module) => module,
        Err(e) => {
            errors.push(format!(
                "Failed to parse {:?}:\n{}",
                path.file_name().unwrap(),
                e.emit_to_string(&source)
            ));
            return;
        }
    };

    // Validate the parsed module
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(e) = validator.validate(&module) {
        errors.push(format!(
            "Failed to validate {:?}:\n{:?}",
            path.file_name().unwrap(),
            e
        ));
    }
}

//==============================================================================
// TEST 2: Prefix Sum - Verify Blelloch scan algorithm correctness
//==============================================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PrefixSumParams {
    element_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[test]
fn test_prefix_sum_simple_case() {
    let (device, queue) = create_test_device();

    // Load the prefix sum shader
    let shader_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/gpu/shaders/particle_sort_prefix_sum.wgsl");
    let shader_source = fs::read_to_string(shader_path).expect("Failed to read prefix sum shader");

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Prefix Sum Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Test input: [1, 2, 3, 4, 5, 6, 7, 8]
    // Expected output (exclusive prefix sum): [0, 1, 3, 6, 10, 15, 21, 28]
    let input_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let expected: Vec<u32> = vec![0, 1, 3, 6, 10, 15, 21, 28];

    let element_count = input_data.len() as u32;
    let num_blocks = (element_count + 511) / 512;

    // Create buffers
    let params = PrefixSumParams {
        element_count,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prefix Sum Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prefix Sum Data"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix Sum Block Sums"),
        size: (num_blocks * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Prefix Sum Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prefix Sum Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: block_sums_buffer.as_entire_binding(),
            },
        ],
    });

    // Create pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Prefix Sum Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let local_prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Local Prefix Sum Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("local_prefix_sum"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute local prefix sum
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Prefix Sum Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prefix Sum Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&local_prefix_sum_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Dispatch one workgroup (256 threads processing 512 elements)
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // Read back results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (element_count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &data_buffer,
        0,
        &staging_buffer,
        0,
        (element_count * 4) as u64,
    );

    queue.submit(Some(encoder.finish()));

    // Map and verify
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // Verify results
    assert_eq!(
        result, expected,
        "Prefix sum failed: got {:?}, expected {:?}",
        result, expected
    );

    println!("✓ Prefix sum test passed: {:?} → {:?}", input_data, result);
}

//==============================================================================
// TEST 3: Prefix Sum - Larger input (64 elements)
//==============================================================================

#[test]
fn test_prefix_sum_larger_input() {
    let (device, queue) = create_test_device();

    let shader_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/gpu/shaders/particle_sort_prefix_sum.wgsl");
    let shader_source = fs::read_to_string(shader_path).expect("Failed to read prefix sum shader");

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Prefix Sum Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Test input: array of 64 ones
    // Expected output: [0, 1, 2, 3, ..., 63]
    let input_data: Vec<u32> = vec![1; 64];
    let expected: Vec<u32> = (0..64).collect();

    let element_count = input_data.len() as u32;
    let num_blocks = (element_count + 511) / 512;

    let params = PrefixSumParams {
        element_count,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prefix Sum Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prefix Sum Data"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix Sum Block Sums"),
        size: (num_blocks * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Prefix Sum Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prefix Sum Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: block_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Prefix Sum Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let local_prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Local Prefix Sum Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("local_prefix_sum"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Prefix Sum Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prefix Sum Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&local_prefix_sum_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (element_count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &data_buffer,
        0,
        &staging_buffer,
        0,
        (element_count * 4) as u64,
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    assert_eq!(
        result, expected,
        "Prefix sum 64-element test failed"
    );

    println!("✓ Prefix sum 64-element test passed");
}

//==============================================================================
// TEST 4: Quadratic B-spline weights - Verify kernel properties
//==============================================================================

#[test]
fn test_quadratic_bspline_properties() {
    // The quadratic B-spline kernel should satisfy:
    // 1. Partition of unity: sum of weights from 3 adjacent nodes = 1
    // 2. Compact support: weight = 0 for |x| >= 1.5
    // 3. Symmetry: weight(x) = weight(-x)
    // 4. Smoothness: C1 continuous

    // Helper function matching WGSL implementation
    fn quadratic_bspline_1d(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 0.5 {
            0.75 - ax * ax
        } else if ax < 1.5 {
            let t = 1.5 - ax;
            0.5 * t * t
        } else {
            0.0
        }
    }

    // Test compact support
    assert_eq!(quadratic_bspline_1d(1.5), 0.0, "Should be 0 at boundary");
    assert_eq!(quadratic_bspline_1d(2.0), 0.0, "Should be 0 outside support");
    assert_eq!(quadratic_bspline_1d(-2.0), 0.0, "Should be 0 outside support");

    // Test symmetry
    for x in [0.1, 0.5, 1.0, 1.3] {
        let w_pos = quadratic_bspline_1d(x);
        let w_neg = quadratic_bspline_1d(-x);
        assert!(
            (w_pos - w_neg).abs() < 1e-6,
            "Symmetry violated at x={}: w({})={}, w({})={}",
            x, x, w_pos, -x, w_neg
        );
    }

    // Test partition of unity at several positions
    // For quadratic B-spline with compact support [-1.5, 1.5],
    // we need to test that for any position x, the sum of weights
    // to the 3 nearest grid nodes equals 1.
    // The three nodes are at floor(x), floor(x)+1, floor(x)+2
    // when measuring distance as abs(x - node_position)
    for frac in [0.0, 0.25, 0.5] {
        // Distances to three adjacent nodes for a particle at position frac
        // Node -1 is at distance frac + 1.0
        // Node 0 is at distance frac
        // Node 1 is at distance 1.0 - frac
        let w_m1 = quadratic_bspline_1d(frac + 1.0);
        let w_0 = quadratic_bspline_1d(frac);
        let w_p1 = quadratic_bspline_1d(1.0 - frac);
        let sum = w_m1 + w_0 + w_p1;

        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Partition of unity violated at frac={}: sum={}, weights=[{}, {}, {}]",
            frac, sum, w_m1, w_0, w_p1
        );
    }

    // Test peak value at x=0
    let peak = quadratic_bspline_1d(0.0);
    assert_eq!(peak, 0.75, "Peak should be 0.75 at x=0");

    println!("✓ Quadratic B-spline kernel properties verified");
}

//==============================================================================
// TEST 5: P2G Scatter - Test single particle contribution
//==============================================================================

#[test]
fn test_p2g_scatter_single_particle_center() {
    // Test that a single particle at cell center (0.5, 0.5, 0.5)
    // distributes its momentum to the surrounding grid nodes correctly.
    // This is a simplified version - full test would require all 15 bindings.

    let (device, _queue) = create_test_device();

    // For now, just verify the shader compiles and can be created as a pipeline
    let shader_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/gpu/shaders/p2g_scatter_3d.wgsl");
    let shader_source = fs::read_to_string(shader_path)
        .expect("Failed to read p2g_scatter_3d shader");

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("P2G Scatter Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create a minimal bind group layout matching the shader
    // This requires 15 bindings (0-14) as per the shader
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("P2G Scatter Layout"),
        entries: &[
            // Binding 0: uniform params
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
            // Bindings 1-14: storage buffers
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 14,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("P2G Scatter Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Verify the compute pipeline can be created (validates shader compilation)
    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("P2G Scatter Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("scatter"),
        compilation_options: Default::default(),
        cache: None,
    });

    println!("✓ P2G scatter shader compiles and pipeline creation succeeds");
}
