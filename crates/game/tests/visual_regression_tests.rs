use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::fluid_renderer::ScreenSpaceFluidRenderer;
use glam::{Mat4, Vec3};
use std::path::Path;

mod headless_harness;

const TEST_WIDTH: u32 = 800;
const TEST_HEIGHT: u32 = 600;
const REFERENCE_DIR: &str = "tests/visual_references";

/// Initialize headless wgpu device and queue for rendering tests
fn init_rendering_context() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_harness::init_device_queue()
}

/// Create a simple camera view matrix looking at the origin
fn create_camera_view(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, center, up)
}

/// Create a perspective projection matrix
fn create_camera_proj(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y, aspect, near, far)
}

/// Render particles to an offscreen texture and return the pixel data
fn render_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &ScreenSpaceFluidRenderer,
    gpu_flip: &GpuFlip3D,
    particle_count: u32,
    camera_view: Mat4,
    camera_proj: Mat4,
    camera_pos: Vec3,
    width: u32,
    height: u32,
) -> Vec<u8> {
    // Create output texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Visual Test Output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());

    // Clear to black background
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Visual Test Encoder"),
    });

    {
        let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    // Render fluid particles
    renderer.render(
        device,
        queue,
        &mut encoder,
        &view,
        gpu_flip,
        particle_count,
        camera_view.to_cols_array_2d(),
        camera_proj.to_cols_array_2d(),
        camera_pos.to_array(),
        width,
        height,
    );

    // Create staging buffer for readback
    let bytes_per_pixel = 4; // RGBA8
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Visual Test Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture to buffer
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Read back pixels
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();

    // Unpad rows if necessary
    let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        let row_end = row_start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&data[row_start..row_end]);
    }

    drop(data);
    staging_buffer.unmap();

    pixels
}

/// Save pixels as PNG
fn save_png(path: &Path, pixels: &[u8], width: u32, height: u32) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header()
        .map_err(|e| format!("Failed to write PNG header: {}", e))?;

    writer.write_image_data(pixels)
        .map_err(|e| format!("Failed to write PNG data: {}", e))?;

    Ok(())
}

/// Load pixels from PNG
fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open file: {}", e))?;

    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)
        .map_err(|e| format!("Failed to read PNG frame: {}", e))?;

    buf.truncate(info.buffer_size());

    Ok((buf, info.width, info.height))
}

/// Compare two pixel buffers with tolerance
/// Returns (passed, diff_percentage, max_pixel_diff)
fn compare_images(expected: &[u8], actual: &[u8], tolerance: f32) -> (bool, f32, u8) {
    assert_eq!(expected.len(), actual.len(), "Image sizes must match");

    let mut diff_pixels = 0;
    let mut max_diff = 0u8;

    for (e, a) in expected.chunks(4).zip(actual.chunks(4)) {
        let mut pixel_differs = false;
        for i in 0..4 {
            let diff = (e[i] as i16 - a[i] as i16).abs() as u8;
            if diff > 1 { // Allow 1 bit tolerance per channel for compression artifacts
                pixel_differs = true;
                max_diff = max_diff.max(diff);
            }
        }
        if pixel_differs {
            diff_pixels += 1;
        }
    }

    let total_pixels = expected.len() / 4;
    let diff_percentage = (diff_pixels as f32 / total_pixels as f32) * 100.0;

    (diff_percentage <= tolerance, diff_percentage, max_diff)
}

/// Helper to set up a basic FLIP simulation with a given particle configuration
fn setup_flip_simulation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut particles: Vec<Vec3>,
) -> (GpuFlip3D, u32) {
    let grid_width = 64;
    let grid_height = 64;
    let grid_depth = 64;
    let cell_size = 0.1;
    let max_particles = 1_000_000;

    let mut gpu_flip = GpuFlip3D::new(
        device,
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        max_particles,
    );

    // Set up cell types (walls on boundaries)
    let mut cell_types = vec![1u32; (grid_width * grid_height * grid_depth) as usize]; // CELL_FLUID
    for z in 0..grid_depth {
        for y in 0..grid_height {
            for x in 0..grid_width {
                if x == 0 || x == grid_width - 1
                    || y == 0 || y == grid_height - 1
                    || z == 0 || z == grid_depth - 1
                {
                    let idx = (z * grid_width * grid_height + y * grid_width + x) as usize;
                    cell_types[idx] = 2; // CELL_SOLID
                }
            }
        }
    }

    // Upload particles via step function
    let particle_count = particles.len() as u32;
    if particle_count > 0 {
        let mut velocities = vec![Vec3::ZERO; particles.len()];
        let mut c_matrices = vec![glam::Mat3::ZERO; particles.len()];
        let densities = vec![1.0f32; particles.len()];

        // Call step with dt=0 to just upload particles without simulation
        gpu_flip.step_no_readback(
            device,
            queue,
            &mut particles,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            0.0,
            0.0,
            0.0,
            0,
        );
    }

    (gpu_flip, particle_count)
}

#[test]
fn test_render_empty_scene() {
    let (device, queue) = match init_rendering_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("Skipping visual test: No GPU available");
            return;
        }
    };

    // Create renderer
    let mut renderer = ScreenSpaceFluidRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.particle_radius = 0.05;
    renderer.resize(&device, TEST_WIDTH, TEST_HEIGHT);

    // Create empty simulation
    let (gpu_flip, particle_count) = setup_flip_simulation(&device, &queue, vec![]);

    // Set up camera
    let eye = Vec3::new(3.2, 2.4, 3.2);
    let center = Vec3::new(3.2, 3.2, 3.2);
    let up = Vec3::Y;
    let view = create_camera_view(eye, center, up);
    let proj = create_camera_proj(
        45.0f32.to_radians(),
        TEST_WIDTH as f32 / TEST_HEIGHT as f32,
        0.1,
        100.0,
    );

    // Render
    let pixels = render_to_texture(
        &device,
        &queue,
        &renderer,
        &gpu_flip,
        particle_count,
        view,
        proj,
        eye,
        TEST_WIDTH,
        TEST_HEIGHT,
    );

    // Compare or save reference
    let ref_path = Path::new(REFERENCE_DIR).join("empty_scene.png");

    if ref_path.exists() {
        let (expected, width, height) = load_png(&ref_path).unwrap();
        assert_eq!(width, TEST_WIDTH, "Reference image width mismatch");
        assert_eq!(height, TEST_HEIGHT, "Reference image height mismatch");

        let (passed, diff_pct, max_diff) = compare_images(&expected, &pixels, 1.0);

        if !passed {
            // Save failed output for debugging
            let fail_path = Path::new(REFERENCE_DIR).join("empty_scene_FAILED.png");
            save_png(&fail_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
            panic!(
                "Visual regression test failed!\n\
                 Difference: {:.2}% of pixels (max channel diff: {})\n\
                 Failed output saved to: {:?}",
                diff_pct, max_diff, fail_path
            );
        }

        println!("Empty scene test passed (diff: {:.4}%)", diff_pct);
    } else {
        // No reference exists, save this as the new reference
        std::fs::create_dir_all(REFERENCE_DIR).unwrap();
        save_png(&ref_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
        println!("Saved new reference image: {:?}", ref_path);
    }
}

#[test]
fn test_render_particles_grid() {
    let (device, queue) = match init_rendering_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("Skipping visual test: No GPU available");
            return;
        }
    };

    // Create renderer
    let mut renderer = ScreenSpaceFluidRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.particle_radius = 0.05;
    renderer.resize(&device, TEST_WIDTH, TEST_HEIGHT);

    // Create particles in a regular 4x4x4 grid pattern
    let mut particles = Vec::new();
    let spacing = 0.2;
    let offset = Vec3::new(2.8, 2.8, 2.8);

    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let pos = offset + Vec3::new(
                    x as f32 * spacing,
                    y as f32 * spacing,
                    z as f32 * spacing,
                );
                particles.push(pos);
            }
        }
    }

    let (gpu_flip, particle_count) = setup_flip_simulation(&device, &queue, particles);

    // Set up camera to see the grid
    let eye = Vec3::new(2.5, 3.2, 4.5);
    let center = Vec3::new(3.0, 3.0, 3.0);
    let up = Vec3::Y;
    let view = create_camera_view(eye, center, up);
    let proj = create_camera_proj(
        45.0f32.to_radians(),
        TEST_WIDTH as f32 / TEST_HEIGHT as f32,
        0.1,
        100.0,
    );

    // Render
    let pixels = render_to_texture(
        &device,
        &queue,
        &renderer,
        &gpu_flip,
        particle_count,
        view,
        proj,
        eye,
        TEST_WIDTH,
        TEST_HEIGHT,
    );

    // Compare or save reference
    let ref_path = Path::new(REFERENCE_DIR).join("particles_grid.png");

    if ref_path.exists() {
        let (expected, width, height) = load_png(&ref_path).unwrap();
        assert_eq!(width, TEST_WIDTH, "Reference image width mismatch");
        assert_eq!(height, TEST_HEIGHT, "Reference image height mismatch");

        let (passed, diff_pct, max_diff) = compare_images(&expected, &pixels, 1.0);

        if !passed {
            let fail_path = Path::new(REFERENCE_DIR).join("particles_grid_FAILED.png");
            save_png(&fail_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
            panic!(
                "Visual regression test failed!\n\
                 Difference: {:.2}% of pixels (max channel diff: {})\n\
                 Failed output saved to: {:?}",
                diff_pct, max_diff, fail_path
            );
        }

        println!("Particles grid test passed (diff: {:.4}%)", diff_pct);
    } else {
        std::fs::create_dir_all(REFERENCE_DIR).unwrap();
        save_png(&ref_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
        println!("Saved new reference image: {:?}", ref_path);
    }
}

#[test]
fn test_render_settled_box() {
    let (device, queue) = match init_rendering_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("Skipping visual test: No GPU available");
            return;
        }
    };

    // Create renderer
    let mut renderer = ScreenSpaceFluidRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.particle_radius = 0.05;
    renderer.resize(&device, TEST_WIDTH, TEST_HEIGHT);

    // Create particles settled at the bottom of container
    let mut particles = Vec::new();
    let cell_size = 0.1;
    let spacing = cell_size / 2.0;

    // Create a 8x4x8 cell box of particles at the bottom
    for cell_x in 4..12 {
        for cell_y in 1..5 {
            for cell_z in 4..12 {
                for px in 0..2 {
                    for py in 0..2 {
                        for pz in 0..2 {
                            let pos = Vec3::new(
                                cell_x as f32 * cell_size + px as f32 * spacing + spacing * 0.5,
                                cell_y as f32 * cell_size + py as f32 * spacing + spacing * 0.5,
                                cell_z as f32 * cell_size + pz as f32 * spacing + spacing * 0.5,
                            );
                            particles.push(pos);
                        }
                    }
                }
            }
        }
    }

    let (gpu_flip, particle_count) = setup_flip_simulation(&device, &queue, particles);

    // Set up camera to see the settled box
    let eye = Vec3::new(2.0, 2.0, 5.0);
    let center = Vec3::new(3.2, 1.5, 3.2);
    let up = Vec3::Y;
    let view = create_camera_view(eye, center, up);
    let proj = create_camera_proj(
        45.0f32.to_radians(),
        TEST_WIDTH as f32 / TEST_HEIGHT as f32,
        0.1,
        100.0,
    );

    // Render
    let pixels = render_to_texture(
        &device,
        &queue,
        &renderer,
        &gpu_flip,
        particle_count,
        view,
        proj,
        eye,
        TEST_WIDTH,
        TEST_HEIGHT,
    );

    // Compare or save reference
    let ref_path = Path::new(REFERENCE_DIR).join("settled_box.png");

    if ref_path.exists() {
        let (expected, width, height) = load_png(&ref_path).unwrap();
        assert_eq!(width, TEST_WIDTH, "Reference image width mismatch");
        assert_eq!(height, TEST_HEIGHT, "Reference image height mismatch");

        let (passed, diff_pct, max_diff) = compare_images(&expected, &pixels, 1.0);

        if !passed {
            let fail_path = Path::new(REFERENCE_DIR).join("settled_box_FAILED.png");
            save_png(&fail_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
            panic!(
                "Visual regression test failed!\n\
                 Difference: {:.2}% of pixels (max channel diff: {})\n\
                 Failed output saved to: {:?}",
                diff_pct, max_diff, fail_path
            );
        }

        println!("Settled box test passed (diff: {:.4}%)", diff_pct);
    } else {
        std::fs::create_dir_all(REFERENCE_DIR).unwrap();
        save_png(&ref_path, &pixels, TEST_WIDTH, TEST_HEIGHT).unwrap();
        println!("Saved new reference image: {:?}", ref_path);
    }
}
