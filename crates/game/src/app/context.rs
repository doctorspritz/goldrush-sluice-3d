//! GPU context management for the unified app framework.
//!
//! Provides `GpuContext` which handles device creation, surface management,
//! and view uniform buffer setup.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use super::ViewUniforms;

/// GPU context containing device, queue, surface, and view-related resources.
///
/// This struct manages the core wgpu infrastructure needed for rendering,
/// including the view uniform buffer and bind group for camera/projection data.
pub struct GpuContext {
    /// The wgpu device for GPU operations
    pub device: Arc<wgpu::Device>,
    /// The wgpu queue for submitting commands
    pub queue: Arc<wgpu::Queue>,
    /// The rendering surface
    pub surface: wgpu::Surface<'static>,
    /// Surface configuration
    pub config: wgpu::SurfaceConfiguration,
    /// Depth buffer texture view
    pub depth_view: wgpu::TextureView,
    /// Bind group layout for view uniforms (group 0)
    pub view_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group containing the view uniform buffer
    pub view_bind_group: wgpu::BindGroup,
    /// Buffer containing view/projection uniforms
    pub view_uniform_buffer: wgpu::Buffer,
}

impl GpuContext {
    /// Returns the required GPU limits for this application.
    ///
    /// Requires:
    /// - At least 16 storage buffers per shader stage
    /// - At least 256MB max storage buffer binding size
    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256 MB
            ..wgpu::Limits::default()
        }
    }

    /// Creates a new GPU context with the given window.
    ///
    /// This is an async function that:
    /// 1. Creates a wgpu instance and surface
    /// 2. Requests an adapter and device with required limits
    /// 3. Configures the surface
    /// 4. Creates depth buffer
    /// 5. Creates view uniform buffer and bind group
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: Self::required_limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create depth texture
        let depth_view = Self::create_depth_texture(&device, width, height);

        // Create view uniform buffer
        let view_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Uniform Buffer"),
            contents: bytemuck::bytes_of(&ViewUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for view uniforms (group 0, binding 0)
        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("View Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create bind group
        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Bind Group"),
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            device,
            queue,
            surface,
            config,
            depth_view,
            view_bind_group_layout,
            view_bind_group,
            view_uniform_buffer,
        }
    }

    /// Resizes the surface and recreates the depth texture.
    ///
    /// Should be called when the window is resized.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        // Recreate depth texture
        self.depth_view = Self::create_depth_texture(&self.device, width, height);
    }

    /// Updates the view uniform buffer with new data.
    pub fn update_view_uniforms(&self, uniforms: &ViewUniforms) {
        self.queue
            .write_buffer(&self.view_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Returns the surface texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns the depth texture format (always Depth32Float).
    pub fn depth_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }

    /// Creates a depth texture view for the given dimensions.
    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_limits() {
        let limits = GpuContext::required_limits();
        assert!(limits.max_storage_buffers_per_shader_stage >= 16);
        assert!(limits.max_storage_buffer_binding_size >= 256 * 1024 * 1024);
    }
}
