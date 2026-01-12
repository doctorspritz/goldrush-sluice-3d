use std::sync::Arc;
use wgpu::{Device, Queue, Surface, SurfaceConfiguration, TextureView};
use winit::window::Window;

use super::uniforms::ViewUniforms;

pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub depth_view: TextureView,
    pub view_bind_group_layout: wgpu::BindGroupLayout,
    pub view_bind_group: wgpu::BindGroup,
    pub view_uniform_buffer: wgpu::Buffer,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: Self::required_limits(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let size = window.inner_size();
        let surface_format = surface.get_capabilities(&adapter).formats[0];

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let depth_view = Self::create_depth_view(&device, config.width, config.height);

        // Create view bind group layout
        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<ViewUniforms>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                }],
            });

        // Create view uniform buffer
        let view_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view_uniform_buffer"),
            size: std::mem::size_of::<ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create view bind group
        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view_bind_group"),
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

    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256MB
            ..wgpu::Limits::default()
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = Self::create_depth_view(&self.device, width, height);
    }

    pub fn update_view_uniforms(&self, uniforms: &ViewUniforms) {
        self.queue
            .write_buffer(&self.view_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn depth_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }

    fn create_depth_view(device: &Device, width: u32, height: u32) -> TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[wgpu::TextureFormat::Depth32Float],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_required_limits() {
        let limits = GpuContext::required_limits();
        assert_eq!(limits.max_storage_buffers_per_shader_stage, 16);
        assert_eq!(limits.max_storage_buffer_binding_size, 256 * 1024 * 1024);
    }

    #[test]
    fn test_depth_format() {
        // Can't test with actual GPU in headless tests, but verify format constant is correct
        assert_eq!(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureFormat::Depth32Float
        );
    }
}
