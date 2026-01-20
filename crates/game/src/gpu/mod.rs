// Disabled: bed_3d was the Drucker-Prager + heightfield system that had threshold tuning issues
// pub mod bed_3d;
pub mod bridge_3d;
pub mod dem_3d;
pub mod dem_render;
pub mod flip_3d;
pub mod fluid_renderer;
pub mod g2p_3d;
pub mod heightfield;
pub mod mgpcg;
pub mod p2g_3d;
pub mod p2g_cell_centric_3d;
pub mod params;
pub mod particle_sort;
pub mod pressure_3d;
pub mod readback;
pub mod sph_3d;
pub mod sph_dfsph;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::SurfaceTarget;
use winit::window::Window;

/// Global flag indicating GPU device was lost
static GPU_DEVICE_LOST: AtomicBool = AtomicBool::new(false);

/// Check if the GPU device has been lost
pub fn is_device_lost() -> bool {
    GPU_DEVICE_LOST.load(Ordering::SeqCst)
}

/// Reset the device lost flag (call after recreating device)
pub fn reset_device_lost() {
    GPU_DEVICE_LOST.store(false, Ordering::SeqCst);
}

/// GPU error type for buffer operations
#[derive(Debug)]
pub enum GpuError {
    DeviceLost,
    BufferMapFailed(wgpu::BufferAsyncError),
    ChannelDisconnected,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceLost => write!(f, "GPU device lost"),
            GpuError::BufferMapFailed(e) => write!(f, "Buffer map failed: {:?}", e),
            GpuError::ChannelDisconnected => write!(f, "Channel disconnected"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Wait for a buffer map operation to complete, returning Result instead of panicking.
/// Use this instead of `rx.recv().unwrap().unwrap()`.
pub fn await_buffer_map(
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
) -> Result<(), GpuError> {
    if is_device_lost() {
        return Err(GpuError::DeviceLost);
    }
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            log::error!("Buffer map failed: {:?}", e);
            Err(GpuError::BufferMapFailed(e))
        }
        Err(_) => {
            log::error!("Buffer map channel disconnected - possible device lost");
            GPU_DEVICE_LOST.store(true, Ordering::SeqCst);
            Err(GpuError::ChannelDisconnected)
        }
    }
}

/// Central GPU context holding device, queue, and surface
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // Create surface from window
        let surface = instance
            .create_surface(SurfaceTarget::from(window.clone()))
            .expect("Failed to create surface");

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        log::info!("Using GPU: {:?}", adapter.get_info());

        // Request device with compute shader support
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..wgpu::Limits::default()
                    }
                    .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Set up device lost callback
        device.on_uncaptured_error(Box::new(|error| {
            log::error!("GPU uncaptured error: {:?}", error);
            if matches!(error, wgpu::Error::OutOfMemory { .. }) {
                GPU_DEVICE_LOST.store(true, Ordering::SeqCst);
            }
        }));

        // Reset device lost flag for fresh device
        reset_device_lost();

        // Configure surface
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

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            surface,
            config,
            size: (width, height),
        }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.size = (new_width, new_height);
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }
}
