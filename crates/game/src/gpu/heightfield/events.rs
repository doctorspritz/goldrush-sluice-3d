//! Read-only event capture for a CPU f64 provenance tracer. These passes never
//! write physical water, concentrations or geology and add no GPU/CPU waits.
use super::{GeologyBuffers, WaterBuffers};
use wgpu::util::DeviceExt;

pub struct TerrainEventRecorder {
    pipeline: wgpu::ComputePipeline,
    inputs: wgpu::BindGroup,
    output_layout: wgpu::BindGroupLayout,
}
pub struct TerrainEventCapture {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    output: wgpu::BindGroup,
    stride: u32,
    cells: u32,
}
impl TerrainEventRecorder {
    pub fn new(device: &wgpu::Device, water: &WaterBuffers, geology: &GeologyBuffers) -> Self {
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Provenance Inputs"),
            entries: &(0..7)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Provenance Inputs"),
            layout: &input_layout,
            entries: &[
                &water.depth,
                &water.flux_x,
                &water.flux_z,
                &water.suspended_sediment_next,
                &water.suspended_paydirt,
                &geology.paydirt,
                &geology.bed_remainder,
            ]
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Provenance Output"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Provenance Capture"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/heightfield_events.wgsl").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Provenance Layout"),
            bind_group_layouts: &[&input_layout, &output_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Provenance Capture"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("record_event"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            inputs,
            output_layout,
        }
    }
    pub fn capture(&self, device: &wgpu::Device, cells: u32, substeps: u32) -> TerrainEventCapture {
        let count = substeps.max(1) + 2;
        let size = cells as u64 * count as u64 * 16;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Provenance Events"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut params = vec![0u32; (stride * count / 4) as usize];
        for event in 0..count {
            let at = (event * stride / 4) as usize;
            params[at..at + 4].copy_from_slice(&[
                cells,
                event * cells,
                u32::from(event >= substeps.max(1)),
                0,
            ]);
        }
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Provenance Event Offsets"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Provenance Event Targets"),
            layout: &self.output_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        });
        TerrainEventCapture {
            buffer,
            size,
            output,
            stride,
            cells,
        }
    }
    pub(super) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        capture: &TerrainEventCapture,
        event: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Record Actual Terrain Event"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.inputs, &[]);
        pass.set_bind_group(1, &capture.output, &[event * capture.stride]);
        pass.dispatch_workgroups(capture.cells.div_ceil(64), 1, 1);
    }
}
