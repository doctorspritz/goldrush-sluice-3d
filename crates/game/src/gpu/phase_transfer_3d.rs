//! Conservative local phase response on the existing quadratic APIC MAC grid.
//!
//! Integrated with the playable step in explicit conservative validation mode.
//! Gravity boundaries, packed beds and operating calibration remain open.
//! It never applies a global momentum correction or reads back between passes.
use super::g2p_3d::SedimentParams3D;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    shape: [u32; 3],
    count: u32,
    cell_size: f32,
    dt: f32,
    gravity: f32,
    reference_volume: f32,
    capacity: u32,
    face_counts: [u32; 3],
    sediment: SedimentParams3D,
}

pub struct BoundaryVolume3D {
    pipeline: wgpu::ComputePipeline,
    geometry: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

pub struct PhaseTransfer3D {
    shape: [u32; 3],
    face_counts: [u32; 3],
    capacity: u32,
    params: wgpu::Buffer,
    moments: wgpu::Buffer,
    /// First plane is the unforced pure-water velocity; an aligned auxiliary
    /// plane contains [water predictor, kw/k, mixture predictor, K/k] per face.
    pub water: [wgpu::Buffer; 3],
    /// Ready header followed by per-face pressure mobility. Shared with pressure.
    pub mobility: [wgpu::Buffer; 3],
    group: wgpu::BindGroup,
    count_cells: wgpu::ComputePipeline,
    decode_cells: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    normalise: wgpu::ComputePipeline,
    assemble: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    validate_particles: wgpu::ComputePipeline,
}

impl PhaseTransfer3D {
    /// Storage fixed by the grid, excluding the shared particle metadata.
    pub fn storage_bytes(shape: [u32; 3]) -> Option<u64> {
        let [nx, ny, nz] = shape.map(u64::from);
        if shape.contains(&0) {
            return None;
        }
        let faces = [
            (nx + 1).checked_mul(ny)?.checked_mul(nz)?,
            nx.checked_mul(ny + 1)?.checked_mul(nz)?,
            nx.checked_mul(ny)?.checked_mul(nz + 1)?,
        ];
        let cells = nx.checked_mul(ny)?.checked_mul(nz)?;
        let moment_bytes = faces
            .iter()
            .try_fold(0u64, |s, &n| s.checked_add(n))?
            .checked_mul(64)?
            .checked_add(cells.checked_mul(16)?)?
            .checked_add(8)?
            .checked_add(std::mem::size_of::<Params>() as u64)?;
        // Once checked above, these dimensions fit all intermediate layouts.
        let water = Self::water_bytes(shape, faces);
        faces
            .into_iter()
            .zip(water)
            .try_fold(moment_bytes, |s, (n, aux)| {
                s.checked_add(aux)?.checked_add((n + 1).checked_mul(4)?)
            })
    }
    pub fn aux_offset(face_count: u64) -> u64 {
        (face_count + 4).div_ceil(64) * 64
    }

    /// Raw water-home counts in the low 31 bits; runtime publishes immutable
    /// solid geometry in the high bit. Standalone transfers retain raw counts.
    pub fn cell_count_offset(shape: [u32; 3]) -> u64 {
        let n = (u64::from(shape[0]) + 1) * u64::from(shape[1]) * u64::from(shape[2]);
        (Self::aux_offset(n) + 4 * n).div_ceil(64) * 64
    }
    pub fn cell_volume_offset(shape: [u32; 3]) -> u64 {
        let cells = shape.into_iter().map(u64::from).product::<u64>();
        (Self::cell_count_offset(shape) + cells).div_ceil(64) * 64
    }
    fn water_bytes(shape: [u32; 3], face_counts: [u64; 3]) -> [u64; 3] {
        let mut bytes = face_counts.map(|n| (Self::aux_offset(n) + 4 * n) * 4);
        bytes[0] = (Self::cell_volume_offset(shape)
            + shape.into_iter().map(u64::from).product::<u64>())
            * 4;
        bytes
    }

    /// Pure preflight for runtime allocation, before any particle/grid buffers.
    pub fn preflight(
        limits: &wgpu::Limits,
        shape: [u32; 3],
        capacity: usize,
    ) -> Result<(), String> {
        let _ = Self::storage_bytes(shape).ok_or("Phase grid size overflow")?;
        if capacity == 0
            || capacity > u32::MAX as usize
            || limits.max_storage_buffers_per_shader_stage < 16
        {
            return Err("Phase particle count or storage binding limit".into());
        }
        let [nx, ny, nz] = shape.map(u64::from);
        let cells = nx * ny * nz;
        let faces = [(nx + 1) * ny * nz, nx * (ny + 1) * nz, nx * ny * (nz + 1)];
        let total = faces.into_iter().sum::<u64>();
        if total * 8 + cells * 2 > u32::MAX as u64 - 1 {
            return Err("Phase moment index overflow".into());
        }
        let mut sizes = vec![total * 64 + cells * 16 + 8, capacity as u64 * 16];
        sizes.extend(Self::water_bytes(shape, faces));
        sizes.extend(faces.map(|n| (n + 1) * 4));
        if sizes.into_iter().any(|n| {
            n > limits.max_buffer_size || n > u64::from(limits.max_storage_buffer_binding_size)
        }) {
            return Err("Phase storage exceeds adapter binding/buffer limit".into());
        }
        Ok(())
    }

    /// Particle inputs are positions (xyz, represented volume), velocity, and
    /// three affine rows. Metadata is 4*capacity floats: density, diameter,
    /// backward-Euler b and c. The latter two planes are GPU outputs.
    pub fn new(
        device: &wgpu::Device,
        shape: [u32; 3],
        capacity: u32,
        particles: [&wgpu::Buffer; 5],
        metadata: &wgpu::Buffer,
        grids: [&wgpu::Buffer; 3],
    ) -> Result<Self, String> {
        Self::preflight(&device.limits(), shape, capacity as usize)?;
        let required = Self::storage_bytes(shape).ok_or("phase grid size overflow")?;
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 16 {
            return Err("phase response needs 16 storage bindings".into());
        }
        if capacity == 0 || metadata.size() < u64::from(capacity) * 16 {
            return Err("phase metadata is smaller than four capacity planes".into());
        }
        let [nx, ny, nz] = shape.map(u64::from);
        let face_sizes = [(nx + 1) * ny * nz, nx * (ny + 1) * nz, nx * ny * (nz + 1)];
        if face_sizes.iter().any(|&n| n > u32::MAX as u64) {
            return Err("phase face index overflow".into());
        }
        let total_faces: u64 = face_sizes.iter().sum();
        let cell_count = nx * ny * nz;
        if total_faces * 8 + cell_count * 2 > u32::MAX as u64 - 1 {
            return Err("phase moment index overflow".into());
        }
        let moment_size = total_faces * 64 + cell_count * 16 + 8;
        let water_sizes = Self::water_bytes(shape, face_sizes);
        // Check every binding before allocating; this is a fixed-grid budget,
        // not an assertion about peak GPU residency or whole-process RSS.
        let mut sizes = vec![moment_size, 96];
        for (n, bytes) in face_sizes.into_iter().zip(water_sizes) {
            sizes.push(bytes);
            sizes.push((n + 1) * 4);
        }
        if sizes.iter().any(|&n| {
            n > limits.max_buffer_size || n > u64::from(limits.max_storage_buffer_binding_size)
        }) {
            return Err(format!(
                "phase grid exceeds device binding limit ({required} total bytes)"
            ));
        }
        for (i, n) in face_sizes.into_iter().enumerate() {
            if grids[i].size() < n * 4 {
                return Err("phase MAC grid buffer is too small".into());
            }
        }
        if particles
            .iter()
            .any(|b| b.size() < u64::from(capacity) * 16)
        {
            return Err("phase particle input is smaller than capacity".into());
        }
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let moments = buffer("Conservative face sums", moment_size, storage);
        let water =
            water_sizes.map(|bytes| buffer("Phase water base and response", bytes, storage));
        let mobility = face_sizes.map(|n| buffer("Phase pressure mobility", (n + 1) * 4, storage));
        let params = buffer(
            "Phase transfer parameters",
            std::mem::size_of::<Params>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Conservative phase transfer"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}",
                    include_str!("shaders/wide_sum.wgsl"),
                    include_str!("shaders/mac_wall_3d.wgsl"),
                    include_str!("shaders/sample_sdf_3d.wgsl"),
                    include_str!("shaders/phase_transfer_3d.wgsl")
                )
                .into(),
            ),
        });
        let entries: Vec<_> = (0..17)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding <= 5,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phase transfer layout"),
            entries: &entries,
        });
        let buffers = [
            &params,
            particles[0],
            particles[1],
            particles[2],
            particles[3],
            particles[4],
            metadata,
            &moments,
            grids[0],
            grids[1],
            grids[2],
            &water[0],
            &water[1],
            &water[2],
            &mobility[0],
            &mobility[1],
            &mobility[2],
        ];
        let bindings: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Phase transfer bindings"),
            layout: &layout,
            entries: &bindings,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            shape,
            face_counts: face_sizes.map(|n| n as u32),
            capacity,
            params,
            moments,
            water,
            mobility,
            group,
            count_cells: pipeline("count_cells"),
            decode_cells: pipeline("decode_cells"),
            scatter: pipeline("scatter"),
            normalise: pipeline("normalise"),
            assemble: pipeline("assemble"),
            publish: pipeline("publish"),
            validate_particles: pipeline("validate_particles"),
        })
    }
    /// All force predictors must be formed before `encode_response`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        count: u32,
        cell_size: f32,
        dt: f32,
        gravity: f32,
        reference_volume: f32,
        sediment: SedimentParams3D,
    ) {
        assert!(
            count <= self.capacity
                && cell_size.is_finite()
                && cell_size > 0.
                && dt.is_finite()
                && dt > 0.
                && gravity.is_finite()
                && reference_volume.is_finite()
                && reference_volume > 0.
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                shape: self.shape,
                count,
                cell_size,
                dt,
                gravity,
                reference_volume,
                capacity: self.capacity,
                face_counts: self.face_counts,
                // Standalone phase operators and gather default to open faces.
                // The runtime writes its actual mask after prepare.
                sediment: SedimentParams3D {
                    _pad: [63.],
                    ..sediment
                },
            }),
        );
    }
    /// Re-run after any particle position repair, using exactly the gather stencil.
    pub fn encode_transfer(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        assert!(count <= self.capacity);
        encoder.clear_buffer(
            &self.moments,
            8,
            Some(self.face_counts.into_iter().map(u64::from).sum::<u64>() * 64),
        );
        for plane in &self.mobility {
            encoder.clear_buffer(plane, 0, Some(4));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Conservative phase moments"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.scatter);
        if count > 0 {
            pass.dispatch_workgroups(count.div_ceil(128), 1, 1);
        }
        pass.set_pipeline(&self.normalise);
        pass.dispatch_workgroups(
            self.face_counts.into_iter().sum::<u32>().div_ceil(128),
            1,
            1,
        );
    }
    /// Begin a step without clearing refusal. The status stays sticky until
    /// authoritative recovery constructs a fresh solver, including substeps
    /// that intentionally have no particle readback.
    pub fn encode_begin_step(&self, encoder: &mut wgpu::CommandEncoder) {
        for p in &self.mobility {
            encoder.clear_buffer(p, 0, Some(4));
        }
    }
    /// Represented total volume and raw water markers use independent planes.
    /// Re-run after any position correction before the phase transfer.
    pub fn encode_cell_volume(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        self.encode_volume(encoder, count, None);
    }
    pub fn encode_cell_volume_with_boundaries(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        boundary: &BoundaryVolume3D,
    ) {
        self.encode_volume(encoder, count, Some(boundary));
    }
    fn encode_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        boundary: Option<&BoundaryVolume3D>,
    ) {
        assert!(count <= self.capacity);
        let first = 8 + self.face_counts.into_iter().map(u64::from).sum::<u64>() * 64;
        encoder.clear_buffer(&self.moments, first, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Phase represented cell volume"),
            timestamp_writes: None,
        });
        if let Some(boundary) = boundary {
            pass.set_bind_group(0, &boundary.group, &[]);
            pass.set_pipeline(&boundary.pipeline);
        } else {
            pass.set_bind_group(0, &self.group, &[]);
            pass.set_pipeline(&self.count_cells);
        }
        if count > 0 {
            pass.dispatch_workgroups(count.div_ceil(128), 1, 1);
        }
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.decode_cells);
        pass.dispatch_workgroups(self.shape.into_iter().product::<u32>().div_ceil(128), 1, 1);
        if let Some(boundary) = boundary {
            pass.set_bind_group(0, &boundary.group, &[]);
            pass.set_pipeline(&boundary.geometry);
            pass.dispatch_workgroups(self.shape.into_iter().product::<u32>().div_ceil(128), 1, 1);
        }
    }

    /// Share the boundary mask between cell support and paired affine transfers.
    /// Call after prepare; physical drag never reads this sediment padding.
    pub fn set_volume_boundaries(&self, queue: &wgpu::Queue, open: u32) {
        assert!(open <= 63);
        queue.write_buffer(
            &self.params,
            (std::mem::size_of::<Params>() - 4) as u64,
            bytemuck::bytes_of(&(open as f32)),
        );
    }
    /// Classify pressure cells from actual carrier support, retaining solids.
    pub fn pressure_support(
        &self,
        device: &wgpu::Device,
        parameters: &wgpu::Buffer,
        cell_types: &wgpu::Buffer,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Carried water pressure support"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("shaders/wide_sum.wgsl"),
                    include_str!("shaders/phase_pressure_cells_3d.wgsl")
                )
                .into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Carried water pressure support"),
            entries: &(0..3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Carried water pressure support"),
            layout: &layout,
            entries: &[parameters, &self.moments, cell_types]
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Carried water pressure support"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Carried water pressure support"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("classify"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, group)
    }

    pub fn boundary_volume(
        &self,
        device: &wgpu::Device,
        positions: &wgpu::Buffer,
        metadata: &wgpu::Buffer,
        cell_types: &wgpu::Buffer,
        sdf: &wgpu::Buffer,
    ) -> BoundaryVolume3D {
        assert!(
            positions.size() >= u64::from(self.capacity) * 16
                && metadata.size() >= u64::from(self.capacity) * 16
        );
        assert!(cell_types.size() >= self.shape.into_iter().map(u64::from).product::<u64>() * 4);
        assert!(sdf.size() >= self.shape.into_iter().map(u64::from).product::<u64>() * 4);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Boundary volume layout"),
            entries: &[0, 1, 6, 7, 11, 17, 18].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 1 || binding == 17 || binding == 18,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boundary volume"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: metadata.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.moments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.water[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: sdf.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Boundary volume shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}",
                    include_str!("shaders/wide_sum.wgsl"),
                    include_str!("shaders/mac_wall_3d.wgsl"),
                    include_str!("shaders/sample_sdf_3d.wgsl"),
                    include_str!("shaders/phase_transfer_3d.wgsl")
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Boundary volume pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_cells_reflected"),
            compilation_options: Default::default(),
            cache: None,
        });
        let geometry = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Publish immutable voxel geometry"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("publish_voxel_geometry"),
            compilation_options: Default::default(),
            cache: None,
        });
        BoundaryVolume3D {
            pipeline,
            geometry,
            group,
        }
    }

    pub fn encode_response(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Implicit phase response"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.assemble);
        pass.dispatch_workgroups(
            self.face_counts.into_iter().sum::<u32>().div_ceil(128),
            1,
            1,
        );
        pass.set_pipeline(&self.publish);
        pass.dispatch_workgroups(1, 1, 1);
    }
    /// Validate transient G2P output before any advection or CPU exposure.
    pub fn encode_validate_particles(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Phase finite particle output"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.validate_particles);
        if count > 0 {
            pass.dispatch_workgroups(count.div_ceil(128), 1, 1);
        }
    }
    /// First u32 is an error bitfield. Join it to the ordinary particle readback
    /// before activating this operator in a worker; never silently ignore it.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.moments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_grid_budget_includes_headers_alignment_and_uniforms() {
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(PhaseTransfer3D::storage_bytes([1, 1, 1]), Some(1888));
        assert_eq!(PhaseTransfer3D::aux_offset(60), 64);
        assert_eq!(PhaseTransfer3D::aux_offset(61), 128);
        assert_eq!(PhaseTransfer3D::storage_bytes([0, 8, 8]), None);
        assert_eq!(PhaseTransfer3D::storage_bytes([u32::MAX; 3]), None);
    }
}
