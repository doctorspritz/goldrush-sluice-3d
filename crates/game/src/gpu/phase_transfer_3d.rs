//! Conservative local phase response on the existing quadratic APIC MAC grid.
//!
//! This operator is deliberately separate from the playable step until the
//! complete pressure/gather/position-repair integration has passed its controls.
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
    scatter: wgpu::ComputePipeline,
    normalise: wgpu::ComputePipeline,
    assemble: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
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
        let moment_bytes = faces
            .iter()
            .try_fold(0u64, |s, &n| s.checked_add(n))?
            .checked_mul(64)?
            .checked_add(8)?
            .checked_add(std::mem::size_of::<Params>() as u64)?;
        faces.into_iter().try_fold(moment_bytes, |s, n| {
            let aux = Self::aux_offset(n)
                .checked_add(n.checked_mul(4)?)?
                .checked_mul(4)?;
            s.checked_add(aux)?.checked_add((n + 1).checked_mul(4)?)
        })
    }
    pub fn aux_offset(face_count: u64) -> u64 {
        (face_count + 4).div_ceil(64) * 64
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
        if total_faces > (u32::MAX as u64 - 1) / 8 {
            return Err("phase moment index overflow".into());
        }
        let moment_size = total_faces * 64 + 8;
        // Check every binding before allocating; this is a fixed-grid budget,
        // not an assertion about peak GPU residency or whole-process RSS.
        let mut sizes = vec![moment_size, 96];
        for n in face_sizes {
            sizes.push((Self::aux_offset(n) + 4 * n) * 4);
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
        let water = face_sizes.map(|n| {
            buffer(
                "Phase water base and response",
                (Self::aux_offset(n) + 4 * n) * 4,
                storage,
            )
        });
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
                    "{}\n{}",
                    include_str!("shaders/wide_sum.wgsl"),
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
            scatter: pipeline("scatter"),
            normalise: pipeline("normalise"),
            assemble: pipeline("assemble"),
            publish: pipeline("publish"),
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
                sediment,
            }),
        );
    }
    /// Re-run after any particle position repair, using exactly the gather stencil.
    pub fn encode_transfer(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        assert!(count <= self.capacity);
        encoder.clear_buffer(&self.moments, 0, None);
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
        assert_eq!(PhaseTransfer3D::storage_bytes([1, 1, 1]), Some(1388));
        assert_eq!(PhaseTransfer3D::aux_offset(60), 64);
        assert_eq!(PhaseTransfer3D::aux_offset(61), 128);
        assert_eq!(PhaseTransfer3D::storage_bytes([0, 8, 8]), None);
        assert_eq!(PhaseTransfer3D::storage_bytes([u32::MAX; 3]), None);
    }
}
