//! Async GPU buffer readback for particle data.
//!
//! Provides double-buffered async readback to avoid GPU stalls when reading
//! particle positions, velocities, and APIC matrices back to CPU.

use std::sync::mpsc;

/// Readback scheduling mode.
#[derive(Copy, Clone)]
pub(crate) enum ReadbackMode {
    None,
    Sync,
    Async,
}

/// A single staging buffer with async map tracking.
struct StagingBuffer {
    buffer: wgpu::Buffer,
    rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl StagingBuffer {
    fn new(device: &wgpu::Device, label: &str, size: u64) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            rx: None,
        }
    }

    fn copy_from(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, byte_size: u64) {
        encoder.copy_buffer_to_buffer(src, 0, &self.buffer, 0, byte_size);
    }

    fn start_map(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.rx = Some(rx);
    }

    fn check_ready(&mut self) -> ReadyState {
        if let Some(receiver) = &self.rx {
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    self.rx = None;
                    ReadyState::Ready
                }
                Ok(Err(_)) => {
                    self.rx = None;
                    ReadyState::Failed
                }
                Err(mpsc::TryRecvError::Empty) => ReadyState::Pending,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx = None;
                    ReadyState::Failed
                }
            }
        } else {
            ReadyState::Ready
        }
    }

    fn read_vec3(&self, out: &mut [glam::Vec3], count: usize) {
        let data = self.buffer.slice(..).get_mapped_range();
        let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
        for i in 0..count {
            out[i] = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
        }
    }

    fn read_vec3_to_mat3_axis(
        &self,
        out: &mut [glam::Mat3],
        count: usize,
        axis: fn(&mut glam::Mat3) -> &mut glam::Vec3,
    ) {
        let data = self.buffer.slice(..).get_mapped_range();
        let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
        for i in 0..count {
            *axis(&mut out[i]) = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
        }
    }

    fn unmap(&self) {
        self.buffer.unmap();
    }
}

#[derive(PartialEq)]
enum ReadyState {
    Ready,
    Pending,
    Failed,
}

/// Double-buffered async readback slot for particle data.
///
/// Handles positions (Vec3), velocities (Vec3), and APIC C matrix columns.
pub(crate) struct ReadbackSlot {
    positions: StagingBuffer,
    velocities: StagingBuffer,
    c_col0: StagingBuffer,
    c_col1: StagingBuffer,
    c_col2: StagingBuffer,
    capacity: usize,
    count: usize,
    pending: bool,
}

impl ReadbackSlot {
    pub fn new(device: &wgpu::Device, max_particles: usize) -> Self {
        let buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        Self {
            positions: StagingBuffer::new(device, "Readback Positions Staging", buffer_size),
            velocities: StagingBuffer::new(device, "Readback Velocities Staging", buffer_size),
            c_col0: StagingBuffer::new(device, "Readback C Col0 Staging", buffer_size),
            c_col1: StagingBuffer::new(device, "Readback C Col1 Staging", buffer_size),
            c_col2: StagingBuffer::new(device, "Readback C Col2 Staging", buffer_size),
            capacity: max_particles,
            count: 0,
            pending: false,
        }
    }

    /// Schedule an async copy from GPU buffers to staging buffers.
    ///
    /// Returns false if a readback is already pending or count is 0.
    pub fn schedule(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        c_col0: &wgpu::Buffer,
        c_col1: &wgpu::Buffer,
        c_col2: &wgpu::Buffer,
        count: usize,
    ) -> bool {
        if self.pending {
            return false;
        }

        let count = count.min(self.capacity);
        if count == 0 {
            return false;
        }

        let byte_size = (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Copy Encoder"),
        });

        self.positions.copy_from(&mut encoder, positions, byte_size);
        self.velocities.copy_from(&mut encoder, velocities, byte_size);
        self.c_col0.copy_from(&mut encoder, c_col0, byte_size);
        self.c_col1.copy_from(&mut encoder, c_col1, byte_size);
        self.c_col2.copy_from(&mut encoder, c_col2, byte_size);

        queue.submit(std::iter::once(encoder.finish()));

        self.count = count;
        self.pending = true;

        self.positions.start_map();
        self.velocities.start_map();
        self.c_col0.start_map();
        self.c_col1.start_map();
        self.c_col2.start_map();

        true
    }

    /// Try to read the staged data if all buffers are mapped.
    ///
    /// Returns None if not ready or failed, Some(count) on success.
    pub fn try_read(
        &mut self,
        positions_out: &mut [glam::Vec3],
        velocities_out: &mut [glam::Vec3],
        c_matrices_out: &mut [glam::Mat3],
    ) -> Option<usize> {
        if !self.pending {
            return None;
        }

        let states = [
            self.positions.check_ready(),
            self.velocities.check_ready(),
            self.c_col0.check_ready(),
            self.c_col1.check_ready(),
            self.c_col2.check_ready(),
        ];

        let failed = states.iter().any(|s| *s == ReadyState::Failed);
        let all_ready = states.iter().all(|s| *s == ReadyState::Ready);

        if failed {
            self.pending = false;
            self.positions.unmap();
            self.velocities.unmap();
            self.c_col0.unmap();
            self.c_col1.unmap();
            self.c_col2.unmap();
            return None;
        }

        if !all_ready {
            return None;
        }

        let count = self
            .count
            .min(positions_out.len())
            .min(velocities_out.len())
            .min(c_matrices_out.len());

        self.positions.read_vec3(positions_out, count);
        self.positions.unmap();

        self.velocities.read_vec3(velocities_out, count);
        self.velocities.unmap();

        self.c_col0
            .read_vec3_to_mat3_axis(c_matrices_out, count, |m| &mut m.x_axis);
        self.c_col0.unmap();

        self.c_col1
            .read_vec3_to_mat3_axis(c_matrices_out, count, |m| &mut m.y_axis);
        self.c_col1.unmap();

        self.c_col2
            .read_vec3_to_mat3_axis(c_matrices_out, count, |m| &mut m.z_axis);
        self.c_col2.unmap();

        self.pending = false;
        self.count = 0;
        Some(count)
    }

    /// Returns true if a readback is currently pending.
    pub fn _is_pending(&self) -> bool {
        self.pending
    }
}
