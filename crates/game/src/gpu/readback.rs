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

    fn start_map(&mut self, bytes: u64) {
        let (tx, rx) = mpsc::channel();
        self.buffer
            .slice(..bytes)
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
/// Five planar copies share one mapping and completion callback. The bound
/// is 80 bytes per particle plus a 16-byte refusal-status tail. Status is
/// inspected before exposing any position, velocity or APIC output.
pub(crate) struct ReadbackSlot {
    data: StagingBuffer,
    capacity: usize,
    count: usize,
    pending: bool,
}

impl ReadbackSlot {
    pub fn new(device: &wgpu::Device, max_particles: usize) -> Self {
        let buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        Self {
            data: StagingBuffer::new(device, "Particle readback staging", buffer_size * 5 + 16),
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
        sources: [&wgpu::Buffer; 5],
        status: Option<&wgpu::Buffer>,
        first: usize,
        count: usize,
    ) -> bool {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Copy Encoder"),
        });
        if !self.encode_copy(&mut encoder, sources, status, first, count) {
            return false;
        }
        queue.submit(Some(encoder.finish()));
        self.begin_map();
        true
    }

    /// Append copies after the producer passes, then map only after submission.
    pub fn encode_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sources: [&wgpu::Buffer; 5],
        status: Option<&wgpu::Buffer>,
        first: usize,
        count: usize,
    ) -> bool {
        if self.pending || first >= self.capacity || count == 0 {
            return false;
        }
        let count = count.min(self.capacity - first);
        let byte_size = (count * 16) as u64;
        let first_byte = (first * 16) as u64;
        for (plane, source) in sources.into_iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                source,
                first_byte,
                &self.data.buffer,
                plane as u64 * byte_size,
                byte_size,
            );
        }
        if let Some(status) = status {
            encoder.copy_buffer_to_buffer(status, 0, &self.data.buffer, byte_size * 5, 16);
        } else {
            encoder.clear_buffer(&self.data.buffer, byte_size * 5, Some(16));
        }
        self.count = count;
        self.pending = true;
        true
    }

    pub fn begin_map(&mut self) {
        self.data.start_map((self.count * 80 + 16) as u64);
    }

    /// Try to read all five staged planes from one completed mapping.
    ///
    /// Returns None if not ready or failed, Some(count) on success.
    pub fn try_read(
        &mut self,
        positions_out: &mut [glam::Vec3],
        velocities_out: &mut [glam::Vec3],
        c_matrices_out: &mut [glam::Mat3],
    ) -> Option<(usize, u32)> {
        if !self.pending {
            return None;
        }

        match self.data.check_ready() {
            ReadyState::Failed => {
                self.pending = false;
                self.data.unmap();
                return None;
            }
            ReadyState::Pending => return None,
            ReadyState::Ready => {}
        }

        let count = self
            .count
            .min(positions_out.len())
            .min(velocities_out.len())
            .min(c_matrices_out.len());

        let status;
        {
            let bytes = self
                .data
                .buffer
                .slice(..(self.count * 80 + 16) as u64)
                .get_mapped_range();
            let data_bytes = self.count * 80;
            status = *bytemuck::from_bytes::<u32>(&bytes[data_bytes..data_bytes + 4]);
            let planes: &[[f32; 4]] = bytemuck::cast_slice(&bytes[..data_bytes]);
            let vector = |plane: usize, i: usize| {
                let v = planes[plane * self.count + i];
                glam::Vec3::new(v[0], v[1], v[2])
            };
            for i in 0..if status == 0 { count } else { 0 } {
                positions_out[i] = vector(0, i);
                velocities_out[i] = vector(1, i);
                c_matrices_out[i] = glam::Mat3::from_cols(vector(2, i), vector(3, i), vector(4, i));
            }
        }
        self.data.unmap();

        self.pending = false;
        self.count = 0;
        Some((count, status))
    }

    /// Returns true if a readback is currently pending.
    pub fn is_pending(&self) -> bool {
        self.pending
    }
}
