//! GPU readback staging for async data transfer.
//!
//! This module handles asynchronous readback of particle data from GPU to CPU,
//! using staging buffers and map_async for non-blocking transfers.

use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone)]
pub(super) enum ReadbackMode {
    None,
    Sync,
    Async,
}

pub(super) struct ReadbackSlot {
    positions_staging: wgpu::Buffer,
    velocities_staging: wgpu::Buffer,
    c_col0_staging: wgpu::Buffer,
    c_col1_staging: wgpu::Buffer,
    c_col2_staging: wgpu::Buffer,
    capacity: usize,
    count: usize,
    pending: bool,
    positions_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    velocities_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col0_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col1_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col2_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl ReadbackSlot {
    pub(super) fn new(device: &wgpu::Device, max_particles: usize) -> Self {
        let buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        Self {
            positions_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Positions Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            velocities_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Velocities Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col0_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col0 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col1_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col1 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col2_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col2 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: max_particles,
            count: 0,
            pending: false,
            positions_rx: None,
            velocities_rx: None,
            c_col0_rx: None,
            c_col1_rx: None,
            c_col2_rx: None,
        }
    }

    pub(super) fn schedule(
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
        encoder.copy_buffer_to_buffer(positions, 0, &self.positions_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(velocities, 0, &self.velocities_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col0, 0, &self.c_col0_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col1, 0, &self.c_col1_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col2, 0, &self.c_col2_staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        self.count = count;
        self.pending = true;

        let (tx, rx) = mpsc::channel();
        self.positions_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.positions_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.velocities_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.velocities_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col0_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col0_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col1_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col1_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col2_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col2_rx = Some(rx);

        true
    }

    pub(super) fn try_read(
        &mut self,
        positions_out: &mut [glam::Vec3],
        velocities_out: &mut [glam::Vec3],
        c_matrices_out: &mut [glam::Mat3],
    ) -> Option<usize> {
        if !self.pending {
            return None;
        }

        let mut failed = false;
        let mut all_ready = true;
        for rx in [
            &mut self.positions_rx,
            &mut self.velocities_rx,
            &mut self.c_col0_rx,
            &mut self.c_col1_rx,
            &mut self.c_col2_rx,
        ] {
            if let Some(receiver) = rx {
                match receiver.try_recv() {
                    Ok(Ok(())) => {
                        *rx = None;
                    }
                    Ok(Err(_)) => {
                        failed = true;
                        *rx = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        all_ready = false;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        failed = true;
                        *rx = None;
                    }
                }
            }
        }

        if failed {
            self.pending = false;
            self.positions_staging.unmap();
            self.velocities_staging.unmap();
            self.c_col0_staging.unmap();
            self.c_col1_staging.unmap();
            self.c_col2_staging.unmap();
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

        {
            let data = self.positions_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                positions_out[i] = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.positions_staging.unmap();

        {
            let data = self.velocities_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                velocities_out[i] = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.velocities_staging.unmap();

        {
            let data = self.c_col0_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].x_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col0_staging.unmap();

        {
            let data = self.c_col1_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].y_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col1_staging.unmap();

        {
            let data = self.c_col2_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].z_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col2_staging.unmap();

        self.pending = false;
        self.count = 0;
        Some(count)
    }
}
