//! Opt-in device timestamps. Stages remain in their original command buffer;
//! query results are copied once at the end, without per-stage CPU/GPU fences.
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

pub const FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;
const QUERIES: u32 = 128;

pub struct GpuStageTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
    labels: Vec<&'static str>,
    receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    period_ns: f64,
}
impl GpuStageTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        assert!(
            device.features().contains(FEATURES),
            "GPU stage timestamps require query features"
        );
        let period_ns = queue.get_timestamp_period() as f64;
        assert!(
            period_ns.is_finite() && period_ns > 0.,
            "invalid GPU timestamp period"
        );
        let bytes = QUERIES as u64 * 8;
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("FLIP stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FLIP timestamp resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FLIP timestamp readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            labels: Vec::with_capacity(QUERIES as usize / 2),
            receiver: None,
            period_ns,
        }
    }
    /// Attach this pair directly to an existing compute pass. No marker
    /// dispatch, separate command buffer or per-stage fence is introduced.
    pub fn pass(&mut self, label: &'static str) -> wgpu::ComputePassTimestampWrites<'_> {
        assert!(
            self.receiver.is_none(),
            "collect the prior GPU timestamp batch first"
        );
        assert!(
            self.labels.len() * 2 + 1 < QUERIES as usize,
            "GPU timestamp batch is full"
        );
        let index = self.labels.len() as u32 * 2;
        self.labels.push(label);
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }
    }
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        assert!(!self.labels.is_empty());
        let count = self.labels.len() as u32 * 2;
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.readback, 0, count as u64 * 8);
    }
    pub fn begin_map(&mut self) {
        assert!(self.receiver.is_none());
        let (tx, rx) = mpsc::channel();
        self.readback
            .slice(..self.labels.len() as u64 * 16)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.receiver = Some(rx);
    }
    /// Call after the required particle readback, before another profiled step.
    /// Only the diagnostic path waits for this small completed-batch mapping.
    pub fn collect(&mut self, device: &wgpu::Device) -> Result<Vec<(&'static str, f64)>, String> {
        if self.receiver.is_none() && self.labels.is_empty() {
            return Ok(Vec::new());
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            device.poll(wgpu::Maintain::Poll);
            match self
                .receiver
                .as_ref()
                .ok_or("GPU timestamps were not mapped")?
                .try_recv()
            {
                Ok(Ok(())) => break,
                Ok(Err(error)) => return Err(format!("GPU timestamp map failed: {error}")),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err("GPU timestamp callback disconnected".into())
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if Instant::now() >= deadline {
                        return Err("GPU timestamp readback timed out".into());
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
        }
        self.receiver = None;
        let mapped = self
            .readback
            .slice(..self.labels.len() as u64 * 16)
            .get_mapped_range();
        let ticks: &[[u64; 2]] = bytemuck::cast_slice(&mapped);
        let result = if ticks.iter().all(|pair| *pair == [0, 0]) {
            Err("GPU counter batch returned only zero timestamps".into())
        } else {
            self.labels
                .iter()
                .zip(ticks)
                .map(|(label, pair)| {
                    let elapsed = pair[1].checked_sub(pair[0]).ok_or_else(|| {
                        format!(
                            "GPU timestamp pass {label} ran backwards: {} -> {}",
                            pair[0], pair[1]
                        )
                    })?;
                    Ok((*label, elapsed as f64 * self.period_ns * 1e-6))
                })
                .collect::<Result<Vec<_>, String>>()
        };
        drop(mapped);
        self.readback.unmap();
        self.labels.clear();
        result
    }
}
