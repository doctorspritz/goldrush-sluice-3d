//! Real compute tests: adapter acquisition failure fails the opt-in test.
use game::gpu::heightfield::GpuHeightfield;

const N: usize = 32;
const CELLS: usize = N * N;
struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    h: GpuHeightfield,
}
impl Fixture {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("real GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Water conservation tests"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .unwrap();
        let h = GpuHeightfield::new(
            &device,
            N as u32,
            N as u32,
            0.25,
            0.,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        Self { device, queue, h }
    }
    fn write(&self, buffer: &wgpu::Buffer, values: &[f32]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(values));
    }
    fn step(&self, boundaries: u32) {
        self.h
            .update_water_params(&self.queue, 0.05, boundaries, 0.036, 4);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.h.dispatch_water_substeps(&mut encoder, 4);
        self.queue.submit(Some(encoder.finish()));
    }
    fn read(&self, buffer: &wgpu::Buffer, offset: u64, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut e = self.device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit(Some(e.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        result
    }
    fn depths(&self) -> Vec<f32> {
        self.read(&self.h.water.depth, 0, CELLS)
    }
    fn aux(&self) -> Vec<f32> {
        self.read(
            &self.h.water.suspended_sediment_next,
            (CELLS * 4) as u64,
            CELLS * 12,
        )
    }
}
fn volume(depth: &[f32]) -> f64 {
    depth.iter().map(|d| *d as f64 * 0.0625).sum()
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn shared_faces_preserve_water_and_all_four_suspended_materials() {
    let f = Fixture::new();
    let mut d = vec![0.; CELLS];
    d[16 * N + 16] = 0.2;
    f.write(&f.h.water.depth, &d);
    let mut vx = vec![0.; CELLS];
    let mut vz = vx.clone();
    vx[16 * N + 16] = 20.;
    vx[16 * N + 15] = -20.;
    vz[16 * N + 16] = 20.;
    vz[15 * N + 16] = -20.;
    f.write(&f.h.water.velocity_x, &vx);
    f.write(&f.h.water.velocity_z, &vz);
    for b in [
        &f.h.water.suspended_sediment,
        &f.h.water.suspended_overburden,
        &f.h.water.suspended_gravel,
        &f.h.water.suspended_paydirt,
    ] {
        f.write(b, &vec![0.1; CELLS]);
    }
    let initial = volume(&d);
    for _ in 0..80 {
        f.step(0);
    }
    let after = f.depths();
    assert!(
        (volume(&after) - initial).abs() < 1e-7,
        "water {} -> {}",
        initial,
        volume(&after)
    );
    for (b, deposited) in [
        (&f.h.water.suspended_sediment, &f.h.geology.sediment),
        (&f.h.water.suspended_overburden, &f.h.geology.overburden),
        (&f.h.water.suspended_gravel, &f.h.geology.gravel),
        (&f.h.water.suspended_paydirt, &f.h.geology.paydirt),
    ] {
        let c = f.read(b, 0, CELLS);
        let mass: f64 = c
            .iter()
            .zip(&after)
            .map(|(c, d)| *c as f64 * *d as f64 * 0.0625)
            .sum::<f64>()
            + volume(&f.read(deposited, 0, CELLS));
        assert!((mass - initial * 0.1).abs() < 1e-7, "material mass {mass}");
    }
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn every_open_edge_accounts_for_discharge_and_closed_edges_do_not_leak() {
    let f = Fixture::new();
    for mask in [0, 1, 2, 4, 8, 15] {
        let depth = vec![0.02; CELLS];
        f.write(&f.h.water.depth, &depth);
        // Sloped surfaces drive outward flow on all four sides, including corners.
        let ground: Vec<f32> = (0..CELLS)
            .map(|i| {
                let x = (i % N) as f32 - 15.5;
                let z = (i / N) as f32 - 15.5;
                0.3 - 0.001 * (x * x + z * z)
            })
            .collect();
        f.write(&f.h.geology.bedrock, &ground);
        f.write(&f.h.water.velocity_x, &vec![0.; CELLS]);
        f.write(&f.h.water.velocity_z, &vec![0.; CELLS]);
        let channels = [
            &f.h.water.suspended_sediment,
            &f.h.water.suspended_overburden,
            &f.h.water.suspended_gravel,
            &f.h.water.suspended_paydirt,
        ];
        for buffer in channels {
            f.write(buffer, &vec![0.001; CELLS]);
        }
        let mut exported = 0.;
        let mut solids_exported = [0.; 4];
        for _ in 0..40 {
            f.step(mask);
            for a in f.aux().chunks_exact(12) {
                exported += a[6] as f64;
                for channel in 0..4 {
                    solids_exported[channel] += a[8 + channel] as f64;
                }
            }
        }
        let depths = f.depths();
        let error = volume(&depths) + exported - volume(&depth);
        assert!(error.abs() < 2e-7, "edge mask {mask}: error {error}");
        assert_eq!(exported > 0., mask != 0);
        for (channel, buffer) in channels.into_iter().enumerate() {
            let stored: f64 = f
                .read(buffer, 0, CELLS)
                .iter()
                .zip(&depths)
                .map(|(c, d)| *c as f64 * *d as f64 * 0.0625)
                .sum();
            assert!(
                (stored + solids_exported[channel] - volume(&depth) * 0.001).abs() < 1e-8,
                "material {channel} at edge mask {mask}"
            );
        }
    }
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn submillimetre_runoff_moves_and_lake_at_rest_remains_still() {
    let f = Fixture::new();
    let bed: Vec<f32> = (0..CELLS).map(|i| (i / N) as f32 * 0.001).collect();
    f.write(&f.h.geology.bedrock, &bed);
    let mut d = vec![0.; CELLS];
    for z in 10..22 {
        for x in 10..22 {
            d[z * N + x] = 0.0005;
        }
    }
    f.write(&f.h.water.depth, &d);
    let initial = volume(&d);
    for _ in 0..80 {
        f.step(0);
    }
    let after = f.depths();
    let moment = |a: &[f32]| {
        a.iter()
            .enumerate()
            .map(|(i, d)| (i / N) as f64 * *d as f64)
            .sum::<f64>()
    };
    assert!(
        moment(&after) < moment(&d) - 0.0001,
        "thin water must move downhill"
    );
    assert!((volume(&after) - initial).abs() < 1e-8);
    let lake: Vec<f32> = bed.iter().map(|b| (0.02 - b).max(0.)).collect();
    f.write(&f.h.water.depth, &lake);
    f.write(&f.h.water.velocity_x, &vec![0.; CELLS]);
    f.write(&f.h.water.velocity_z, &vec![0.; CELLS]);
    for _ in 0..20 {
        f.step(0);
    }
    assert!(f
        .depths()
        .iter()
        .zip(&lake)
        .all(|(a, b)| (a - b).abs() < 1e-7));
}
