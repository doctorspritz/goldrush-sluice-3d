//! Isolated production bed stages: no water advection/collapse can hide exchange
//! error or a slope read from a neighbour that already changed this dispatch.
use game::gpu::heightfield::{BedExchangeStage, GpuHeightfield};

const N: usize = 32;
const CELLS: usize = N * N;
const AREA: f64 = 0.25 * 0.25;
struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    h: GpuHeightfield,
}
impl Fixture {
    fn new() -> Self {
        Self::with_cell(0.25)
    }
    fn with_cell(cell: f32) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("real GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Bed exchange regressions"),
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
            cell,
            0.,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        Self { device, queue, h }
    }
    fn layers(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.h.geology.sediment,
            &self.h.geology.overburden,
            &self.h.geology.gravel,
            &self.h.geology.paydirt,
        ]
    }
    fn concentrations(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.h.water.suspended_sediment,
            &self.h.water.suspended_overburden,
            &self.h.water.suspended_gravel,
            &self.h.water.suspended_paydirt,
        ]
    }
    fn targets(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.h.water.suspended_sediment_next,
            &self.h.water.suspended_overburden_next,
            &self.h.water.suspended_gravel_next,
            &self.h.water.suspended_paydirt_next,
        ]
    }
    fn surface(&self, material: u32) {
        self.queue.write_buffer(
            &self.h.geology.surface_material,
            0,
            bytemuck::cast_slice(&vec![material; CELLS]),
        );
    }
    fn write(&self, buffer: &wgpu::Buffer, values: &[f32]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(values));
    }
    fn fill(&self, buffer: &wgpu::Buffer, value: f32) {
        self.write(buffer, &vec![value; CELLS]);
    }
    fn clear_exchange_error(&self) {
        self.queue.write_buffer(
            &self.h.geology.bed_remainder,
            0,
            bytemuck::cast_slice(&vec![[0f32; 4]; CELLS]),
        );
        self.queue.write_buffer(
            &self.h.geology.bed_exchange_error,
            0,
            bytemuck::cast_slice(&vec![[0f32; 4]; CELLS]),
        );
    }
    fn stage(&self, settling: bool, tile: [u32; 4], dt: f32) {
        self.h
            .update_params_tile(&self.queue, dt, tile[0], tile[1], tile[2], tile[3]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.h.dispatch_bed_exchange_tile(
            &mut encoder,
            tile[2],
            tile[3],
            if settling {
                BedExchangeStage::Settling
            } else {
                BedExchangeStage::Erosion
            },
        );
        self.queue.submit(Some(encoder.finish()));
    }
    fn read(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        self.read_range(buffer, 0, CELLS)
    }
    fn read_range(&self, buffer: &wgpu::Buffer, offset: u64, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed stage readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut e = self.device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(buffer, offset, &read, 0, size);
        self.queue.submit(Some(e.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec();
        read.unmap();
        out
    }
    fn mass(&self, channel: usize) -> f64 {
        let bed = self.read(self.layers()[channel]);
        let c = self.read(self.concentrations()[channel]);
        let h = self.read(&self.h.water.depth);
        let remainder = self.read_range(&self.h.geology.bed_remainder, 0, CELLS * 4);
        bed.into_iter()
            .zip(c)
            .zip(h)
            .enumerate()
            .map(|(i, ((b, c), h))| {
                (b as f64 + remainder[i * 4 + channel] as f64 + c as f64 * h as f64) * AREA
            })
            .sum()
    }
}

// Independent f64 evaluation of the retained size/density terminal velocity.
// Deposition references below integrate the well-mixed column balance in f64.
fn settling_speed(d: f64) -> f64 {
    let stokes = 9.81 * (2650. - 1000.) * d * d / (18. * 0.001);
    let turbulent = (4. * 9.81 * d * (2650. - 1000.) / (3. * 1000. * 0.44)).sqrt();
    if d < 0.0001 {
        stokes
    } else if d > 0.001 {
        turbulent
    } else {
        let t = (d - 0.0001) / (0.001 - 0.0001);
        stokes * (1. - t) + turbulent * t
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn settling_matches_the_mixed_column_reference_across_workgroups() {
    let f = Fixture::new();
    let cs = [0.01_f32, 0.1, 0.1, 0.1];
    for b in f.layers() {
        f.fill(b, 0.0625);
    }
    for (buffer, c) in f.concentrations().into_iter().zip(cs) {
        f.fill(buffer, c);
    }
    f.fill(&f.h.water.depth, 0.5);
    f.fill(&f.h.water.velocity_x, 0.25);
    f.stage(true, [0, 0, N as u32, N as u32], 0.05);
    let mut maximum = 0.0_f64;
    for (ch, d) in [0.0001_f64, 0.001, 0.01, 0.002].into_iter().enumerate() {
        let deposit = cs[ch] as f64 * 0.5 * (1. - (-settling_speed(d) * 0.05 / 0.5).exp());
        let expected = 0.0625 + deposit;
        for actual in f.read(f.layers()[ch]) {
            maximum = maximum.max((actual as f64 - expected).abs());
        }
    }
    println!("flat-bed snapshot reference maximum layer error: {maximum:.12e} m");
    assert!(
        maximum < 2e-7,
        "mixed-column deposition reference: {maximum}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn settling_and_erosion_transfer_actual_stored_layer_changes() {
    let f = Fixture::new();
    let mut maximum = 0.0_f64;
    for channel in 0..4 {
        f.clear_exchange_error();
        for buffer in f.layers() {
            f.fill(buffer, 0.);
        }
        for buffer in f.concentrations() {
            f.fill(buffer, 0.);
        }
        f.fill(f.layers()[channel], 1024.);
        f.fill(f.concentrations()[channel], 0.1);
        f.fill(&f.h.water.depth, 0.5);
        f.fill(&f.h.water.velocity_x, 0.);
        let before = f.mass(channel);
        f.stage(true, [0, 0, N as u32, N as u32], 0.5);
        assert!(f.read(f.layers()[channel]).iter().all(|v| *v > 1024.));
        let deposit_error = f.mass(channel) - before;
        maximum = maximum.max(deposit_error.abs());
        let material = [4u32, 3, 2, 1][channel];
        f.queue.write_buffer(
            &f.h.geology.surface_material,
            0,
            bytemuck::cast_slice(&vec![material; CELLS]),
        );
        f.fill(&f.h.water.velocity_x, 8.);
        let before = f.mass(channel);
        f.stage(false, [0, 0, N as u32, N as u32], 0.05);
        let erosion_error = f.mass(channel) - before;
        maximum = maximum.max(erosion_error.abs());
        println!("channel {channel}: deposit residual {deposit_error:.12e}, erosion residual {erosion_error:.12e} m3");
    }
    // Absolute exchange budget: a huge background layer must not hide loss.
    assert!(maximum < 1e-6, "represented exchange lost {maximum} m3");
}

fn slope(ground: &[f64], i: usize) -> f64 {
    let x = i % N;
    let z = i / N;
    let dx = if x == 0 {
        (ground[i + 1] - ground[i]) / 0.25
    } else if x == N - 1 {
        (ground[i] - ground[i - 1]) / 0.25
    } else {
        (ground[i + 1] - ground[i - 1]) / 0.5
    };
    let dz = if z == 0 {
        (ground[i + N] - ground[i]) / 0.25
    } else if z == N - 1 {
        (ground[i] - ground[i - N]) / 0.25
    } else {
        (ground[i + N] - ground[i - N]) / 0.5
    };
    dx.hypot(dz)
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn nonflat_erosion_matches_f64_reference_and_repeats_exactly() {
    let f = Fixture::new();
    let mut max_error = 0.0_f64;
    for (ch, d) in [0.0001_f64, 0.001, 0.01, 0.002].into_iter().enumerate() {
        let scale = 2. * 0.045 * 1.65 * d / 0.02;
        // Dyadic heights avoid hiding the race behind a difference between
        // scalar f64 and GPU f32 ground-height summation order.
        let rock: Vec<f32> = (0..CELLS)
            .map(|i| {
                let h = scale
                    * 0.25
                    * (3. * ((i % N) as f64 / 3.).sin() + 2. * ((i / N) as f64 / 4.).cos());
                ((h * 1_048_576.).round() / 1_048_576.) as f32
            })
            .collect();
        let ground: Vec<f64> = rock
            .iter()
            .map(|v| *v as f64 + 0.125 + (3 - ch) as f64 * 0.0625)
            .collect();
        let mut previous = None;
        for _ in 0..2 {
            f.clear_exchange_error();
            f.write(&f.h.geology.bedrock, &rock);
            for (n, b) in f.layers().into_iter().enumerate() {
                f.fill(
                    b,
                    if n == ch {
                        0.125
                    } else if n > ch {
                        0.0625
                    } else {
                        0.
                    },
                );
            }
            for b in f.concentrations() {
                f.fill(b, 0.02);
            }
            f.fill(&f.h.water.depth, 0.02);
            f.fill(&f.h.water.velocity_x, 0.1);
            let times: Vec<u32> = (0..CELLS).map(|i| (i % 31) as u32 * 50_000).collect();
            f.queue
                .write_buffer(&f.h.geology.settling_time, 0, bytemuck::cast_slice(&times));
            f.surface([4, 3, 2, 1][ch]);
            f.stage(false, [0, 0, N as u32, N as u32], 0.05);
            let bed = f.read(f.layers()[ch]);
            let concentration = f.read(f.concentrations()[ch]);
            for i in 0..CELLS {
                let stress = 1000. * (9.81 * 0.02 * slope(&ground, i) + 0.003 * 0.1_f64.powi(2));
                let shields = stress / (9.81 * 1650. * d);
                let critical = 0.045 * (1. + 0.5 * (times[i] as f64 / 1_500_000.).min(1.));
                let requested = if shields > critical {
                    (0.01 * (shields / critical - 1.) / [0.5, 1., 2., 5.][ch] * 0.05)
                        .min(0.05 * 0.05)
                } else {
                    0.
                };
                max_error = max_error.max((bed[i] as f64 - (0.125 - requested)).abs());
                let actual = 0.125 - bed[i] as f64;
                assert!(
                    (concentration[i] as f64 - (0.02_f32 as f64 + actual / 0.02_f32 as f64)).abs()
                        < 2e-7
                );
            }
            for (n, b) in f.layers().into_iter().enumerate() {
                if n != ch {
                    assert!(f
                        .read(b)
                        .iter()
                        .all(|v| *v == if n > ch { 0.0625 } else { 0. }));
                }
            }
            if let Some(previous) = previous {
                assert_eq!((bed.clone(), concentration.clone()), previous);
            }
            previous = Some((bed, concentration));
        }
    }
    println!("nonflat four-material erosion maximum f64 error: {max_error:.12e} m");
    assert!(
        max_error < 2e-7,
        "snapshot slope/erosion mismatch: {max_error}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn moving_bed_tiles_restore_transport_targets_and_preserve_aux_ledgers() {
    let f = Fixture::new();
    for b in f.layers() {
        f.fill(b, 0.0625);
    }
    for b in f.concentrations() {
        f.fill(b, 0.08);
    }
    f.fill(&f.h.water.depth, 0.5);
    let aux: Vec<f32> = (0..CELLS * 12)
        .map(|i| (i % 12 + 1) as f32 * 0.125)
        .collect();
    f.queue.write_buffer(
        &f.h.water.suspended_sediment_next,
        (CELLS * 4) as u64,
        bytemuck::cast_slice(&aux),
    );
    for tile in [[10, 11, 11, 13], [0, 0, 3, 4], [28, 28, 8, 8]] {
        let before: Vec<_> = f.layers().into_iter().map(|b| f.read(b)).collect();
        let old_c: Vec<_> = f.concentrations().into_iter().map(|b| f.read(b)).collect();
        for b in f.targets() {
            f.fill(b, 12345.);
        }
        f.stage(true, tile, 0.05);
        for ch in 0..4 {
            let bed = f.read(f.layers()[ch]);
            let c = f.read(f.concentrations()[ch]);
            assert_eq!(
                c,
                f.read(f.targets()[ch]),
                "old-layer scratch escaped the bed stage"
            );
            for i in 0..CELLS {
                let x = (i % N) as u32;
                let z = (i / N) as u32;
                let active =
                    x >= tile[0] && x < tile[0] + tile[2] && z >= tile[1] && z < tile[1] + tile[3];
                if !active {
                    assert_eq!((bed[i], c[i]), (before[ch][i], old_c[ch][i]));
                }
                assert!(bed[i] >= 0. && c[i] >= 0. && bed[i].is_finite() && c[i].is_finite());
            }
        }
        let after = f.read_range(
            &f.h.water.suspended_sediment_next,
            (CELLS * 4) as u64,
            CELLS * 12,
        );
        for i in 0..CELLS * 12 {
            if i % 12 != 7 {
                assert_eq!(after[i], aux[i], "water/export ledger overwritten");
            }
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn tiled_full_steps_preserve_cpu_edits_outside_the_active_transport_tile() {
    let f = Fixture::new();
    f.fill(&f.h.water.depth, 0.5);
    for (ch, b) in f.concentrations().into_iter().enumerate() {
        f.fill(b, 0.01 * (ch + 1) as f32);
    }
    for b in f.targets() {
        f.fill(b, 12345.);
    }
    for tile in [[0, 0, 5, 7], [12, 11, 5, 7]] {
        let mut expected = Vec::new();
        for (ch, b) in f.concentrations().into_iter().enumerate() {
            let mut c = f.read(b);
            c[CELLS - 1] = 0.1234 * (ch + 1) as f32;
            f.write(b, &c);
            expected.push(c);
        }
        f.h.update_params_tile(&f.queue, 0.05, tile[0], tile[1], tile[2], tile[3]);
        let mut encoder = f.device.create_command_encoder(&Default::default());
        f.h.dispatch_tile(&mut encoder, tile[2], tile[3]);
        f.queue.submit(Some(encoder.finish()));
        for (ch, b) in f.concentrations().into_iter().enumerate() {
            let current = f.read(b);
            assert_eq!(current, f.read(f.targets()[ch]));
            for i in 0..CELLS {
                let x = (i % N) as u32;
                let z = (i / N) as u32;
                if x < tile[0] || x >= tile[0] + tile[2] || z < tile[1] || z >= tile[1] + tile[3] {
                    assert_eq!(
                        current[i], expected[ch][i],
                        "inactive suspension corrupted at {i}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn unrepresentable_recipients_and_dry_cells_cannot_lose_material() {
    let f = Fixture::new();
    f.fill(&f.h.geology.sediment, 1_048_576.);
    f.fill(&f.h.water.suspended_sediment, 0.1);
    f.fill(&f.h.water.depth, 0.5);
    f.stage(true, [0, 0, N as u32, N as u32], 0.05);
    assert!(f
        .read(&f.h.geology.sediment)
        .iter()
        .all(|v| *v == 1_048_576.));
    let c = f.read(&f.h.water.suspended_sediment);
    let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
    for i in 0..CELLS {
        assert!(
            c[i] < 0.1,
            "large background bed must not suspend fines forever"
        );
        let mobile = 0.1_f32 as f64 * 0.5;
        assert!(
            (low[i * 4] as f64 + c[i] as f64 * 0.5 - mobile).abs() < mobile * 1e-6,
            "unrepresentable geometry lost its counted deposit"
        );
    }
    f.clear_exchange_error();
    f.fill(&f.h.geology.sediment, 0.125);
    f.fill(&f.h.water.suspended_sediment, 1_000_000.);
    f.fill(&f.h.water.velocity_x, 8.);
    f.surface(4);
    f.stage(false, [0, 0, N as u32, N as u32], 0.05);
    assert!(f.read(&f.h.geology.sediment).iter().all(|v| *v == 0.125));
    assert!(f
        .read(&f.h.water.suspended_sediment)
        .iter()
        .all(|v| *v == 1_000_000.));
    // A full thin-cover clear must stop before exposing the next stratum when
    // suspension cannot represent the receiving transfer.
    f.fill(&f.h.geology.sediment, 0.00001);
    f.fill(&f.h.geology.overburden, 0.125);
    f.stage(false, [0, 0, N as u32, N as u32], 0.05);
    assert!(f.read(&f.h.geology.sediment).iter().all(|v| *v == 0.00001));
    assert!(f.read(&f.h.geology.overburden).iter().all(|v| *v == 0.125));
    f.fill(&f.h.geology.sediment, 0.125);
    f.fill(&f.h.water.depth, 0.);
    f.stage(true, [0, 0, N as u32, N as u32], 0.05);
    f.stage(false, [0, 0, N as u32, N as u32], 0.05);
    assert!(f.read(&f.h.geology.sediment).iter().all(|v| *v == 0.125));
    assert!(f
        .read(&f.h.water.suspended_sediment)
        .iter()
        .all(|v| *v == 1_000_000.));
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn repeated_nonbinary_depth_exchange_closes_each_mobile_material_budget() {
    let f = Fixture::new();
    let mut residuals = Vec::new();
    for ch in 0..4 {
        f.clear_exchange_error();
        for b in f.layers() {
            f.fill(b, 0.);
        }
        for b in f.concentrations() {
            f.fill(b, 0.);
        }
        f.fill(f.layers()[ch], 0.125);
        f.fill(f.concentrations()[ch], 0.1);
        f.fill(&f.h.water.depth, 0.3);
        f.surface([4, 3, 2, 1][ch]);
        let initial = f.mass(ch);
        // Fix tolerance against initial MOBILE material, never the bedrock or
        // a huge background layer that would conceal the exchanged load.
        let mobile = 0.1_f32 as f64 * 0.3_f32 as f64 * AREA * CELLS as f64;
        for cycle in 0..10_000 {
            if cycle % 250 == 0 {
                f.device.poll(wgpu::Maintain::Wait);
            }
            f.fill(&f.h.water.velocity_x, 0.);
            f.stage(true, [0, 0, N as u32, N as u32], 0.05);
            f.fill(&f.h.water.velocity_x, 8.);
            f.stage(false, [0, 0, N as u32, N as u32], 0.05);
            if [0, 1, 10, 49, 99, 999, 9999].contains(&cycle) {
                println!(
                    "material {ch} cycle {cycle}: residual {:.12e} bed {:.12e} C {:.12e}",
                    f.mass(ch) - initial,
                    f.read(f.layers()[ch])[0],
                    f.read(f.concentrations()[ch])[0]
                );
            }
        }
        let error = f.mass(ch) - initial;
        println!(
            "material {ch}: 20000 stage residual {error:.12e} m3, mobile fraction {:.6e}",
            error / mobile
        );
        residuals.push((error, mobile));
    }
    assert!(
        residuals
            .iter()
            .all(|(error, mobile)| error.abs() < mobile * 1e-6),
        "long exchange budgets: {residuals:?}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn constant_forcing_erosion_agrees_across_time_steps() {
    let f = Fixture::new();
    let mut maximum = 0.0_f64;
    for (ch, d) in [0.0001_f64, 0.001, 0.01, 0.002].into_iter().enumerate() {
        let velocity = (0.2_f64 * (d / 0.0001).sqrt()) as f32;
        let shields = 1000. * 0.003 * (velocity as f64).powi(2) / (9.81 * 1650. * d);
        let rate = 0.01 * (shields / 0.045 - 1.) / [0.5, 1., 2., 5.][ch];
        assert!(rate > 0. && rate < 0.05);
        for count in [20, 40, 80] {
            f.clear_exchange_error();
            for b in f.layers() {
                f.fill(b, 0.);
            }
            for b in f.concentrations() {
                f.fill(b, 0.02);
            }
            f.fill(f.layers()[ch], 0.125);
            f.fill(&f.h.water.depth, 0.3);
            f.fill(&f.h.water.velocity_x, velocity);
            f.surface([4, 3, 2, 1][ch]);
            f.queue.write_buffer(
                &f.h.geology.settling_time,
                0,
                bytemuck::cast_slice(&vec![0u32; CELLS]),
            );
            for _ in 0..count {
                f.stage(false, [0, 0, N as u32, N as u32], 1. / count as f32);
            }
            for actual in f.read(f.layers()[ch]) {
                maximum = maximum.max((actual as f64 - (0.125 - rate)).abs());
            }
        }
    }
    println!("20/40/80 Hz erosion maximum constant-forcing error: {maximum:.12e} m");
    assert!(maximum < 2e-6, "erosion changed with time step: {maximum}");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn erosion_uses_the_new_geometry_after_deposition() {
    let f = Fixture::new();
    f.fill(&f.h.geology.sediment, 0.125);
    let c: Vec<f32> = (0..CELLS)
        .map(|i| 0.12 + 0.03 * ((i % N) as f32 * 0.4).sin())
        .collect();
    f.write(&f.h.water.suspended_sediment, &c);
    f.fill(&f.h.water.depth, 0.5);
    f.stage(true, [0, 0, N as u32, N as u32], 0.2);
    let deposited = f.read(&f.h.geology.sediment);
    let ground: Vec<f64> = deposited.iter().map(|v| *v as f64).collect();
    f.fill(&f.h.water.velocity_x, 0.1);
    f.stage(false, [0, 0, N as u32, N as u32], 0.05);
    let after = f.read(&f.h.geology.sediment);
    let mut maximum = 0.0_f64;
    let mut eroded = 0.0_f64;
    for i in 0..CELLS {
        let stress = 1000. * (9.81 * 0.5 * slope(&ground, i) + 0.003 * 0.1_f64.powi(2));
        let shields = stress / (9.81 * 1650. * 0.0001);
        let request = if shields > 0.045 {
            (0.01 * (shields / 0.045 - 1.) / 0.5 * 0.05).min(0.0025)
        } else {
            0.
        };
        maximum = maximum.max((after[i] as f64 - (ground[i] - request)).abs());
        eroded = eroded.max(ground[i] - after[i] as f64);
    }
    // The original flat slope has stress below threshold at this velocity.
    // Using it again would produce no erosion anywhere after deposition.
    assert!(
        eroded > 0.001,
        "erosion reused the pre-deposition flat geometry"
    );
    assert!(maximum < 2e-7, "post-deposition reference error: {maximum}");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_zero_elapsed_time_cannot_deposit_erode_or_age_the_bed() {
    let f = Fixture::new();
    f.fill(&f.h.geology.sediment, 0.125);
    f.fill(&f.h.water.suspended_sediment, 0.2);
    f.fill(&f.h.water.depth, 0.5);
    f.surface(4);
    let before = f.read(&f.h.geology.sediment);
    let c = f.read(&f.h.water.suspended_sediment);
    let age = f.read(&f.h.geology.settling_time);
    let errors = f.read_range(&f.h.geology.bed_exchange_error, 0, CELLS * 4);
    f.stage(true, [0, 0, N as u32, N as u32], 0.);
    f.fill(&f.h.water.velocity_x, 8.);
    f.stage(false, [0, 0, N as u32, N as u32], 0.);
    assert!(
        before == f.read(&f.h.geology.sediment),
        "zero elapsed time moved bed material"
    );
    assert_eq!(c, f.read(&f.h.water.suspended_sediment));
    assert_eq!(age, f.read(&f.h.geology.settling_time));
    assert_eq!(
        errors,
        f.read_range(&f.h.geology.bed_exchange_error, 0, CELLS * 4)
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_dilute_columns_follow_size_depth_and_elapsed_time() {
    let f = Fixture::new();
    let mut maximum = 0.0_f64;
    for (ch, d) in [0.0001_f64, 0.001, 0.01, 0.002].into_iter().enumerate() {
        for h in [0.05_f32, 0.3, 1.] {
            for steps in [4, 8, 16] {
                f.clear_exchange_error();
                for b in f.layers() {
                    f.fill(b, 0.);
                }
                for b in f.concentrations() {
                    f.fill(b, 0.);
                }
                f.fill(f.concentrations()[ch], 0.0001);
                f.fill(&f.h.water.depth, h);
                f.fill(&f.h.water.velocity_x, 0.);
                let before = f.mass(ch);
                for _ in 0..steps {
                    f.stage(true, [0, 0, N as u32, N as u32], 0.2 / steps as f32);
                }
                let expected = 0.0001_f32 as f64 * (-settling_speed(d) * 0.2 / h as f64).exp();
                let cs = f.read(f.concentrations()[ch]);
                let bed = f.read(f.layers()[ch]);
                for i in 0..CELLS {
                    maximum = maximum.max((cs[i] as f64 - expected).abs() / 0.0001);
                    assert!(
                        bed[i] > 0. && bed[i] < 0.001,
                        "dilute submillimetre deposit missing"
                    );
                }
                assert!(
                    (f.mass(ch) - before).abs() < before * 1e-6,
                    "dilute component budget"
                );
            }
        }
    }
    println!("dilute four-size/three-depth timestep relative error: {maximum:.12e}");
    assert!(
        maximum < 2e-5,
        "settling is gated or depends on step count: {maximum}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_thin_cover_washes_out_then_exposes_the_next_layer() {
    let f = Fixture::new();
    f.fill(&f.h.geology.sediment, 0.00002);
    f.fill(&f.h.geology.overburden, 0.01);
    f.fill(&f.h.water.depth, 0.3);
    f.fill(&f.h.water.velocity_x, 8.);
    // A stale/coarse visual tag must not make erosion skip the real cover.
    f.surface(3);
    let before = [f.mass(0), f.mass(1)];
    f.stage(false, [0, 0, N as u32, N as u32], 0.05);
    let sediment = f.read(&f.h.geology.sediment);
    let over = f.read(&f.h.geology.overburden);
    assert!(
        sediment.iter().all(|v| *v == 0.),
        "thin cover never washed out"
    );
    // Both rates are capped at .05m/s here. Removing 20um of cover consumes
    // .0004s, leaving .0496s to erode 2.48mm from the exposed overburden.
    let expected = 0.01_f32 as f64 - (0.05_f32 as f64 - 0.00002_f32 as f64 / 0.05) * 0.05;
    assert!(
        over.iter().all(|v| (*v as f64 - expected).abs() < 2e-8),
        "cover consumed a full timestep or was bypassed"
    );
    for (ch, initial) in before.into_iter().enumerate() {
        assert!(
            (f.mass(ch) - initial).abs() < initial * 1e-6,
            "thin-layer component budget"
        );
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_bed_age_uses_elapsed_microseconds_including_still_water() {
    let f = Fixture::new();
    for steps in [4, 8, 32] {
        f.clear_exchange_error();
        f.fill(&f.h.geology.sediment, 0.125);
        f.fill(&f.h.water.depth, 0.5);
        f.fill(&f.h.water.velocity_x, 0.);
        f.surface(4);
        f.queue.write_buffer(
            &f.h.geology.settling_time,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS]),
        );
        for _ in 0..steps {
            f.stage(false, [0, 0, N as u32, N as u32], 0.4 / steps as f32);
        }
        let age = f.read(&f.h.geology.settling_time);
        let age: &[u32] = bytemuck::cast_slice(&age);
        assert!(
            age.iter().all(|v| v.abs_diff(400_000) <= 2),
            "bed age counts ticks, not physical time: {}",
            age[0]
        );
        f.fill(&f.h.water.velocity_x, 0.165);
        f.stage(false, [0, 0, N as u32, N as u32], 0.01);
        assert!(
            f.read(&f.h.geology.sediment).iter().all(|v| *v == 0.125),
            "same elapsed rest produced different erosion resistance"
        );
        f.fill(&f.h.water.velocity_x, 0.3);
        f.stage(false, [0, 0, N as u32, N as u32], 0.01);
        let age = f.read(&f.h.geology.settling_time);
        assert!(
            bytemuck::cast_slice::<f32, u32>(&age)
                .iter()
                .all(|v| *v == 0),
            "actual erosion did not reset age"
        );
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_settling_and_scour_converge_to_the_mixed_column_balance() {
    let f = Fixture::new();
    let mut errors = Vec::new();
    let h = 0.3_f32;
    let c0 = 0.001_f32;
    let v = 0.2_f32;
    let shields = 1000. * 0.003 * (v as f64).powi(2) / (9.81 * 1650. * 0.0001);
    let erosion = (0.01 * (shields / 0.045 - 1.) / 0.5).min(0.05);
    let ws = settling_speed(0.0001);
    let expected = erosion / ws + (c0 as f64 - erosion / ws) * (-ws * 0.2 / h as f64).exp();
    for steps in [4, 8, 16] {
        f.clear_exchange_error();
        f.fill(&f.h.geology.sediment, 0.125);
        f.fill(&f.h.water.suspended_sediment, c0);
        f.fill(&f.h.water.depth, h);
        f.fill(&f.h.water.velocity_x, v);
        f.surface(4);
        f.queue.write_buffer(
            &f.h.geology.settling_time,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS]),
        );
        let before = f.mass(0);
        for _ in 0..steps {
            f.stage(true, [0, 0, N as u32, N as u32], 0.2 / steps as f32);
            f.stage(false, [0, 0, N as u32, N as u32], 0.2 / steps as f32);
        }
        let c = f.read(&f.h.water.suspended_sediment);
        errors.push((c[0] as f64 - expected).abs());
        let accounted_transfer = (c0 as f64 * h as f64 + erosion * 0.2) * AREA * CELLS as f64;
        assert!(
            (f.mass(0) - before).abs() < accounted_transfer * 1e-6,
            "settling/scour budget"
        );
    }
    println!("coupled settling/scour concentration errors: {errors:?}");
    assert!(
        errors[1] < errors[0] * 0.7 && errors[2] < errors[1] * 0.7,
        "coupled physical-time solution does not converge"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_full_terrain_step_settles_fines_without_changing_water_volume() {
    let f = Fixture::new();
    f.fill(&f.h.water.depth, 0.3);
    f.fill(&f.h.water.suspended_sediment, 0.0001);
    let water = f.read(&f.h.water.depth);
    let before = f.mass(0);
    for _ in 0..40 {
        f.h.update_water_params(&f.queue, 0.05, 0, 0.036, 2);
        let mut e = f.device.create_command_encoder(&Default::default());
        f.h.dispatch_water_substeps(&mut e, 2);
        f.queue.submit(Some(e.finish()));
    }
    let expected = 0.0001_f32 as f64 * (-settling_speed(0.0001) * 2. / 0.3_f32 as f64).exp();
    assert!(
        f.read(&f.h.water.suspended_sediment)
            .iter()
            .all(|c| (*c as f64 - expected).abs() < 2e-9),
        "full pipeline retains dilute fines indefinitely"
    );
    assert_eq!(
        water,
        f.read(&f.h.water.depth),
        "settling created or removed water"
    );
    assert!(
        (f.mass(0) - before).abs() < before * 1e-6,
        "full pipeline fine budget"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_long_residence_exhausts_the_representable_mobile_load() {
    let f = Fixture::new();
    let c: Vec<f32> = (0..CELLS)
        .map(|i| (1 + (i * 37) % 101) as f32 * 1e-5)
        .collect();
    let h: Vec<f32> = (0..CELLS)
        .map(|i| 0.03 + ((i * 53) % 97) as f32 * 0.0075)
        .collect();
    f.write(&f.h.water.depth, &h);
    f.write(&f.h.water.suspended_gravel, &c);
    let initial = f.mass(2);
    // The coarse class has >90 e-folding residence times even in the deepest
    // column. Only floating-point residue, not a complete load, may remain.
    f.stage(true, [0, 0, N as u32, N as u32], 100.);
    let after = f.read(&f.h.water.suspended_gravel);
    let residual = after
        .iter()
        .zip(&h)
        .map(|(c, h)| *c as f64 * *h as f64 * AREA)
        .sum::<f64>();
    println!(
        "long-residence mobile fraction: {:.12e}",
        residual / initial
    );
    assert!(
        residual < initial * 2e-6,
        "full deposition rolled back: {}",
        residual / initial
    );
    assert!(
        (f.mass(2) - initial).abs() < initial * 1e-6,
        "long-residence component budget"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn physical_recipients_match_independently_rounded_component_balances() {
    let f = Fixture::new();
    let h: Vec<f32> = (0..CELLS)
        .map(|i| 0.03 + ((i * 53) % 97) as f32 * 0.0075)
        .collect();
    f.write(&f.h.water.depth, &h);
    f.fill(&f.h.water.velocity_x, 8.);
    for ch in 0..4 {
        for settling in [true, false] {
            f.clear_exchange_error();
            for (i, b) in f.layers().into_iter().enumerate() {
                f.fill(b, if i == ch { 0.125 } else { 0. });
            }
            for (i, b) in f.concentrations().into_iter().enumerate() {
                f.fill(b, if i == ch { 0.1 } else { 0. });
            }
            f.surface([4, 3, 2, 1][ch]);
            f.stage(settling, [0, 0, N as u32, N as u32], 0.001);
            let bed = f.read(f.layers()[ch]);
            let c = f.read(f.concentrations()[ch]);
            let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
            for i in 0..CELLS {
                // Count both parts of actual bed inventory in the independent
                // f64 receiver balance, excluding numerical error feedback.
                let expected = (0.1_f32 as f64
                    + ((0.125 - bed[i] as f64) - low[i * 4 + ch] as f64) / h[i] as f64)
                    as f32;
                assert_eq!(
                    c[i].to_bits(),
                    expected.to_bits(),
                    "recipient rounding: class {ch}, settling {settling}, cell {i}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn submillimetre_films_settle_without_a_depth_gate_or_unphysical_suspension_tail() {
    let f = Fixture::new();
    let h = 0.00025f32;
    for b in f.layers() {
        f.fill(b, 0.125);
    }
    for b in f.concentrations() {
        f.fill(b, 0.01);
    }
    f.fill(&f.h.water.depth, h);
    let before: Vec<_> = (0..4).map(|ch| f.mass(ch)).collect();
    for _ in 0..100 {
        f.stage(true, [0, 0, N as u32, N as u32], 0.05);
    }
    for (ch, start) in before.into_iter().enumerate() {
        let c = f.read(f.concentrations()[ch]);
        let remaining = c.iter().map(|c| *c as f64 * h as f64 * AREA).sum::<f64>();
        let mobile = 0.01f32 as f64 * h as f64 * AREA * CELLS as f64;
        println!(
            "THIN FILM class={ch} remaining_fraction={} budget_error_m3={}",
            remaining / mobile,
            f.mass(ch) - start
        );
        assert!(
            remaining / mobile < 1e-5,
            "class {ch} cannot settle out of a 0.25 mm film"
        );
        assert!(
            (f.mass(ch) - start).abs() < mobile * 1e-6,
            "class {ch} thin-film mass budget"
        );
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn deposition_pair_preserves_the_nonbinary_product_residual() {
    use wgpu::util::DeviceExt;
    let f = Fixture::new();
    let source = include_str!("../../game/src/gpu/shaders/heightfield_erosion.wgsl");
    let difference = &source[source.find("fn stored_difference(").unwrap()
        ..source.find("fn concentration_after_exchange(").unwrap()];
    let receiver = &source
        [source.find("fn stored_positive(").unwrap()..source.find("fn settle_column(").unwrap()];
    let shader = format!(
        "{difference}\n{receiver}\n{}",
        r#"
@group(0) @binding(0) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>>;
@compute @workgroup_size(16,16) fn probe(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.y*32u+g.x;let v=input[i];
 let pair=receive_deposit(v.x,0.,v.y,v.z,v.w);
 let difference=stored_difference(v.y,v.z);
 let amount=stored_positive(difference*v.w);
 output[i]=vec4(pair,amount,fma(difference,v.w,-amount));
}"#
    );
    let values: Vec<[f32; 4]> = (0..CELLS)
        .map(|i| [[0., 1e-9, 0.125, 1.][i % 4], 0.1, 0.09851249, 0.3])
        .collect();
    let input = f
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = f.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (CELLS * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let module = f.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = f
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let group = f.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
        ],
    });
    let mut encoder = f.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(2, 2, 1);
    }
    f.queue.submit(Some(encoder.finish()));
    let result = f.read_range(&output, 0, CELLS * 4);
    for i in 0..4 {
        let v = values[i];
        let r = &result[i * 4..][..4];
        let expected = (v[1] as f64 - v[2] as f64) * v[3] as f64;
        let error = (r[0] as f64 - v[0] as f64) + r[1] as f64 - expected;
        println!(
            "PAIR old={} high={} low={} product={} fma={} relative_error={}",
            v[0],
            r[0],
            r[1],
            r[2],
            r[3],
            error / expected
        );
        assert!(
            error.abs() < expected * 1e-10,
            "stored deposit pair loses its product residual"
        );
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn mixed_column_deposition_is_continuous_across_film_depth_timestep_and_cell_size() {
    let mut max_concentration_error = 0f64;
    let mut max_material_fraction = 0f64;
    let depths = [
        0.0000001f32,
        0.0001,
        0.00025,
        0.000999,
        0.001001,
        0.003,
        0.1,
    ];
    for cell in [0.125, 0.25, 0.5] {
        let f = Fixture::with_cell(cell);
        let h: Vec<f32> = (0..CELLS).map(|i| depths[i % depths.len()]).collect();
        f.write(&f.h.water.depth, &h);
        for steps in [1, 2, 4, 8] {
            f.clear_exchange_error();
            for b in f.layers() {
                f.fill(b, 0.125);
            }
            for b in f.concentrations() {
                f.fill(b, 0.01);
            }
            for _ in 0..steps {
                f.stage(true, [0, 0, N as u32, N as u32], 0.0001 / steps as f32);
            }
            let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
            for (ch, d) in [0.0001, 0.001, 0.01, 0.002].into_iter().enumerate() {
                let bed = f.read(f.layers()[ch]);
                let c = f.read(f.concentrations()[ch]);
                for i in 0..CELLS {
                    let initial = 0.01f32 as f64 * h[i] as f64;
                    let expected =
                        0.01f32 as f64 * (-settling_speed(d) * 0.0001 / h[i] as f64).exp();
                    let error = (c[i] as f64 - expected).abs() / 0.01f32 as f64;
                    max_concentration_error = max_concentration_error.max(error);
                    let residual = ((bed[i] as f64 - 0.125)
                        + low[i * 4 + ch] as f64
                        + c[i] as f64 * h[i] as f64
                        - initial)
                        .abs()
                        / initial;
                    max_material_fraction = max_material_fraction.max(residual);
                    assert!(
                        error < 3e-6,
                        "cell {cell}, steps {steps}, h {}, class {ch}: reference",
                        h[i]
                    );
                    assert!(
                        residual < 1e-6,
                        "cell {cell}, steps {steps}, h {}, class {ch}: budget {residual}",
                        h[i]
                    );
                }
            }
        }
    }
    println!("FILM MATRIX max_concentration_relative={max_concentration_error} max_mobile_budget_fraction={max_material_fraction}");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn prescribed_drying_preserves_deposits_and_rewet_flow_can_remobilize_each_class() {
    let f = Fixture::new();
    for ch in 0..4 {
        f.clear_exchange_error();
        for b in f.layers() {
            f.fill(b, 0.);
        }
        for b in f.concentrations() {
            f.fill(b, 0.);
        }
        f.fill(&f.h.geology.bedrock, 0.125);
        f.fill(&f.h.water.depth, 0.00025);
        f.fill(f.concentrations()[ch], 0.01);
        let initial = f.mass(ch);
        for _ in 0..100 {
            f.stage(true, [0, 0, N as u32, N as u32], 0.05);
        }
        assert!(f.read(f.concentrations()[ch]).iter().all(|v| *v == 0.));
        f.fill(&f.h.water.depth, 0.);
        f.fill(&f.h.water.velocity_x, 0.);
        for settling in [true, false] {
            f.stage(settling, [0, 0, N as u32, N as u32], 0.05);
        }
        assert!(
            (f.mass(ch) - initial).abs() < initial * 1e-6,
            "dry bed lost class {ch}"
        );
        f.fill(&f.h.water.depth, 0.01);
        for settling in [true, false] {
            f.stage(settling, [0, 0, N as u32, N as u32], 0.05);
        }
        assert!(
            f.read(f.concentrations()[ch]).iter().all(|v| *v == 0.),
            "quiet rewet resuspended class {ch}"
        );
        f.fill(&f.h.water.velocity_x, 2.);
        f.stage(false, [0, 0, N as u32, N as u32], 0.05);
        let mobile = f
            .read(f.concentrations()[ch])
            .iter()
            .map(|c| *c as f64 * 0.01f32 as f64 * AREA)
            .sum::<f64>();
        assert!(
            mobile / initial > 1. - 1e-6,
            "flow failed to remobilize class {ch}"
        );
        assert!(
            (f.mass(ch) - initial).abs() < initial * 1e-6,
            "rewet exchange budget class {ch}"
        );
        println!(
            "REWET class={ch} mobile_fraction={} material_fraction={}",
            mobile / initial,
            (f.mass(ch) - initial) / initial
        );
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn provenance_records_actual_substeps_without_changing_any_physical_channel() {
    use game::gpu::heightfield::TerrainEventRecorder;
    for boundaries in [0, 15] {
        let f = Fixture::new();
        let traced = GpuHeightfield::new(
            &f.device,
            N as u32,
            N as u32,
            0.25,
            0.,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        for (a, b, value) in [
            (&f.h.geology.paydirt, &traced.geology.paydirt, 0.125),
            (&f.h.geology.overburden, &traced.geology.overburden, 0.03),
            (&f.h.geology.gravel, &traced.geology.gravel, 0.01),
            (&f.h.geology.sediment, &traced.geology.sediment, 0.02),
            (&f.h.water.velocity_x, &traced.water.velocity_x, 0.7),
            (&f.h.water.velocity_z, &traced.water.velocity_z, -0.3),
        ] {
            f.fill(a, value);
            f.fill(b, value);
        }
        let initial_depth: Vec<f32> = (0..CELLS).map(|i| 0.2 + 0.001 * (i % N) as f32).collect();
        let ground: Vec<f32> = (0..CELLS)
            .map(|i| 0.002 * ((i % N) + (i / N)) as f32)
            .collect();
        for b in [&f.h.water.depth, &traced.water.depth] {
            f.write(b, &initial_depth);
        }
        for b in [&f.h.geology.bedrock, &traced.geology.bedrock] {
            f.write(b, &ground);
        }
        for (ch, (a, b)) in f
            .concentrations()
            .into_iter()
            .zip([
                &traced.water.suspended_sediment,
                &traced.water.suspended_overburden,
                &traced.water.suspended_gravel,
                &traced.water.suspended_paydirt,
            ])
            .enumerate()
        {
            let values: Vec<f32> = (0..CELLS)
                .map(|i| 0.01 * (ch + 1) as f32 * (1. + (i % N) as f32 / N as f32))
                .collect();
            f.write(a, &values);
            f.write(b, &values);
        }
        let recorder = TerrainEventRecorder::new(&f.device, &traced.water, &traced.geology);
        let capture = recorder.capture(&f.device, CELLS as u32, 4);
        for h in [&f.h, &traced] {
            h.update_water_params(&f.queue, 0.05, boundaries, 0.02, 4);
        }
        let mut encoder = f.device.create_command_encoder(&Default::default());
        f.h.dispatch_water_substeps(&mut encoder, 4);
        traced.dispatch_water_substeps_traced(&mut encoder, 4, &recorder, &capture);
        f.queue.submit(Some(encoder.finish()));
        for (a, b, count) in [
            (&f.h.water.depth, &traced.water.depth, CELLS),
            (&f.h.water.velocity_x, &traced.water.velocity_x, CELLS),
            (&f.h.water.velocity_z, &traced.water.velocity_z, CELLS),
            (&f.h.water.flux_x, &traced.water.flux_x, CELLS),
            (&f.h.water.flux_z, &traced.water.flux_z, CELLS),
            (&f.h.geology.bedrock, &traced.geology.bedrock, CELLS),
            (&f.h.geology.paydirt, &traced.geology.paydirt, CELLS),
            (&f.h.geology.overburden, &traced.geology.overburden, CELLS),
            (&f.h.geology.gravel, &traced.geology.gravel, CELLS),
            (&f.h.geology.sediment, &traced.geology.sediment, CELLS),
            (
                &f.h.geology.bed_remainder,
                &traced.geology.bed_remainder,
                CELLS * 4,
            ),
            (
                &f.h.geology.bed_exchange_error,
                &traced.geology.bed_exchange_error,
                CELLS * 4,
            ),
            (
                &f.h.geology.settling_time,
                &traced.geology.settling_time,
                CELLS,
            ),
        ] {
            assert_eq!(
                f.read_range(a, 0, count),
                f.read_range(b, 0, count),
                "recorder changed physical channel"
            );
        }
        for (a, b) in f.concentrations().into_iter().zip([
            &traced.water.suspended_sediment,
            &traced.water.suspended_overburden,
            &traced.water.suspended_gravel,
            &traced.water.suspended_paydirt,
        ]) {
            assert_eq!(f.read(a), f.read(b));
        }
        let raw = f.read_range(&capture.buffer, 0, (capture.size / 4) as usize);
        let events = raw.as_chunks::<4>().0;
        let n = CELLS;
        for (a, b) in events[..n].iter().map(|v| v[0]).zip(&initial_depth) {
            assert_eq!(a, *b);
        }
        for step in 0..4 {
            let event = &events[step * n..(step + 1) * n];
            for i in 0..n {
                let x = i % N;
                let z = i / N;
                let outgoing = event[i][1].max(0.)
                    + event[i][2].max(0.)
                    + event[i][3]
                    + if x > 0 { -event[i - 1][1].min(0.) } else { 0. }
                    + if z > 0 { -event[i - N][2].min(0.) } else { 0. };
                assert!(outgoing as f64 <= event[i][0] as f64 * AREA * (1. + 1e-7));
                if step < 3 {
                    let mut net = -event[i][1] - event[i][2] - event[i][3];
                    if x > 0 {
                        net += event[i - 1][1];
                    }
                    if z > 0 {
                        net += event[i - N][2];
                    }
                    let predicted = (event[i][0] + net / (AREA as f32)).max(0.);
                    assert!(
                        (predicted - events[(step + 1) * n + i][0]).abs() < 1e-6 * initial_depth[i]
                    );
                }
            }
        }
        let after_scour = &events[5 * n..6 * n];
        let pay = f.read(&traced.geology.paydirt);
        let c = f.read(&traced.water.suspended_paydirt);
        for i in 0..n {
            assert_eq!(after_scour[i][0], pay[i]);
            assert_eq!(after_scour[i][2], c[i]);
        }
        println!("READ-ONLY EVENTS boundaries={boundaries}: four actual water substeps and separate bed stages; physical channels bit-identical");
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn moving_shallow_paydirt_settles_during_its_actual_transport_residence() {
    use game::gpu::heightfield::TerrainEventRecorder;
    let f = Fixture::new();
    for b in f.layers() {
        f.fill(b, 0.125);
    }
    f.fill(&f.h.water.depth, 0.0008);
    f.fill(&f.h.water.velocity_x, 2.);
    let pulse: Vec<f32> = (0..CELLS)
        .map(|i| if i % N == N / 2 { 0.01 } else { 0. })
        .collect();
    f.write(&f.h.water.suspended_paydirt, &pulse);
    let recorder = TerrainEventRecorder::new(&f.device, &f.h.water, &f.h.geology);
    let capture = recorder.capture(&f.device, CELLS as u32, 2);
    f.h.update_water_params(&f.queue, 0.05, 0, 0.036, 2);
    let mut encoder = f.device.create_command_encoder(&Default::default());
    f.h.dispatch_water_substeps_traced(&mut encoder, 2, &recorder, &capture);
    f.queue.submit(Some(encoder.finish()));
    let raw = f.read_range(&capture.buffer, 0, (capture.size / 4) as usize);
    let events = raw.as_chunks::<4>().0;
    let depth = f.read(&f.h.water.depth);
    let c = f.read(&f.h.water.suspended_paydirt);
    let layer = f.read(&f.h.geology.paydirt);
    let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
    let row = N / 2;
    let receiver = row * N + N / 2 + 1;
    let ws = (4. * 9.81 * 0.002 * 1650_f64 / (3. * 1000. * 0.44)).sqrt();
    let mut mobile = vec![0.; N];
    mobile[N / 2] = 0.01_f32 as f64 * 0.0008_f32 as f64;
    let mut deposited = vec![0.; N];
    for step in 0..2 {
        let current = &events[step * CELLS + row * N..step * CELLS + (row + 1) * N];
        let next_h: Vec<f64> = (0..N)
            .map(|x| {
                if step == 0 {
                    events[CELLS + row * N + x][0] as f64
                } else {
                    depth[row * N + x] as f64
                }
            })
            .collect();
        let h: Vec<f64> = (0..N)
            .map(|x| (current[x][0] as f64 + next_h[x]) * 0.5)
            .collect();
        let q: Vec<f64> = current.iter().map(|v| v[1] as f64 / AREA / 0.025).collect();
        let derivative = |m: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut flow = vec![0.; N];
            let mut sink = vec![0.; N];
            for x in 0..N {
                sink[x] = ws * m[x] / h[x].max(1e-30);
                flow[x] -= sink[x];
            }
            for x in 0..N - 1 {
                let donor = if q[x] >= 0. { x } else { x + 1 };
                let flux = q[x] * m[donor] / h[donor].max(1e-30);
                flow[x] -= flux;
                flow[x + 1] += flux;
            }
            (flow, sink)
        };
        let dt = 0.025 / 512.;
        for _ in 0..512 {
            let (a, sa) = derivative(&mobile);
            let (b, sb) = derivative(
                &(0..N)
                    .map(|i| mobile[i] + 0.5 * dt * a[i])
                    .collect::<Vec<_>>(),
            );
            let (c, sc) = derivative(
                &(0..N)
                    .map(|i| mobile[i] + 0.5 * dt * b[i])
                    .collect::<Vec<_>>(),
            );
            let (d, sd) = derivative(&(0..N).map(|i| mobile[i] + dt * c[i]).collect::<Vec<_>>());
            for i in 0..N {
                mobile[i] += dt * (a[i] + 2. * b[i] + 2. * c[i] + d[i]) / 6.;
                deposited[i] += dt * (sa[i] + 2. * sb[i] + 2. * sc[i] + sd[i]) / 6.;
            }
        }
    }
    let expected = mobile[N / 2 + 1] + deposited[N / 2 + 1];
    let actual = (layer[receiver] as f64 - 0.125)
        + low[receiver * 4 + 3] as f64
        + c[receiver] as f64 * depth[receiver] as f64;
    println!("RESIDENCE actual_receiver_height={actual} continuous_reference={expected} relative_error={} settling_time_s={}",(actual/expected-1.).abs(),0.0008/ws);
    assert!(
        (actual / expected - 1.).abs() < 0.05,
        "full-tick advection overtransports fast settling paydirt"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn reactive_transport_converges_to_translated_decaying_and_deposited_gaussian_profiles() {
    // Independently prescribed steady clean-inlet / open-outlet flow. The
    // analytic ADR profile supplies both the mobile and deposited reference.
    // This isolates transport convergence from the hydraulic approximation.
    let f = Fixture::new();
    let (extent, speed, h0, total, sigma) = (8_f64, 0.5_f64, 0.1_f64, 0.5_f64, 0.4_f64);
    let ws = (4. * 9.81 * 0.002 * 1650_f64 / (3. * 1000. * 0.44)).sqrt();
    let k = ws / h0;
    let initial = |x: f64| 0.01 * (-0.5 * ((x - 2.) / sigma).powi(2)).exp();
    let average = |x: f64, dx: f64, t: f64| {
        let mut sum = 0.;
        for p in 0..=8 {
            let weight = if p == 0 || p == 8 {
                1.
            } else if p % 2 == 0 {
                2.
            } else {
                4.
            };
            sum += weight * initial(x + dx * (p as f64 / 8. - 0.5) - speed * t);
        }
        sum / 24.
    };
    let mut errors = Vec::new();
    for size in [32_usize, 64, 128] {
        let cell = (extent / size as f64) as f32;
        let n = size * size;
        let area = (cell as f64).powi(2);
        let steps = size / 8;
        let dt = total / steps as f64;
        let h = GpuHeightfield::new(
            &f.device,
            size as u32,
            size as u32,
            cell,
            0.,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        for b in [
            &h.geology.sediment,
            &h.geology.overburden,
            &h.geology.gravel,
            &h.geology.paydirt,
        ] {
            f.write(b, &vec![0.125; n]);
        }
        f.write(&h.water.depth, &vec![h0 as f32; n]);
        let c: Vec<f32> = (0..n)
            .map(|i| {
                average(
                    (i % size) as f64 * cell as f64 + 0.5 * cell as f64,
                    cell as f64,
                    0.,
                ) as f32
            })
            .collect();
        f.write(&h.water.suspended_paydirt, &c);
        let q = (speed * h0 * cell as f64 * dt) as f32;
        let flux: Vec<f32> = (0..n)
            .map(|i| if i % size + 1 == size { 0. } else { q })
            .collect();
        f.write(&h.water.flux_x, &flux);
        let mut aux = vec![0_f32; n * 13];
        for i in 0..n {
            let a = n + i * 12;
            aux[a] = h0 as f32;
            if i % size + 1 == size {
                aux[a + 2] = q;
            }
        }
        f.write(&h.water.suspended_sediment_next, &aux);
        h.update_water_params(&f.queue, dt as f32, 2, 0., 1);
        let mut encoder = f.device.create_command_encoder(&Default::default());
        for _ in 0..steps {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&h.simulation.sediment_transport);
                pass.set_bind_group(0, &h.bind_groups.params, &[]);
                pass.set_bind_group(1, &h.bind_groups.water, &[]);
                pass.set_bind_group(2, &h.bind_groups.terrain, &[]);
                pass.dispatch_workgroups((size as u32).div_ceil(16), (size as u32).div_ceil(16), 1);
            }
            for (next, current) in [
                (
                    &h.water.suspended_sediment_next,
                    &h.water.suspended_sediment,
                ),
                (
                    &h.water.suspended_overburden_next,
                    &h.water.suspended_overburden,
                ),
                (&h.water.suspended_gravel_next, &h.water.suspended_gravel),
                (&h.water.suspended_paydirt_next, &h.water.suspended_paydirt),
            ] {
                encoder.copy_buffer_to_buffer(next, 0, current, 0, (n * 4) as u64);
            }
        }
        f.queue.submit(Some(encoder.finish()));
        let c_after = f.read_range(&h.water.suspended_paydirt, 0, n);
        let layer = f.read_range(&h.geology.paydirt, 0, n);
        let low = f.read_range(&h.geology.bed_remainder, 0, n * 4);
        let aux = f.read_range(&h.water.suspended_sediment_next, (n * 4) as u64, n * 12);
        let mut mobile_error = 0.;
        let mut bed_error = 0.;
        let mut reference_mobile = 0.;
        let mut reference_bed = 0.;
        let row = size / 2;
        for x in 0..size {
            let at = (x as f64 + 0.5) * cell as f64;
            let exact_mobile = average(at, cell as f64, total) * (-k * total).exp() * h0;
            let mut integral = 0.;
            for p in 0..=128 {
                let t = total * p as f64 / 128.;
                let weight = if p == 0 || p == 128 {
                    1.
                } else if p % 2 == 0 {
                    2.
                } else {
                    4.
                };
                integral += weight * k * h0 * average(at, cell as f64, t) * (-k * t).exp();
            }
            let exact_bed = integral * total / (3. * 128.);
            let i = row * size + x;
            mobile_error += (c_after[i] as f64 * h0 - exact_mobile).abs();
            bed_error += ((layer[i] as f64 - 0.125) + low[i * 4 + 3] as f64 - exact_bed).abs();
            reference_mobile += exact_mobile;
            reference_bed += exact_bed;
        }
        let initial_mass = c
            .iter()
            .map(|c| *c as f64 * h0 as f32 as f64 * area)
            .sum::<f64>();
        let final_mass = (0..n)
            .map(|i| {
                c_after[i] as f64 * h0 as f32 as f64 * area
                    + (layer[i] as f64 - 0.125) * area
                    + low[i * 4 + 3] as f64 * area
                    + aux[i * 12 + 11] as f64
            })
            .sum::<f64>();
        assert!(
            (final_mass - initial_mass).abs() < initial_mass * 1e-6,
            "moving reaction material budget at {size}: {}",
            final_mass - initial_mass
        );
        let mobile_error = mobile_error / reference_mobile;
        let bed_error = bed_error / reference_bed;
        println!("ADR GAUSSIAN size={size} dx={cell} dt={dt} mobile_l1={mobile_error} bed_l1={bed_error} mobile_budget_relative={}",(final_mass-initial_mass)/initial_mass);
        errors.push((mobile_error, bed_error));
    }
    assert!(
        errors[1].0 < errors[0].0 * 0.7 && errors[2].0 < errors[1].0 * 0.7,
        "mobile spatial/time convergence: {errors:?}"
    );
    assert!(
        errors[1].1 < errors[0].1 * 0.7 && errors[2].1 < errors[1].1 * 0.7,
        "deposited spatial/time convergence: {errors:?}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn full_reactive_steps_match_quiet_columns_and_exact_zero_time_without_losing_material() {
    let f = Fixture::new();
    let mut worst_c = 0_f64;
    let mut worst_mass = 0_f64;
    for h0 in [1e-7_f32, 0.00025, 0.001001, 0.1] {
        for steps in [1, 2, 8] {
            for dt in [0_f32, 0.0001] {
                for b in f.layers() {
                    f.fill(b, 0.125);
                }
                f.clear_exchange_error();
                f.fill(&f.h.water.depth, h0);
                f.fill(&f.h.water.velocity_x, 0.);
                f.fill(&f.h.water.velocity_z, 0.);
                let initial = [0.1_f32, 0.09, 0.08, 0.07];
                for (b, c) in f.concentrations().into_iter().zip(initial) {
                    f.fill(b, c);
                }
                f.h.update_water_params(&f.queue, dt, 0, 0.036, steps);
                let mut encoder = f.device.create_command_encoder(&Default::default());
                f.h.dispatch_water_substeps(&mut encoder, steps);
                f.queue.submit(Some(encoder.finish()));
                let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
                let depth = f.read(&f.h.water.depth);
                for (ch, (cb, layer)) in f.concentrations().into_iter().zip(f.layers()).enumerate()
                {
                    let c = f.read(cb);
                    let bed = f.read(layer);
                    let d = [0.0001, 0.001, 0.01, 0.002][ch];
                    let stokes = 9.81 * 1650. * d * d / (18. * 0.001);
                    let turbulent = (4. * 9.81 * d * 1650_f64 / (3. * 1000. * 0.44)).sqrt();
                    let ws = if d < 0.0001 {
                        stokes
                    } else if d > 0.001 {
                        turbulent
                    } else {
                        let t = (d - 0.0001) / 0.0009;
                        stokes * (1. - t) + turbulent * t
                    };
                    let expected = initial[ch] as f64 * (-ws * dt as f64 / h0 as f64).exp();
                    let c_error = c
                        .iter()
                        .map(|v| (*v as f64 - expected).abs() / initial[ch] as f64)
                        .fold(0_f64, f64::max);
                    let error = (0..CELLS)
                        .map(|i| {
                            ((bed[i] as f64 - 0.125)
                                + low[i * 4 + ch] as f64
                                + c[i] as f64 * depth[i] as f64
                                - initial[ch] as f64 * h0 as f64)
                                * AREA
                        })
                        .sum::<f64>();
                    let mass_error =
                        error.abs() / (initial[ch] as f64 * h0 as f64 * AREA * CELLS as f64);
                    assert!(
                        c_error < 1e-6,
                        "quiet C error h={h0} steps={steps} dt={dt} ch={ch}: {c_error}"
                    );
                    assert!(
                        mass_error < 1e-6,
                        "quiet material h={h0} steps={steps} dt={dt} ch={ch}: {mass_error}"
                    );
                    if dt == 0. {
                        assert!(c.iter().all(|v| *v == initial[ch]));
                        assert!(bed.iter().all(|v| *v == 0.125));
                    }
                    worst_c = worst_c.max(c_error);
                    worst_mass = worst_mass.max(mass_error);
                }
            }
        }
    }
    println!("FULL REACTIVE QUIET maximum_C_relative={worst_c} maximum_mobile_budget_relative={worst_mass}");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn reactive_cross_depth_arrivals_match_time_quadrature_for_every_class() {
    for gravity in [9.81, 0.] {
        let f = Fixture::new();
        for b in f.layers() {
            f.fill(b, 0.125);
        }
        let donor = 16 * N + 16;
        let receiver = donor + 1;
        let dt = 0.025_f32;
        let (old_a, old_b, new_a, new_b) = (0.01_f32, 0.005_f32, 0.008_f32, 0.007_f32);
        let mut depths = vec![0.; CELLS];
        depths[donor] = new_a;
        depths[receiver] = new_b;
        f.write(&f.h.water.depth, &depths);
        let mut q = vec![0.; CELLS];
        q[donor] = 0.002 * AREA as f32;
        f.write(&f.h.water.flux_x, &q);
        let mut aux = vec![0.; CELLS * 13];
        aux[CELLS + donor * 12] = old_a;
        aux[CELLS + receiver * 12] = old_b;
        f.write(&f.h.water.suspended_sediment_next, &aux);
        for (ch, b) in f.concentrations().into_iter().enumerate() {
            let mut c = vec![0.; CELLS];
            c[donor] = 0.01 * (ch + 1) as f32;
            f.write(b, &c);
        }
        f.h.update_water_params(&f.queue, dt, 0, 0.036, 1);
        f.queue.write_buffer(
            &f.h.params_buffer,
            40,
            bytemuck::bytes_of(&(gravity as f32)),
        );
        let mut encoder = f.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&f.h.simulation.sediment_transport);
            pass.set_bind_group(0, &f.h.bind_groups.params, &[]);
            pass.set_bind_group(1, &f.h.bind_groups.water, &[]);
            pass.set_bind_group(2, &f.h.bind_groups.terrain, &[]);
            pass.dispatch_workgroups(2, 2, 1);
        }
        f.queue.submit(Some(encoder.finish()));
        let low = f.read_range(&f.h.geology.bed_remainder, 0, CELLS * 4);
        for (ch, (target, layer)) in f.targets().into_iter().zip(f.layers()).enumerate() {
            let c = f.read(target);
            let bed = f.read(layer);
            let initial_c = (0.01_f32 * (ch + 1) as f32) as f64;
            let old = initial_c * old_a as f64;
            let d = [0.0001, 0.001, 0.01, 0.002][ch];
            let stokes = gravity * 1650. * d * d / (18. * 0.001);
            let turbulent = (4. * gravity * d * 1650_f64 / (3. * 1000. * 0.44)).sqrt();
            let ws = if d < 0.0001 {
                stokes
            } else if d > 0.001 {
                turbulent
            } else {
                let t = (d - 0.0001) / 0.0009;
                stokes * (1. - t) + turbulent * t
            };
            let a = ws * dt as f64 / (0.5 * (old_a + new_a)) as f64;
            let b = ws * dt as f64 / (0.5 * (old_b + new_b)) as f64;
            let mut arrival = 0.;
            let mut departure = 0.;
            for p in 0..=1024 {
                let t = p as f64 / 1024.;
                let w = if p == 0 || p == 1024 {
                    1.
                } else if p % 2 == 0 {
                    2.
                } else {
                    4.
                };
                arrival += w * (-a * t - b * (1. - t)).exp();
                departure += w * (-a * t).exp();
            }
            let entering = initial_c * q[donor] as f64 / AREA * departure / (3. * 1024.);
            let surviving = initial_c * q[donor] as f64 / AREA * arrival / (3. * 1024.);
            let actual_mobile = c[receiver] as f64 * new_b as f64;
            let actual_bed = bed[receiver] as f64 - 0.125 + low[receiver * 4 + ch] as f64;
            assert!(
                (actual_mobile - surviving).abs() < old * 1e-6,
                "cross-depth mobile {ch}"
            );
            assert!(
                (actual_bed - (entering - surviving)).abs() < old * 1e-6,
                "cross-depth bed {ch}"
            );
            let total = c[donor] as f64 * new_a as f64 + actual_mobile + bed[donor] as f64 - 0.125
                + low[donor * 4 + ch] as f64
                + actual_bed;
            assert!(
                (total - old).abs() < old * 1e-6,
                "cross-depth material {ch}"
            );
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn fixed_grid_reactive_time_refinement_matches_independent_poisson_reference() {
    let f = Fixture::new();
    let size = 64_usize;
    let n = size * size;
    let cell = 0.125_f32;
    let h0 = 0.1_f32;
    let (speed, total, sigma) = (0.5_f64, 0.5_f64, 0.4_f64);
    let area = (cell as f64).powi(2);
    let k = (4. * 9.81 * 0.002 * 1650_f64 / (3. * 1000. * 0.44)).sqrt() / h0 as f64;
    let initial: Vec<f32> = (0..size)
        .map(|x| {
            let mut sum = 0.;
            for p in 0..=8 {
                let at = (x as f64 + p as f64 / 8.) * cell as f64;
                let w = if p == 0 || p == 8 {
                    1.
                } else if p % 2 == 0 {
                    2.
                } else {
                    4.
                };
                sum += w * 0.01 * (-0.5 * ((at - 2.) / sigma).powi(2)).exp();
            }
            (sum / 24.) as f32
        })
        .collect();
    let exact = |x: usize, t: f64| {
        // Exact semidiscrete donor-cell solution, independent of Euler steps.
        let lambda = speed * t / cell as f64;
        let mut weight = (-lambda).exp();
        let mut sum = 0.;
        for j in 0..=x {
            sum += weight * initial[x - j] as f64;
            weight *= lambda / (j + 1) as f64;
        }
        sum * (-k * t).exp() * h0 as f64
    };
    let mut errors = Vec::new();
    for steps in [4, 8, 16] {
        let dt = total / steps as f64;
        let h = GpuHeightfield::new(
            &f.device,
            size as u32,
            size as u32,
            cell,
            0.,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        for b in [
            &h.geology.sediment,
            &h.geology.overburden,
            &h.geology.gravel,
            &h.geology.paydirt,
        ] {
            f.write(b, &vec![0.125; n]);
        }
        f.write(&h.water.depth, &vec![h0; n]);
        f.write(
            &h.water.suspended_paydirt,
            &(0..n).map(|i| initial[i % size]).collect::<Vec<_>>(),
        );
        let q = (speed * h0 as f64 * cell as f64 * dt) as f32;
        let flow: Vec<f32> = (0..n)
            .map(|i| if i % size + 1 == size { 0. } else { q })
            .collect();
        f.write(&h.water.flux_x, &flow);
        let mut aux = vec![0.; n * 13];
        for i in 0..n {
            aux[n + i * 12] = h0;
            if i % size + 1 == size {
                aux[n + i * 12 + 2] = q;
            }
        }
        f.write(&h.water.suspended_sediment_next, &aux);
        h.update_water_params(&f.queue, dt as f32, 2, 0., 1);
        let mut encoder = f.device.create_command_encoder(&Default::default());
        for _ in 0..steps {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&h.simulation.sediment_transport);
                pass.set_bind_group(0, &h.bind_groups.params, &[]);
                pass.set_bind_group(1, &h.bind_groups.water, &[]);
                pass.set_bind_group(2, &h.bind_groups.terrain, &[]);
                pass.dispatch_workgroups(4, 4, 1);
            }
            for (next, current) in [
                (
                    &h.water.suspended_sediment_next,
                    &h.water.suspended_sediment,
                ),
                (
                    &h.water.suspended_overburden_next,
                    &h.water.suspended_overburden,
                ),
                (&h.water.suspended_gravel_next, &h.water.suspended_gravel),
                (&h.water.suspended_paydirt_next, &h.water.suspended_paydirt),
            ] {
                encoder.copy_buffer_to_buffer(next, 0, current, 0, (n * 4) as u64);
            }
        }
        f.queue.submit(Some(encoder.finish()));
        let c = f.read_range(&h.water.suspended_paydirt, 0, n);
        let bed = f.read_range(&h.geology.paydirt, 0, n);
        let low = f.read_range(&h.geology.bed_remainder, 0, n * 4);
        let mut mobile_error = 0.;
        let mut bed_error = 0.;
        let mut mobile_ref = 0.;
        let mut bed_ref = 0.;
        for x in 0..size {
            let i = (size / 2) * size + x;
            let m = exact(x, total);
            let mut b = 0.;
            for p in 0..=256 {
                let t = total * p as f64 / 256.;
                let w = if p == 0 || p == 256 {
                    1.
                } else if p % 2 == 0 {
                    2.
                } else {
                    4.
                };
                b += w * k * exact(x, t);
            }
            b *= total / (3. * 256.);
            mobile_error += (c[i] as f64 * h0 as f64 - m).abs();
            bed_error += (bed[i] as f64 - 0.125 + low[i * 4 + 3] as f64 - b).abs();
            mobile_ref += m;
            bed_ref += b;
        }
        let m = mobile_error / mobile_ref;
        let b = bed_error / bed_ref;
        let initial_mass = initial
            .iter()
            .map(|c| *c as f64 * h0 as f64 * area * size as f64)
            .sum::<f64>();
        let final_mass = (0..n)
            .map(|i| {
                ((c[i] as f64 * h0 as f64) + (bed[i] as f64 - 0.125) + low[i * 4 + 3] as f64) * area
            })
            .sum::<f64>();
        assert!((final_mass - initial_mass).abs() < initial_mass * 1e-6);
        println!("ADR TIME fixed_dx={cell} dt={dt} mobile_l1={m} bed_l1={b}");
        errors.push((m, b));
    }
    assert!(
        errors[1].0 < errors[0].0 * 0.7 && errors[2].0 < errors[1].0 * 0.7,
        "fixed-grid mobile time convergence {errors:?}"
    );
    assert!(
        errors[1].1 < errors[0].1 * 0.7 && errors[2].1 < errors[1].1 * 0.7,
        "fixed-grid bed time convergence {errors:?}"
    );
}
