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
            0.25,
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
        bed.into_iter()
            .zip(c)
            .zip(h)
            .map(|((b, c), h)| (b as f64 + c as f64 * h as f64) * AREA)
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
    assert!(f
        .read(&f.h.water.suspended_sediment)
        .iter()
        .all(|v| *v == 0.1));
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
            for i in 0..CELLS {
                // The reference observes final stored bed values, then rounds
                // the independently evaluated f64 conservation equation once.
                let expected = (0.1_f32 as f64 + (0.125 - bed[i] as f64) / h[i] as f64) as f32;
                assert_eq!(
                    c[i].to_bits(),
                    expected.to_bits(),
                    "recipient rounding: class {ch}, settling {settling}, cell {i}"
                );
            }
        }
    }
}
