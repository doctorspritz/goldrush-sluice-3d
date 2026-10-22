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

// Independent f64 evaluation of the documented size/density terminal-velocity
// and current flat-bed deposition law. The deliberately retained 1mm gate and
// per-step calm boost are separate physical-model work, not this race fix.
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
fn settling_reads_one_immutable_pre_stage_slope_across_workgroups() {
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
        let shields = 1000. * 0.003 * 0.25_f64.powi(2) / (9.81 * 1650. * d);
        let threshold = 0.045 * [0.4, 0.5, 0.8, 0.6][ch];
        let deposit = if shields < threshold {
            let frac = (settling_speed(d) * 0.05 / 0.5).min([0.25, 0.25, 0.35, 0.25][ch]);
            let request = cs[ch] as f64 * frac * 0.5;
            if request > 0.001 {
                request
            } else {
                0.
            }
        } else {
            0.
        };
        let expected = 0.0625 + deposit;
        for actual in f.read(f.layers()[ch]) {
            maximum = maximum.max((actual as f64 - expected).abs());
        }
    }
    println!("flat-bed snapshot reference maximum layer error: {maximum:.12e} m");
    assert!(
        maximum < 2e-7,
        "neighbour writes changed the deposition slope: {maximum}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn settling_and_erosion_transfer_actual_stored_layer_changes() {
    let f = Fixture::new();
    let mut maximum = 0.0_f64;
    for channel in 0..4 {
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
        f.stage(true, [0, 0, N as u32, N as u32], 0.05);
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
            let times: Vec<u32> = (0..CELLS).map(|i| (i % 31) as u32).collect();
            f.queue
                .write_buffer(&f.h.geology.settling_time, 0, bytemuck::cast_slice(&times));
            f.surface([4, 3, 2, 1][ch]);
            f.stage(false, [0, 0, N as u32, N as u32], 0.05);
            let bed = f.read(f.layers()[ch]);
            let concentration = f.read(f.concentrations()[ch]);
            for i in 0..CELLS {
                let stress = 1000. * (9.81 * 0.02 * slope(&ground, i) + 0.003 * 0.1_f64.powi(2));
                let shields = stress / (9.81 * 1650. * d);
                let critical = 0.045 * (1. + 0.5 * (times[i] as f64 / 30.).min(1.));
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
    for ch in 0..4 {
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
        }
        let error = f.mass(ch) - initial;
        println!(
            "material {ch}: 20000 stage residual {error:.12e} m3, mobile fraction {:.6e}",
            error / mobile
        );
        assert!(
            error.abs() < mobile * 1e-6,
            "long exchange budget: {error} / {mobile}"
        );
    }
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
    f.stage(true, [0, 0, N as u32, N as u32], 0.05);
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
