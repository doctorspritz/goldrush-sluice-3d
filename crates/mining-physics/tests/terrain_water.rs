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
    fn collapse(&self, tile: [u32; 4], dt: f32) {
        self.h
            .update_params_tile(&self.queue, dt, tile[0], tile[1], tile[2], tile[3]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.h
            .dispatch_collapse_tile(&mut encoder, tile[2], tile[3]);
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
fn dry_collapse_preserves_material_when_many_donors_share_receivers() {
    let f = Fixture::new();
    let initial: Vec<f32> = (0..CELLS)
        .map(|i| {
            if ((i % N) + (i / N)).is_multiple_of(2) {
                2.
            } else {
                0.
            }
        })
        .collect();
    f.write(&f.h.geology.overburden, &initial);
    let before = volume(&initial);
    for _ in 0..20 {
        f.step(0);
    }
    let after = f.read(&f.h.geology.overburden, 0, CELLS);
    let total = volume(&after);
    println!(
        "dry collapse volume: before={before} after={total} residual={}",
        total - before
    );
    assert!(after.iter().all(|v| v.is_finite() && *v >= 0.));
    assert!(
        (total - before).abs() < before * 1e-6,
        "collapse lost {} m3",
        before - total
    );
}

// Independent f64 snapshot/gather reference for a single exposed overburden
// layer. Compare every cell, so a conserved total cannot hide wrong directions.
fn reference_collapse(old: &[f32], tile: [u32; 4], dt: f64) -> Vec<f64> {
    let mut next: Vec<f64> = old.iter().map(|v| *v as f64).collect();
    for z in tile[1] as usize..(tile[1] + tile[3]) as usize {
        for x in tile[0] as usize..(tile[0] + tile[2]) as usize {
            if x >= N || z >= N {
                continue;
            }
            let i = z * N + x;
            let source = old[i] as f64;
            if source <= 0.001 {
                continue;
            }
            let neighbours: Vec<_> = (-1i32..=1)
                .flat_map(|dz| (-1i32..=1).map(move |dx| (dx, dz)))
                .filter(|(dx, dz)| *dx != 0 || *dz != 0)
                .filter_map(|(dx, dz)| {
                    let (nx, nz) = (x as i32 + dx, z as i32 + dz);
                    if !(0..N as i32).contains(&nx) || !(0..N as i32).contains(&nz) {
                        return None;
                    }
                    let j = nz as usize * N + nx as usize;
                    let distance = ((dx * dx + dz * dz) as f64).sqrt();
                    Some((j, (source - old[j] as f64).max(0.), distance))
                })
                .collect();
            let total: f64 = neighbours
                .iter()
                .map(|(_, difference, distance)| difference / distance)
                .sum();
            if total <= 0. {
                continue;
            }
            for (j, difference, distance) in neighbours {
                let weight = difference / distance / total;
                let excess = (difference - 0.7002 * 0.25 * distance).max(0.) * 0.5;
                let amount = (excess * weight * 4. * dt).min(source * 0.5 * weight);
                next[i] -= amount;
                next[j] += amount;
            }
        }
    }
    next
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn collapse_gathers_across_tile_halos_and_ignores_stale_donors() {
    let f = Fixture::new();
    let mut current = vec![0.; CELLS];
    current[15 * N + 15] = 2.;
    current[0] = 1.;
    current[CELLS - 1] = 1.5;
    f.write(&f.h.geology.overburden, &current);
    let initial = volume(&current);
    for tile in [
        [15, 15, 1, 1],
        [16, 15, 1, 1],
        [0, 0, 2, 2],
        [31, 31, 1, 1],
        [14, 14, 5, 4],
    ] {
        let expected = reference_collapse(&current, tile, 0.05);
        f.collapse(tile, 0.05);
        let next = f.read(&f.h.geology.overburden, 0, CELLS);
        let max_error = next
            .iter()
            .zip(expected)
            .map(|(v, e)| (*v as f64 - e).abs())
            .fold(0., f64::max);
        assert!(max_error < 5e-7, "tile {tile:?}: {max_error}");
        assert!(
            (volume(&next) - initial).abs() < 1e-7,
            "tile {tile:?} volume"
        );
        if tile == [15, 15, 1, 1] {
            for (dx, dz) in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                assert!(
                    next[(15 + dz) as usize * N + (15 + dx) as usize] > 0.,
                    "halo receiver missing"
                );
            }
        }
        current = next;
    }
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn collapse_preserves_each_layer_and_respects_repose_and_burial() {
    let f = Fixture::new();
    let layers = [
        &f.h.geology.sediment,
        &f.h.geology.overburden,
        &f.h.geology.gravel,
    ];
    for (layer, angle) in layers.into_iter().zip([0.6249, 0.7002, 0.7813]) {
        for slope in [angle - 0.01, angle + 0.2] {
            for b in layers {
                f.write(b, &vec![0.; CELLS]);
            }
            let initial: Vec<_> = (0..CELLS)
                .map(|i| 1. + (i % N) as f32 * 0.25 * slope)
                .collect();
            f.write(layer, &initial);
            for _ in 0..20 {
                f.collapse([0, 0, N as u32, N as u32], 0.05);
            }
            let next = f.read(layer, 0, CELLS);
            assert!(next.iter().all(|v| v.is_finite() && *v >= 0.));
            assert!((volume(&next) - volume(&initial)).abs() < volume(&initial) * 1e-6);
            let change = next
                .iter()
                .zip(&initial)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            if slope < angle {
                assert_eq!(change, 0.);
            } else {
                assert!(change > 0.001);
            }
        }
    }
    let checker: Vec<f32> = (0..CELLS)
        .map(|i| {
            if ((i % N) + (i / N)).is_multiple_of(2) {
                1.
            } else {
                0.
            }
        })
        .collect();
    for b in [
        &f.h.geology.bedrock,
        &f.h.geology.paydirt,
        &f.h.geology.gravel,
        &f.h.geology.overburden,
    ] {
        f.write(b, &checker);
    }
    f.write(&f.h.geology.sediment, &vec![0.1; CELLS]);
    // An exposed sediment cover cannot let buried material collapse through it.
    f.collapse([0, 0, N as u32, N as u32], 0.05);
    for b in [
        &f.h.geology.bedrock,
        &f.h.geology.paydirt,
        &f.h.geology.gravel,
        &f.h.geology.overburden,
    ] {
        assert_eq!(f.read(b, 0, CELLS), checker);
    }
    let before: Vec<_> = layers
        .iter()
        .map(|b| volume(&f.read(b, 0, CELLS)))
        .collect();
    for _ in 0..120 {
        f.collapse([0, 0, N as u32, N as u32], 0.05);
    }
    for (b, initial) in layers.into_iter().zip(before) {
        let next = f.read(b, 0, CELLS);
        assert!(next.iter().all(|v| v.is_finite() && *v >= 0.));
        assert!((volume(&next) - initial).abs() < initial * 1e-6);
    }
}

#[test]
#[ignore = "requires native GPU; run explicitly with --ignored --test-threads=1"]
fn collapse_shape_converges_as_the_timestep_decreases() {
    let f = Fixture::new();
    let mut pile = vec![0.; CELLS];
    for z in 12..20 {
        for x in 12..20 {
            pile[z * N + x] = 2.;
        }
    }
    let mut profiles = Vec::new();
    for steps in [40, 80, 160] {
        f.write(&f.h.geology.overburden, &pile);
        for _ in 0..steps {
            f.collapse([0, 0, N as u32, N as u32], 2. / steps as f32);
        }
        let next = f.read(&f.h.geology.overburden, 0, CELLS);
        assert!((volume(&next) - volume(&pile)).abs() < volume(&pile) * 1e-6);
        profiles.push(next);
    }
    let difference = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs() * 0.0625)
            .sum::<f64>()
            / volume(&pile)
    };
    let coarse = difference(&profiles[0], &profiles[1]);
    let fine = difference(&profiles[1], &profiles[2]);
    println!("collapse normalized shape error: dt .05/.025={coarse}, .025/.0125={fine}");
    assert!(
        coarse < 0.03 && fine < coarse * 0.8,
        "timestep convergence {coarse} -> {fine}"
    );
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
            for a in f.aux().as_chunks::<12>().0 {
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
