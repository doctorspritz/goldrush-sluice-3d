//! Raw transport controls: preserve every bit, padding, bounded prefix and
//! untouched tail before running any physical shader.
use game::gpu::{flip_3d::GpuFlip3D, p2g_3d::GpuP2g3D};
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}
fn words(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raw particle transport control"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}
fn vector(i: usize) -> Vec3 {
    // Transport must also retain signed zero and payload bits. No physical
    // shader is dispatched on these deliberately nonphysical sentinel values.
    Vec3::new(
        if i == 0 { -0. } else { i as f32 * 0.125 },
        if i == 2 {
            f32::from_bits(0x7fc0_0abc)
        } else {
            -(i as f32) * 0.75
        },
        i as f32 + 0.375,
    )
}
fn matrix(i: usize) -> Mat3 {
    Mat3::from_cols(vector(i), vector(i + 1), vector(i + 2))
}
fn packed(v: Vec3, w: f32) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), w.to_bits()]
}

#[test]
#[ignore = "requires actual GPU; explicitly run with --ignored"]
fn full_upload_preserves_raw_planes_short_inputs_and_untouched_tails() {
    let (device, queue) = device();
    let capacity = 64;
    let mut expected: Vec<Vec<u32>> = (0..6)
        .map(|plane| vec![0xa5a5_a5a5; capacity * if plane == 2 { 1 } else { 4 }])
        .collect();
    let buffers: Vec<_> = expected
        .iter()
        .map(|data| {
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("seeded upload plane"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }),
            )
        })
        .collect();
    let mut p2g = GpuP2g3D::new(
        &device,
        4,
        4,
        4,
        capacity,
        true,
        false,
        buffers[0].clone(),
        buffers[1].clone(),
        buffers[2].clone(),
        buffers[3].clone(),
        buffers[4].clone(),
        buffers[5].clone(),
    );
    for (stage, (np, nv, nc, nd, nw)) in [
        (64, 64, 64, 64, 64),
        (7, 4, 3, 2, 5),
        (0, 0, 0, 0, 0),
        (83, 80, 83, 88, 2),
        (1, 1, 1, 1, 1),
        (33, 0, 0, 0, 0),
        (0, 4, 3, 5, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let p: Vec<_> = (0..np).map(vector).collect();
        let v: Vec<_> = (0..nv).map(|i| vector(i + stage)).collect();
        let c: Vec<_> = (0..nc).map(|i| matrix(i + stage)).collect();
        let d: Vec<_> = (0..nd).map(|i| 1. + i as f32 * 0.25).collect();
        p2g.volume_weights = (0..nw).map(|i| i as f32 * 0.375).collect();
        let n = np.min(capacity);
        assert_eq!(p2g.upload_particles(&queue, &p, &v, &d, &c, 0.01), n as u32);
        for i in 0..n {
            expected[0][i * 4..i * 4 + 4].copy_from_slice(&packed(
                p[i],
                p2g.volume_weights.get(i).copied().unwrap_or(1.),
            ));
        }
        for i in 0..nv.min(n) {
            expected[1][i * 4..i * 4 + 4].copy_from_slice(&packed(v[i], 0.));
        }
        for i in 0..nd.min(capacity) {
            expected[2][i] = d[i].to_bits();
        }
        for i in 0..nc.min(n) {
            for axis in 0..3 {
                expected[axis + 3][i * 4..i * 4 + 4].copy_from_slice(&packed(c[i].col(axis), 0.));
            }
        }
        for (plane, buffer) in buffers.iter().enumerate() {
            assert_eq!(
                words(&device, &queue, buffer),
                expected[plane],
                "stage {stage}, plane {plane}"
            );
        }
    }
}

#[test]
#[ignore = "requires actual GPU; explicitly run with --ignored"]
fn range_upload_preserves_prefix_suffix_affine_columns_and_volume_defaults() {
    let (device, queue) = device();
    let mut sim = GpuFlip3D::new(&device, 4, 4, 4, 0.01, 64);
    let mut p = vec![Vec3::ZERO; 64];
    let mut v = p.clone();
    let mut c = vec![Mat3::ZERO; 64];
    let weights: Vec<_> = (0..23).map(|i| 0.125 + i as f32 * 0.25).collect();
    sim.set_particle_volumes(weights.clone());
    sim.upload_particle_range(&queue, 0, &p, &v, &c);
    for (stage, (first, np, nv, nc)) in
        [(13, 21, 17, 11), (63, 1, 1, 1), (0, 4, 7, 6), (64, 0, 0, 0)]
            .into_iter()
            .enumerate()
    {
        let positions: Vec<_> = (0..np).map(|i| vector(i + stage)).collect();
        let velocities: Vec<_> = (0..nv).map(|i| vector(i + stage + 1)).collect();
        let affine: Vec<_> = (0..nc).map(|i| matrix(i + stage)).collect();
        sim.upload_particle_range(&queue, first, &positions, &velocities, &affine);
        if np > 0 {
            p[first..first + np].copy_from_slice(&positions);
            v[first..first + nv].copy_from_slice(&velocities);
            c[first..first + nc].copy_from_slice(&affine);
        }
        assert!(sim.request_particle_range(&device, &queue, 0, 64));
        device.poll(wgpu::Maintain::Wait);
        let mut got_p = vec![Vec3::ZERO; 64];
        let mut got_v = got_p.clone();
        let mut got_c = vec![Mat3::ZERO; 64];
        assert_eq!(
            sim.try_readback(&device, &mut got_p, &mut got_v, &mut got_c),
            Some(64)
        );
        let bits = |v: &[Vec3]| {
            v.iter()
                .flat_map(|v| v.to_array().map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&p), bits(&got_p), "positions stage {stage}");
        assert_eq!(bits(&v), bits(&got_v), "velocities stage {stage}");
        let bits = |v: &[Mat3]| {
            v.iter()
                .flat_map(|v| v.to_cols_array().map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&c), bits(&got_c), "affine stage {stage}");
        let expected: Vec<_> = p
            .iter()
            .enumerate()
            .flat_map(|(i, p)| packed(*p, weights.get(i).copied().unwrap_or(1.)))
            .collect();
        assert_eq!(
            words(&device, &queue, &sim.positions_buffer),
            expected,
            "volume/padding stage {stage}"
        );
    }
}
