//! Presentation of the saved processing circuit. Wire capsules and receiving
//! sheets come directly from the contact model; animation interpolates recorded
//! drive clocks and stable particle IDs, never an independent render oscillator.
use bevy::{
    asset::RenderAssetUsages, camera::visibility::NoFrustumCulling, math::DVec3, mesh::Indices,
    prelude::*, render::render_resource::PrimitiveTopology,
};
use mining_sim::{
    screen::{Screen, ScreenConfig},
    sluice::{Grain, HOPPER_X, HOPPER_Z},
};
use std::collections::HashMap;

use crate::{app::Game, motion_capture::MotionCapture, scene::Palette, sediment::Geometry};

/// The original low hopper is retained for legacy circuits only.
#[derive(Component)]
pub(crate) struct LegacyHopper;

#[derive(Component)]
pub(crate) struct ScreenPart;

const PARTS: usize = 8;
const WIRES: usize = 0;

fn empty() -> Geometry {
    Geometry {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    }
}

fn render_position(point: [f64; 3]) -> Vec3 {
    Vec3::from_array(point.map(|coordinate| coordinate as f32))
}

fn triangle(out: &mut Geometry, points: [Vec3; 3], double_sided: bool) {
    let normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize_or_zero();
    let base = out.positions.len() as u32;
    out.positions.extend(points.map(|p| p.to_array()));
    out.normals.extend([normal.to_array(); 3]);
    out.indices.extend([base, base + 1, base + 2]);
    if double_sided {
        let base = out.positions.len() as u32;
        out.positions.extend(points.map(|p| p.to_array()));
        out.normals.extend([(-normal).to_array(); 3]);
        out.indices.extend([base, base + 2, base + 1]);
    }
}

fn quad(out: &mut Geometry, points: [Vec3; 4], double_sided: bool) {
    triangle(out, [points[0], points[1], points[2]], double_sided);
    triangle(out, [points[0], points[2], points[3]], double_sided);
}

fn cuboid(out: &mut Geometry, center: Vec3, axes: [Vec3; 3], size: Vec3) {
    let [x, y, z] = axes;
    let point = |a: f32, b: f32, c: f32| {
        center + x * (a * size.x * 0.5) + y * (b * size.y * 0.5) + z * (c * size.z * 0.5)
    };
    for face in [
        [
            point(1., -1., -1.),
            point(1., 1., -1.),
            point(1., 1., 1.),
            point(1., -1., 1.),
        ],
        [
            point(-1., -1., 1.),
            point(-1., 1., 1.),
            point(-1., 1., -1.),
            point(-1., -1., -1.),
        ],
        [
            point(-1., 1., -1.),
            point(-1., 1., 1.),
            point(1., 1., 1.),
            point(1., 1., -1.),
        ],
        [
            point(-1., -1., 1.),
            point(-1., -1., -1.),
            point(1., -1., -1.),
            point(1., -1., 1.),
        ],
        [
            point(-1., -1., 1.),
            point(1., -1., 1.),
            point(1., 1., 1.),
            point(-1., 1., 1.),
        ],
        [
            point(1., -1., -1.),
            point(-1., -1., -1.),
            point(-1., 1., -1.),
            point(1., 1., -1.),
        ],
    ] {
        // A screen's downstream/up/across frame has reflected handedness.
        // Two-sided faces preserve the outer and inner thin-sheet appearance.
        quad(out, face, true);
    }
}

fn beam(out: &mut Geometry, a: Vec3, b: Vec3, width: f32, depth: f32) {
    let along = (b - a).normalize_or_zero();
    if along == Vec3::ZERO {
        return;
    }
    let reference = if along.dot(Vec3::Y).abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let across = along.cross(reference).normalize();
    let up = across.cross(along).normalize();
    cuboid(
        out,
        (a + b) * 0.5,
        [along, up, across],
        Vec3::new(a.distance(b), depth, width),
    );
}

/// A tessellated capsule, including the rounded wire ends used by contact.
/// Vertices lie on the actual radius; sub-pixel wire visibility is left to the
/// renderer rather than enlarging the aperture geometry.
fn capsule(out: &mut Geometry, a: Vec3, b: Vec3, radius: f32) {
    const SIDES: usize = 8;
    const HEMISPHERE: usize = 3;
    let axis = (b - a).normalize_or_zero();
    if axis == Vec3::ZERO || radius <= 0. {
        return;
    }
    let reference = if axis.dot(Vec3::Y).abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let u = axis.cross(reference).normalize();
    let v = axis.cross(u);
    let base = out.positions.len() as u32;
    let mut rings = Vec::with_capacity(2 * (HEMISPHERE + 1));
    for end in [false, true] {
        for k in 0..=HEMISPHERE {
            let angle = if end {
                k as f32 / HEMISPHERE as f32 * std::f32::consts::FRAC_PI_2
            } else {
                -std::f32::consts::FRAC_PI_2
                    + k as f32 / HEMISPHERE as f32 * std::f32::consts::FRAC_PI_2
            };
            rings.push((if end { b } else { a }, angle));
        }
    }
    for (center, angle) in &rings {
        for j in 0..SIDES {
            let theta = j as f32 / SIDES as f32 * std::f32::consts::TAU;
            let radial = u * theta.cos() + v * theta.sin();
            let normal = axis * angle.sin() + radial * angle.cos();
            out.positions.push((*center + normal * radius).to_array());
            out.normals.push(normal.to_array());
        }
    }
    for ring in 0..rings.len() - 1 {
        for j in 0..SIDES {
            let next = (j + 1) % SIDES;
            let a = base + (ring * SIDES + j) as u32;
            let b = base + (ring * SIDES + next) as u32;
            let c = base + ((ring + 1) * SIDES + j) as u32;
            let d = base + ((ring + 1) * SIDES + next) as u32;
            out.indices.extend([a, b, c, b, d, c]);
        }
    }
}

#[derive(Clone, PartialEq)]
struct RestKey {
    generation: u64,
    origin: mining_sim::Vec3,
    yaw: f32,
    config: ScreenConfig,
    hopper_height: f32,
}

#[derive(Clone)]
struct ParticleSample {
    grain: Grain,
    position: DVec3,
}

struct Frame {
    drive_time: f64,
    elapsed_time: f64,
    particles: HashMap<u64, ParticleSample>,
    queued: Vec<Grain>,
}

impl Frame {
    fn from_screen(screen: &Screen) -> Self {
        Self {
            drive_time: screen.time,
            elapsed_time: screen.elapsed_time_s,
            particles: screen
                .particles
                .iter()
                .map(|p| {
                    (
                        p.id,
                        ParticleSample {
                            grain: p.grain.clone(),
                            position: DVec3::from_array(p.position.to_array()),
                        },
                    )
                })
                .collect(),
            queued: screen
                .undersize
                .iter()
                .chain(&screen.oversize)
                .chain(&screen.spills)
                .cloned()
                .collect(),
        }
    }
}

#[derive(Default)]
pub(crate) struct Visuals {
    handles: Option<[Handle<Mesh>; PARTS]>,
    entities: Vec<Entity>,
    rest_key: Option<RestKey>,
    snapshot_key: Option<(u64, u64)>,
    previous: Option<Frame>,
    current: Option<Frame>,
    received_wall: f64,
    wall_interval: f64,
    drawn: Option<(u64, u64, u64)>,
}

fn geometry(screen: &Screen, site: &mining_sim::Site) -> [Geometry; 5] {
    let mut out = std::array::from_fn(|_| empty());
    let at = |x: f64, y: f64, z: f64| render_position(screen.to_world([x, y, z].into()).to_array());
    let displacement = render_position(screen.offset_at(screen.time).0.to_array());
    for (a, b, radius) in screen.wire_segments(screen.time) {
        capsule(
            &mut out[0],
            render_position(a.to_array()) - displacement,
            render_position(b.to_array()) - displacement,
            radius as f32,
        );
    }
    for points in screen
        .gutter_triangles()
        .into_iter()
        .chain(screen.bin_triangles())
    {
        triangle(
            &mut out[1],
            points.map(|p| render_position(p.to_array())),
            true,
        );
    }
    let slope = screen.config.slope_deg.to_radians().tan();
    let length = screen.config.length_m;
    let half = screen.config.width_m * 0.5;
    // Dark beams sit outside the true screening aperture and below its wires.
    for z in [-half - 0.035, half + 0.035] {
        beam(
            &mut out[3],
            at(0., -0.035, z),
            at(length, -length * slope - 0.035, z),
            0.045,
            0.055,
        );
        for x in [0.04, length - 0.04] {
            let top = at(x, -0.10 - x * slope, z);
            let ground = site.height(top.x, top.z);
            beam(
                &mut out[3],
                Vec3::new(top.x, ground + 0.015, top.z),
                top,
                0.035,
                0.035,
            );
        }
    }
    // Motor housing and its real supporting bracket. Only the wire assembly
    // translates with its contact oscillator; the housing stays on the frame.
    let downstream = (at(1., 0., 0.) - at(0., 0., 0.)).normalize();
    let across = (at(0., 0., 1.) - at(0., 0., 0.)).normalize();
    cuboid(
        &mut out[3],
        at(length * 0.45, -0.06 - length * 0.45 * slope, half + 0.12),
        [downstream, Vec3::Y, across],
        Vec3::new(0.18, 0.11, 0.16),
    );
    cuboid(
        &mut out[2],
        at(length * 0.45, -0.065 - length * 0.45 * slope, half + 0.205),
        [downstream, Vec3::Y, across],
        Vec3::new(0.13, 0.08, 0.025),
    );
    let h = site.height(HOPPER_X, HOPPER_Z);
    let mouth = h + 2.10;
    let bottom = h + 1.62;
    let corner = |i: usize, y: f32, r: f32| {
        let (x, z) = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)][i % 4];
        Vec3::new(HOPPER_X + x * r, y, HOPPER_Z + z * r)
    };
    // Four open tapering sheets and a narrow rim; no opaque cap hides feed.
    for i in 0..4 {
        quad(
            &mut out[2],
            [
                corner(i, mouth, 0.8),
                corner(i + 1, mouth, 0.8),
                corner(i + 1, bottom, 0.175),
                corner(i, bottom, 0.175),
            ],
            true,
        );
        beam(
            &mut out[2],
            corner(i, mouth, 0.8),
            corner(i + 1, mouth, 0.8),
            0.045,
            0.045,
        );
        let top = corner(i, mouth - 0.03, 0.72);
        let ground = site.height(top.x, top.z);
        beam(
            &mut out[3],
            Vec3::new(top.x, ground + 0.015, top.z),
            top,
            0.065,
            0.065,
        );
    }
    // An open feeder chute connects the fixed hopper outlet to the actual birth
    // footprint; it is presentation of the feeder, not a second screen deck.
    let start = Vec3::new(HOPPER_X, bottom - 0.015, HOPPER_Z);
    // The lowest birth centre is y=.053; keep the open chute floor
    // below it, including the largest physical feed radius.
    let end = at(0.035, 0.035, 0.);
    let half_chute = 0.17;
    let left = across * half_chute;
    quad(
        &mut out[4],
        [start - left, end - left, end + left, start + left],
        true,
    );
    for sign in [-1., 1.] {
        let edge = left * sign;
        quad(
            &mut out[4],
            [
                start + edge,
                end + edge,
                end + edge + Vec3::Y * 0.065,
                start + edge + Vec3::Y * 0.065,
            ],
            true,
        );
    }
    out
}

fn upload(meshes: &mut Assets<Mesh>, handle: &Handle<Mesh>, data: Geometry) {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_indices(Indices::U32(data.indices));
    if let Some(existing) = meshes.get_mut(handle) {
        *existing = mesh;
    }
}

fn sampled_grains(previous: Option<&Frame>, current: &Frame, alpha: f64) -> Vec<Grain> {
    let mut particles: Vec<_> = current.particles.iter().collect();
    particles.sort_unstable_by_key(|(id, _)| **id);
    let mut out = Vec::with_capacity(particles.len() + current.queued.len());
    for (id, sample) in particles {
        let mut grain = sample.grain.clone();
        let position = if let Some(old) = previous.and_then(|frame| frame.particles.get(id)) {
            grain.rotation = old
                .grain
                .rotation
                .slerp(sample.grain.rotation, alpha as f32);
            old.position.lerp(sample.position, alpha)
        } else {
            sample.position
        };
        grain.position = mining_sim::Vec3::from_array(position.as_vec3().to_array());
        out.push(grain);
    }
    out.extend(current.queued.iter().cloned());
    out
}

/// One recorded snapshot of presentation delay smooths the high-frequency deck
/// and keeps its particles at the same physical instant. Stalls clamp to the
/// newest sample; nothing extrapolates or keeps vibrating after a paused worker.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct RenderContext<'w, 's> {
    commands: Commands<'w, 's>,
    game: Res<'w, Game>,
    capture: Res<'w, MotionCapture>,
    time: Res<'w, Time>,
    palette: Res<'w, Palette>,
    meshes: ResMut<'w, Assets<Mesh>>,
    visuals: Local<'s, Visuals>,
    parts: Query<'w, 's, (&'static mut Transform, &'static mut Visibility), With<ScreenPart>>,
    legacy: Query<'w, 's, &'static mut Visibility, (With<LegacyHopper>, Without<ScreenPart>)>,
}

pub(crate) fn sync(context: RenderContext) {
    let RenderContext {
        mut commands,
        game,
        capture,
        time,
        palette,
        mut meshes,
        mut visuals,
        mut parts,
        mut legacy,
    } = context;
    let enabled = game.session.plant.is_some() && !capture.water_only;
    for mut visibility in &mut legacy {
        *visibility = if game.session.plant.is_some() {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
    for (_, mut visibility) in &mut parts {
        *visibility = if enabled {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    let Some(plant) = game.session.plant.as_ref().filter(|_| enabled) else {
        visuals.rest_key = None;
        visuals.snapshot_key = None;
        visuals.previous = None;
        visuals.current = None;
        visuals.drawn = None;
        return;
    };
    let screen = &plant.screen;
    if visuals.handles.is_none() {
        let materials = [
            palette.chrome.clone(),
            palette.steel.clone(),
            palette.yellow.clone(),
            palette.dark.clone(),
            palette.steel.clone(),
            palette.dirt.clone(),
            palette.sand.clone(),
            palette.gold.clone(),
        ];
        let mut entities = Vec::with_capacity(PARTS);
        let handles = std::array::from_fn(|i| {
            let handle = meshes.add(Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            ));
            let transform = if i == WIRES {
                Transform::from_translation(render_position(
                    screen.offset_at(screen.time).0.to_array(),
                ))
            } else {
                Transform::default()
            };
            let entity = commands
                .spawn((
                    Mesh3d(handle.clone()),
                    MeshMaterial3d(materials[i].clone()),
                    transform,
                    NoFrustumCulling,
                    ScreenPart,
                ))
                .id();
            entities.push(entity);
            handle
        });
        visuals.entities = entities;
        visuals.handles = Some(handles);
    }
    let key = RestKey {
        generation: game.simulation_generation,
        origin: screen.origin,
        yaw: screen.yaw,
        config: screen.config.clone(),
        hopper_height: game.session.site.height(HOPPER_X, HOPPER_Z),
    };
    if visuals.rest_key.as_ref() != Some(&key) {
        for (handle, data) in visuals
            .handles
            .as_ref()
            .unwrap()
            .iter()
            .zip(geometry(screen, &game.session.site))
        {
            upload(&mut meshes, handle, data);
        }
        visuals.rest_key = Some(key);
        visuals.snapshot_key = None;
        visuals.previous = None;
        visuals.current = None;
        visuals.drawn = None;
    }
    let wall = time.elapsed_secs_f64();
    let snapshot_key = (game.simulation_generation, game.snapshot_id);
    if visuals.snapshot_key != Some(snapshot_key) {
        let sample = Frame::from_screen(screen);
        let monotone = visuals.current.as_ref().is_some_and(|old| {
            sample.elapsed_time > old.elapsed_time && sample.drive_time >= old.drive_time
        });
        visuals.wall_interval = if monotone {
            (wall - visuals.received_wall).max(0.)
        } else {
            0.
        };
        visuals.previous = if monotone {
            visuals.current.take()
        } else {
            None
        };
        visuals.current = Some(sample);
        visuals.received_wall = wall;
        visuals.snapshot_key = Some(snapshot_key);
    }
    let latest = game.session.paused
        || capture.frozen
        || visuals.previous.is_none()
        || visuals.wall_interval <= 0.;
    let alpha = if latest {
        1.
    } else {
        ((wall - visuals.received_wall) / visuals.wall_interval).clamp(0., 1.)
    };
    let draw_key = (
        game.simulation_generation,
        game.snapshot_id,
        alpha.to_bits(),
    );
    if visuals.drawn == Some(draw_key) {
        return;
    }
    let current = visuals.current.as_ref().unwrap();
    let drive_time = if screen.motor_on {
        visuals.previous.as_ref().map_or(current.drive_time, |old| {
            old.drive_time + (current.drive_time - old.drive_time) * alpha
        })
    } else {
        current.drive_time
    };
    let offset = render_position(screen.offset_at(drive_time).0.to_array());
    if let Ok((mut transform, _)) = parts.get_mut(visuals.entities[WIRES]) {
        transform.translation = offset;
    }
    let grains = sampled_grains(visuals.previous.as_ref(), current, alpha);
    for (handle, data) in visuals.handles.as_ref().unwrap()[5..]
        .iter()
        .zip(crate::sediment::world_geometry(grains.iter()))
    {
        upload(&mut meshes, handle, data);
    }
    visuals.drawn = Some(draw_key);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_proxy_vertices_preserve_the_contact_capsule_radius() {
        let a = Vec3::new(1., 2., 3.);
        let b = Vec3::new(1.8, 1.83, 3.);
        let radius = 0.001;
        let mut mesh = empty();
        capsule(&mut mesh, a, b, radius);
        for point in mesh.positions {
            let p = Vec3::from_array(point);
            let fraction = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0., 1.);
            assert!((p.distance(a.lerp(b, fraction)) - radius).abs() < 3e-7);
        }
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn presentation_uses_the_same_open_receiving_triangles_and_wire_count() {
        let screen = Screen::new(mining_sim::Vec3::new(17., 2., 18.), 0.7);
        let site = mining_sim::Site::default();
        let mesh = geometry(&screen, &site);
        let contact: Vec<_> = screen
            .gutter_triangles()
            .into_iter()
            .chain(screen.bin_triangles())
            .collect();
        assert_eq!(mesh[1].positions.len(), contact.len() * 6);
        for (rendered, triangle) in mesh[1].positions.chunks_exact(6).zip(contact) {
            assert_eq!(&rendered[..3], &triangle.map(|p| p.as_vec3().to_array()));
        }
        assert_eq!(
            mesh[0].positions.len(),
            screen.wire_segments(screen.time).len() * 8 * 8
        );
    }
}
