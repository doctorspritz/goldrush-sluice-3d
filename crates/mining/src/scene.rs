use crate::app::{bvec, Game, GameCamera};
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::tonemapping::Tonemapping,
    light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use mining_sim::{
    site::{CELL, WIDTH},
    sluice::{HOPPER_X, HOPPER_Z},
    ControlMode, Site, Vec3 as SVec,
};

#[derive(Resource)]
pub struct Palette {
    pub cube: Handle<Mesh>,
    pub sphere: Handle<Mesh>,
    pub cylinder: Handle<Mesh>,
    pub yellow: Handle<StandardMaterial>,
    pub dark: Handle<StandardMaterial>,
    pub steel: Handle<StandardMaterial>,
    pub chrome: Handle<StandardMaterial>,
    pub glass: Handle<StandardMaterial>,
    pub dirt: Handle<StandardMaterial>,
    pub sand: Handle<StandardMaterial>,
    pub gold: Handle<StandardMaterial>,
    pub water: Handle<StandardMaterial>,
    pub blue: Handle<StandardMaterial>,
    pub skin: Handle<StandardMaterial>,
    pub white: Handle<StandardMaterial>,
}
#[derive(Component)]
pub(crate) struct TerrainChunk {
    x: usize,
    z: usize,
    revision: u64,
}
#[derive(Component)]
pub(crate) struct WaterSurface;
#[derive(Clone, Copy)]
enum Mount {
    Chassis,
    Upper,
    Bucket,
}
#[derive(Component)]
pub(crate) struct Mounted {
    mount: Mount,
    local: Transform,
}
#[derive(Component)]
pub(crate) struct Link {
    kind: u8,
    offset: f32,
    width: f32,
}
#[derive(Component)]
pub(crate) struct BucketFill;
#[derive(Component)]
pub(crate) struct PersonPart {
    local: Transform,
    limb: u8,
}
#[derive(Component)]
pub(crate) struct SluiceVisual;
#[derive(Component)]
pub(crate) struct HopperFill;
#[derive(Component)]
pub(crate) struct TailingsPile;
#[derive(Component)]
pub(crate) struct FallingVisual {
    index: usize,
}
#[derive(Resource, Default)]
pub(crate) struct PlantRevision(
    Option<(SVec, f32, f32, mining_sim::sluice::HydraulicProfile, bool)>,
);

fn material(
    materials: &mut Assets<StandardMaterial>,
    color: Color,
    roughness: f32,
    metallic: f32,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        perceptual_roughness: roughness,
        metallic,
        ..default()
    })
}
fn box_part(
    commands: &mut Commands,
    p: &Palette,
    pos: Vec3,
    size: Vec3,
    mat: Handle<StandardMaterial>,
) -> Entity {
    commands
        .spawn((
            Mesh3d(p.cube.clone()),
            MeshMaterial3d(mat),
            Transform::from_translation(pos).with_scale(size),
        ))
        .id()
}
fn mounted(
    commands: &mut Commands,
    p: &Palette,
    mount: Mount,
    position: Vec3,
    size: Vec3,
    mat: Handle<StandardMaterial>,
    rotation: Quat,
) {
    let local = Transform::from_translation(position)
        .with_scale(size)
        .with_rotation(rotation);
    commands.spawn((
        Mesh3d(p.cube.clone()),
        MeshMaterial3d(mat),
        local,
        Mounted { mount, local },
    ));
}
fn cylinder_between(
    commands: &mut Commands,
    p: &Palette,
    a: Vec3,
    b: Vec3,
    r: f32,
    mat: Handle<StandardMaterial>,
) {
    let delta = b - a;
    commands.spawn((
        Mesh3d(p.cylinder.clone()),
        MeshMaterial3d(mat),
        Transform::from_translation((a + b) * 0.5)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize_or_zero()))
            .with_scale(Vec3::new(r, delta.length(), r)),
    ));
}

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    game: Res<Game>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.70, 0.79, 0.83)));
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.75, 0.84, 1.0),
        brightness: 650.0,
        ..default()
    });
    commands.insert_resource(DirectionalLightShadowMap { size: 2048 });
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-20.0, 35.0, -15.0).looking_at(Vec3::new(17.0, 0.0, 17.0), Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 16.0,
            maximum_distance: 80.0,
            num_cascades: 3,
            ..default()
        }
        .build(),
    ));
    commands.spawn((
        Camera3d {
            depth_texture_usages: (bevy::render::render_resource::TextureUsages::RENDER_ATTACHMENT
                | bevy::render::render_resource::TextureUsages::TEXTURE_BINDING)
                .into(),
            ..default()
        },
        Msaa::Off,
        Tonemapping::TonyMcMapface,
        Transform::from_xyz(28.0, 6.0, 17.0).looking_at(Vec3::new(19.0, 1.5, 18.0), Vec3::Y),
        DistanceFog {
            color: Color::srgb(0.70, 0.79, 0.83),
            falloff: FogFalloff::Linear {
                start: 35.0,
                end: 115.0,
            },
            ..default()
        },
        GameCamera,
    ));
    let mut glass = StandardMaterial {
        base_color: Color::srgba(0.33, 0.62, 0.68, 0.22),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.07,
        metallic: 0.15,
        cull_mode: None,
        ..default()
    };
    glass.reflectance = 0.8;
    let p = Palette {
        cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        sphere: meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap()),
        cylinder: meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(12)),
        yellow: material(&mut materials, Color::srgb(0.91, 0.52, 0.075), 0.38, 0.35),
        dark: material(&mut materials, Color::srgb(0.055, 0.065, 0.066), 0.75, 0.35),
        steel: material(&mut materials, Color::srgb(0.29, 0.35, 0.35), 0.37, 0.75),
        chrome: material(&mut materials, Color::srgb(0.63, 0.70, 0.72), 0.20, 0.9),
        glass: materials.add(glass),
        dirt: material(&mut materials, Color::srgb(0.29, 0.17, 0.085), 0.96, 0.0),
        sand: material(&mut materials, Color::srgb(0.50, 0.38, 0.23), 0.9, 0.0),
        gold: material(&mut materials, Color::srgb(0.98, 0.66, 0.10), 0.23, 0.8),
        water: materials.add(StandardMaterial {
            base_color: Color::srgba(0.11, 0.31, 0.30, 0.73),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.13,
            metallic: 0.18,
            reflectance: 0.65,
            cull_mode: None,
            ..default()
        }),
        blue: material(&mut materials, Color::srgb(0.12, 0.23, 0.28), 0.85, 0.0),
        skin: material(&mut materials, Color::srgb(0.64, 0.40, 0.24), 0.8, 0.0),
        white: material(&mut materials, Color::srgb(0.88, 0.84, 0.71), 0.8, 0.0),
    };
    let terrain_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.96,
        ..default()
    });
    for z in (0..WIDTH - 1).step_by(16) {
        for x in (0..WIDTH - 1).step_by(16) {
            let mesh = terrain_mesh(&game.session.site, x, z);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(terrain_mat.clone()),
                Transform::default(),
                TerrainChunk {
                    x,
                    z,
                    revision: u64::MAX,
                },
            ));
        }
    }
    commands.spawn((
        Mesh3d(meshes.add(water_mesh(&game.session.site, 0.0))),
        MeshMaterial3d(p.water.clone()),
        Transform::default(),
        WaterSurface,
    ));
    // Larger silhouettes conceal the finite claim edge without expanding simulation cost.
    let hill_mat = material(&mut materials, Color::srgb(0.24, 0.31, 0.23), 1.0, 0.0);
    for i in 0..16 {
        let a = i as f32 * std::f32::consts::TAU / 16.0;
        let position = Vec3::new(16.0 + a.sin() * 55.0, -1.0, 16.0 + a.cos() * 55.0);
        commands.spawn((
            Mesh3d(p.sphere.clone()),
            MeshMaterial3d(hill_mat.clone()),
            Transform::from_translation(position).with_scale(Vec3::new(
                19.0 + (i % 3) as f32 * 5.0,
                9.0 + (i % 4) as f32 * 3.0,
                19.0,
            )),
        ));
    }
    vegetation(
        &mut commands,
        &mut meshes,
        &mut materials,
        &p,
        &game.session.site,
    );
    machine(&mut commands, &p);
    person(&mut commands, &p);
    equipment(&mut commands, &mut meshes, &p, &game.session.site);
    commands.insert_resource(PlantRevision::default());
    commands.insert_resource(p);
}

pub fn terrain_mesh(site: &Site, x0: usize, z0: usize) -> Mesh {
    let nx = 16.min(WIDTH - 1 - x0);
    let nz = 16.min(WIDTH - 1 - z0);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for z in 0..=nz {
        for x in 0..=nx {
            let ix = x0 + x;
            let iz = z0 + z;
            let px = ix as f32 * CELL;
            let pz = iz as f32 * CELL;
            let y = site.world.ground_height(ix, iz);
            positions.push([px, y, pz]);
            let normal = Vec3::new(
                site.height(px - 0.12, pz) - site.height(px + 0.12, pz),
                0.24,
                site.height(px, pz - 0.12) - site.height(px, pz + 0.12),
            )
            .normalize();
            normals.push(normal.to_array());
            uvs.push([px, pz]);
            let i = site.world.idx(ix, iz);
            let wet = site.world.water_depth(ix, iz) > 0.015;
            let mut c = if site.world.terrain_sediment[i] > 0.015 {
                [0.43, 0.34, 0.22, 1.0]
            } else if site.world.overburden_thickness[i] > 0.025 {
                [0.34, 0.29, 0.18, 1.0]
            } else if site.world.paydirt_thickness[i] > 0.01 {
                [0.58, 0.43, 0.24, 1.0]
            } else {
                [0.30, 0.32, 0.30, 1.0]
            };
            let noise = 0.88 + hash(ix as u32, iz as u32) * 0.22;
            for v in &mut c[..3] {
                *v *= noise * if wet { 0.67 } else { 1.0 };
            }
            if site.world.overburden_thickness[i] > 0.025
                && !((11.0..25.0).contains(&px) && (12.0..26.0).contains(&pz))
            {
                c = [0.20 * noise, 0.29 * noise, 0.15 * noise, 1.0];
            }
            colors.push(Color::srgb(c[0], c[1], c[2]).to_linear().to_f32_array());
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            let i = (z * (nx + 1) + x) as u32;
            let w = (nx + 1) as u32;
            indices.extend_from_slice(&[i, i + w, i + 1, i + 1, i + w, i + w + 1]);
        }
    }
    let mut m = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    m.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    m.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    m.insert_indices(Indices::U32(indices));
    m
}
fn water_mesh(site: &Site, time: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            let px = x as f32 * CELL;
            let pz = z as f32 * CELL;
            let depth = site.world.water_depth(x, z);
            let wave = (px * 8.0 + time * 1.7).sin()
                * (pz * 5.0 - time * 2.6).cos()
                * 0.009
                * depth.min(0.1)
                * 10.0;
            positions.push([
                px,
                site.world.water_surface[site.world.idx(x, z)] + 0.012 + wave,
                pz,
            ]);
            normals.push(
                Vec3::new(
                    (px * 8.0 + time * 1.7).cos() * 0.05,
                    1.0,
                    (pz * 5.0 - time * 2.6).sin() * 0.05,
                )
                .normalize()
                .to_array(),
            );
        }
    }
    for z in 0..WIDTH - 1 {
        for x in 0..WIDTH - 1 {
            if site.world.water_depth(x, z) < 0.008 || site.world.water_depth(x + 1, z + 1) < 0.008
            {
                continue;
            }
            let i = (z * WIDTH + x) as u32;
            let w = WIDTH as u32;
            indices.extend_from_slice(&[i, i + w, i + 1, i + 1, i + w, i + w + 1]);
        }
    }
    let mut m = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    m.insert_indices(Indices::U32(indices));
    m
}
fn hash(x: u32, z: u32) -> f32 {
    let n = x
        .wrapping_mul(374761393)
        .wrapping_add(z.wrapping_mul(668265263));
    let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as f32 / u32::MAX as f32
}

fn vegetation(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    p: &Palette,
    site: &Site,
) {
    let bark = material(materials, Color::srgb(0.22, 0.15, 0.10), 0.95, 0.0);
    let green = material(materials, Color::srgb(0.12, 0.25, 0.18), 0.95, 0.0);
    let grass = material(materials, Color::srgb(0.30, 0.34, 0.18), 1.0, 0.0);
    let rock = material(materials, Color::srgb(0.34, 0.37, 0.34), 0.95, 0.0);
    let cone = meshes.add(Cone::new(1.0, 1.0).mesh().resolution(7));
    for i in 0..180u32 {
        let x = 1.0 + hash(i, 71) * 29.5;
        let z = 1.0 + hash(i, 23) * 29.5;
        if (x - Site::river_center(z)).abs() < 2.0
            || ((10.5..25.0).contains(&x) && (11.0..26.0).contains(&z))
        {
            continue;
        }
        let h = site.height(x, z);
        let scale = 0.7 + hash(i, 11) * 0.8;
        if i < 75 {
            commands.spawn((
                Mesh3d(p.cylinder.clone()),
                MeshMaterial3d(bark.clone()),
                Transform::from_xyz(x, h + scale * 1.5, z).with_scale(Vec3::new(
                    0.14 * scale,
                    3.0 * scale,
                    0.14 * scale,
                )),
            ));
            for tier in 0..3 {
                commands.spawn((
                    Mesh3d(cone.clone()),
                    MeshMaterial3d(green.clone()),
                    Transform::from_xyz(x, h + (2.1 + tier as f32 * 0.70) * scale, z).with_scale(
                        Vec3::new(
                            (1.1 - tier as f32 * 0.17) * scale,
                            2.0 * scale,
                            (1.1 - tier as f32 * 0.17) * scale,
                        ),
                    ),
                ));
            }
        } else {
            commands.spawn((
                Mesh3d(p.sphere.clone()),
                MeshMaterial3d(rock.clone()),
                Transform::from_xyz(x, h + scale * 0.10, z)
                    .with_rotation(Quat::from_euler(
                        EulerRot::XYZ,
                        i as f32,
                        scale,
                        i as f32 * 0.7,
                    ))
                    .with_scale(Vec3::new(scale * 0.32, scale * 0.22, scale * 0.45)),
            ));
        }
    }
    for i in 0..600u32 {
        let x = hash(i, 53) * 31.0;
        let z = hash(i, 89) * 31.0;
        if (x - Site::river_center(z)).abs() < 2.1
            || ((11.0..25.0).contains(&x) && (12.0..26.0).contains(&z))
        {
            continue;
        }
        let h = site.height(x, z);
        commands.spawn((
            Mesh3d(cone.clone()),
            MeshMaterial3d(grass.clone()),
            Transform::from_xyz(x, h + 0.15, z).with_scale(Vec3::new(
                0.12,
                0.30 + hash(i, 13) * 0.2,
                0.12,
            )),
        ));
    }
    // Claim stakes and a small service camp establish scale at ground level.
    for (x, z) in [(11.0, 12.0), (25.0, 12.0), (25.0, 26.0), (11.0, 26.0)] {
        let h = site.height(x, z);
        box_part(
            commands,
            p,
            Vec3::new(x, h + 0.5, z),
            Vec3::new(0.07, 1.0, 0.07),
            bark.clone(),
        );
        box_part(
            commands,
            p,
            Vec3::new(x + 0.16, h + 0.90, z),
            Vec3::new(0.35, 0.22, 0.02),
            p.yellow.clone(),
        );
    }
    let h = site.height(24.0, 13.0);
    box_part(
        commands,
        p,
        Vec3::new(24.0, h + 0.9, 12.8),
        Vec3::new(2.3, 1.8, 2.0),
        p.blue.clone(),
    );
    box_part(
        commands,
        p,
        Vec3::new(24.0, h + 1.9, 12.8),
        Vec3::new(2.7, 0.15, 2.4),
        p.steel.clone(),
    );
    box_part(
        commands,
        p,
        Vec3::new(24.0, h + 0.7, 13.83),
        Vec3::new(0.65, 1.4, 0.08),
        bark.clone(),
    );
    for x in [22.4, 22.9] {
        commands.spawn((
            Mesh3d(p.cylinder.clone()),
            MeshMaterial3d(p.yellow.clone()),
            Transform::from_xyz(x, h + 0.4, 13.0).with_scale(Vec3::new(0.25, 0.8, 0.25)),
        ));
    }
}

fn machine(commands: &mut Commands, p: &Palette) {
    for side in [-1.0, 1.0] {
        mounted(
            commands,
            p,
            Mount::Chassis,
            Vec3::new(side * 0.93, 0.38, 0.0),
            Vec3::new(0.55, 0.65, 3.25),
            p.dark.clone(),
            Quat::IDENTITY,
        );
        for i in 0..7 {
            let local = Transform::from_xyz(side * 1.20, 0.38, -1.28 + i as f32 * 0.43)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::new(0.24, 0.045, 0.24));
            commands.spawn((
                Mesh3d(p.cylinder.clone()),
                MeshMaterial3d(p.steel.clone()),
                local,
                Mounted {
                    mount: Mount::Chassis,
                    local,
                },
            ));
        }
        for i in 0..17 {
            for y in [0.10, 0.69] {
                mounted(
                    commands,
                    p,
                    Mount::Chassis,
                    Vec3::new(side * 0.93, y, -1.5 + i as f32 * 0.185),
                    Vec3::new(0.59, 0.055, 0.10),
                    p.steel.clone(),
                    Quat::IDENTITY,
                );
            }
        }
    }
    mounted(
        commands,
        p,
        Mount::Chassis,
        Vec3::new(0.0, 0.70, 0.0),
        Vec3::new(1.7, 0.35, 1.9),
        p.dark.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(0.0, 1.10, -0.1),
        Vec3::new(2.05, 0.60, 2.3),
        p.yellow.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(0.30, 1.58, -0.75),
        Vec3::new(1.30, 0.58, 0.85),
        p.yellow.clone(),
        Quat::IDENTITY,
    );
    for i in 0..7 {
        mounted(
            commands,
            p,
            Mount::Upper,
            Vec3::new(0.30, 1.60, -1.19 + i as f32 * 0.05),
            Vec3::new(0.95, 0.08, 0.025),
            p.dark.clone(),
            Quat::IDENTITY,
        );
    }
    // Open framed cab: transparent glazing and a readable interior in first person.
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 1.45, 0.22),
        Vec3::new(0.95, 0.12, 1.42),
        p.dark.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 2.78, 0.22),
        Vec3::new(1.06, 0.12, 1.55),
        p.yellow.clone(),
        Quat::IDENTITY,
    );
    for x in [-1.12, -0.16] {
        for z in [-0.45, 0.87] {
            mounted(
                commands,
                p,
                Mount::Upper,
                Vec3::new(x, 2.10, z),
                Vec3::new(0.055, 1.3, 0.055),
                p.dark.clone(),
                Quat::IDENTITY,
            );
        }
    }
    for x in [-1.1, -0.18] {
        mounted(
            commands,
            p,
            Mount::Upper,
            Vec3::new(x, 2.18, 0.22),
            Vec3::new(0.015, 0.95, 1.2),
            p.glass.clone(),
            Quat::IDENTITY,
        );
    }
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 2.18, 0.87),
        Vec3::new(0.90, 0.95, 0.015),
        p.glass.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 1.72, -0.12),
        Vec3::new(0.43, 0.16, 0.45),
        p.dark.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 1.96, -0.33),
        Vec3::new(0.43, 0.46, 0.12),
        p.dark.clone(),
        Quat::IDENTITY,
    );
    for x in [-0.99, -0.29] {
        mounted(
            commands,
            p,
            Mount::Upper,
            Vec3::new(x, 1.98, 0.25),
            Vec3::new(0.045, 0.25, 0.045),
            p.dark.clone(),
            Quat::from_rotation_x(0.2),
        );
    }
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 1.85, 0.68),
        Vec3::new(0.84, 0.22, 0.20),
        p.dark.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Upper,
        Vec3::new(-0.64, 1.99, 0.65),
        Vec3::new(0.22, 0.02, 0.12),
        p.blue.clone(),
        Quat::from_rotation_x(-0.3),
    );
    // Arm members and two parallel hydraulic rams are updated from actual joint positions.
    for (kind, width, offset, mat) in [
        (0, 0.30, 0.0, p.yellow.clone()),
        (1, 0.22, 0.0, p.yellow.clone()),
        (0, 0.09, 0.24, p.dark.clone()),
        (0, 0.045, 0.24, p.chrome.clone()),
        (1, 0.065, -0.20, p.chrome.clone()),
    ] {
        commands.spawn((
            Mesh3d(p.cube.clone()),
            MeshMaterial3d(mat),
            Transform::default(),
            Link {
                kind,
                offset,
                width,
            },
        ));
    }
    mounted(
        commands,
        p,
        Mount::Bucket,
        Vec3::new(0.0, -0.20, 0.43),
        Vec3::new(0.82, 0.075, 0.88),
        p.steel.clone(),
        Quat::IDENTITY,
    );
    mounted(
        commands,
        p,
        Mount::Bucket,
        Vec3::new(0.0, 0.02, 0.02),
        Vec3::new(0.82, 0.50, 0.075),
        p.steel.clone(),
        Quat::IDENTITY,
    );
    for x in [-0.40, 0.40] {
        mounted(
            commands,
            p,
            Mount::Bucket,
            Vec3::new(x, 0.0, 0.40),
            Vec3::new(0.065, 0.43, 0.84),
            p.steel.clone(),
            Quat::IDENTITY,
        );
    }
    for x in [-0.30, -0.15, 0.0, 0.15, 0.30] {
        mounted(
            commands,
            p,
            Mount::Bucket,
            Vec3::new(x, -0.18, 0.94),
            Vec3::new(0.10, 0.10, 0.25),
            p.chrome.clone(),
            Quat::IDENTITY,
        );
    }
    commands.spawn((
        Mesh3d(p.sphere.clone()),
        MeshMaterial3d(p.dirt.clone()),
        Transform::default(),
        BucketFill,
    ));
}

fn person(commands: &mut Commands, p: &Palette) {
    for (position, scale, material, limb, sphere) in [
        (
            Vec3::new(0.0, 1.17, 0.0),
            Vec3::new(0.45, 0.55, 0.25),
            p.blue.clone(),
            0,
            false,
        ),
        (
            Vec3::new(0.0, 1.61, 0.0),
            Vec3::splat(0.135),
            p.skin.clone(),
            0,
            true,
        ),
        (
            Vec3::new(0.0, 1.75, 0.0),
            Vec3::new(0.34, 0.12, 0.32),
            p.yellow.clone(),
            0,
            false,
        ),
        (
            Vec3::new(-0.13, 0.55, 0.0),
            Vec3::new(0.17, 0.65, 0.18),
            p.blue.clone(),
            1,
            false,
        ),
        (
            Vec3::new(0.13, 0.55, 0.0),
            Vec3::new(0.17, 0.65, 0.18),
            p.blue.clone(),
            2,
            false,
        ),
        (
            Vec3::new(-0.13, 0.12, 0.06),
            Vec3::new(0.22, 0.18, 0.34),
            p.dark.clone(),
            1,
            false,
        ),
        (
            Vec3::new(0.13, 0.12, 0.06),
            Vec3::new(0.22, 0.18, 0.34),
            p.dark.clone(),
            2,
            false,
        ),
        (
            Vec3::new(-0.30, 1.03, 0.0),
            Vec3::new(0.13, 0.52, 0.13),
            p.skin.clone(),
            3,
            false,
        ),
        (
            Vec3::new(0.30, 1.03, 0.0),
            Vec3::new(0.13, 0.52, 0.13),
            p.skin.clone(),
            4,
            false,
        ),
        (
            Vec3::new(0.36, 0.86, 0.25),
            Vec3::new(0.04, 1.0, 0.04),
            p.white.clone(),
            4,
            false,
        ),
        (
            Vec3::new(0.36, 0.30, 0.25),
            Vec3::new(0.24, 0.30, 0.07),
            p.steel.clone(),
            4,
            false,
        ),
    ] {
        let local = Transform::from_translation(position).with_scale(scale);
        commands.spawn((
            Mesh3d(if sphere {
                p.sphere.clone()
            } else {
                p.cube.clone()
            }),
            MeshMaterial3d(material),
            local,
            PersonPart { local, limb },
        ));
    }
}

fn equipment(commands: &mut Commands, meshes: &mut Assets<Mesh>, p: &Palette, site: &Site) {
    let h = site.height(HOPPER_X, HOPPER_Z);
    for x in [-0.65, 0.65] {
        for z in [-0.65, 0.65] {
            box_part(
                commands,
                p,
                Vec3::new(HOPPER_X + x, h + 0.58, HOPPER_Z + z),
                Vec3::new(0.085, 1.16, 0.085),
                p.steel.clone(),
            );
        }
    }
    // Four sloping sheets form a genuinely open hopper.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    let mut pos = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for i in 0..4 {
        let a = i as f32 * std::f32::consts::FRAC_PI_2;
        let b = (i + 1) as f32 * std::f32::consts::FRAC_PI_2;
        let normal = Vec3::new((a + 0.785).sin(), 0.35, (a + 0.785).cos()).normalize();
        for (angle, y, r) in [
            (a, 1.55, 1.06),
            (b, 1.55, 1.06),
            (a, 0.73, 0.28),
            (b, 0.73, 0.28),
        ] {
            pos.push([angle.sin() * r, h + y, angle.cos() * r]);
            normals.push(normal.to_array());
        }
        let s = i * 4;
        indices.extend_from_slice(&[
            s,
            s + 2,
            s + 1,
            s + 1,
            s + 2,
            s + 3,
            s,
            s + 1,
            s + 2,
            s + 1,
            s + 3,
            s + 2,
        ]);
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(p.yellow.clone()),
        Transform::from_xyz(HOPPER_X, 0.0, HOPPER_Z),
    ));
    box_part(
        commands,
        p,
        Vec3::new(HOPPER_X, h + 0.84, HOPPER_Z + 0.59),
        Vec3::new(0.35, 0.12, 0.65),
        p.steel.clone(),
    );
    commands.spawn((
        Mesh3d(p.sphere.clone()),
        MeshMaterial3d(p.dirt.clone()),
        Transform::default(),
        HopperFill,
    ));
    commands.spawn((
        Mesh3d(p.sphere.clone()),
        MeshMaterial3d(p.sand.clone()),
        Transform::default(),
        TailingsPile,
    ));
    // Pump, intake strainer and supply hose follow the bank to the processing plant.
    let pump = Vec3::new(9.5, site.height(9.5, HOPPER_Z), HOPPER_Z);
    box_part(
        commands,
        p,
        pump + Vec3::Y * 0.18,
        Vec3::new(0.85, 0.35, 0.70),
        p.dark.clone(),
    );
    box_part(
        commands,
        p,
        pump + Vec3::Y * 0.49,
        Vec3::new(0.60, 0.30, 0.52),
        p.yellow.clone(),
    );
    let intake = Vec3::new(Site::river_center(HOPPER_Z), 0.20, HOPPER_Z);
    let end = Vec3::new(HOPPER_X, h + 1.20, HOPPER_Z + 0.80);
    for (a, b) in [
        (intake, pump + Vec3::Y * 0.3),
        (pump + Vec3::Y * 0.3, Vec3::new(12.3, 0.48, 17.8)),
        (Vec3::new(12.3, 0.48, 17.8), end),
    ] {
        cylinder_between(commands, p, a, b, 0.055, p.dark.clone());
    }
    for i in 0..100 {
        commands.spawn((
            Mesh3d(p.sphere.clone()),
            MeshMaterial3d(p.dirt.clone()),
            Transform::default(),
            Visibility::Hidden,
            FallingVisual { index: i },
        ));
    }
}

pub fn sync_terrain(
    game: Res<Game>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunks: Query<(&Mesh3d, &mut TerrainChunk)>,
    water: Query<&Mesh3d, (With<WaterSurface>, Without<TerrainChunk>)>,
    mut clock: Local<f32>,
    time: Res<Time>,
) {
    let site = &game.session.site;
    for (handle, mut chunk) in &mut chunks {
        if chunk.revision != site.revision {
            if let Some(mesh) = meshes.get_mut(&handle.0) {
                *mesh = terrain_mesh(site, chunk.x, chunk.z);
            }
            chunk.revision = site.revision;
        }
    }
    *clock += time.delta_secs();
    if *clock > 0.065 {
        *clock = 0.0;
        if let Ok(handle) = water.single() {
            if let Some(mesh) = meshes.get_mut(&handle.0) {
                *mesh = water_mesh(site, game.session.sim_time as f32);
            }
        }
    }
}

#[allow(
    clippy::type_complexity,
    reason = "Bevy ParamSet expresses overlapping component access"
)]
pub fn sync_machine(
    game: Res<Game>,
    mut sets: ParamSet<(
        Query<(&Mounted, &mut Transform)>,
        Query<(&Link, &mut Transform)>,
        Query<(&mut Transform, &mut Visibility), With<BucketFill>>,
        Query<(&PersonPart, &mut Transform, &mut Visibility)>,
    )>,
) {
    let s = &game.session;
    let m = &s.machine;
    let arm = m.arm();
    let chassis = Transform::from_translation(bvec(m.position))
        .with_rotation(Quat::from_rotation_y(m.heading));
    let upper = Transform::from_translation(bvec(m.position))
        .with_rotation(Quat::from_rotation_y(m.heading + m.swing));
    let bucket = Transform::from_translation(bvec(arm.wrist)).with_rotation(
        Quat::from_rotation_y(m.heading + m.swing) * Quat::from_rotation_x(-arm.angle),
    );
    for (part, mut transform) in &mut sets.p0() {
        *transform = match part.mount {
            Mount::Chassis => chassis,
            Mount::Upper => upper,
            Mount::Bucket => bucket,
        }
        .mul_transform(part.local);
    }
    let right = Vec3::new(arm.forward.z, 0.0, -arm.forward.x);
    for (link, mut transform) in &mut sets.p1() {
        let (mut a, mut b) = if link.kind == 0 {
            (bvec(arm.shoulder), bvec(arm.elbow))
        } else {
            (bvec(arm.elbow), bvec(arm.wrist))
        };
        if link.offset != 0.0 {
            let start = a;
            a = a.lerp(b, 0.15) + right * link.offset;
            b = start.lerp(b, 0.75) + right * link.offset + Vec3::Y * 0.12;
        }
        let delta = b - a;
        *transform = Transform::from_translation((a + b) * 0.5)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()))
            .with_scale(Vec3::new(link.width, delta.length(), link.width * 1.3));
    }
    for (mut t, mut v) in &mut sets.p2() {
        let fill = (m.bucket.volume() / mining_sim::machine::BUCKET_CAPACITY) as f32;
        *v = if fill > 0.002 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *t = bucket.mul_transform(Transform::from_xyz(0.0, -0.10, 0.4).with_scale(Vec3::new(
            0.35,
            0.18 * fill,
            0.38,
        )));
    }
    let visible = s.player.mode == ControlMode::OnFoot && s.player.third_person;
    let person = Transform::from_translation(bvec(s.player.position))
        .with_rotation(Quat::from_rotation_y(s.player.yaw));
    let walk = (s.sim_time as f32 * 8.0).sin()
        * (game.controls.walk.abs() + game.controls.strafe.abs()).min(1.0);
    for (part, mut t, mut v) in &mut sets.p3() {
        *v = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let mut local = part.local;
        if part.limb == 1 {
            local.translation.z += walk * 0.15;
        }
        if part.limb == 2 {
            local.translation.z -= walk * 0.15;
        }
        if part.limb == 4 {
            local.translation.z += s.player.tool_swing.sin() * 0.5;
            local.rotation = Quat::from_rotation_x(-s.player.tool_swing * 0.6);
        }
        *t = person.mul_transform(local);
    }
}

pub fn sync_plant(
    mut commands: Commands,
    game: Res<Game>,
    p: Res<Palette>,
    mut revision: ResMut<PlantRevision>,
    mut meshes: ResMut<Assets<Mesh>>,
    existing: Query<Entity, With<SluiceVisual>>,
) {
    let key = game.session.sluice.placement.map(|p| {
        (
            p.inlet,
            p.yaw,
            p.slope_deg,
            game.session.sluice.profile,
            game.session.sluice.riffles_enabled,
        )
    });
    if revision.0 == key {
        return;
    }
    for e in &existing {
        commands.entity(e).despawn();
    }
    revision.0 = key;
    let Some(placement) = game.session.sluice.placement else {
        return;
    };
    let (positions, indices) = mining_physics::geometry::mesh(&game.session.sluice);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh.duplicate_vertices();
    mesh.compute_normals();
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(p.steel.clone()),
        Transform::default(),
        SluiceVisual,
    ));
    for x in [0.15, game.session.sluice.profile.length() - 0.15] {
        let half_width = game.session.sluice.profile.width() * 0.5 + 0.04;
        for z in [-half_width, half_width] {
            let top = bvec(placement.to_world(SVec::new(x, -0.05, z)));
            let ground = game.session.site.height(top.x, top.z);
            let height = (top.y - ground).max(0.10);
            commands.spawn((
                Mesh3d(p.cube.clone()),
                MeshMaterial3d(p.steel.clone()),
                Transform::from_xyz(top.x, ground + height * 0.5, top.z)
                    .with_scale(Vec3::new(0.06, height, 0.06)),
                SluiceVisual,
            ));
        }
    }
}

#[allow(
    clippy::type_complexity,
    reason = "Bevy ParamSet expresses overlapping component access"
)]
pub fn sync_material(
    game: Res<Game>,
    mut sets: ParamSet<(
        Query<(&mut Transform, &mut Visibility), With<HopperFill>>,
        Query<(&mut Transform, &mut Visibility), With<TailingsPile>>,
        Query<(&FallingVisual, &mut Transform, &mut Visibility)>,
    )>,
) {
    let s = &game.session;
    for (mut t, mut v) in &mut sets.p0() {
        let amount = (s.sluice.hopper.volume() / 0.85) as f32;
        *v = if amount > 0.001 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *t = Transform::from_xyz(
            HOPPER_X,
            SluiceTop::height(&s.site) - 0.45 + amount * 0.25,
            HOPPER_Z,
        )
        .with_scale(Vec3::new(0.50, 0.28 * amount.max(0.1), 0.50));
    }
    for (mut t, mut v) in &mut sets.p1() {
        let amount = s.sluice.tailings.volume() as f32;
        if let Some(placement) = s.sluice.placement {
            let pos = bvec(placement.outlet() + placement.forward() * 0.8);
            let r = amount.cbrt() * 0.85;
            *t = Transform::from_xyz(pos.x, s.site.height(pos.x, pos.z) + r * 0.12, pos.z)
                .with_scale(Vec3::new(r, r * 0.45, r * 1.2));
        }
        *v = if amount > 0.0001 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    for (parcel, mut t, mut v) in &mut sets.p2() {
        if let Some(load) = s.falling.get(parcel.index) {
            let size = (load.material.volume() as f32).cbrt() * 0.65;
            *v = Visibility::Visible;
            *t = Transform::from_translation(bvec(load.position))
                .with_scale(Vec3::splat(size.max(0.025)));
        } else {
            *v = Visibility::Hidden;
        }
    }
}
struct SluiceTop;
impl SluiceTop {
    fn height(site: &Site) -> f32 {
        mining_sim::Sluice::hopper_top(site)
    }
}

pub fn guides(game: Res<Game>, mut gizmos: Gizmos) {
    if let Some(p) = game.placement {
        let valid = game.session.validate_placement(p).is_ok();
        let color = if valid {
            Color::srgb(0.45, 0.95, 0.55)
        } else {
            Color::srgb(1.0, 0.25, 0.15)
        };
        let rotation =
            Quat::from_rotation_y(p.yaw) * Quat::from_rotation_x(p.slope_deg.to_radians());
        gizmos.cuboid(
            Transform::from_translation(bvec(p.to_world(SVec::new(
                game.session.sluice.profile.length() * 0.5,
                0.10,
                0.0,
            ))))
            .with_rotation(rotation)
            .with_scale(Vec3::new(
                game.session.sluice.profile.width(),
                0.25,
                game.session.sluice.profile.length(),
            )),
            color,
        );
        gizmos.line(
            bvec(p.inlet),
            Vec3::new(
                HOPPER_X,
                mining_sim::Sluice::hopper_top(&game.session.site) - 0.6,
                HOPPER_Z,
            ),
            color,
        );
        gizmos.arrow(
            bvec(p.inlet) + Vec3::Y * 0.5,
            bvec(p.to_world(SVec::new(game.session.sluice.profile.length(), 0.0, 0.0)))
                + Vec3::Y * 0.5,
            color,
        );
    } else if game.session.player.mode == ControlMode::OnFoot {
        let s = &game.session;
        if let Some(p) = s
            .site
            .raycast(s.player.position + SVec::Y * 1.5, s.look_direction(), 3.0)
        {
            gizmos.circle(
                Isometry3d::new(
                    bvec(p) + Vec3::Y * 0.035,
                    Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                ),
                0.22,
                Color::srgb(1.0, 0.75, 0.30),
            );
        }
    }
}
