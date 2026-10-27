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
    pub soil: Handle<StandardMaterial>,
    pub bucket_load: Handle<StandardMaterial>,
    pub hopper_load: Handle<StandardMaterial>,
    pub sand: Handle<StandardMaterial>,
    pub gold: Handle<StandardMaterial>,
    pub blue: Handle<StandardMaterial>,
    pub skin: Handle<StandardMaterial>,
    pub white: Handle<StandardMaterial>,
}
#[derive(Component)]
pub(crate) struct TerrainChunk {
    x: usize,
    z: usize,
    revision: u64,
    generation: u64,
    source: Vec<[u32; 2]>,
    subdivisions: usize,
}
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
pub(crate) struct BucketGeometry(mining_sim::bucket::Model);
#[derive(Component)]
pub(crate) struct PersonPart {
    local: Transform,
    limb: u8,
}
#[derive(Component)]
pub(crate) struct SluiceVisual;
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
    let mut entity = commands.spawn((
        Mesh3d(p.cube.clone()),
        MeshMaterial3d(mat),
        local,
        Mounted { mount, local },
    ));
    if matches!(mount, Mount::Bucket) {
        entity.insert(BucketGeometry(mining_sim::bucket::Model::LegacyFloor));
    }
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
    mut images: ResMut<Assets<Image>>,
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
    let terrain_material = crate::ground_material::material(&mut images);
    let mut soil_material = terrain_material.clone();
    soil_material.base_color = Color::srgb(0.29, 0.17, 0.085);
    soil_material.perceptual_roughness = 0.96;
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
        soil: materials.add(soil_material.clone()),
        bucket_load: material(&mut materials, Color::srgb(0.29, 0.17, 0.085), 0.96, 0.0),
        hopper_load: materials.add(soil_material),
        sand: material(&mut materials, Color::srgb(0.50, 0.38, 0.23), 0.9, 0.0),
        gold: material(&mut materials, Color::srgb(0.98, 0.66, 0.10), 0.23, 0.8),
        blue: material(&mut materials, Color::srgb(0.12, 0.23, 0.28), 0.85, 0.0),
        skin: material(&mut materials, Color::srgb(0.64, 0.40, 0.24), 0.8, 0.0),
        white: material(&mut materials, Color::srgb(0.88, 0.84, 0.71), 0.8, 0.0),
    };
    let terrain_mat = materials.add(terrain_material);
    for z in (0..WIDTH - 1).step_by(16) {
        for x in (0..WIDTH - 1).step_by(16) {
            let mesh = terrain_mesh(&game.session.site, x, z, 1);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(terrain_mat.clone()),
                Transform::default(),
                TerrainChunk {
                    x,
                    z,
                    revision: u64::MAX,
                    generation: u64::MAX,
                    source: Vec::new(),
                    subdivisions: 1,
                },
            ));
        }
    }
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
        &mut images,
        &p,
        &game.session.site,
    );
    machine(&mut commands, &mut meshes, &p);
    person(&mut commands, &p);
    equipment(&mut commands, &mut meshes, &p, &game.session.site);
    commands.insert_resource(PlantRevision::default());
    commands.insert_resource(p);
}

// A thin film wets soil before it is 15 mm deep. Quantize only this visual
// albedo coefficient, so negligible water changes do not upload whole chunks.
fn ground_wetness_code(depth: f32) -> u32 {
    let t = (depth / 0.003).clamp(0., 1.);
    (t * t * (3. - 2. * t) * 63.).round() as u32
}
fn ground_wet_tone(code: u32) -> f32 {
    1. - 0.33 * code as f32 / 63.
}

// Deposited fines cover the original ground continuously. A binary material
// switch at 15 mm painted cell-sized mud patches and made deposition pop.
// Quantize only the visual coverage, with the same code in the mesh cache key.
fn ground_sediment_code(thickness: f32) -> u32 {
    let t = (thickness / 0.015).clamp(0., 1.);
    (t * t * (3. - 2. * t) * 63.).round() as u32
}

// Shape-preserving cubic interpolation retains every measured cell value.
// Unlike a blur it neither spreads a dry/dry interval nor overshoots the local
// mud cover. Shared neighbour samples give adjacent chunks the same tangent.
fn monotone_cover(values: [f32; 4], t: f32) -> f32 {
    let [a, b, c, d] = values;
    let tangent = |left: f32, right: f32| {
        if left * right > 0. {
            2. * left * right / (left + right)
        } else {
            0.
        }
    };
    let m0 = tangent(b - a, c - b);
    let m1 = tangent(c - b, d - c);
    let t2 = t * t;
    let t3 = t2 * t;
    (2. * t3 - 3. * t2 + 1.) * b
        + (t3 - 2. * t2 + t) * m0
        + (-2. * t3 + 3. * t2) * c
        + (t3 - t2) * m1
}
fn deposited_cover(site: &Site, x: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iz = z.floor() as i32;
    let rows = std::array::from_fn(|dz| {
        let values = std::array::from_fn(|dx| {
            let x = (ix + dx as i32 - 1).clamp(0, WIDTH as i32 - 1) as usize;
            let z = (iz + dz as i32 - 1).clamp(0, WIDTH as i32 - 1) as usize;
            ground_sediment_code(site.world.terrain_sediment[site.world.idx(x, z)]) as f32 / 63.
        });
        monotone_cover(values, x.fract())
    });
    monotone_cover(rows, z.fract())
}

pub fn terrain_mesh(site: &Site, x0: usize, z0: usize, subdivisions: usize) -> Mesh {
    let nx = 16.min(WIDTH - 1 - x0);
    let nz = 16.min(WIDTH - 1 - z0);
    let mut source_height = Vec::new();
    let mut source_normal = Vec::new();
    let mut source_color = Vec::new();
    let mut source_mud = Vec::new();
    for z in 0..=nz {
        for x in 0..=nx {
            let ix = x0 + x;
            let iz = z0 + z;
            let px = ix as f32 * CELL;
            let pz = iz as f32 * CELL;
            let y = site.world.ground_height(ix, iz);
            source_height.push(y);
            let normal = Vec3::new(
                site.height(px - 0.12, pz) - site.height(px + 0.12, pz),
                0.24,
                site.height(px, pz - 0.12) - site.height(px, pz + 0.12),
            )
            .normalize();
            source_normal.push(normal.to_array());
            let i = site.world.idx(ix, iz);
            let wet = ground_wet_tone(ground_wetness_code(site.world.water_depth(ix, iz)));
            let mut c = if site.world.overburden_thickness[i] > 0.025 {
                [0.34, 0.29, 0.18, 1.0]
            } else if site.world.paydirt_thickness[i] > 0.01 {
                [0.58, 0.43, 0.24, 1.0]
            } else {
                [0.30, 0.32, 0.30, 1.0]
            };
            let noise = 0.88 + hash(ix as u32, iz as u32) * 0.22;
            for v in &mut c[..3] {
                *v *= noise;
            }
            if site.world.overburden_thickness[i] > 0.025
                && !((11.0..25.0).contains(&px) && (12.0..26.0).contains(&pz))
            {
                c = [0.20 * noise, 0.29 * noise, 0.15 * noise, 1.0];
            }
            for value in &mut c[..3] {
                *value *= wet;
            }
            let base = Color::srgb(c[0], c[1], c[2]).to_linear().to_f32_array();
            let sediment = Color::srgb(0.43 * noise * wet, 0.34 * noise * wet, 0.22 * noise * wet)
                .to_linear()
                .to_f32_array();
            source_color.push(base);
            source_mud.push(sediment);
        }
    }
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let fine_x = nx * subdivisions;
    let fine_z = nz * subdivisions;
    for z in 0..=fine_z {
        for x in 0..=fine_x {
            let cx = (x / subdivisions).min(nx - 1);
            let cz = (z / subdivisions).min(nz - 1);
            let u = (x - cx * subdivisions) as f32 / subdivisions as f32;
            let v = (z - cz * subdivisions) as f32 / subdivisions as f32;
            let i = cz * (nx + 1) + cx;
            let ids = [i, i + 1, i + nx + 1, i + nx + 2];
            let sample = |values| crate::environment::interpolate(values, u, v);
            let px = (x0 as f32 + x as f32 / subdivisions as f32) * CELL;
            let pz = (z0 as f32 + z as f32 / subdivisions as f32) * CELL;
            let y = sample(ids.map(|i| source_height[i]));
            let n = Vec3::from_array(std::array::from_fn(|c| {
                sample(ids.map(|i| source_normal[i][c]))
            }))
            .normalize();
            let tangent = (Vec3::X - n * n.x).normalize();
            positions.push([px, y, pz]);
            normals.push(n.to_array());
            tangents.push([tangent.x, tangent.y, tangent.z, -1.]);
            let base = Vec4::from_array(std::array::from_fn(|c| {
                sample(ids.map(|i| source_color[i][c]))
            }));
            let mud = Vec4::from_array(std::array::from_fn(|c| {
                sample(ids.map(|i| source_mud[i][c]))
            }));
            colors.push(
                base.lerp(mud, deposited_cover(site, px / CELL, pz / CELL))
                    .to_array(),
            );
            uvs.push([px, pz]);
        }
    }
    for z in 0..fine_z {
        for x in 0..fine_x {
            let i = (z * (fine_x + 1) + x) as u32;
            let w = (fine_x + 1) as u32;
            indices.extend_from_slice(&[i, i + w, i + 1, i + 1, i + w, i + w + 1]);
        }
    }
    let mut m = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    m.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    m.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    m.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
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
    images: &mut Assets<Image>,
    p: &Palette,
    site: &Site,
) {
    let bark = material(materials, Color::srgb(0.22, 0.15, 0.10), 0.95, 0.0);
    let foliage = materials.add(crate::foliage::material(images));
    let canopies: Vec<_> = (0..8)
        .map(|seed| meshes.add(crate::foliage::canopy(seed)))
        .collect();
    let trunk = meshes.add(Cone::new(0.075, 4.4).mesh().resolution(10));
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
                Mesh3d(trunk.clone()),
                MeshMaterial3d(bark.clone()),
                Transform::from_xyz(x, h + scale * 2.2, z).with_scale(Vec3::splat(scale)),
            ));
            commands.spawn((
                Mesh3d(canopies[i as usize % canopies.len()].clone()),
                MeshMaterial3d(foliage.clone()),
                Transform::from_xyz(x, h, z)
                    .with_scale(Vec3::splat(scale))
                    .with_rotation(Quat::from_rotation_y(hash(i, 19) * std::f32::consts::TAU)),
            ));
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

fn machine(commands: &mut Commands, meshes: &mut Assets<Mesh>, p: &Palette) {
    for side in [-1.0, 1.0] {
        mounted(
            commands,
            p,
            Mount::Chassis,
            Vec3::new(side * mining_sim::tracks::TRACK_GAUGE * 0.5, 0.325, 0.0),
            Vec3::new(
                mining_sim::tracks::TRACK_WIDTH,
                0.65,
                mining_sim::tracks::TRACK_LENGTH,
            ),
            p.dark.clone(),
            Quat::IDENTITY,
        );
        for i in 0..7 {
            let local = Transform::from_xyz(side * 1.18, 0.325, -1.18 + i as f32 * 0.393)
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
            for y in [0.0275, 0.6225] {
                mounted(
                    commands,
                    p,
                    Mount::Chassis,
                    Vec3::new(
                        side * mining_sim::tracks::TRACK_GAUGE * 0.5,
                        y,
                        -1.37 + i as f32 * 0.17125,
                    ),
                    Vec3::new(mining_sim::tracks::TRACK_WIDTH, 0.055, 0.10),
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
        (
            1,
            mining_sim::machine::STICK_VISUAL_WIDTH_M,
            0.0,
            p.yellow.clone(),
        ),
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
    // Contacts, mass integration and presentation share these exact vertices.
    for model in [
        mining_sim::bucket::Model::ShellV1,
        mining_sim::bucket::Model::MountedShellV2,
    ] {
        for part in &model.geometry().parts {
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            for f in &part.triangles {
                let v = f.map(|i| bvec(part.vertices[i as usize]));
                let n = (v[1] - v[0]).cross(v[2] - v[0]).normalize();
                positions.extend(v.map(|p| p.to_array()));
                normals.extend([n.to_array(); 3]);
            }
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(if part.tooth {
                    p.chrome.clone()
                } else {
                    p.steel.clone()
                }),
                Transform::default(),
                Mounted {
                    mount: Mount::Bucket,
                    local: Transform::IDENTITY,
                },
                BucketGeometry(model),
            ));
        }
    }
    commands.spawn((
        Mesh3d(p.sphere.clone()),
        MeshMaterial3d(p.bucket_load.clone()),
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
            let leg = box_part(
                commands,
                p,
                Vec3::new(HOPPER_X + x, h + 0.58, HOPPER_Z + z),
                Vec3::new(0.085, 1.16, 0.085),
                p.steel.clone(),
            );
            commands
                .entity(leg)
                .insert(crate::screen_render::LegacyHopper);
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
        crate::screen_render::LegacyHopper,
    ));
    let chute = box_part(
        commands,
        p,
        Vec3::new(HOPPER_X, h + 0.84, HOPPER_Z + 0.59),
        Vec3::new(0.35, 0.12, 0.65),
        p.steel.clone(),
    );
    commands
        .entity(chute)
        .insert(crate::screen_render::LegacyHopper);
    commands.spawn((
        Mesh3d(meshes.add(crate::hopper_render::empty_mesh())),
        MeshMaterial3d(p.hopper_load.clone()),
        Transform::default(),
        crate::hopper_render::Fill::default(),
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

// Exact mesh inputs, including the one-cell halo used by normal sampling.
// Water-only solver revisions should not upload unchanged bank meshes again.
fn terrain_source(site: &Site, x0: usize, z0: usize) -> Vec<[u32; 2]> {
    let mut key = Vec::new();
    for z in z0.saturating_sub(1)..=(z0 + 17).min(WIDTH - 1) {
        for x in x0.saturating_sub(1)..=(x0 + 17).min(WIDTH - 1) {
            let i = site.world.idx(x, z);
            let flags = ground_wetness_code(site.world.water_depth(x, z))
                | (ground_sediment_code(site.world.terrain_sediment[i]) << 6)
                | (u32::from(site.world.overburden_thickness[i] > 0.025) << 12)
                | (u32::from(site.world.paydirt_thickness[i] > 0.01) << 13);
            key.push([site.world.ground_height(x, z).to_bits(), flags]);
        }
    }
    key
}
pub fn sync_terrain(
    game: Res<Game>,
    detail: Res<crate::environment::Detail>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunks: Query<(&Mesh3d, &mut TerrainChunk)>,
) {
    let site = &game.session.site;
    for (handle, mut chunk) in &mut chunks {
        let subdivisions = detail.subdivisions(chunk.x, chunk.z);
        if chunk.revision != site.revision
            || chunk.subdivisions != subdivisions
            || chunk.generation != game.simulation_generation
        {
            let source = terrain_source(site, chunk.x, chunk.z);
            if source != chunk.source || chunk.subdivisions != subdivisions {
                if let Some(mesh) = meshes.get_mut(&handle.0) {
                    *mesh = terrain_mesh(site, chunk.x, chunk.z, subdivisions);
                }
                chunk.source = source;
                chunk.subdivisions = subdivisions;
            }
            chunk.revision = site.revision;
            chunk.generation = game.simulation_generation;
        }
    }
}

// Preserve the actual hinge-plane roll. A shortest-arc rotation points the
// beam along the stick but twists its rectangular cross-section as the house
// turns, invalidating the attachment clearance measured in that plane.
fn stick_beam_rotation(direction: Vec3, right: Vec3) -> Quat {
    let y = direction.normalize();
    let x = (right - y * right.dot(y)).normalize();
    Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y)))
}

#[allow(
    clippy::type_complexity,
    reason = "Bevy ParamSet expresses overlapping component access"
)]
pub fn sync_machine(
    game: Res<Game>,
    mut sets: ParamSet<(
        Query<(
            &Mounted,
            &mut Transform,
            &mut Visibility,
            Option<&BucketGeometry>,
        )>,
        Query<(&Link, &mut Transform)>,
        Query<(&mut Transform, &mut Visibility), With<BucketFill>>,
        Query<(&PersonPart, &mut Transform, &mut Visibility)>,
    )>,
) {
    let s = &game.session;
    let m = &s.machine;
    let arm = m.arm();
    let chassis = Transform::from_translation(bvec(m.position))
        .with_rotation(Quat::from_array(m.chassis_rotation().to_array()));
    let upper = Transform::from_translation(bvec(m.position))
        .with_rotation(Quat::from_array(m.upper_rotation().to_array()));
    let bucket = Transform::from_translation(bvec(arm.wrist)).with_rotation(
        if m.bucket_model().is_shell() {
            Quat::from_array(m.bucket_rotation().to_array())
        } else {
            Quat::from_array(m.upper_rotation().to_array()) * Quat::from_rotation_x(-arm.angle)
        },
    );
    for (part, mut transform, mut visibility, geometry) in &mut sets.p0() {
        if let Some(geometry) = geometry {
            *visibility = if geometry.0 == m.bucket_model() {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
        *transform = match part.mount {
            Mount::Chassis => chassis,
            Mount::Upper => upper,
            Mount::Bucket => bucket,
        }
        .mul_transform(part.local);
    }
    let right = bvec(arm.right);
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
            .with_rotation(
                if m.bucket_model() == mining_sim::bucket::Model::MountedShellV2
                    && link.kind == 1
                    && link.offset == 0.
                {
                    stick_beam_rotation(delta, right)
                } else {
                    Quat::from_rotation_arc(Vec3::Y, delta.normalize())
                },
            )
            .with_scale(Vec3::new(
                link.width,
                delta.length(),
                link.width * mining_sim::machine::LINK_VISUAL_DEPTH_RATIO,
            ));
    }
    for (mut t, mut v) in &mut sets.p2() {
        let fill = (m.bucket.volume() / mining_sim::machine::BUCKET_CAPACITY) as f32;
        *v = if fill > 0.002 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if let Some(moments) = m.articulation.as_ref().and_then(|a| a.payload_moments) {
            let (q, e) = moments.visual_ellipsoid();
            *t = bucket.mul_transform(
                Transform::from_translation(bvec(moments.center.as_vec3()))
                    .with_rotation(Quat::from_xyzw(
                        q.x as f32, q.y as f32, q.z as f32, q.w as f32,
                    ))
                    .with_scale(bvec(e.as_vec3())),
            );
        } else {
            *t = bucket.mul_transform(Transform::from_xyz(0.0, -0.10, 0.4).with_scale(Vec3::new(
                0.35,
                0.18 * fill,
                0.38,
            )));
        }
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
    palette: Res<Palette>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sets: ParamSet<(
        Query<(&mut Transform, &mut Visibility), With<TailingsPile>>,
        Query<(&FallingVisual, &mut Transform, &mut Visibility)>,
    )>,
) {
    let s = &game.session;
    for (handle, load) in [
        (&palette.bucket_load, s.machine.bucket),
        (&palette.hopper_load, s.sluice.hopper),
    ] {
        // Continuous display response to actual carried moisture. The palette
        // is an art approximation; it does not alter particles or soil strength.
        let wet = load.saturation() as f32;
        let color = Color::srgb(0.29 - 0.14 * wet, 0.17 - 0.07 * wet, 0.085 - 0.03 * wet);
        let roughness = 0.96 - 0.35 * wet;
        if materials
            .get(handle)
            .is_some_and(|m| m.base_color != color || m.perceptual_roughness != roughness)
        {
            if let Some(material) = materials.get_mut(handle) {
                material.base_color = color;
                material.perceptual_roughness = roughness;
            }
        }
    }

    for (mut t, mut v) in &mut sets.p0() {
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
    for (parcel, mut t, mut v) in &mut sets.p1() {
        if let Some(load) = s.falling.get(parcel.index) {
            let size = (load.material.volume() as f32).cbrt() * 0.65;
            *v = Visibility::Visible;
            *t = if let Some(body) = &load.rigid {
                let (axes, extent) = body.moments.visual_ellipsoid();
                let q = body.rotation * axes;
                Transform::from_translation(bvec(load.position))
                    .with_rotation(Quat::from_xyzw(
                        q.x as f32, q.y as f32, q.z as f32, q.w as f32,
                    ))
                    .with_scale(Vec3::new(extent.x as f32, extent.y as f32, extent.z as f32))
            } else {
                Transform::from_translation(bvec(load.position))
                    .with_scale(Vec3::splat(size.max(0.025)))
            };
        } else {
            *v = Visibility::Hidden;
        }
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

#[cfg(test)]
mod terrain_detail_tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;
    #[test]
    fn shallow_wetting_has_no_fifteen_millimetre_colour_jump_and_updates_mesh_key() {
        assert_eq!(ground_wetness_code(0.), 0);
        assert_eq!(ground_wetness_code(0.003), 63);
        assert_eq!(ground_wetness_code(0.014999), ground_wetness_code(0.015001));
        let mut previous = 1.;
        for i in 0..=600 {
            let tone = ground_wet_tone(ground_wetness_code(i as f32 * 0.000005));
            assert!(tone <= previous && previous - tone < 0.006);
            previous = tone;
        }
        let mut site = Site::new();
        let i = site.world.idx(64, 64);
        let ground = site.world.ground_height(64, 64);
        site.world.water_surface[i] = ground;
        let dry = terrain_source(&site, 64, 64);
        site.world.water_surface[i] = ground + 0.0015;
        assert_ne!(dry, terrain_source(&site, 64, 64));
        // Display sampling must not alter the physical site.
        let before = serde_json::to_string(&site.snapshot()).unwrap();
        let _ = terrain_mesh(&site, 64, 64, 4);
        assert_eq!(before, serde_json::to_string(&site.snapshot()).unwrap());
    }

    #[test]
    fn mud_cover_is_bounded_exact_at_cells_and_smooth_across_chunk_edges() {
        let mut site = Site::new();
        site.world.terrain_sediment.fill(0.);
        for z in 0..WIDTH {
            let i = site.world.idx(64, z);
            site.world.terrain_sediment[i] = 0.02;
        }
        let before = serde_json::to_string(&site.snapshot()).unwrap();
        for z in [0., 16.3, 64., 126.8, 127.] {
            assert_eq!(deposited_cover(&site, 63., z), 0.);
            assert_eq!(deposited_cover(&site, 64., z), 1.);
            assert_eq!(deposited_cover(&site, 65., z), 0.);
            assert_eq!(
                deposited_cover(&site, 62.5, z),
                0.,
                "dry interval acquired mud"
            );
            for step in 0..=256 {
                let value = deposited_cover(&site, 62. + step as f32 / 64., z);
                assert!(
                    (0.0..=1.0).contains(&value),
                    "mud interpolation overshot: {value}"
                );
            }
            let h = 0.001;
            let left = (1. - deposited_cover(&site, 64. - h, z)) / h;
            let right = (deposited_cover(&site, 64. + h, z) - 1.) / h;
            assert!(
                (right - left).abs() < 0.01,
                "chunk-edge mud slope kink: {left}/{right}"
            );
            // Negative control: interpolate the same nodal peak with the
            // former linear ramp. Its one-sided slopes disagree by two.
            let old = |x: f32| if x <= 64. { x - 63. } else { 65. - x };
            let old_left = (old(64.) - old(64. - h)) / h;
            let old_right = (old(64. + h) - old(64.)) / h;
            assert!((old_right - old_left).abs() > 1.99);
        }
        let a = terrain_mesh(&site, 48, 64, 4);
        let b = terrain_mesh(&site, 64, 64, 4);
        let colour = |mesh: &Mesh| -> Vec<[f32; 4]> {
            let Some(VertexAttributeValues::Float32x4(v)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR)
            else {
                panic!("terrain colour")
            };
            v.clone()
        };
        let (a, b) = (colour(&a), colour(&b));
        for z in 0..65 {
            assert_eq!(a[z * 65 + 64], b[z * 65], "chunk colour seam");
        }
        assert_eq!(before, serde_json::to_string(&site.snapshot()).unwrap());
    }
    #[test]
    fn deposited_fines_colour_is_continuous_and_invalidates_cached_ground() {
        let mut site = Site::new();
        let i = site.world.idx(64, 64);
        let colour = |site: &Site| {
            let mesh = terrain_mesh(site, 64, 64, 1);
            let Some(VertexAttributeValues::Float32x4(values)) =
                mesh.attribute(Mesh::ATTRIBUTE_COLOR)
            else {
                panic!("terrain colours")
            };
            Vec4::from_array(values[0])
        };
        site.world.terrain_sediment[i] = 0.;
        let dry = colour(&site);
        let key = terrain_source(&site, 64, 64);
        site.world.terrain_sediment[i] = 0.0075;
        assert_ne!(
            key,
            terrain_source(&site, 64, 64),
            "partial deposit left stale colour"
        );
        let half = colour(&site);
        site.world.terrain_sediment[i] = 0.02;
        let full = colour(&site);
        assert!(
            (dry - full).length() > 0.02,
            "negative control needs distinct soil/mud endpoints"
        );
        assert!(
            (half - (dry + full) * 0.5).length() < 0.002,
            "partial cover must blend in linear light"
        );
        site.world.terrain_sediment[i] = 0.014999;
        let before_threshold = colour(&site);
        site.world.terrain_sediment[i] = 0.015001;
        assert_eq!(
            before_threshold,
            colour(&site),
            "old 15 mm material switch remains"
        );
        let mut previous = dry;
        for n in 1..=64 {
            site.world.terrain_sediment[i] = n as f32 * 0.015 / 64.;
            let current = colour(&site);
            assert!(
                (current - previous).length() < 0.006,
                "deposition colour popped at sample {n}"
            );
            previous = current;
        }
        let state = serde_json::to_string(&site.snapshot()).unwrap();
        let _ = terrain_mesh(&site, 64, 64, 4);
        assert_eq!(state, serde_json::to_string(&site.snapshot()).unwrap());
    }

    #[test]
    fn fine_chunk_edges_match_coarse_neighbours_without_cracks() {
        let site = Site::new();
        let fine = terrain_mesh(&site, 64, 64, 4);
        let coarse = terrain_mesh(&site, 80, 64, 1);
        let positions = |mesh: &Mesh| match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("terrain positions must be float3"),
        };
        let a = positions(&fine);
        let b = positions(&coarse);
        assert_eq!(a.len(), 65 * 65);
        for z in 0..=64 {
            let lo = (z / 4).min(15);
            let t = (z - lo * 4) as f32 / 4.;
            let expected = Vec3::from_array(b[lo * 17]).lerp(Vec3::from_array(b[(lo + 1) * 17]), t);
            assert!((Vec3::from_array(a[z * 65 + 64]) - expected).length() < 0.000002);
        }
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            fine.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("fine terrain needs normal-map tangents")
        };
        assert!(tangents
            .iter()
            .all(|t| (Vec3::new(t[0], t[1], t[2]).length() - 1.).abs() < 0.00001 && t[3] == -1.));
    }
}

#[derive(Component)]
pub(crate) struct SoilVisual {
    index: usize,
    generation: u64,
    vertices: [SVec; 6],
}
/// Draw the actual convex clod geometry and solved pose, not decorative balls
/// or an unaccounted pile. Geometry is rebuilt only on activation/load.
pub fn sync_soil(
    mut commands: Commands,
    game: Res<Game>,
    palette: Res<Palette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut existing: Query<(Entity, &SoilVisual, &mut Transform)>,
) {
    let soil = game.session.active_soil.as_ref();
    let mut visible = std::collections::HashSet::new();
    for (entity, visual, mut transform) in &mut existing {
        if let Some(clod) = soil.and_then(|s| s.clods.get(visual.index)).filter(|c| {
            visual.generation == game.simulation_generation && visual.vertices == *c.vertices()
        }) {
            *transform =
                Transform::from_translation(bvec(clod.center)).with_rotation(Quat::from_xyzw(
                    clod.rotation.x,
                    clod.rotation.y,
                    clod.rotation.z,
                    clod.rotation.w,
                ));
            visible.insert(visual.index);
        } else {
            commands.entity(entity).despawn();
        }
    }
    let Some(soil) = soil else {
        return;
    };
    for (index, clod) in soil
        .clods
        .iter()
        .enumerate()
        .filter(|(i, _)| !visible.contains(i))
    {
        let vertices = clod.vertices();
        let mesh = soil_mesh(clod);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(palette.soil.clone()),
            Transform::from_translation(bvec(clod.center)).with_rotation(Quat::from_xyzw(
                clod.rotation.x,
                clod.rotation.y,
                clod.rotation.z,
                clod.rotation.w,
            )),
            SoilVisual {
                index,
                generation: game.simulation_generation,
                vertices: *vertices,
            },
        ));
    }
}

fn soil_mesh(clod: &mining_sim::soil::Clod) -> Mesh {
    let vertices = clod.vertices();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    for face in mining_sim::soil::FACES {
        let [a, mut b, mut c] = face.map(|i| vertices[i]);
        let mut n = (b - a).cross(c - a);
        if n.length_squared() < 1e-16 {
            continue;
        }
        if n.dot((a + b + c) / 3.) < 0. {
            std::mem::swap(&mut b, &mut c);
            n = -n;
        }
        let n = n.normalize();
        let t = (b - a).normalize();
        let bitangent = n.cross(t);
        for p in [a, b, c] {
            positions.push(p.to_array());
            normals.push(n.to_array());
            uvs.push([p.dot(t), p.dot(bitangent)]);
            tangents.push([t.x, t.y, t.z, 1.]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32((0..positions.len() as u32).collect()));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh
}
#[cfg(test)]
mod soil_render_tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;
    #[test]
    fn rendered_hull_encloses_the_same_physical_prism_with_outward_faces() {
        let source = [
            SVec::ZERO,
            SVec::new(0.4, 0., 0.),
            SVec::new(0., 0., 0.3),
            SVec::new(0., 0.2, 0.),
            SVec::new(0.4, 0.2, 0.),
            SVec::new(0., 0.2, 0.3),
        ];
        for reverse in [false, true] {
            let mut v = source;
            if reverse {
                v.swap(1, 2);
                v.swap(4, 5);
            }
            let clod = mining_sim::soil::Clod::prism(
                mining_sim::Material {
                    soil_m3: 0.012,
                    ..default()
                },
                v.map(|p| p.as_dvec3()),
            )
            .unwrap();
            let mesh = soil_mesh(&clod);
            let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("render positions");
            };
            let Some(VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!("render normals");
            };
            assert_eq!(positions.len(), 24);
            assert_eq!(mesh.indices().unwrap().len(), 24);
            let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
            else {
                panic!("soil UVs");
            };
            let Some(VertexAttributeValues::Float32x4(tangents)) =
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
            else {
                panic!("soil tangents");
            };
            let mut volume = 0.;
            for (i, (face, normal)) in positions
                .chunks_exact(3)
                .zip(normals.chunks_exact(3))
                .enumerate()
            {
                let p = face
                    .iter()
                    .map(|v| SVec::from_array(*v).as_dvec3())
                    .collect::<Vec<_>>();
                let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                assert!(cross.dot((p[0] + p[1] + p[2]) / 3.) > 0.);
                assert!(
                    cross
                        .normalize()
                        .dot(SVec::from_array(normal[0]).as_dvec3())
                        > 0.99999
                );
                let uv = |n| Vec2::from_array(uvs[i * 3 + n]);
                let du = uv(1) - uv(0);
                let dv = uv(2) - uv(0);
                let det = du.x * dv.y - du.y * dv.x;
                assert!(det > 0., "normal map must retain outward handedness");
                for n in [1, 2] {
                    assert!(
                        ((uv(n) - uv(0)).length() as f64 - (p[n] - p[0]).length()).abs() < 0.000001,
                        "soil detail must retain metre scale"
                    );
                }
                let t = ((p[1] - p[0]) * dv.y as f64 - (p[2] - p[0]) * du.y as f64) / det as f64;
                let mapped_t =
                    SVec::from_array(tangents[i * 3][..3].try_into().unwrap()).as_dvec3();
                assert!(t.normalize().dot(mapped_t) > 0.99999 && tangents[i * 3][3] == 1.);
                volume += p[0].dot(p[1].cross(p[2])) / 6.;
            }
            assert!((volume - clod.geometry_volume_m3()).abs() < 1e-8);
        }
    }
}

#[cfg(test)]
mod machine_render_tests {
    use super::*;
    #[test]
    fn mounted_stick_cross_section_keeps_the_physical_hinge_plane_on_tilt_and_swing() {
        let mut old_twist = 0f32;
        for (heading, swing, roll) in [(0.0, 0.0, 0.0_f32), (1.2, 0.7, 0.25), (2.4, -1.1, -0.3)] {
            let mut m = mining_sim::Excavator::new(&mining_sim::Site::new());
            m.heading = heading;
            m.swing = swing;
            m.tracks.tilt.x = 0.;
            m.tracks.tilt.y = 0.;
            m.tracks.tilt.z = (roll * 0.5).sin();
            m.tracks.tilt.w = (roll * 0.5).cos();
            m.boom = 0.4;
            m.stick = -1.2;
            m.enable_articulation().unwrap();
            let arm = m.arm();
            let direction = bvec(arm.wrist - arm.elbow).normalize();
            let right = bvec(arm.right);
            let q = stick_beam_rotation(direction, right);
            assert!((q * Vec3::Y).distance(direction) < 1e-6);
            assert!(
                (q * Vec3::X).distance(right) < 1e-6,
                "beam twisted out of its hinge plane"
            );
            let old = Quat::from_rotation_arc(Vec3::Y, direction);
            old_twist = old_twist.max((old * Vec3::X).distance(right));
        }
        assert!(
            old_twist > 0.5,
            "old render-frame control did not reproduce twist"
        );
    }
}
