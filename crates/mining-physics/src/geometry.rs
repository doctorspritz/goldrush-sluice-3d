//! The same original geometry builder supplies both collision solids and the
//! visible sluice. Placement changes the transform, never an invisible proxy.
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder};
use mining_sim::{sluice::HydraulicProfile, Sluice, Vec3};
pub fn config(s: &Sluice) -> SluiceConfig {
    let profile = s.profile;
    let original = profile == HydraulicProfile::OriginalDemo;
    SluiceConfig {
        grid_width: (profile.length() / profile.cell()).round() as usize,
        grid_height: s.floor_cells() + if original { 22 } else { 18 },
        grid_depth: (profile.width() / profile.cell()).round() as usize + 2,
        cell_size: profile.cell(),
        floor_height_left: s.floor_cells(),
        floor_height_right: 4,
        riffle_spacing: if original { 32 } else { 19 },
        riffle_height: if original { 3 } else { 2 },
        riffle_thickness: 2,
        riffle_start_x: if !s.riffles_enabled {
            usize::MAX
        } else if original {
            40
        } else {
            27
        },
        riffle_end_pad: 12,
        wall_margin: 8,
        exit_width_fraction: 1.0,
        exit_height: if original { 12 } else { 14 },
        ..Default::default()
    }
}

/// The solver's phase-presence mask, in the original grain order. This is the
/// existing neighbouring-water predicate, not a saturation or submergence test.
pub fn grain_wetness(s: &Sluice) -> Vec<bool> {
    let c = config(s);
    let width = c.grid_width + s.profile.outlet_buffer();
    let height = c.grid_height;
    let depth = c.grid_depth;
    let dx = s.profile.cell();
    let index = |p: Vec3| -> Option<usize> {
        let cell = (p / dx).floor().as_ivec3();
        if cell.min_element() < 0
            || cell.x >= width as i32
            || cell.y >= height as i32
            || cell.z >= depth as i32
        {
            None
        } else {
            Some(cell.z as usize * width * height + cell.y as usize * width + cell.x as usize)
        }
    };
    let mut water_cells = vec![0_u32; width * height * depth];
    for water in &s.water {
        if let Some(i) = index(water.position) {
            water_cells[i] += 1;
        }
    }
    s.grains
        .iter()
        .map(|g| neighbouring_water(g.position, dx, &water_cells, &index))
        .collect()
}

pub(crate) fn neighbouring_water(
    position: Vec3,
    dx: f32,
    water_cells: &[u32],
    index: &impl Fn(Vec3) -> Option<usize>,
) -> bool {
    let mut wet = false;
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if let Some(i) = index(position + Vec3::new(x as f32, y as f32, z as f32) * dx) {
                    wet |= water_cells[i] > 0;
                }
            }
        }
    }
    wet
}

pub fn mesh(s: &Sluice) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut b = SluiceGeometryBuilder::new(config(s));
    b.build_mesh_from_config();
    let Some(p) = s.placement else {
        return (vec![], vec![]);
    };
    (
        b.vertices()
            .iter()
            .map(|v| {
                p.to_world(s.grid_to_local(Vec3::from_array(v.position)))
                    .to_array()
            })
            .collect(),
        // The legacy builder winds its Z faces inward. Repair those faces,
        // then account for the reflected (downstream, up, right) world basis.
        b.indices()
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|t| {
                let v = |i: u32| Vec3::from_array(b.vertices()[i as usize].position);
                let normal = (v(t[1]) - v(t[0])).cross(v(t[2]) - v(t[0]));
                if normal.z.abs() > 1e-8 {
                    [t[0], t[1], t[2]]
                } else {
                    [t[0], t[2], t[1]]
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Quat};
    use mining_sim::{
        sluice::{Grain, WaterParticle},
        GrainKind, Material,
    };
    use std::cell::Cell;

    fn grain(position: Vec3) -> Grain {
        Grain {
            volume_resolved: true,
            kind: GrainKind::Sand,
            position,
            velocity: Vec3::ZERO,
            material: Material::sand_from_solid_volume(1e-7),
            resting: false,
            diameter_m: 0.0001,
            affine: Mat3::ZERO,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        }
    }

    fn water(position: Vec3) -> WaterParticle {
        WaterParticle {
            id: 0,
            position,
            velocity: Vec3::ZERO,
            affine: Mat3::ZERO,
        }
    }

    // Retain the pre-extraction solver loop as a separate oracle. It does not
    // call neighbouring_water or use the public mask's occupancy allocation.
    fn original_solver_mask(s: &Sluice) -> Vec<bool> {
        let c = config(s);
        let grid_width = c.grid_width + s.profile.outlet_buffer();
        let grid_height = c.grid_height;
        let grid_depth = c.grid_depth;
        let dx = s.profile.cell();
        let index = |p: Vec3| -> Option<usize> {
            let cell = (p / dx).floor().as_ivec3();
            if cell.min_element() < 0
                || cell.x >= grid_width as i32
                || cell.y >= grid_height as i32
                || cell.z >= grid_depth as i32
            {
                None
            } else {
                Some(
                    cell.z as usize * grid_width * grid_height
                        + cell.y as usize * grid_width
                        + cell.x as usize,
                )
            }
        };
        let mut water_cells = vec![0_u32; grid_width * grid_height * grid_depth];
        for p in &s.water {
            if let Some(i) = index(p.position) {
                water_cells[i] += 1;
            }
        }
        s.grains
            .iter()
            .map(|g| {
                let mut wet = false;
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            if let Some(i) =
                                index(g.position + Vec3::new(x as f32, y as f32, z as f32) * dx)
                            {
                                wet |= water_cells[i] > 0;
                            }
                        }
                    }
                }
                wet
            })
            .collect()
    }

    #[test]
    fn grain_wetness_matches_original_loop_at_floor_and_domain_edges() {
        for profile in [HydraulicProfile::OriginalDemo, HydraulicProfile::Legacy] {
            let mut s = Sluice::default();
            s.profile = profile;
            let c = config(&s);
            let dims = Vec3::new(
                (c.grid_width + profile.outlet_buffer()) as f32,
                c.grid_height as f32,
                c.grid_depth as f32,
            );
            let dx = profile.cell();
            let low = Vec3::splat(0.25);
            let high = dims - Vec3::splat(0.25);
            let floor = Vec3::new(3.25, s.floor_cells() as f32 + 1.25, 5.25);
            for p in [low, high, floor] {
                s.water.push(water(p * dx));
            }
            for p in [
                low,
                low - Vec3::X * 0.5,
                low - Vec3::Y * 0.5,
                low - Vec3::Z * 0.5,
                low - Vec3::X * 1.5,
                low - Vec3::Y * 1.5,
                low - Vec3::Z * 1.5,
                high,
                high + Vec3::X * 0.5,
                high + Vec3::Y * 0.5,
                high + Vec3::Z * 0.5,
                high + Vec3::X * 1.5,
                high + Vec3::Y * 1.5,
                high + Vec3::Z * 1.5,
                floor - Vec3::Y,
                floor - Vec3::Y * 2.,
                Vec3::new(c.grid_width as f32 - 0.25, 5.25, 5.25),
            ] {
                s.grains.push(grain(p * dx));
            }
            let expected = vec![
                true, true, true, true, false, false, false, true, true, true, true, false, false,
                false, true, false, false,
            ];
            assert_eq!(original_solver_mask(&s), expected, "{profile:?}");
            assert_eq!(grain_wetness(&s), expected, "{profile:?}");
        }
    }

    #[test]
    fn grain_wetness_includes_outlet_buffer_without_wrapping_outside_water() {
        for profile in [HydraulicProfile::OriginalDemo, HydraulicProfile::Legacy] {
            let mut s = Sluice::default();
            s.profile = profile;
            let c = config(&s);
            let dx = profile.cell();
            let buffer = Vec3::new(c.grid_width as f32 + 0.25, 5.25, 5.25) * dx;
            s.water.push(water(buffer));
            s.grains.push(grain(buffer));
            assert_eq!(grain_wetness(&s), original_solver_mask(&s));
            assert_eq!(grain_wetness(&s), vec![profile.outlet_buffer() > 0]);

            s.water = vec![
                water(Vec3::new(-0.25, 5.25, 5.25) * dx),
                water(
                    Vec3::new(
                        (c.grid_width + profile.outlet_buffer()) as f32 + 0.25,
                        5.25,
                        5.25,
                    ) * dx,
                ),
                water(Vec3::new(5.25, -0.25, 5.25) * dx),
                water(Vec3::new(5.25, c.grid_height as f32 + 0.25, 5.25) * dx),
                water(Vec3::new(5.25, 5.25, -0.25) * dx),
                water(Vec3::new(5.25, 5.25, c.grid_depth as f32 + 0.25) * dx),
            ];
            s.grains = s.water.iter().map(|p| grain(p.position)).collect();
            assert_eq!(grain_wetness(&s), original_solver_mask(&s));
            assert_eq!(grain_wetness(&s), vec![false; 6]);
        }
    }

    #[test]
    fn grain_wetness_covers_every_grain_and_duplicate_water_is_idempotent() {
        let mut s = Sluice::default();
        let p = Vec3::new(5.25, 12.25, 5.25) * s.profile.cell();
        for kind in [GrainKind::Sand, GrainKind::Gravel, GrainKind::Gold] {
            for resting in [false, true] {
                let mut g = grain(p);
                g.kind = kind;
                g.resting = resting;
                s.grains.push(g);
            }
        }
        assert_eq!(grain_wetness(&s), vec![false; 6]);
        s.water.push(water(p));
        let expected = grain_wetness(&s);
        assert_eq!(expected, vec![true; 6]);
        s.water.extend(std::iter::repeat_n(water(p), 7));
        assert_eq!(grain_wetness(&s), original_solver_mask(&s));
        assert_eq!(grain_wetness(&s), expected);
    }

    #[test]
    fn neighbouring_water_evaluates_all_27_queries_even_after_a_hit() {
        for occupied in [0, 1] {
            let queries = Cell::new(0);
            let index = |_| {
                queries.set(queries.get() + 1);
                Some(0)
            };
            assert_eq!(
                neighbouring_water(Vec3::new(0.015, 0.025, 0.035), 0.01, &[occupied], &index),
                occupied > 0
            );
            assert_eq!(queries.get(), 27);
        }
    }
}
