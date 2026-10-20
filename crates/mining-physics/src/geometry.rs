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
        riffle_start_x: if original { 40 } else { 27 },
        riffle_end_pad: 12,
        wall_margin: 8,
        exit_width_fraction: 1.0,
        exit_height: if original { 12 } else { 14 },
        ..Default::default()
    }
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
