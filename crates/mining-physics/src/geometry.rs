//! The same original geometry builder supplies both collision solids and the
//! visible sluice. Placement changes the transform, never an invisible proxy.
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder};
use mining_sim::{
    sluice::{CHANNEL_WIDTH, FLUID_CELL, LENGTH},
    Sluice, Vec3,
};
pub fn config(s: &Sluice) -> SluiceConfig {
    SluiceConfig {
        grid_width: (LENGTH / FLUID_CELL).round() as usize,
        grid_height: s.floor_cells() + 18,
        grid_depth: (CHANNEL_WIDTH / FLUID_CELL).round() as usize + 2,
        cell_size: FLUID_CELL,
        floor_height_left: s.floor_cells(),
        floor_height_right: 4,
        riffle_spacing: 19,
        riffle_height: 2,
        riffle_thickness: 2,
        riffle_start_x: 27,
        riffle_end_pad: 12,
        wall_margin: 8,
        exit_width_fraction: 1.0,
        exit_height: 14,
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
        b.indices().to_vec(),
    )
}
