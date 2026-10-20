use mining_physics::geometry;
use mining_sim::{sluice::FLUID_CELL, Placement, Session, Vec3};

#[test]
fn world_space_floor_faces_up_after_grid_to_placement_transform() {
    let mut session = Session::new();
    let placement = Placement::suggested(&session.site);
    session.sluice.placement = Some(placement);
    let (vertices, indices) = geometry::mesh(&session.sluice);
    let floor_offset = (session.sluice.floor_cells() + 1) as f32 * FLUID_CELL;
    let mut faces = 0;
    for triangle in indices.chunks_exact(3) {
        let a = Vec3::from_array(vertices[triangle[0] as usize]);
        let b = Vec3::from_array(vertices[triangle[1] as usize]);
        let c = Vec3::from_array(vertices[triangle[2] as usize]);
        let normal = (b - a).cross(c - a).normalize_or_zero();
        let midpoint = (a + b + c) / 3.0 - placement.inlet;
        if (0.2..0.4).contains(&midpoint.dot(placement.forward()))
            && midpoint.dot(placement.right()).abs() < 0.2
            && midpoint.y + floor_offset > 0.1
            && normal.y.abs() > 0.99
        {
            assert!(normal.y > 0.99, "floor top was inside-out: {normal:?}");
            faces += 1;
        }
    }
    assert!(
        faces > 10,
        "must inspect actual original-builder floor faces"
    );
}

#[test]
fn every_visible_face_points_from_collision_solid_into_air() {
    let mut s = Session::new();
    let mut p = Placement::suggested(&s.site);
    p.yaw = 0.7;
    s.sluice.placement = Some(p);
    let config = geometry::config(&s.sluice);
    let offset = (s.sluice.floor_cells() + 1) as f32 * FLUID_CELL;
    let solid_at = |point: Vec3| {
        let relative = point - p.inlet;
        let grid = Vec3::new(
            relative.dot(p.forward()),
            relative.y + offset,
            relative.dot(p.right()) + mining_sim::sluice::CHANNEL_WIDTH * 0.5 + FLUID_CELL,
        );
        let cell = (grid / FLUID_CELL).floor();
        cell.x >= 0.0
            && cell.y >= 0.0
            && cell.z >= 0.0
            && cell.x < config.grid_width as f32
            && cell.y < config.grid_height as f32
            && cell.z < config.grid_depth as f32
            && config.is_solid(cell.x as usize, cell.y as usize, cell.z as usize)
    };
    let (vertices, indices) = geometry::mesh(&s.sluice);
    for triangle in indices.chunks_exact(3) {
        let a = Vec3::from_array(vertices[triangle[0] as usize]);
        let b = Vec3::from_array(vertices[triangle[1] as usize]);
        let c = Vec3::from_array(vertices[triangle[2] as usize]);
        let normal = (b - a).cross(c - a).normalize();
        let center = (a + b + c) / 3.0;
        assert!(
            !solid_at(center + normal * 0.003),
            "face points into solid: {center:?} / {normal:?}"
        );
        assert!(
            solid_at(center - normal * 0.003),
            "face must have solid behind it"
        );
    }
}
