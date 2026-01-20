// Minimal test to debug coordinate systems

use game::editor::EditorLayout;
use glam::Vec3;

#[test]
fn test_coordinate_systems() {
    let layout = EditorLayout::new_connected();
    let gutter = &layout.gutters[0];

    let cell_size = 0.025;
    let margin = cell_size * 4.0;
    let max_width = gutter.max_width();

    let width = ((gutter.length + margin * 2.0) / cell_size).ceil() as usize;
    let height = ((gutter.wall_height + margin + 0.5) / cell_size).ceil() as usize;
    let depth = ((max_width + margin * 2.0) / cell_size).ceil() as usize;

    let grid_offset = Vec3::new(
        gutter.position.x - gutter.length / 2.0 - margin,
        gutter.position.y - margin,
        gutter.position.z - max_width / 2.0 - margin,
    );

    let dem_bounds_min = grid_offset;
    let dem_bounds_max = grid_offset
        + Vec3::new(
            width as f32 * cell_size,
            height as f32 * cell_size,
            depth as f32 * cell_size,
        );

    println!("Gutter position: {:?}", gutter.position);
    println!(
        "Gutter dimensions: length={}, width={}, wall_height={}",
        gutter.length, gutter.width, gutter.wall_height
    );
    println!("Grid offset: {:?}", grid_offset);
    println!("Grid size: {}x{}x{} cells", width, height, depth);
    println!(
        "Grid world extents: {:?} to {:?}",
        dem_bounds_min, dem_bounds_max
    );

    // Check if gutter.position is inside grid bounds
    let inside = gutter.position.x >= dem_bounds_min.x
        && gutter.position.x <= dem_bounds_max.x
        && gutter.position.y >= dem_bounds_min.y
        && gutter.position.y <= dem_bounds_max.y
        && gutter.position.z >= dem_bounds_min.z
        && gutter.position.z <= dem_bounds_max.z;

    println!("Gutter position inside grid bounds: {}", inside);
    assert!(inside, "Gutter position should be inside grid bounds!");
}
