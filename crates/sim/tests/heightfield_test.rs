//! Profiling/Feature test for Bed Heightfield
//!
//! Verified behavior:
//! The `bed_height` field on the Grid can be populated and accessed.
//! This is a precursor to using it for O(1) floor collision.

use sim::FlipSimulation;

#[test]
fn test_bed_heightfield_access() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Verify initialization
    assert_eq!(sim.grid.bed_height.len(), WIDTH, "Bed heightfield wrong size");
    for &h in &sim.grid.bed_height {
        assert_eq!(h, 0.0, "Bed height should initialize to 0.0");
    }
    
    // 2. Simulate setting a bed profile (e.g. from terrain or sediment)
    // Create a ramp
    for i in 0..WIDTH {
        sim.grid.bed_height[i] = (i as f32) * 0.5 * CELL_SIZE;
    }
    
    // 3. Verify values
    for i in 0..WIDTH {
        let expected = (i as f32) * 0.5 * CELL_SIZE;
        assert!((sim.grid.bed_height[i] - expected).abs() < 1e-5);
    }
    
    // 4. (Future) Benchmarking logic could go here
    // checking access speed vs SDF sampling.
}
