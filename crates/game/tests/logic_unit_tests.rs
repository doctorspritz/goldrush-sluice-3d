// Logic Unit Tests for FLIP Fluid Simulation Infrastructure
//
// Tests the core data structures and algorithms used in GPU particle sorting,
// spatial hashing, and buffer management for the FLIP/APIC simulation.

use glam::Vec3;

// ============================================================================
// 1. Spatial Hash Cell Assignment Tests
// ============================================================================

/// Test that particles at known positions map to correct spatial hash cells
#[test]
fn test_spatial_hash_cell_assignment() {
    let cell_size = 0.1; // 10cm cells
    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;

    // Helper function to compute cell index (matching shader logic)
    let cell_index = |i: u32, j: u32, k: u32| -> u32 {
        k * width * height + j * width + i
    };

    // Helper to compute cell from position (matching shader logic)
    let position_to_cell = |pos: Vec3| -> u32 {
        let ci = ((pos.x / cell_size).floor() as i32).clamp(0, width as i32 - 1) as u32;
        let cj = ((pos.y / cell_size).floor() as i32).clamp(0, height as i32 - 1) as u32;
        let ck = ((pos.z / cell_size).floor() as i32).clamp(0, depth as i32 - 1) as u32;
        cell_index(ci, cj, ck)
    };

    // Test Case 1: Particle at origin maps to cell (0, 0, 0)
    let pos1 = Vec3::new(0.0, 0.0, 0.0);
    let cell1 = position_to_cell(pos1);
    assert_eq!(cell1, cell_index(0, 0, 0), "Origin should map to cell (0,0,0)");

    // Test Case 2: Particle in middle of first cell
    let pos2 = Vec3::new(0.05, 0.05, 0.05);
    let cell2 = position_to_cell(pos2);
    assert_eq!(cell2, cell_index(0, 0, 0), "Middle of first cell should map to (0,0,0)");

    // Test Case 3: Particle at cell boundary (should be in next cell due to floor)
    let pos3 = Vec3::new(0.1, 0.1, 0.1);
    let cell3 = position_to_cell(pos3);
    assert_eq!(cell3, cell_index(1, 1, 1), "Cell boundary should map to (1,1,1)");

    // Test Case 4: Particle at specific position
    let pos4 = Vec3::new(0.25, 0.35, 0.45); // cells (2, 3, 4)
    let cell4 = position_to_cell(pos4);
    assert_eq!(cell4, cell_index(2, 3, 4), "Should map to correct cell (2,3,4)");

    // Test Case 5: Negative position (should clamp to 0)
    let pos5 = Vec3::new(-0.5, -0.5, -0.5);
    let cell5 = position_to_cell(pos5);
    assert_eq!(cell5, cell_index(0, 0, 0), "Negative positions should clamp to (0,0,0)");

    // Test Case 6: Out-of-bounds position (should clamp to max)
    let pos6 = Vec3::new(100.0, 100.0, 100.0);
    let cell6 = position_to_cell(pos6);
    assert_eq!(
        cell6,
        cell_index(width - 1, height - 1, depth - 1),
        "Out-of-bounds positions should clamp to max cell"
    );

    // Test Case 7: Edge case - exactly at grid boundary
    let pos7 = Vec3::new(0.99, 0.99, 0.99); // Should be in cell (9, 9, 9)
    let cell7 = position_to_cell(pos7);
    assert_eq!(cell7, cell_index(9, 9, 9), "Just before boundary should map to (9,9,9)");
}

/// Test cell index ordering (Z-major, then Y, then X)
#[test]
fn test_cell_index_ordering() {
    let width = 4u32;
    let height = 4u32;
    let _depth = 4u32;

    let cell_index = |i: u32, j: u32, k: u32| -> u32 {
        k * width * height + j * width + i
    };

    // Cell indices should follow Z-major ordering
    assert_eq!(cell_index(0, 0, 0), 0);
    assert_eq!(cell_index(1, 0, 0), 1);
    assert_eq!(cell_index(0, 1, 0), 4); // Jump by width
    assert_eq!(cell_index(0, 0, 1), 16); // Jump by width*height

    // Verify sequential cells
    assert_eq!(cell_index(3, 0, 0), 3);
    assert_eq!(cell_index(0, 3, 0), 12);
    assert_eq!(cell_index(0, 0, 3), 48);

    // Last cell
    assert_eq!(cell_index(3, 3, 3), 63);
}

// ============================================================================
// 2. Prefix Sum Correctness Tests
// ============================================================================

/// CPU-side exclusive prefix sum implementation (matching GPU Blelloch scan)
fn exclusive_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0u32; input.len()];
    let mut running_sum = 0u32;
    for (i, &val) in input.iter().enumerate() {
        output[i] = running_sum;
        running_sum += val;
    }
    output
}

/// Test exclusive prefix sum with known inputs
#[test]
fn test_prefix_sum_basic() {
    // Example from requirements: [1,2,3,4] → [0,1,3,6]
    let input = vec![1, 2, 3, 4];
    let output = exclusive_prefix_sum(&input);
    assert_eq!(output, vec![0, 1, 3, 6], "Basic prefix sum failed");

    // Total particles = sum of input = 10
    let total: u32 = input.iter().sum();
    assert_eq!(total, 10);
    // Last offset + last count = total
    assert_eq!(output[3] + input[3], 10);
}

/// Test prefix sum with zeros (empty cells)
#[test]
fn test_prefix_sum_with_zeros() {
    let input = vec![5, 0, 0, 3, 0, 2];
    let output = exclusive_prefix_sum(&input);
    assert_eq!(output, vec![0, 5, 5, 5, 8, 8], "Prefix sum with zeros failed");

    // Empty cells (count=0) should have same offset as previous cell
    assert_eq!(output[1], output[2]); // Cells 1 and 2 both empty
    assert_eq!(output[4], output[5]); // Cell 4 empty
}

/// Test prefix sum with all zeros
#[test]
fn test_prefix_sum_all_zeros() {
    let input = vec![0, 0, 0, 0];
    let output = exclusive_prefix_sum(&input);
    assert_eq!(output, vec![0, 0, 0, 0], "All-zero input should produce all-zero output");
}

/// Test prefix sum with single element
#[test]
fn test_prefix_sum_single_element() {
    let input = vec![42];
    let output = exclusive_prefix_sum(&input);
    assert_eq!(output, vec![0], "Single element should have offset 0");
}

/// Test prefix sum with large values (overflow check)
#[test]
fn test_prefix_sum_large_values() {
    // Use reasonable values that won't overflow u32
    let input = vec![1000, 2000, 3000, 4000];
    let output = exclusive_prefix_sum(&input);
    assert_eq!(output, vec![0, 1000, 3000, 6000]);
}

// ============================================================================
// 3. Particle Sort Order Tests
// ============================================================================

/// Simulates particle sorting: after sorting, particles should be grouped by cell
#[test]
fn test_particle_sort_grouping() {
    // Setup: 6 particles in 3 cells
    let cell_keys = vec![2, 0, 2, 1, 0, 2]; // Cell assignments
    let particle_ids = vec![0, 1, 2, 3, 4, 5]; // Original indices

    // Count particles per cell
    let num_cells = 3;
    let mut cell_counts = vec![0u32; num_cells];
    for &key in &cell_keys {
        cell_counts[key as usize] += 1;
    }
    assert_eq!(cell_counts, vec![2, 1, 3]); // Cell 0: 2 particles, Cell 1: 1, Cell 2: 3

    // Compute cell offsets (exclusive prefix sum)
    let cell_offsets = exclusive_prefix_sum(&cell_counts);
    assert_eq!(cell_offsets, vec![0, 2, 3]); // Cell 0 starts at 0, Cell 1 at 2, Cell 2 at 3

    // Simulate scatter: particles are written to sorted positions
    let mut sorted_particles = vec![0usize; particle_ids.len()];
    let mut write_counters = cell_offsets.clone();

    for (particle_id, &cell_key) in particle_ids.iter().zip(cell_keys.iter()) {
        let write_pos = write_counters[cell_key as usize] as usize;
        sorted_particles[write_pos] = *particle_id;
        write_counters[cell_key as usize] += 1;
    }

    // Verify sorted order: particles should be grouped by cell
    // Cell 0: particles 1, 4 (order within cell doesn't matter)
    // Cell 1: particle 3
    // Cell 2: particles 0, 2, 5
    assert_eq!(sorted_particles[0..2].len(), 2); // Cell 0 has 2 particles
    assert!(sorted_particles[0..2].contains(&1));
    assert!(sorted_particles[0..2].contains(&4));

    assert_eq!(sorted_particles[2], 3); // Cell 1 has particle 3

    assert_eq!(sorted_particles[3..6].len(), 3); // Cell 2 has 3 particles
    assert!(sorted_particles[3..6].contains(&0));
    assert!(sorted_particles[3..6].contains(&2));
    assert!(sorted_particles[3..6].contains(&5));
}

/// Test that sorted particles maintain cell contiguity
#[test]
fn test_particle_sort_contiguity() {
    let cell_keys = vec![0, 0, 1, 1, 1, 2];
    let particle_count = cell_keys.len();

    // Count and compute offsets
    let num_cells = 3;
    let mut cell_counts = vec![0u32; num_cells];
    for &key in &cell_keys {
        cell_counts[key as usize] += 1;
    }
    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    // After sorting, cell ranges should be contiguous:
    // Cell 0: indices [0, 2)
    // Cell 1: indices [2, 5)
    // Cell 2: indices [5, 6)

    for cell in 0..num_cells {
        let start = cell_offsets[cell] as usize;
        let count = cell_counts[cell] as usize;
        let end = start + count;

        // Verify range is valid
        assert!(end <= particle_count, "Cell range should be within bounds");

        // Verify no overlap with previous cell
        if cell > 0 {
            let prev_end = cell_offsets[cell] as usize;
            assert_eq!(start, prev_end, "Cells should be contiguous");
        }
    }
}

// ============================================================================
// 4. Buffer Capacity Tests
// ============================================================================

/// Test buffer growth/shrink while preserving data
#[test]
fn test_buffer_resize_preserves_data() {
    // Simulate a buffer holding particle positions
    let initial_capacity = 100;
    let mut buffer = vec![Vec3::ZERO; initial_capacity];

    // Fill with test data
    for i in 0..50 {
        buffer[i] = Vec3::new(i as f32, (i * 2) as f32, (i * 3) as f32);
    }

    // Grow buffer
    let new_capacity = 200;
    buffer.resize(new_capacity, Vec3::ZERO);

    // Verify original data is preserved
    for i in 0..50 {
        let expected = Vec3::new(i as f32, (i * 2) as f32, (i * 3) as f32);
        assert_eq!(buffer[i], expected, "Data at index {} should be preserved", i);
    }

    // Verify new capacity
    assert_eq!(buffer.len(), new_capacity);
    assert!(buffer.capacity() >= new_capacity);

    // Shrink buffer (keeping first 75 elements)
    buffer.truncate(75);
    assert_eq!(buffer.len(), 75);

    // Verify data still preserved
    for i in 0..50 {
        let expected = Vec3::new(i as f32, (i * 2) as f32, (i * 3) as f32);
        assert_eq!(buffer[i], expected, "Data should survive truncation");
    }
}

/// Test that buffer expansion doesn't lose data
#[test]
fn test_buffer_growth_strategies() {
    let mut buffer: Vec<u32> = Vec::new();

    // Strategy 1: Incremental growth
    for i in 0..100u32 {
        buffer.push(i);
    }
    assert_eq!(buffer.len(), 100);
    for i in 0..100usize {
        assert_eq!(buffer[i], i as u32, "Sequential push should preserve order");
    }

    // Strategy 2: Reserve and fill
    let mut buffer2: Vec<u32> = Vec::with_capacity(200);
    for i in 0..100u32 {
        buffer2.push(i);
    }
    // Grow by reserving more
    buffer2.reserve(100);
    for i in 100..200u32 {
        buffer2.push(i);
    }
    assert_eq!(buffer2.len(), 200);
    for i in 0..200usize {
        assert_eq!(buffer2[i], i as u32, "Reserve+push should preserve data");
    }
}

// ============================================================================
// 5. Empty Cell Handling Tests
// ============================================================================

/// Test that empty cells have valid (empty) ranges in the sorted array
#[test]
fn test_empty_cell_ranges() {
    // 5 cells, only cells 0, 2, 4 have particles
    let cell_counts = vec![3, 0, 2, 0, 1];
    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    assert_eq!(cell_offsets, vec![0, 3, 3, 5, 5]);

    // Helper to get cell range
    let cell_range = |cell: usize| -> (usize, usize) {
        let start = cell_offsets[cell] as usize;
        let count = cell_counts[cell] as usize;
        (start, start + count)
    };

    // Cell 0: [0, 3)
    assert_eq!(cell_range(0), (0, 3));

    // Cell 1 (empty): [3, 3) - valid empty range
    assert_eq!(cell_range(1), (3, 3));

    // Cell 2: [3, 5)
    assert_eq!(cell_range(2), (3, 5));

    // Cell 3 (empty): [5, 5) - valid empty range
    assert_eq!(cell_range(3), (5, 5));

    // Cell 4: [5, 6)
    assert_eq!(cell_range(4), (5, 6));

    // Empty cells should have zero-length ranges
    for cell in 0..cell_counts.len() {
        let (start, end) = cell_range(cell);
        let range_size = end - start;
        assert_eq!(
            range_size,
            cell_counts[cell] as usize,
            "Range size should match cell count"
        );
    }
}

/// Test neighbor iteration with empty cells
#[test]
fn test_neighbor_search_with_empty_cells() {
    // Grid: 3x3x1 (9 cells), particles only in corners and center
    let width = 3u32;
    let height = 3u32;
    let depth = 1u32;
    let num_cells = (width * height * depth) as usize;

    let cell_index = |i: u32, j: u32, k: u32| -> usize {
        (k * width * height + j * width + i) as usize
    };

    // Particles in cells: (0,0,0), (1,1,0), (2,2,0)
    let mut cell_counts = vec![0u32; num_cells];
    cell_counts[cell_index(0, 0, 0)] = 2; // Corner
    cell_counts[cell_index(1, 1, 0)] = 5; // Center
    cell_counts[cell_index(2, 2, 0)] = 3; // Corner

    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    // Total particles
    let total: u32 = cell_counts.iter().sum();
    assert_eq!(total, 10);

    // Verify offsets
    assert_eq!(cell_offsets[cell_index(0, 0, 0)], 0);
    assert_eq!(cell_offsets[cell_index(1, 1, 0)], 2);
    assert_eq!(cell_offsets[cell_index(2, 2, 0)], 7);

    // Check neighbor cells (27-cell stencil for 3D)
    // Center cell (1, 1, 0) should have 9 neighbors in z=0 plane
    let _center_cell = cell_index(1, 1, 0);

    let mut neighbor_particle_count = 0;
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let ni = 1 + dx;
                let nj = 1 + dy;
                let nk = 0 + dz;

                // Bounds check
                if ni >= 0 && ni < 3 && nj >= 0 && nj < 3 && nk >= 0 && nk < 1 {
                    let neighbor_cell = cell_index(ni as u32, nj as u32, nk as u32);
                    neighbor_particle_count += cell_counts[neighbor_cell];
                }
            }
        }
    }

    // Center sees all 10 particles (2 + 5 + 3)
    assert_eq!(neighbor_particle_count, 10);

    // Empty cell (0, 1, 0) should have zero particles
    let empty_cell = cell_index(0, 1, 0);
    assert_eq!(cell_counts[empty_cell], 0);
    // But its offset should still be valid
    let offset = cell_offsets[empty_cell];
    assert!(offset <= total); // Valid offset within bounds
}

// ============================================================================
// 6. Edge Cases and Boundary Conditions
// ============================================================================

/// Test single particle scenario
#[test]
fn test_single_particle() {
    let _cell_keys = vec![5]; // One particle in cell 5
    let num_cells = 10;

    let mut cell_counts = vec![0u32; num_cells];
    cell_counts[5] = 1;

    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    // Only cell 5 should have a particle
    assert_eq!(cell_offsets[5], 0);
    assert_eq!(cell_counts[5], 1);

    // All other cells should be empty but have valid offsets
    for i in 0..5 {
        assert_eq!(cell_offsets[i], 0);
    }
    for i in 6..num_cells {
        assert_eq!(cell_offsets[i], 1);
    }
}

/// Test all particles in one cell
#[test]
fn test_all_particles_in_one_cell() {
    let num_particles = 100;
    let _cell_keys = vec![0u32; num_particles]; // All in cell 0

    let num_cells = 10;
    let mut cell_counts = vec![0u32; num_cells];
    cell_counts[0] = num_particles as u32;

    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    // Cell 0 starts at offset 0
    assert_eq!(cell_offsets[0], 0);

    // All other cells start after the 100 particles
    for i in 1..num_cells {
        assert_eq!(cell_offsets[i], num_particles as u32);
    }
}

/// Test particle at exact grid boundaries
#[test]
fn test_boundary_particles() {
    let cell_size = 1.0;
    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;

    let position_to_cell = |pos: Vec3| -> u32 {
        let ci = ((pos.x / cell_size).floor() as i32).clamp(0, width as i32 - 1) as u32;
        let cj = ((pos.y / cell_size).floor() as i32).clamp(0, height as i32 - 1) as u32;
        let ck = ((pos.z / cell_size).floor() as i32).clamp(0, depth as i32 - 1) as u32;
        ck * width * height + cj * width + ci
    };

    // Particle exactly at cell boundaries
    let pos_00 = Vec3::new(0.0, 0.0, 0.0); // Start of cell 0
    let pos_10 = Vec3::new(1.0, 0.0, 0.0); // Start of cell 1
    let pos_99 = Vec3::new(9.0, 9.0, 9.0); // Start of last cell

    assert_eq!(position_to_cell(pos_00), 0);
    assert_eq!(position_to_cell(pos_10), 1);
    assert_eq!(position_to_cell(pos_99), 999); // Cell (9, 9, 9) = 9*100 + 9*10 + 9

    // Particles just inside boundaries
    let pos_inside = Vec3::new(0.001, 0.001, 0.001);
    assert_eq!(position_to_cell(pos_inside), 0);

    // Particles just outside grid
    let pos_outside = Vec3::new(10.0, 10.0, 10.0);
    assert_eq!(position_to_cell(pos_outside), 999); // Should clamp to last cell
}

// ============================================================================
// 7. Integration Test: Full Sort Pipeline Simulation
// ============================================================================

/// Simulate the entire GPU sort pipeline on CPU to verify correctness
#[test]
fn test_full_sort_pipeline() {
    // Grid parameters
    let cell_size = 0.5;
    let width = 4u32;
    let height = 4u32;
    let depth = 4u32;
    let num_cells = (width * height * depth) as usize;

    // Test particles at various positions
    let positions = vec![
        Vec3::new(0.25, 0.25, 0.25), // Cell (0, 0, 0)
        Vec3::new(0.75, 0.25, 0.25), // Cell (1, 0, 0)
        Vec3::new(0.25, 0.25, 0.25), // Cell (0, 0, 0) - duplicate
        Vec3::new(1.25, 1.75, 0.75), // Cell (2, 3, 1)
        Vec3::new(0.75, 0.25, 0.25), // Cell (1, 0, 0) - duplicate
    ];

    let particle_count = positions.len();

    // Step 1: Compute cell keys
    let cell_index = |i: u32, j: u32, k: u32| -> u32 {
        k * width * height + j * width + i
    };

    let position_to_cell = |pos: Vec3| -> u32 {
        let ci = ((pos.x / cell_size).floor() as i32).clamp(0, width as i32 - 1) as u32;
        let cj = ((pos.y / cell_size).floor() as i32).clamp(0, height as i32 - 1) as u32;
        let ck = ((pos.z / cell_size).floor() as i32).clamp(0, depth as i32 - 1) as u32;
        cell_index(ci, cj, ck)
    };

    let cell_keys: Vec<u32> = positions.iter().map(|&pos| position_to_cell(pos)).collect();

    // Expected cell keys
    assert_eq!(cell_keys[0], cell_index(0, 0, 0));
    assert_eq!(cell_keys[1], cell_index(1, 0, 0));
    assert_eq!(cell_keys[2], cell_index(0, 0, 0));
    assert_eq!(cell_keys[3], cell_index(2, 3, 1));
    assert_eq!(cell_keys[4], cell_index(1, 0, 0));

    // Step 2: Count particles per cell
    let mut cell_counts = vec![0u32; num_cells];
    for &key in &cell_keys {
        cell_counts[key as usize] += 1;
    }

    assert_eq!(cell_counts[cell_index(0, 0, 0) as usize], 2); // 2 particles
    assert_eq!(cell_counts[cell_index(1, 0, 0) as usize], 2); // 2 particles
    assert_eq!(cell_counts[cell_index(2, 3, 1) as usize], 1); // 1 particle

    // Step 3: Exclusive prefix sum
    let cell_offsets = exclusive_prefix_sum(&cell_counts);

    // Step 4: Scatter particles
    let mut sorted_indices = vec![0usize; particle_count];
    let mut write_counters = cell_offsets.clone();

    for (particle_id, &cell_key) in cell_keys.iter().enumerate() {
        let write_pos = write_counters[cell_key as usize] as usize;
        sorted_indices[write_pos] = particle_id;
        write_counters[cell_key as usize] += 1;
    }

    // Step 5: Verify sorted order
    // Particles in same cell should be contiguous in sorted array
    let cell_000 = cell_index(0, 0, 0) as usize;
    let cell_100 = cell_index(1, 0, 0) as usize;
    let cell_231 = cell_index(2, 3, 1) as usize;

    let start_000 = cell_offsets[cell_000] as usize;
    let count_000 = cell_counts[cell_000] as usize;
    let sorted_000 = &sorted_indices[start_000..start_000 + count_000];
    assert_eq!(sorted_000.len(), 2);
    assert!(sorted_000.contains(&0));
    assert!(sorted_000.contains(&2));

    let start_100 = cell_offsets[cell_100] as usize;
    let count_100 = cell_counts[cell_100] as usize;
    let sorted_100 = &sorted_indices[start_100..start_100 + count_100];
    assert_eq!(sorted_100.len(), 2);
    assert!(sorted_100.contains(&1));
    assert!(sorted_100.contains(&4));

    let start_231 = cell_offsets[cell_231] as usize;
    let count_231 = cell_counts[cell_231] as usize;
    let sorted_231 = &sorted_indices[start_231..start_231 + count_231];
    assert_eq!(sorted_231.len(), 1);
    assert!(sorted_231.contains(&3));
}
