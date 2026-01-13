
use glam::Vec3;

// COPY of TestFloor from sim3d/src/test_geometry.rs to avoid dependency issues if needed,
// but better to import if possible. Let's try importing first.
// If not possible (private fields?), we copypasta.
// Structs are pub.

// Mocking the structs to avoid complex build deps if simple.
// Actually, using the crate is better if I can run it as an example.
// But `sim3d` is a lib. I can add an example to `game` that uses `sim3d`.

fn main() {
    println!("Testing SDF Generation...");
    
    let w = 40;
    let h = 60;
    let d = 40;
    let cell_size = 0.025;
    let offset = Vec3::new(
        -(w as f32 * cell_size) * 0.5,
        -(h as f32 * cell_size) * 0.5,
        -(d as f32 * cell_size) * 0.5,
    );
    
    println!("Grid: {}x{}x{}", w, h, d);
    println!("Offset: {:?}", offset);
    
    // Manual calculation of expected Sample[0]
    // Index 0 -> x=0, y=0, z=0
    let x = 0; let y = 0; let z = 0;
    let pos = offset + Vec3::new(
        (x as f32 + 0.5) * cell_size,
        (y as f32 + 0.5) * cell_size,
        (z as f32 + 0.5) * cell_size,
    );
    println!("Sample[0] Pos: {:?}", pos);
    
    let floor_y = 0.0;
    let thickness = 0.5;
    let floor_top = floor_y;
    let floor_bottom = floor_y - thickness;
    
    let dist = if pos.y >= floor_top {
        pos.y - floor_top
    } else if pos.y <= floor_bottom {
        floor_bottom - pos.y
    } else {
        let d_top = floor_top - pos.y;
        let d_bottom = pos.y - floor_bottom;
        -d_top.min(d_bottom)
    };
    
    println!("Calculated Dist: {}", dist);
    
    // Check against 0.3875 seen in logs
    if (dist - 0.3875).abs() < 0.001 {
        println!("MATCHES LOG! Logic produces 0.3875.");
    } else {
        println!("MISMATCH! Logic produced {}, log had 0.3875.", dist);
    }
}
