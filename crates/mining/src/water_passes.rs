//! Shared by the renderer and its complete-chain GPU regression.
//! Four geometry filters precede two optical exchanges. All geometry filters
//! must retain resolved channel crests; a broad blur here reintroduces domes.
pub fn surface_passes<T: Copy>(geometry_x: T, geometry_y: T, optical_x: T, optical_y: T) -> [T; 6] {
    [
        geometry_x, geometry_y, geometry_x, geometry_y, optical_x, optical_y,
    ]
}
