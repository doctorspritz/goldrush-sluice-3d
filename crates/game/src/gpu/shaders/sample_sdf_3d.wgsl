override SDF_PERIODIC_Z: bool = false;
// Shared cell-centred geometry query for collision and volume admission.
fn sdf_at(i: i32, j: i32, k: i32) -> f32 {
    var kk = k;
    if SDF_PERIODIC_Z { kk = ((k % i32(params.depth)) + i32(params.depth)) % i32(params.depth); }
    if (kk < 0 || kk >= i32(params.depth)) { return -params.cell_size; }
    if (i < 0 || j < 0) { return -params.cell_size; }
    if (i >= i32(params.width) || j >= i32(params.height)) { return params.cell_size; }
    return sdf[cell_index(u32(i), u32(j), u32(kk))];
}

fn sample_sdf(pos: vec3<f32>) -> f32 {
    let inv_dx = 1.0 / params.cell_size;
    let fx = pos.x * inv_dx - 0.5;
    let fy = pos.y * inv_dx - 0.5;
    let fz = pos.z * inv_dx - 0.5;

    let i0 = i32(floor(fx));
    let j0 = i32(floor(fy));
    let k0 = i32(floor(fz));

    let tx = fx - f32(i0);
    let ty = fy - f32(j0);
    let tz = fz - f32(k0);

    var result = 0.0;
    for (var dk: i32 = 0; dk < 2; dk++) {
        for (var dj: i32 = 0; dj < 2; dj++) {
            for (var di: i32 = 0; di < 2; di++) {
                let ii = i0 + di;
                let jj = j0 + dj;
                let kk = k0 + dk;

                let val = sdf_at(ii, jj, kk);

                let wx = select(tx, 1.0 - tx, di == 0);
                let wy = select(ty, 1.0 - ty, dj == 0);
                let wz = select(tz, 1.0 - tz, dk == 0);

                result += val * wx * wy * wz;
            }
        }
    }

    return result;
}
