//! Independent controls and opt-in timing of the corrected S C S support.
//! Ordinary tests use bounded synthetic geometry and never load a saved scene.
use super::*;
use bevy::math::{DMat3, DVec3};
use std::cell::RefCell;

const MAX_CAPTURE: usize = 65_536;
const TOLERANCE: f64 = 2e-5;
std::thread_local! {
    static CAPTURE: RefCell<Option<Vec<Kernel>>> = const { RefCell::new(None) };
}

/// Called only by the test build's actual water-instance construction. Direct
/// old/new support timings contain neither this hook nor a TLS lookup.
pub(crate) fn record_kernel(kernel: Kernel) {
    CAPTURE.with(|capture| {
        if let Some(kernels) = capture.borrow_mut().as_mut() {
            assert!(
                kernels.len() < MAX_CAPTURE,
                "unbounded render-kernel capture"
            );
            kernels.push(kernel);
        }
    });
}
fn tensor(axes: [DVec3; 3]) -> DMat3 {
    axes.into_iter()
        .map(|a| DMat3::from_cols(a * a.x, a * a.y, a * a.z))
        .fold(DMat3::ZERO, |a, b| a + b)
}
fn max_abs(matrix: DMat3) -> f64 {
    matrix
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0., f64::max)
}
fn reference(kernel: Kernel) -> DMat3 {
    let axes = kernel.axes.map(|axis| axis.as_dvec3());
    let n = kernel.normal.as_dvec3().normalize_or_zero();
    let thickness = f64::from(1. - 0.5 * kernel.planarity * kernel.bulk);
    let s = DMat3::IDENTITY - DMat3::from_cols(n * n.x, n * n.y, n * n.z) * (1. - thickness);
    s * tensor(axes) * s
}
#[derive(Default)]
struct Errors {
    cases: usize,
    tensor_abs: f64,
    cosine: f64,
    determinant_relative: f64,
    volume_orthogonality_relative: f64,
    shader_radius_relative: f64,
    tangent_abs: f64,
}
impl Errors {
    fn check(&mut self, kernel: Kernel, output: [Vec3; 3]) {
        let axes = output.map(|axis| axis.as_dvec3());
        assert!(
            axes.iter().all(|axis| axis.is_finite()
                && (0.225 - TOLERANCE..=1.8 + TOLERANCE).contains(&axis.length())),
            "support left its admitted render-axis bounds"
        );
        let actual = tensor(axes);
        let expected = reference(kernel);
        let error = max_abs(actual - expected);
        assert!(error < TOLERANCE, "S C S tensor mismatch: {error}");
        self.tensor_abs = self.tensor_abs.max(error);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let cosine = (axes[a].dot(axes[b]) / (axes[a].length() * axes[b].length())).abs();
            assert!(
                cosine < TOLERANCE,
                "shader support axes are not orthogonal: {cosine}"
            );
            self.cosine = self.cosine.max(cosine);
        }
        let input = kernel.axes.map(|axis| axis.as_dvec3());
        let factor = if kernel.normal == Vec3::ZERO {
            1.
        } else {
            f64::from(1. - 0.5 * kernel.planarity * kernel.bulk)
        };
        let expected_determinant = DMat3::from_cols(input[0], input[1], input[2])
            .determinant()
            .abs()
            * factor;
        let determinant = DMat3::from_cols(axes[0], axes[1], axes[2])
            .determinant()
            .abs();
        let product = axes.iter().map(|a| a.length()).product::<f64>();
        let determinant_error = (determinant / expected_determinant - 1.).abs();
        let volume_error = (product / determinant - 1.).abs();
        assert!(
            determinant_error < TOLERANCE,
            "normal compression changed determinant: {determinant_error}"
        );
        assert!(
            volume_error < TOLERANCE,
            "length product is not physical support determinant: {volume_error}"
        );
        self.determinant_relative = self.determinant_relative.max(determinant_error);
        self.volume_orthogonality_relative = self.volume_orthogonality_relative.max(volume_error);
        // The shader divides each axis projection by that axis's squared length.
        // Tensor identity alone would not detect a non-diagonalized Gram matrix;
        // this independently evaluates the actual ellipsoid ray interpretation.
        for direction in [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(0.3, 0.8, 0.6).normalize(),
            DVec3::new(-0.7, 0.2, 0.5).normalize(),
        ] {
            let shader_radius = axes
                .iter()
                .map(|a| (direction.dot(*a) / a.length_squared()).powi(2))
                .sum::<f64>()
                .sqrt()
                .recip();
            let expected_radius = direction.dot(expected.inverse() * direction).sqrt().recip();
            let radius_error = (shader_radius / expected_radius - 1.).abs();
            assert!(
                radius_error < TOLERANCE,
                "shader radial support mismatch: {radius_error}"
            );
            self.shader_radius_relative = self.shader_radius_relative.max(radius_error);
        }
        let normal = kernel.normal.as_dvec3().normalize_or_zero();
        if normal != DVec3::ZERO {
            let t = normal
                .cross(if normal.x.abs() < 0.8 {
                    DVec3::X
                } else {
                    DVec3::Y
                })
                .normalize();
            for tangent in [t, normal.cross(t)] {
                let difference =
                    (tangent.dot(actual * tangent) - tangent.dot(tensor(input) * tangent)).abs();
                assert!(
                    difference < TOLERANCE,
                    "tangent footprint changed: {difference}"
                );
                self.tangent_abs = self.tangent_abs.max(difference);
            }
        }
        self.cases += 1;
    }
    fn json(&self) -> serde_json::Value {
        serde_json::json!({"cases":self.cases,"maximum_tensor_absolute_error":self.tensor_abs,
            "maximum_axis_pair_cosine":self.cosine,"maximum_determinant_relative_error":self.determinant_relative,
            "maximum_length_product_vs_determinant_relative_error":self.volume_orthogonality_relative,
            "maximum_shader_radial_relative_error":self.shader_radius_relative,"maximum_tangent_quadratic_error":self.tangent_abs,
            "unchanged_absolute_tensor_acceptance":TOLERANCE,"normalized_shape_acceptance":TOLERANCE})
    }
}
fn kernel(axes: [Vec3; 3], normal: Vec3, planarity: f32, bulk: f32) -> Kernel {
    Kernel {
        axes,
        normal,
        planarity,
        bulk,
        center: Vec3::ZERO,
        channel_center: Vec3::ZERO,
    }
}
fn bits(axes: [Vec3; 3]) -> [[u32; 3]; 3] {
    axes.map(|axis| axis.to_array().map(f32::to_bits))
}

// This undiagonalized support has the correct tensor but is not a valid
// orthogonal shader frame. Keep a negative control for the acceptance oracle.
#[test]
#[should_panic(expected = "shader support axes are not orthogonal")]
fn tensor_identity_alone_rejects_an_invalid_shader_frame() {
    let k = kernel([Vec3::X, Vec3::Y, Vec3::Z], Vec3::new(1., 1., 0.), 1., 1.);
    Errors::default().check(
        k,
        [
            Vec3::new(0.75, -0.25, 0.),
            Vec3::new(-0.25, 0.75, 0.),
            Vec3::Z,
        ],
    );
}

#[test]
fn right_gram_support_matches_independent_tensors_and_shader_rays() {
    let mut current = Errors::default();
    let mut previous = Errors::default();
    let lengths = [
        Vec3::splat(0.45),
        Vec3::new(0.45, 0.45, 1.8),
        Vec3::new(1.8, 0.72, 0.72),
        Vec3::new(0.45, 0.85, 1.8),
        Vec3::new(0.7, 1.1, 1.7),
    ];
    for phase in 0..6 {
        let rotation = Mat3::from_quat(Quat::from_axis_angle(
            Vec3::new(0.3, 0.7, -0.2).normalize(),
            phase as f32 * 0.37,
        ));
        for length in lengths {
            let original = [
                rotation.x_axis * length.x,
                rotation.y_axis * length.y,
                rotation.z_axis * length.z,
            ];
            for axes in [
                original,
                [original[2], -original[0], original[1]],
                [-original[1], original[2], original[0]],
            ] {
                for normal in [
                    Vec3::Y,
                    Vec3::new(0.2, 0.9, -0.4).normalize(),
                    Vec3::new(-0.8, 0.1, 0.2).normalize(),
                    Vec3::ZERO,
                ] {
                    for (planarity, bulk) in [(1., 1.), (0.3, 0.8), (0., 1.), (1., 0.)] {
                        let k = kernel(axes, rotation * normal, planarity, bulk);
                        let a = k.channel_axes();
                        current.check(k, a);
                        previous.check(k, k.channel_axes_world_reference());
                        assert!(
                            max_abs(
                                tensor(a.map(|v| v.as_dvec3()))
                                    - tensor(
                                        k.channel_axes_world_reference().map(|v| v.as_dvec3())
                                    )
                            ) < TOLERANCE
                        );
                        if bulk == 0. || planarity == 0. || normal == Vec3::ZERO {
                            assert_eq!(bits(a), bits(axes), "inactive support must bypass exactly");
                        } else {
                            for transformed_normal in [-k.normal, k.normal * 3.] {
                                let alternative = Kernel {
                                    normal: transformed_normal,
                                    ..k
                                };
                                current.check(alternative, alternative.channel_axes());
                            }
                        }
                    }
                }
            }
        }
    }
    // Sweep through the normal/tangent eigenvalue tie while the arbitrary
    // repeated-eigenvalue frame rotates. Compare tensor increments, not basis
    // signs or ordering, which are free to change at degeneracy.
    let mut prior = None;
    for phase in 0..129 {
        let angle = phase as f32 * 0.03125;
        let rotation = Mat3::from_rotation_z(0.6) * Mat3::from_rotation_y(angle);
        let frame = Mat3::from_rotation_x(angle * 0.43);
        let delta = (phase as f32 - 64.) * 0.000001;
        let axes =
            rotation * frame * Mat3::from_diagonal(Vec3::new(1.8, 0.72 + delta, 0.72 - delta));
        let k = kernel(
            [axes.x_axis, axes.y_axis, axes.z_axis],
            rotation * Vec3::Y,
            1.,
            1.,
        );
        let output = k.channel_axes();
        current.check(k, output);
        previous.check(k, k.channel_axes_world_reference());
        let actual = tensor(output.map(|v| v.as_dvec3()));
        let expected = reference(k);
        if let Some((last_actual, last_expected)) = prior {
            assert!(
                max_abs((actual - last_actual) - (expected - last_expected)) < TOLERANCE * 2.,
                "support jumps through a continuous eigenframe phase"
            );
        }
        prior = Some((actual, expected));
    }
    println!(
        "SUPPORT PROPERTY current={} previous={}",
        current.json(),
        previous.json()
    );
}

fn fingerprint(kernels: &[Kernel]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for k in kernels {
        for byte in k
            .axes
            .into_iter()
            .flat_map(|axis| axis.to_array())
            .chain(k.normal.to_array())
            .chain([k.planarity, k.bulk])
            .flat_map(f32::to_le_bytes)
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}
fn timed(kernels: &[Kernel], previous: bool) -> f64 {
    let start = std::time::Instant::now();
    let mut checksum = 0_f32;
    for &k in std::hint::black_box(kernels) {
        let output = if previous {
            k.channel_axes_world_reference()
        } else {
            k.channel_axes()
        };
        let output = std::hint::black_box(output);
        checksum += output[0].x + output[1].y + output[2].z;
    }
    std::hint::black_box(checksum);
    start.elapsed().as_secs_f64() * 1000.
}
#[test]
#[ignore = "explicit saved-scene CPU support profile; no GPU or physical stepping"]
fn profile_saved_native_corrected_support() {
    let load = std::env::var_os("SLUICE_SURFACE_LOAD").expect("exact saved native Session input");
    let source = std::fs::read_to_string(load).unwrap();
    let mut session = mining_sim::Session::load_json(&source).unwrap();
    session.sluice.ensure_water_ids();
    CAPTURE.with(|capture| {
        assert!(capture.borrow().is_none());
        *capture.borrow_mut() = Some(Vec::with_capacity(MAX_CAPTURE));
    });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CAPTURE.with(|capture| {
                capture.borrow_mut().take();
            });
        }
    }
    let reset = Reset;
    let instances = crate::water::samples(&session.sluice);
    let kernels = CAPTURE.with(|capture| capture.borrow_mut().take().unwrap());
    drop(reset);
    assert!(!kernels.is_empty() && kernels.len() <= MAX_CAPTURE);
    assert_eq!(
        kernels.len(),
        instances.iter().filter(|(_, p)| p.shape[2] == 0.).count()
    );
    let mut current = Errors::default();
    let mut previous = Errors::default();
    let mut maximum_difference = 0_f64;
    let mut input_cosine = 0_f64;
    for &k in &kernels {
        let a = k.channel_axes();
        let b = k.channel_axes_world_reference();
        current.check(k, a);
        previous.check(k, b);
        maximum_difference = maximum_difference.max(max_abs(
            tensor(a.map(|v| v.as_dvec3())) - tensor(b.map(|v| v.as_dvec3())),
        ));
        for (x, y) in [(0, 1), (0, 2), (1, 2)] {
            input_cosine = input_cosine.max(f64::from(
                k.axes[x].normalize().dot(k.axes[y].normalize()).abs(),
            ));
        }
    }
    assert!(maximum_difference < TOLERANCE);
    // Validation above warms both routes. Alternate which complete dataset is
    // timed first, and report every batch rather than asserting a flaky speed.
    let mut current_ms = Vec::new();
    let mut previous_ms = Vec::new();
    for iteration in 0..8 {
        if iteration % 2 == 0 {
            current_ms.push(timed(&kernels, false));
            previous_ms.push(timed(&kernels, true));
        } else {
            previous_ms.push(timed(&kernels, true));
            current_ms.push(timed(&kernels, false));
        }
    }
    let median = |times: &[f64]| {
        let mut sorted = times.to_vec();
        sorted.sort_by(f64::total_cmp);
        (sorted[3] + sorted[4]) * 0.5
    };
    let report = serde_json::json!({"reference":"a48a22c corrected world S C S including exact bypasses, same five Jacobi sweeps",
        "dataset_kernels":kernels.len(),"selected_instances":instances.len(),"outlet_instances_not_modified":instances.iter().filter(|(_,p)|p.shape[2]==2.).count(),
        "input_kernel_fnv1a64":fingerprint(&kernels),"maximum_input_normalized_axis_dot":input_cosine,
        "current":current.json(),"previous":previous.json(),"maximum_current_vs_previous_tensor_error":maximum_difference,
        "current_ms":current_ms,"previous_ms":previous_ms,"current_median_ms":median(&current_ms),"previous_median_ms":median(&previous_ms),
        "alternating_order":true,"tls_capture_disabled_before_validation_and_timing":true,"timed_methods_have_no_capture_hook":true,
        "cpu_math_only_no_native_render_speed_claim":true,"physical_simulation_advanced":false});
    println!("SUPPORT PROFILE: {report}");
    if let Some(path) = std::env::var_os("SLUICE_SUPPORT_PROFILE") {
        assert!(
            !std::path::Path::new(&path).exists(),
            "profile report must be fresh"
        );
        std::fs::write(path, serde_json::to_vec(&report).unwrap()).unwrap();
    }
}
