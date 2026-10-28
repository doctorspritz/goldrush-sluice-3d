//! Rejected test-only experiments: both candidates were slower than production.
//! Reproduction and measured limits: docs/physics/friction-sign-experiment.md.
//! The production projection and saved numerical policies remain unchanged.
#![deny(clippy::all)]
use super::{ScalarType, friction_disk};
use crate::math::{Real, SIMD_WIDTH, SimdReal};
use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdValue};
use std::hint::black_box;
use std::time::Instant;
use std::{eprintln, vec, vec::Vec};
fn candidate_friction_disk<N: ScalarType>(
    k11: N,
    k22: N,
    k12: N,
    old: na::Vector2<N>,
    v: na::Vector2<N>,
    limit: N,
) -> na::Vector2<N> {
    // Unilateral contacts often have zero normal impulse in every SIMD lane.
    // Their Coulomb disk is exactly {0}; no inverse mass or root solve is needed.
    // Return the new impulse, not from the caller: its old warm-start impulse
    // must still be removed from both bodies by the ordinary delta update.
    let positive = limit.simd_gt(N::zero());
    if !positive.any() {
        return na::Vector2::zeros();
    }
    let b = na::vector![
        k11 * old.x + k12 * old.y - v.x,
        k12 * old.x + k22 * old.y - v.y
    ];
    let inverse = crate::utils::simd_inv(k11 * k22 - k12 * k12);
    let trial = na::vector![
        (k22 * b.x - k12 * b.y) * inverse,
        (k11 * b.y - k12 * b.x) * inverse
    ];
    let sliding = (trial.x * trial.x + trial.y * trial.y).simd_gt(limit * limit);
    if !sliding.any() {
        return trial;
    }
    // A frictionless lane can be outside its zero-radius disk while every
    // positive-radius lane sticks. It must not trigger 24 bisections for all
    // lanes. Mixed batches containing real sliding keep the original solve.
    if !(sliding & positive).any() {
        return na::vector![
            trial.x.select(positive, N::zero()),
            trial.y.select(positive, N::zero())
        ];
    }
    let bx = b.x.select(positive, N::zero());
    let by = b.y.select(positive, N::zero());
    let safe_limit = limit.select(positive, N::one());
    let mut low = N::zero();
    let mut high = (bx * bx + by * by).simd_sqrt() / safe_limit;
    let solve = |alpha: N| {
        let a = k11 + alpha;
        let d = k22 + alpha;
        let inv = crate::utils::simd_inv(a * d - k12 * k12);
        na::vector![(d * bx - k12 * by) * inv, (a * by - k12 * bx) * inv]
    };
    for _ in 0..24 {
        let mid = (low + high) * N::splat(0.5);
        let a = k11 + mid;
        let d = k22 + mid;
        let det = a * d - k12 * k12;
        let nx = d * bx - k12 * by;
        let ny = a * by - k12 * bx;
        let q = det * det;
        let x = nx * nx + ny * ny;
        let r = limit * limit;
        let y = r * q;
        // Explicit bounds protect both scalar and vector floating-point modes.
        // Normal-only guards are insufficient when a minor p² is flushed.
        let lower = N::splat(f32::from_bits(0x2780_0000)); // 2^-48
        let upper = N::splat(f32::from_bits(0x5780_0000)); // 2^48
        let within = |v: N| v.simd_ge(lower) & v.simd_le(upper);
        let certified = within(x)
            & within(y)
            & within(r)
            & within(q)
            & det.simd_gt(N::zero())
            & (x - y)
                .simd_abs()
                .simd_gt((x + y) * N::splat(64.0 * f32::EPSILON));
        // A masked zero-right-hand-side lane has an identically false old
        // outside test and must not force every other lane through division.
        let zero = bx.simd_eq(N::zero()) & by.simd_eq(N::zero());
        let outside = if (certified | zero).all() {
            x.simd_gt(y)
        } else {
            let p = solve(mid);
            (p.x * p.x + p.y * p.y).simd_gt(r)
        };
        low = mid.select(outside, low);
        high = high.select(outside, mid);
    }
    // Feasible-side root, followed only by a floating-point norm guard.
    let boundary = solve(high).simd_cap_magnitude(limit);
    na::vector![
        boundary.x.select(sliding, trial.x),
        boundary.y.select(sliding, trial.y)
    ]
}

#[inline(always)]
fn qualified<N: ScalarType>(
    matrix: [N; 3],
    b: [N; 2],
    limit: N,
    high: N,
    relevant: N::SimdBool,
) -> bool {
    let [a, d, c] = matrix;
    let [bx, by] = b;
    let lower = N::splat(f32::from_bits(0x2780_0000));
    let upper = N::splat(f32::from_bits(0x5780_0000));
    let within = |v: N| v.simd_ge(lower) & v.simd_le(upper);
    let trace = a + d;
    let det = a * d - c * c;
    let norm = bx * bx + by * by;
    let ratio = det / trace;
    let ratio2 = ratio * ratio;
    let lower_x = ratio2 * norm * N::splat(1. / 16.);
    let span = trace + N::splat(2.) * high;
    let upper_x = N::splat(4.) * span * span * norm;
    let end_det = (a + high) * (d + high) - c * c;
    let q0 = det * det;
    let qh = end_det * end_det;
    let r = limit * limit;
    let valid = within(trace)
        & within(norm)
        & within(ratio2)
        & within(lower_x)
        & within(upper_x)
        & within(q0)
        & within(qh)
        & within(r * q0)
        & within(r * qh)
        & within(r)
        & a.simd_gt(N::zero())
        & d.simd_gt(N::zero())
        & high.simd_ge(N::zero())
        & det.simd_gt(N::splat(32. * f32::EPSILON) * (trace * trace));
    (valid | !relevant).all()
}
fn hoisted_friction_disk<N: ScalarType>(
    k11: N,
    k22: N,
    k12: N,
    old: na::Vector2<N>,
    v: na::Vector2<N>,
    limit: N,
) -> na::Vector2<N> {
    // Unilateral contacts often have zero normal impulse in every SIMD lane.
    // Their Coulomb disk is exactly {0}; no inverse mass or root solve is needed.
    // Return the new impulse, not from the caller: its old warm-start impulse
    // must still be removed from both bodies by the ordinary delta update.
    let positive = limit.simd_gt(N::zero());
    if !positive.any() {
        return na::Vector2::zeros();
    }
    let b = na::vector![
        k11 * old.x + k12 * old.y - v.x,
        k12 * old.x + k22 * old.y - v.y
    ];
    let inverse = crate::utils::simd_inv(k11 * k22 - k12 * k12);
    let trial = na::vector![
        (k22 * b.x - k12 * b.y) * inverse,
        (k11 * b.y - k12 * b.x) * inverse
    ];
    let sliding = (trial.x * trial.x + trial.y * trial.y).simd_gt(limit * limit);
    if !sliding.any() {
        return trial;
    }
    // A frictionless lane can be outside its zero-radius disk while every
    // positive-radius lane sticks. It must not trigger 24 bisections for all
    // lanes. Mixed batches containing real sliding keep the original solve.
    if !(sliding & positive).any() {
        return na::vector![
            trial.x.select(positive, N::zero()),
            trial.y.select(positive, N::zero())
        ];
    }
    let bx = b.x.select(positive, N::zero());
    let by = b.y.select(positive, N::zero());
    let safe_limit = limit.select(positive, N::one());
    let mut low = N::zero();
    let mut high = (bx * bx + by * by).simd_sqrt() / safe_limit;
    let solve = |alpha: N| {
        let a = k11 + alpha;
        let d = k22 + alpha;
        let inv = crate::utils::simd_inv(a * d - k12 * k12);
        na::vector![(d * bx - k12 * by) * inv, (a * by - k12 * bx) * inv]
    };
    let relevant = sliding & positive;
    if !qualified([k11, k22, k12], [bx, by], limit, high, relevant) {
        for _ in 0..24 {
            let mid = (low + high) * N::splat(0.5);
            let p = solve(mid);
            let outside = (p.x * p.x + p.y * p.y).simd_gt(limit * limit);
            low = mid.select(outside, low);
            high = high.select(outside, mid);
        }
    } else {
        for _ in 0..24 {
            let mid = (low + high) * N::splat(0.5);
            let a = k11 + mid;
            let d = k22 + mid;
            let det = a * d - k12 * k12;
            let nx = d * bx - k12 * by;
            let ny = a * by - k12 * bx;
            let x = nx * nx + ny * ny;
            let y = (limit * limit) * (det * det);
            let certain = (x - y)
                .simd_abs()
                .simd_gt((x + y) * N::splat(64.0 * f32::EPSILON));
            let outside = if (certain | !relevant).all() {
                x.simd_gt(y)
            } else {
                let p = solve(mid);
                (p.x * p.x + p.y * p.y).simd_gt(limit * limit)
            };
            low = mid.select(outside, low);
            high = high.select(outside, mid);
        }
    }
    // Feasible-side root, followed only by a floating-point norm guard.
    let boundary = solve(high).simd_cap_magnitude(limit);
    na::vector![
        boundary.x.select(sliding, trial.x),
        boundary.y.select(sliding, trial.y)
    ]
}
type Case = [Real; 8];
fn cases() -> Vec<Case> {
    let mut result = vec![
        [2., 3., 0.2, 0.1, -0.1, -3., 4., 0.],
        [2., 3., 0.2, 0., 0., -0.1, 0.2, 10.],
        [1., 9., 0.3, 0., 0., -2., -2., 0.4],
        [9., 1., -0.7, 0.2, -0.1, 3., 4., 0.3],
        [0.002, 30., 0.01, 0.5, 0.2, 1., -1., 0.0001],
        [1e-10, 1e-10, 0., 0., 0., 1e-10, 1e-10, 1.],
        [1e-12, 1e-12, 0., 1., 0., 1e-12, 1e-12, 1.],
    ];
    // Forced qualified sliding, warm-start removal, and +/-1 ULP stick/slide.
    for bits in [1f32.to_bits() - 1, 1f32.to_bits(), 1f32.to_bits() + 1] {
        result.push([1., 1., 0., 0., 0., -f32::from_bits(bits), 0., 1.]);
    }
    result.extend([
        [1., 2., 0.25, 0., 0., -3., 1., 1.],
        [1., 2., 0.25, 2., -1., -3., 1., 1.],
        [1., 2., 0.25, 2., -1., -3., 1., 0.],
    ]);
    let mut seed = 0x897c_662b_c0d5_a31du64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 32) as u32) as f64 / (u32::MAX as f64)
    };
    for index in 0..32_768 {
        let scale = 2f64.powi((random() * 30.).floor() as i32 - 15);
        let ratio = 2f64.powi((random() * 18.).floor() as i32);
        let angle = random() * std::f64::consts::TAU;
        let (sn, cs) = angle.sin_cos();
        let limit = if index % 13 == 0 {
            0.
        } else {
            2f64.powi((random() * 36.).floor() as i32 - 18)
        };
        result.push([
            (scale * (cs * cs + ratio * sn * sn)) as Real,
            (scale * (sn * sn + ratio * cs * cs)) as Real,
            (scale * (1. - ratio) * cs * sn) as Real,
            ((random() * 4. - 2.) * limit) as Real,
            ((random() * 4. - 2.) * limit) as Real,
            ((random() * 8. - 4.) * scale * limit) as Real,
            ((random() * 8. - 4.) * scale * limit) as Real,
            limit as Real,
        ]);
    }
    result
}
fn scalar(c: Case, mode: u8) -> na::Vector2<Real> {
    if mode == 2 {
        hoisted_friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    } else if mode == 1 {
        candidate_friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    } else {
        friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    }
}
fn pack(cases: &[Case], offset: usize) -> [SimdReal; 8] {
    std::array::from_fn(|field| {
        SimdReal::from(std::array::from_fn::<_, SIMD_WIDTH, _>(|lane| {
            cases[(offset + lane) % cases.len()][field]
        }))
    })
}
fn simd(c: [SimdReal; 8], mode: u8) -> na::Vector2<SimdReal> {
    if mode == 2 {
        hoisted_friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    } else if mode == 1 {
        candidate_friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    } else {
        friction_disk(
            c[0],
            c[1],
            c[2],
            na::vector![c[3], c[4]],
            na::vector![c[5], c[6]],
            c[7],
        )
    }
}
#[test]
fn certified_sign_candidate_matches_existing_scalar_and_simd_impulse_bits() {
    let cases = cases();
    for mode in [1, 2] {
        let mut count = 0;
        for (index, &c) in cases.iter().enumerate() {
            let a = scalar(c, 0);
            let b = scalar(c, mode);
            for axis in 0..2 {
                assert!(a[axis].is_finite());
                assert_eq!(
                    a[axis].to_bits(),
                    b[axis].to_bits(),
                    "mode={mode} scalar case={index} axis={axis}"
                );
            }
            let packed = pack(&cases, index);
            let a = simd(packed, 0);
            let b = simd(packed, mode);
            for lane in 0..SIMD_WIDTH {
                for axis in 0..2 {
                    let a = a[axis].extract(lane);
                    let b = b[axis].extract(lane);
                    assert!(a.is_finite());
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "mode={mode} SIMD case={index} lane={lane} axis={axis}"
                    );
                    count += 1;
                }
            }
        }
        eprintln!(
            "EXACT DIFFERENTIAL: mode={mode} {} scalar cases, {} SIMD components, width={SIMD_WIDTH}",
            cases.len(),
            count
        );
    }
}

#[test]
#[ignore = "bounded CPU microbenchmark; execute explicitly after the differential check"]
fn certified_sign_candidate_timing() {
    let broad = cases();
    let bounded: Vec<_> = broad
        .iter()
        .copied()
        .filter(|c| {
            let trace = c[0] + c[1];
            let det = c[0] * c[1] - c[2] * c[2];
            c[0] >= 0.125
                && c[0] <= 8.
                && c[1] >= 0.125
                && c[1] <= 8.
                && trace * trace / det < 128.
                && c[7] >= 0.01
                && c[7] <= 4.
        })
        .collect();
    assert!(bounded.len() > 32);
    for (label, cases) in [("broad-stress", broad), ("bounded-synthetic", bounded)] {
        let batches: Vec<_> = (0..cases.len())
            .step_by(SIMD_WIDTH)
            .map(|i| pack(&cases, i))
            .collect();
        let repetitions = 1_048_576_usize.div_ceil(batches.len());
        for repeat in 0..5 {
            for mode in if repeat % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            } {
                let start = Instant::now();
                let mut checksum = 0f64;
                for _ in 0..repetitions {
                    for &c in &batches {
                        let p = black_box(simd(black_box(c), mode));
                        checksum += black_box(p.x.extract(0)) as f64;
                    }
                }
                eprintln!(
                    "TIMING dataset={label} repeat={repeat} mode={mode} batches={} seconds={:.9} checksum={checksum:.15e}",
                    batches.len() * repetitions,
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}

#[test]
fn hoisted_qualification_covers_qualified_and_fallback_root_decisions() {
    let all = cases();
    let mut sliding_batches = 0;
    let mut qualified_batches = 0;
    let mut certified_steps = 0;
    let mut ambiguous_steps = 0;
    for index in 0..all.len() {
        let c = pack(&all, index);
        let [k11, k22, k12, ox, oy, vx, vy, limit] = c;
        let zero = SimdReal::splat(0.);
        let positive = limit.simd_gt(zero);
        let b = na::vector![k11 * ox + k12 * oy - vx, k12 * ox + k22 * oy - vy];
        let inv = crate::utils::simd_inv(k11 * k22 - k12 * k12);
        let trial = na::vector![(k22 * b.x - k12 * b.y) * inv, (k11 * b.y - k12 * b.x) * inv];
        let relevant = (trial.x * trial.x + trial.y * trial.y).simd_gt(limit * limit) & positive;
        if !relevant.any() {
            continue;
        }
        sliding_batches += 1;
        let bx = b.x.select(positive, zero);
        let by = b.y.select(positive, zero);
        let safe_limit = limit.select(positive, SimdReal::splat(1.));
        let mut low = zero;
        let mut high = (bx * bx + by * by).simd_sqrt() / safe_limit;
        if !qualified([k11, k22, k12], [bx, by], limit, high, relevant) {
            continue;
        }
        qualified_batches += 1;
        for step in 0..24 {
            let mid = (low + high) * SimdReal::splat(0.5);
            let a = k11 + mid;
            let d = k22 + mid;
            let det = a * d - k12 * k12;
            let nx = d * bx - k12 * by;
            let ny = a * by - k12 * bx;
            let q = det * det;
            let r = limit * limit;
            let x = nx * nx + ny * ny;
            let y = r * q;
            let inv = crate::utils::simd_inv(det);
            let px = nx * inv;
            let py = ny * inv;
            let original = (px * px + py * py).simd_gt(r);
            let certain = (x - y)
                .simd_abs()
                .simd_gt((x + y) * SimdReal::splat(64. * f32::EPSILON));
            let use_sign = (certain | !relevant).all();
            if use_sign {
                certified_steps += 1;
            } else {
                ambiguous_steps += 1;
            }
            for lane in 0..SIMD_WIDTH {
                if !relevant.extract(lane) {
                    continue;
                }
                for value in [x, y, q, r] {
                    let value = value.extract(lane);
                    assert!(
                        (f32::from_bits(0x2780_0000)..=f32::from_bits(0x5780_0000))
                            .contains(&value),
                        "range case={index} step={step} lane={lane} value={value}"
                    );
                }
                if use_sign {
                    assert_eq!(
                        x.extract(lane) > y.extract(lane),
                        original.extract(lane),
                        "predicate case={index} step={step} lane={lane}"
                    );
                }
            }
            // Induct on the original root history, independently of shortcut.
            low = mid.select(original, low);
            high = high.select(original, mid);
        }
    }
    assert!(qualified_batches > 0 && qualified_batches < sliding_batches);
    assert!(certified_steps > 0 && ambiguous_steps > 0);
    let c = [1., 2., 0.25, 0., 0., -3., 1., 1.];
    let b = [3., -1.];
    assert!(qualified([c[0], c[1], c[2]], b, c[7], 10f32.sqrt(), true));
    eprintln!(
        "QUALIFICATION sliding_batches={sliding_batches} qualified_batches={qualified_batches} certified_steps={certified_steps} ambiguous_steps={ambiguous_steps}"
    );
}

#[test]
fn normal_only_certificate_is_insufficient_with_flush_to_zero() {
    let det = 2f32.powi(60);
    let nx = 1f32;
    let ny = 2f32.powi(-4);
    let r = 2f32.powi(-120);
    let q = det * det;
    let x = nx * nx + ny * ny;
    let y = r * q;
    assert!([q, x, y, r].into_iter().all(f32::is_normal));
    assert!((x - y).abs() > 64. * f32::EPSILON * (x + y));
    assert!(x > y);
    let flush = |v: f32| if v.abs() < f32::MIN_POSITIVE { 0. } else { v };
    let inv = 1. / det;
    let px = nx * inv;
    let py = ny * inv;
    let legacy_norm_squared = flush(px * px) + flush(py * py);
    assert_eq!(legacy_norm_squared.to_bits(), r.to_bits());
    assert!(r < f32::from_bits(0x2780_0000));
}
