//! CPU microbenchmark for the native target sampler (diagnosis only).
//!
//! This is an ignored integration test: it changes no production behaviour and
//! exists to quantify the host cost of the exact full-vocabulary selector for
//! the five measured decode profiles. Run it in release mode:
//!
//! ```text
//! cargo test --release -p ds41rt-core --test target_sampling_bench -- \
//!   --ignored --nocapture
//! ```
//!
//! It reports microseconds per row for each profile over a realistic peaked
//! logit row and a fully tied row, the internal path taken (greedy argmax, the
//! linear fast path, or the ordered sort path), and extrapolated per-step cost
//! for representative lane row counts.

use ds41rt_core::TargetSamplingParams;
use std::time::Instant;

const VOCAB: usize = 129_280;
const WARMUP_ROWS: usize = 3;
const TIMED_ROWS: usize = 50;

/// A peaked, realistic-ish row: four strong logits over a broad tail.
fn spread_logits() -> Vec<f32> {
    let mut logits = vec![0.0f32; VOCAB];
    for (index, logit) in logits.iter_mut().enumerate() {
        let mut hash = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hash ^= hash >> 31;
        let unit = (hash >> 40) as f32 / 16_777_216.0;
        let mut value = -8.0 + 10.0 * unit;
        if index < 4 {
            value += 20.0 - index as f32 * 2.0;
        }
        *logit = value;
    }
    logits
}

/// A fully tied row: every token has the same logit.
fn tied_logits() -> Vec<f32> {
    vec![0.0f32; VOCAB]
}

/// A near-tied row: ties broken by a tiny index-dependent epsilon.
fn near_tied_logits() -> Vec<f32> {
    (0..VOCAB).map(|index| (index as f32) * 1.0e-6).collect()
}

fn path(params: TargetSamplingParams) -> &'static str {
    if params.is_greedy() {
        "greedy-argmax"
    } else if params.top_k().is_none() && params.top_p() >= 1.0 {
        "fast-linear-categorical"
    } else {
        "ordered-sort"
    }
}

fn micros_per_row(params: TargetSamplingParams, logits: &[f32]) -> f64 {
    for position in 0..WARMUP_ROWS {
        let _ = params
            .select_token(logits, None, position as u64)
            .expect("sampler accepts the synthetic row");
    }
    let started = Instant::now();
    let mut checksum = 0usize;
    for position in 0..TIMED_ROWS {
        checksum = checksum.wrapping_add(
            params
                .select_token(logits, None, position as u64)
                .expect("sampler accepts the synthetic row"),
        );
    }
    let elapsed = started.elapsed();
    std::hint::black_box(checksum);
    elapsed.as_secs_f64() * 1.0e6 / TIMED_ROWS as f64
}

fn profile(name: &str, params: TargetSamplingParams) -> f64 {
    let spread = micros_per_row(params, &spread_logits());
    let tied = micros_per_row(params, &tied_logits());
    let near_tied = micros_per_row(params, &near_tied_logits());
    println!(
        "{name:<26} path={:<24} spread={spread:>9.1} us/row  tied={tied:>9.1}  near_tied={near_tied:>9.1}",
        path(params)
    );
    spread
}

#[test]
#[ignore = "CPU microbenchmark; run with --release --ignored --nocapture"]
fn target_sampling_profile_cost() {
    let profiles = [
        ("greedy", TargetSamplingParams::greedy()),
        (
            "temp0.7+min_p0.05",
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 1).unwrap(),
        ),
        (
            "temp0.7+top_p0.9",
            TargetSamplingParams::new(0.7, 0.9, None, 0.0, 1).unwrap(),
        ),
        (
            "temp0.2+top_p0.95",
            TargetSamplingParams::new(0.2, 0.95, None, 0.0, 1).unwrap(),
        ),
        (
            "temp0.7+top_k40",
            TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 1).unwrap(),
        ),
        // Diagnostic: the same ordered path as top_p, but a strong min_p shrinks
        // the ranked set before the sort, isolating the sort's contribution.
        (
            "diag:min_p0.5+top_p0.9",
            TargetSamplingParams::new(0.7, 0.9, None, 0.5, 1).unwrap(),
        ),
    ];

    println!("vocab={VOCAB} timed_rows={TIMED_ROWS} warmup_rows={WARMUP_ROWS}");
    println!(
        "profile                    {:^24} {:>12} {:>12} {:>12}",
        "internal path", "spread", "tied", "near_tied"
    );
    let mut measured: Vec<(&'static str, f64)> = Vec::new();
    for (name, params) in profiles {
        measured.push((name, profile(name, params)));
    }

    // Pre-change (full comparison sort) spread-row costs recorded before this
    // optimization, for the honest before/after comparison.
    const PRE_CHANGE_TOP_K40: f64 = 7_612.0;
    const PRE_CHANGE_TOP_P09: f64 = 7_853.0;
    const PRE_CHANGE_TOP_P095: f64 = 8_055.0;
    println!("\nbefore/after (spread us/row):");
    for (name, after) in &measured {
        let before = match *name {
            "temp0.7+top_p0.9" => Some(PRE_CHANGE_TOP_P09),
            "temp0.2+top_p0.95" => Some(PRE_CHANGE_TOP_P095),
            "temp0.7+top_k40" => Some(PRE_CHANGE_TOP_K40),
            _ => None,
        };
        if let Some(before) = before {
            println!(
                "  {name:<22} before={before:>8.1}  after={after:>7.1}  speedup={:>5.1}x",
                before / after
            );
        }
    }
    let after_of = |name: &str| {
        measured
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, value)| *value)
            .expect("profile measured")
    };
    // Target ~0.4 ms/row for top_k and ~1.0 ms/row for top_p; the asserts leave
    // headroom for slower hosts while still pinning the order-of-magnitude win.
    assert!(
        after_of("temp0.7+top_k40") < 500.0,
        "top_k40 spread cost regressed: {:.1} us/row (pre-change {PRE_CHANGE_TOP_K40:.1})",
        after_of("temp0.7+top_k40")
    );
    assert!(
        after_of("temp0.7+top_p0.9") < 1_500.0,
        "top_p0.9 spread cost regressed: {:.1} us/row (pre-change {PRE_CHANGE_TOP_P09:.1})",
        after_of("temp0.7+top_p0.9")
    );
    assert!(
        after_of("temp0.2+top_p0.95") < 1_500.0,
        "top_p0.95 spread cost regressed: {:.1} us/row (pre-change {PRE_CHANGE_TOP_P095:.1})",
        after_of("temp0.2+top_p0.95")
    );

    // Extrapolate the spread-row cost at representative verification widths.
    // A lane round carries `members * (1 + accepted_drafts)` target rows.
    println!("\nper-step sampler cost (spread row, ms) at lane row counts:");
    print!("{:<26}", "profile");
    for rows in [1usize, 8, 16, 40, 80, 128] {
        print!(" {rows:>9}");
    }
    println!();
    for (name, params) in profiles {
        let per_row_us = micros_per_row(params, &spread_logits());
        print!("{name:<26}");
        for rows in [1usize, 8, 16, 40, 80, 128] {
            let ms = per_row_us * rows as f64 / 1000.0;
            print!(" {ms:>9.3}");
        }
        println!();
    }
}
