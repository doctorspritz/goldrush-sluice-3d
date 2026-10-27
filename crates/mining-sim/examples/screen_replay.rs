//! Bounded deterministic CPU replay of actual saved screen stock.
//! No new feed, wet-sluice or terrain advancement; this isolates screen cost.
//! Step timers exclude JSON, disk I/O and cloning. Every first-run step is
//! recorded in full for byte-for-byte comparison between numerical variants.
use mining_sim::screen::Screen;
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(args.len() == 5 || args.len() == 6 && args[5] == "--metrics-only") {
        return Err("screen_replay INPUT_SAVE OUTPUT_DIR TICKS REPEATS [--metrics-only]; fresh directory, 1..240 ticks, 1..8 repeats".into());
    }
    let ticks: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    if !(1..=240).contains(&ticks) || !(1..=8).contains(&repeats) {
        return Err("bounded replay: 1..240 ticks and 1..8 repeats".into());
    }
    let input = Path::new(&args[1]);
    if fs::metadata(input)?.len() > 64 * 1024 * 1024 {
        return Err("input exceeds 64 MiB".into());
    }
    let initial: Screen = {
        let value: serde_json::Value = serde_json::from_reader(fs::File::open(input)?)?;
        let screen = value.pointer("/plant/screen").unwrap_or(&value);
        serde_json::from_value(screen.clone())?
    };
    if !initial.valid() {
        return Err("invalid saved screen".into());
    }
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    fs::write(
        output.join("initial-screen.json"),
        serde_json::to_vec(&initial)?,
    )?;
    let mut trace = if args.len() == 5 {
        Some(BufWriter::new(fs::File::create(
            output.join("states.jsonl"),
        )?))
    } else {
        None
    };
    let dt = 1.0_f32 / 60.;
    let mut runs = Vec::new();
    let mut first_final = None;
    let mut trace_bytes = 0;
    for repeat in 0..repeats {
        let mut screen = initial.clone();
        let mut step_seconds = 0.;
        let mut intervals = 0usize;
        let mut queries = 0usize;
        let mut contacts = 0usize;
        for tick in 0..ticks {
            let start = Instant::now();
            let result = std::hint::black_box(&mut screen).step(dt);
            step_seconds += start.elapsed().as_secs_f64();
            result.map_err(|e| format!("repeat {repeat}, tick {tick}: {e}"))?;
            intervals += screen.last_step.intervals;
            queries += screen.last_step.contact_queries;
            contacts += screen.last_step.contacts;
            if let Some(trace) = trace.as_mut().filter(|_| repeat == 0) {
                let bytes = serde_json::to_vec(&screen)?;
                trace_bytes += bytes.len() + 1;
                if trace_bytes > 128 * 1024 * 1024 {
                    return Err("trace exceeds 128 MiB".into());
                }
                trace.write_all(&bytes)?;
                trace.write_all(b"\n")?;
            }
        }
        let final_state = serde_json::to_vec(&screen)?;
        if let Some(reference) = &first_final {
            if reference != &final_state {
                return Err("repeated final state differs".into());
            }
        } else {
            first_final = Some(final_state);
        }
        runs.push(serde_json::json!({"repeat":repeat,"step_seconds":step_seconds,"intervals":intervals,"queries":queries,"contacts":contacts,
            "active":screen.particles.len(),"undersize":screen.undersize.len(),"oversize":screen.oversize.len(),"spills":screen.spills.len()}));
    }
    if let Some(trace) = trace.as_mut() {
        trace.flush()?;
    }
    fs::write(
        output.join("final-screen.json"),
        first_final.ok_or("no replay result")?,
    )?;
    let result = serde_json::json!({"completed":true,"scope":"Finite screen stock only; no new feed, wet-sluice or terrain advancement. Not a native FPS or integrated real-time result.",
        "ticks":ticks,"repeats":repeats,"dt_s":dt,"config":initial.config,"motor_on":initial.motor_on,"initial_time_s":initial.time,
        "initial_active":initial.particles.len(),"initial_undersize":initial.undersize.len(),"initial_oversize":initial.oversize.len(),
        "runs":runs,"trace_bytes":trace_bytes,"full_state_trace":trace.as_ref().map(|_| "states.jsonl"),"timing_excludes":"cloning, JSON serialization, disk I/O"});
    fs::write(
        output.join("metrics.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{result}");
    Ok(())
}
