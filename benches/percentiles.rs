//! Tail latency of draws through a running `lotto` process, which Criterion's means cannot show.
//!
//! Prints the p50, p99, p99.9 and maximum of N single-draw round trips and of N / 1,000
//! batches of 1,000 draws, through one process on the same input and picks as `benches/lotto.rs`:
//!
//! ```sh
//! LOTTO_BENCH_PERCENTILES=1000000 cargo bench --bench percentiles
//! ```
//!
//! Without `LOTTO_BENCH_PERCENTILES` it measures nothing, so a plain `cargo bench`, which runs
//! every benchmark target, neither starts the process nor generates a player file.
//! `LOTTO_BENCH_BINARY` works as in `benches/lotto.rs`.

use std::{env, num, time};

#[path = "support/shared.rs"]
mod shared;

/// The value at the `numerator` / `denominator` quantile of `sorted` durations, nearest-rank.
fn quantile(sorted: &[time::Duration], numerator: usize, denominator: usize) -> time::Duration {
    let rank = (numerator * sorted.len()).div_ceil(denominator);
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Print the distribution of `samples` under `name`.
#[expect(clippy::print_stderr, reason = "the distribution is this run's output")]
fn report(name: &str, mut samples: Vec<time::Duration>) {
    samples.sort_unstable();
    eprintln!(
        "{name}: n={} p50 {:?} p99 {:?} p99.9 {:?} max {:?}",
        samples.len(),
        quantile(&samples, 1, 2),
        quantile(&samples, 99, 100),
        quantile(&samples, 999, 1000),
        samples[samples.len() - 1],
    );
}

/// Time `runs` single-draw round trips, then `runs / 1,000` batches of 1,000 draws.
fn main() {
    let Some(runs) = env::var_os("LOTTO_BENCH_PERCENTILES") else {
        return;
    };
    // Zero round trips would leave no distribution to report and no answer to check.
    let runs = runs
        .to_str()
        .and_then(|runs| runs.parse::<num::NonZeroUsize>().ok())
        .expect("LOTTO_BENCH_PERCENTILES must be a positive number of round trips")
        .get();
    let requests = shared::request_lines();
    let mut child = shared::Running::start(&shared::binary(), &shared::input_file());
    let single = requests
        .iter()
        .cycle()
        .take(runs)
        .map(|request| {
            let start = time::Instant::now();
            child.exchange(request.as_bytes());
            start.elapsed()
        })
        .collect();
    report("pipe", single);
    let batch = requests.concat();
    let batches: Vec<_> = (0..runs / shared::PICKS)
        .map(|_| {
            let start = time::Instant::now();
            child.exchange_lines(batch.as_bytes(), shared::PICKS);
            start.elapsed()
        })
        .collect();
    if !batches.is_empty() {
        report("pipe_batch", batches);
    }
    child.finish();
}
