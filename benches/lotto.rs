//! Criterion benchmarks of the `lotto` application on a ten-million-player file.
//!
//! - `query`: one draw against a loaded engine, cycling through 1,000 random picks.
//! - `query_large`: one draw cycling through one million distinct, dispersed picks.
//! - `pipe`: one draw through a running `lotto` process, the way the task submits it: write the
//!   pick to its stdin and read the answer from its stdout, cycling through the same picks.
//! - `pipe_batch`: 1,000 draws sent in one write, until all their answers are back.
//! - `load`: loading the player file in process, with the default worker count.
//! - `load_chunk`: `load` with the file read in chunks of 128 KiB to 10 MiB instead of 512 KiB.
//! - `ready`: a fresh `lotto` process from spawn until it prints READY, the way a user starts
//!   it before the draw.
//! - `first_response`: the first answer after READY, immediately or after an idle interval.
//!   `LOTTO_BENCH_IDLE_MS` selects the interval; the default is 100 ms. Process startup and
//!   idle time are excluded from the reported response duration.
//!
//! `LOTTO_BENCH_COLD`, set to any value, evicts the player file from the page cache before every
//! start of `ready`, which then reports as `ready/cold`; this needs GNU coreutils' `dd`. Criterion
//! reports means; the percentiles of the pipe round trips come from `benches/percentiles.rs`.
//!
//! Run with `cargo bench`, or `cargo bench -- query` for the draws alone. Each run also writes
//! HTML reports, starting at `target/criterion/report/index.html` (see README.md). The player
//! file is ten million random players, which the first run generates in Cargo's temporary target
//! directory. `LOTTO_BENCH_BINARY` overrides the `lotto` binary that the groups with a child
//! process start. Every group uses the defaults: in-process loads use one worker per CPU, up to
//! 32, and child processes get only the input file.

use std::io::BufRead;
use std::{env, hint, io, num, path, process, sync, thread, time};

use lotto::{combinatorics, engine, ticket};

#[path = "support/shared.rs"]
mod shared;

/// The binary's allocator, so that the in-process `load` measures what ships.
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// A larger lookup working set, covering every table rather than a repeated small subset.
const LARGE_PICKS: usize = 1_000_000;
/// Coprime with C(90,5), so successive ranks are dispersed and cannot repeat before a full cycle.
const RANK_STRIDE: usize = 1_000_003;

/// The player file, chosen and generated if need be only when a selected benchmark first reads
/// it, so listing or filtering out every benchmark never writes ten million players.
static INPUT: sync::LazyLock<path::PathBuf> = sync::LazyLock::new(shared::input_file);

/// The [`shared::PICKS`] random picks that the pipe benchmarks send too.
fn small_picks() -> Vec<ticket::Ticket> {
    let mut random = shared::random::Random(0xdeca_fbad);
    (0..shared::PICKS)
        .map(|_| ticket::Ticket::new(random.numbers(90)).expect("random picks are valid"))
        .collect()
}

/// A million distinct picks dispersed across the rank universe.
fn large_picks() -> Vec<ticket::Ticket> {
    let universe = combinatorics::SIZES[5];
    let mut rank = 0;
    (0..LARGE_PICKS)
        .map(|_| {
            let balls = combinatorics::unrank(rank, 5).expect("rank is in the universe");
            rank = (rank + RANK_STRIDE) % universe;
            let numbers = std::array::from_fn(|index| balls[index] + 1);
            ticket::Ticket::new(numbers).expect("unrank produces a valid pick")
        })
        .collect()
}

/// Compare warm repeated draws with a much larger, dispersed lookup working set.
///
/// Setup is lazy: Criterion invokes the closure only for a selected benchmark. Filtering for
/// `ready` therefore cannot load an engine through this group. State persists across samples,
/// so the large stream advances instead of restarting at the same picks for each sample.
fn query(criterion: &mut criterion::Criterion) {
    for (name, picks) in [
        ("query", small_picks as fn() -> _),
        ("query_large", large_picks),
    ] {
        let mut state = None;
        criterion.bench_function(name, |bencher| {
            let (engine, picks, next) = state.get_or_insert_with(|| {
                // The generated file takes 32-bit counters, as `engine::load` chooses for up to
                // about 1.4 billion valid players; `Engine::<u64>` here measures the draws of more.
                let engine =
                    engine::Engine::<u32>::load(INPUT.as_path(), engine::default_workers());
                (engine.expect("the input should load"), picks(), 0)
            });
            bencher.iter(|| {
                let pick = picks[*next];
                *next += 1;
                if *next == picks.len() {
                    *next = 0;
                }
                hint::black_box(engine.query(hint::black_box(pick)))
            });
        });
    }
}

/// In-process load time; dropping the result is not timed. `ready`, which times a fresh process, is
/// the primary startup metric.
fn load(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("load");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.bench_function("load", |bencher| {
        let input = INPUT.as_path();
        bencher.iter_batched(
            || (),
            |()| engine::load(input, engine::default_workers()).expect("the input should load"),
            criterion::BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Chunk sizes that [`load_chunk`] compares, in KiB: a quarter of the program's 512 KiB to 20
/// times it.
const CHUNK_KIB: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 10240];

/// In-process load time at each of [`CHUNK_KIB`], with the default worker count; `load` reads the
/// program's 512 KiB chunks.
fn load_chunk(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("load_chunk");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);
    for kib in CHUNK_KIB {
        let chunk_bytes = num::NonZeroUsize::new(kib * 1024).expect("chunk sizes are positive");
        group.bench_function(format!("{kib}KiB"), |bencher| {
            let input = INPUT.as_path();
            bencher.iter_batched(
                || (),
                |()| {
                    engine::load_in_chunks(input, engine::default_workers(), chunk_bytes)
                        .expect("the input should load")
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// Draw round trips through one running `lotto` process, after its READY.
///
/// Each iteration writes one pick to the child's stdin and reads one answer line; the request
/// lines are formatted before timing starts. Closing stdin afterwards ends the process, which
/// must then exit successfully.
fn pipe(criterion: &mut criterion::Criterion) {
    criterion.bench_function("pipe", |bencher| {
        let mut child = shared::Running::start(&shared::binary(), &INPUT);
        let requests = shared::request_lines();
        let mut next = requests.iter().cycle();
        bencher.iter(|| {
            child.exchange(next.next().expect("cycles forever").as_bytes());
        });
        child.finish();
    });
}

/// A batch of draws sent in one write through one running `lotto` process, after its READY.
///
/// Each iteration writes [`shared::PICKS`] picks at once, which fit the pipe, and reads their
/// answers.
fn pipe_batch(criterion: &mut criterion::Criterion) {
    criterion.bench_function("pipe_batch", |bencher| {
        let mut child = shared::Running::start(&shared::binary(), &INPUT);
        let requests = shared::request_lines().concat();
        bencher.iter(|| child.exchange_lines(requests.as_bytes(), shared::PICKS));
        child.finish();
    });
}

/// First request/answer duration with startup, idle and cleanup outside the measured interval.
fn first_response(criterion: &mut criterion::Criterion) {
    let idle_ms = env::var_os("LOTTO_BENCH_IDLE_MS").map_or(100, |value| {
        value
            .to_str()
            .and_then(|value| value.parse::<u64>().ok())
            .expect("LOTTO_BENCH_IDLE_MS must be an integer")
    });
    let mut group = criterion.benchmark_group("first_response");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);
    for (phase, delay_ms) in [
        ("ready".to_owned(), 0),
        (format!("idle_{idle_ms}ms"), idle_ms),
    ] {
        group.bench_function(phase, |bencher| {
            let delay = time::Duration::from_millis(delay_ms);
            let binary = shared::binary();
            let mut random = shared::random::Random(0xdeca_fbad);
            bencher.iter_custom(|runs| {
                let mut elapsed = time::Duration::ZERO;
                for _ in 0..runs {
                    let request = shared::line(random.numbers(90));
                    let mut child = shared::Running::start(&binary, &INPUT);
                    thread::sleep(delay);
                    let start = time::Instant::now();
                    child.exchange(request.as_bytes());
                    elapsed += start.elapsed();
                    child.finish();
                }
                elapsed
            });
        });
    }
    group.finish();
}

/// Evict `input` from the page cache, so that the next load reads it from the disk.
///
/// `dd` with `iflag=nocache` and no blocks to copy asks the kernel to drop the file's cached
/// pages, without root; `LOTTO_BENCH_COLD` is documented as needing GNU coreutils.
fn evict(input: &path::Path) {
    let status = process::Command::new("dd")
        .arg(format!("if={}", input.display()))
        .args(["iflag=nocache", "count=0", "status=none"])
        .status()
        .expect("dd should be on the PATH for LOTTO_BENCH_COLD");
    assert!(status.success(), "dd could not evict the file: {status}");
}

/// Time from spawning `binary` on `input` until it prints READY, after evicting the file from
/// the page cache when `cold` is set.
///
/// Closing stdin afterwards ends the process, which must then exit successfully.
fn ready_time(binary: &path::Path, input: &path::Path, cold: bool) -> time::Duration {
    if cold {
        evict(input);
    }
    let mut command = shared::command(binary, input);
    let start = time::Instant::now();
    let mut child = command.spawn().expect("the lotto binary should start");
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut line = String::new();
    io::BufReader::new(stdout)
        .read_line(&mut line)
        .expect("the first line should be readable");
    let elapsed = start.elapsed();
    assert_eq!(line, "READY\n", "the first line should be READY");
    let status = child.wait().expect("the lotto process should be waitable");
    assert!(
        status.success(),
        "lotto should exit cleanly at end of input: {status}"
    );
    elapsed
}

/// Spawn-to-READY time of a fresh process, with the file cached or, under `LOTTO_BENCH_COLD`,
/// evicted before every start.
fn ready(criterion: &mut criterion::Criterion) {
    let binary = shared::binary();
    let cold = env::var_os("LOTTO_BENCH_COLD").is_some();
    let mut group = criterion.benchmark_group("ready");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.bench_function(if cold { "cold" } else { "ready" }, |bencher| {
        bencher.iter_custom(|runs| (0..runs).map(|_| ready_time(&binary, &INPUT, cold)).sum());
    });
    group.finish();
}

criterion::criterion_group!(
    benches,
    query,
    pipe,
    pipe_batch,
    load,
    load_chunk,
    ready,
    first_response
);
criterion::criterion_main!(benches);
