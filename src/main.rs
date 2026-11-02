//! The `lotto` command line: `lotto input.txt` loads the players, prints READY, then answers
//! each draw read from stdin. Everything else lives in the `lotto` library; see README.md.

use std::{fs, io, num, path, process, sync};

// `Parser` is a trait, imported by name so that `Cli::parse()` resolves; its derive is spelled
// `clap::Parser` below.
use clap::Parser;
use eyre::WrapErr;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// jemalloc shortens the load compared with glibc's allocator; see docs/DESIGN.md.
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Report how many players match exactly 2, 3, 4 and 5 numbers of each lottery draw.
///
/// Loads the player file and prints READY, then reads one draw per line from stdin and answers
/// each with the four winner counts on stdout. Invalid lines are ignored. Logging is off by
/// default; fatal errors always go to stderr.
#[derive(Debug, clap::Parser)]
#[command(version)]
struct Cli {
    /// Player file: one ticket per line, five distinct numbers in 1..=90.
    input: path::PathBuf,
    /// Load workers, default the CPUs this process may use, up to 32.
    #[arg(long, value_name = "COUNT")]
    threads: Option<num::NonZeroUsize>,
    /// What to log: off, error, warn, info, debug or trace. `warn` logs each invalid line.
    #[arg(long, value_name = "LEVEL", default_value = "off")]
    log_level: tracing::level_filters::LevelFilter,
    /// Log file, used unless the level is off; a run adds to what it holds.
    #[arg(long, value_name = "PATH", default_value = "log.txt")]
    log_name: path::PathBuf,
}

/// Translate errors into a nonzero exit without Rust's panic machinery or stdout diagnostics.
///
/// Command-line mistakes never reach [`run`]: `Cli::parse` prints them to stderr and exits with
/// status 2 itself.
fn main() -> process::ExitCode {
    let cli = Cli::parse();
    let log = open_log(cli.log_level, &cli.log_name);
    start_logging(cli.log_level, log.as_ref().ok().and_then(Option::clone));
    match log.and_then(|_| run(&cli)) {
        Ok(()) => process::ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("{error:#}");
            process::ExitCode::FAILURE
        }
    }
}

/// Open the log file, unless logging is off, before the load, so that a path that cannot be
/// written stops the run at once rather than losing the log.
fn open_log(
    level: tracing::level_filters::LevelFilter,
    log_name: &path::Path,
) -> eyre::Result<Option<sync::Arc<fs::File>>> {
    if level == tracing::level_filters::LevelFilter::OFF {
        return Ok(None);
    }
    let file = fs::File::options()
        .create(true)
        .append(true)
        .open(log_name)
        .wrap_err_with(|| format!("opening the log file {}", log_name.display()))?;
    Ok(Some(sync::Arc::new(file)))
}

/// Send what `level` allows to `log` and fatal errors to stderr.
///
/// A record is one write, so it is in the file however the process ends, and the file cannot stall
/// the answers the way a full stderr pipe can.
fn start_logging(level: tracing::level_filters::LevelFilter, log: Option<sync::Arc<fs::File>>) {
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
        .without_time()
        .with_filter(tracing::level_filters::LevelFilter::ERROR);
    let log = log.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .with_filter(level)
    });
    tracing_subscriber::registry().with(stderr).with(log).init();
}

/// Load the player file, then serve the protocol on stdin and stdout.
fn run(cli: &Cli) -> eyre::Result<()> {
    let threads = cli
        .threads
        .map_or_else(lotto::engine::default_workers, num::NonZero::get);
    let (input, output) = (io::stdin().lock(), io::stdout().lock());
    match lotto::engine::load(&cli.input, threads)? {
        lotto::engine::Loaded::Narrow(engine) => lotto::protocol::serve(&engine, input, output),
        lotto::engine::Loaded::Wide(engine) => lotto::protocol::serve(&engine, input, output),
    }
}
