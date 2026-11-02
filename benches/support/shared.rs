//! What the benchmark targets share: the player file, the random picks, and a running `lotto`
//! process to send draws to. Each target includes this file with `#[path]`, so Cargo does not
//! take it for a benchmark of its own.

use std::io::{BufRead, Write};
use std::{env, fs, io, path, process, thread};

#[path = "random.rs"]
pub(crate) mod random;

/// Picks cycled through by the draw benchmarks.
pub(crate) const PICKS: usize = 1_000;
/// Players in a generated input file: the task's peak load.
const GENERATED_PLAYERS: usize = 10_000_000;

/// The player file to load: [`GENERATED_PLAYERS`] random players, written once into Cargo's
/// temporary target directory and reused by later runs. The benchmarks read no other file.
pub(crate) fn input_file() -> path::PathBuf {
    let directory = path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let generated = directory.join("players-10m.txt");
    if !generated.is_file() {
        // Written under another name and renamed when complete, so an interrupted run never
        // leaves a short file that later runs would take for the full input.
        let partial = directory.join("players-10m.txt.partial");
        write_players(&partial).expect("the generated player file should be writable");
        fs::rename(&partial, &generated).expect("the complete player file should be renamable");
    }
    generated
}

/// Write [`GENERATED_PLAYERS`] random players, one canonical line each, to `path`.
fn write_players(path: &path::Path) -> io::Result<()> {
    let mut random = random::Random(0x1077_0bad);
    let mut writer = io::BufWriter::new(fs::File::create(path)?);
    for _ in 0..GENERATED_PLAYERS {
        let [n1, n2, n3, n4, n5] = random.numbers(90);
        writeln!(writer, "{n1} {n2} {n3} {n4} {n5}")?;
    }
    writer.flush()
}

/// The `lotto` binary that child processes run: `LOTTO_BENCH_BINARY` or Cargo's build.
pub(crate) fn binary() -> path::PathBuf {
    env::var_os("LOTTO_BENCH_BINARY")
        .map_or_else(|| env!("CARGO_BIN_EXE_lotto").into(), path::PathBuf::from)
}

/// A command that runs `binary` on `input` with piped stdin and stdout.
pub(crate) fn command(binary: &path::Path, input: &path::Path) -> process::Command {
    let mut command = process::Command::new(binary);
    command
        .arg(input)
        .stdin(process::Stdio::piped())
        .stdout(process::Stdio::piped())
        .stderr(process::Stdio::null());
    command
}

/// An interactive child whose pipes stay open between readiness and every answer.
pub(crate) struct Running {
    /// Child retained until it exits or a benchmark assertion fails.
    child: process::Child,
    /// Buffered reader for one protocol line at a time.
    stdout: io::BufReader<process::ChildStdout>,
    /// Reused response storage; avoids allocating for each draw.
    response: String,
}

impl Running {
    /// Start a child and wait for READY before issuing any picks.
    pub(crate) fn start(binary: &path::Path, input: &path::Path) -> Self {
        let mut child = command(binary, input)
            .spawn()
            .expect("the lotto binary should start");
        let stdout = io::BufReader::new(child.stdout.take().expect("stdout is piped"));
        let mut running = Self {
            child,
            stdout,
            response: String::with_capacity(64),
        };
        running
            .stdout
            .read_line(&mut running.response)
            .expect("the first line should be readable");
        assert_eq!(
            running.response, "READY\n",
            "the first line should be READY"
        );
        running
    }

    /// Write one request and read its complete answer without closing stdin.
    pub(crate) fn exchange(&mut self, request: &[u8]) {
        self.exchange_lines(request, 1);
    }

    /// Write `request`, holding `answers` picks, and read that many answer lines.
    pub(crate) fn exchange_lines(&mut self, request: &[u8], answers: usize) {
        self.child
            .stdin
            .as_mut()
            .expect("stdin stays open until finish")
            .write_all(request)
            .expect("lotto should accept the picks");
        for _ in 0..answers {
            self.response.clear();
            let bytes = self
                .stdout
                .read_line(&mut self.response)
                .expect("lotto should answer every pick");
            assert!(bytes > 0, "lotto should answer before closing stdout");
        }
    }

    /// Validate the last answer and wait for the child to exit cleanly at EOF.
    pub(crate) fn finish(&mut self) {
        assert_eq!(
            self.response.split_ascii_whitespace().count(),
            4,
            "the last answer should hold four counts: {:?}",
            self.response
        );
        let status = self
            .child
            .wait()
            .expect("the lotto process should be waitable");
        assert!(status.success(), "lotto should exit cleanly: {status}");
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // Failure assertions must not leave a child waiting forever for its next pick. `kill`
        // succeeds on a child that has exited; a second panic while unwinding would abort.
        let reaped = self.child.kill().and_then(|()| self.child.wait());
        assert!(
            thread::panicking() || reaped.is_ok(),
            "the child should be reaped: {reaped:?}"
        );
    }
}

/// [`PICKS`] draws as protocol lines, formatted before timing starts.
pub(crate) fn request_lines() -> Vec<String> {
    let mut random = random::Random(0xdeca_fbad);
    (0..PICKS).map(|_| line(random.numbers(90))).collect()
}

/// `numbers` as one protocol or player line, newline included.
pub(crate) fn line([n1, n2, n3, n4, n5]: [u8; 5]) -> String {
    format!("{n1} {n2} {n3} {n4} {n5}\n")
}
