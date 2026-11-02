//! End-to-end checks of the `lotto` binary against an independent oracle.
//!
//! The oracle parses lines with its own naive tokenizer and counts each draw by AND and
//! popcount over one 128-bit mask per player. It shares neither the parsers, the colex ranks nor
//! the inclusion-exclusion with the application, so agreement checks all three.
//!
//! Every test generates its own player file into a temporary directory: a small one with every
//! kind of line, and a larger one that spans several chunks.

// `Vec<u8>` takes `writeln!` through `io::Write`, `String` through `fmt::Write`.
use std::fmt::Write as FmtWrite;
use std::io::{BufRead, Write};
use std::{fs, io, path, process, sync, thread, time};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

#[path = "../benches/support/random.rs"]
mod random;

/// Valid players in the fixture from [`fixture`].
const FIXTURE_VALID: usize = 701;
/// Invalid lines in the fixture from [`fixture`].
const FIXTURE_INVALID: usize = 6;
/// Lines of the fixture that are not canonical tickets: six invalid ones, a blank line and a
/// valid last line without a newline.
const FIXTURE_TAIL: &[u8] =
    b"1 1 3 4 5\n5 4 3 2 5\n1 2 3 4 5 6\n1 2 3 4\n1 2 3 4 93\n1 2 3 4 -7\n\n1 2 3 4 5";
/// Valid players in [`many_players`].
const MANY_VALID: usize = 100_000;
/// Invalid lines of [`many_players`], by line number, past the first chunk: a repeated number,
/// six numbers, four, one out of range and a negative one.
const MANY_INVALID: [(u64, &str); 6] = [
    (50_001, "70 6 20 46 70"),
    (50_016, "84 12 68 26 47 40"),
    (50_021, "40 11 57 36 40"),
    (50_028, "5 34 24 42"),
    (50_036, "85 21 67 93 2"),
    (50_051, "10 85 69 -7 50"),
];
/// Draws sent to the binary: a ticket, a blank line, an invalid pick (line 3), and two picks
/// with CRLF and extra whitespace. Only the three valid picks get an answer.
const PROTOCOL_PICKS: &[u8] = b"1 2 3 4 5\n\n1 1 2 3 4\n4 79 13 80 56\r\n  1\t2 3 4 5";
/// The valid picks of [`PROTOCOL_PICKS`], in order.
const PROTOCOL_ANSWERED: [[u8; 5]; 3] = [[1, 2, 3, 4, 5], [4, 79, 13, 80, 56], [1, 2, 3, 4, 5]];
/// What the oracle reads from one line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parsed {
    /// Only whitespace.
    Blank,
    /// Anything but five distinct numbers in 1..=90.
    Invalid,
    /// Five distinct numbers in 1..=90, sorted.
    Ticket([u8; 5]),
}

/// Parse a line the plain way: whitespace-separated tokens, each an optional `+` and digits.
fn parse(line: &[u8]) -> Parsed {
    let tokens: Vec<&[u8]> = line
        .split(u8::is_ascii_whitespace)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        return Parsed::Blank;
    }
    if tokens.len() != 5 {
        return Parsed::Invalid;
    }
    let mut numbers = [0; 5];
    for (number, token) in numbers.iter_mut().zip(tokens) {
        let digits = token.strip_prefix(b"+").unwrap_or(token);
        if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
            return Parsed::Invalid;
        }
        let value = digits.iter().try_fold(0_u8, |value, &digit| {
            let value = u16::from(value) * 10 + u16::from(digit - b'0');
            u8::try_from(value).ok().filter(|&value| value <= 90)
        });
        match value {
            Some(value @ 1..=90) => *number = value,
            _ => return Parsed::Invalid,
        }
    }
    numbers.sort_unstable();
    if numbers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Parsed::Invalid;
    }
    Parsed::Ticket(numbers)
}

/// One bit per drawn number.
fn mask(numbers: [u8; 5]) -> u128 {
    numbers.iter().fold(0, |mask, &number| mask | 1 << number)
}

/// Brute-force winner counts: one mask per valid player, duplicates kept.
struct Oracle {
    /// Masks of the valid players, in file order.
    masks: Vec<u128>,
    /// Invalid nonblank lines.
    invalid: usize,
    /// The first 100 valid players, which make draws with guaranteed five-number matches.
    first: Vec<[u8; 5]>,
}

impl Oracle {
    /// Read a whole player file, one line per `\n`.
    fn from_bytes(bytes: &[u8]) -> Self {
        let lines: Vec<Parsed> = bytes.par_split(|&byte| byte == b'\n').map(parse).collect();
        let tickets: Vec<[u8; 5]> = lines
            .iter()
            .filter_map(|line| match line {
                Parsed::Ticket(numbers) => Some(*numbers),
                Parsed::Blank | Parsed::Invalid => None,
            })
            .collect();
        Self {
            masks: tickets.par_iter().map(|&numbers| mask(numbers)).collect(),
            invalid: lines
                .iter()
                .filter(|&&line| line == Parsed::Invalid)
                .count(),
            first: tickets.into_iter().take(100).collect(),
        }
    }

    /// Valid players.
    fn valid(&self) -> usize {
        self.masks.len()
    }

    /// Winners with 2, 3, 4 and 5 matches for `pick`.
    fn query(&self, pick: [u8; 5]) -> [u64; 4] {
        let pick = mask(pick);
        self.masks
            .par_chunks(1 << 14)
            .map(|chunk| {
                let mut counts = [0; 4];
                for &player in chunk {
                    let matches = (player & pick).count_ones() as usize;
                    if let Some(count) = matches.checked_sub(2).and_then(|i| counts.get_mut(i)) {
                        *count += 1;
                    }
                }
                counts
            })
            .reduce(|| [0; 4], |a, b| std::array::from_fn(|i| a[i] + b[i]))
    }
}

/// A small player file with every kind of line.
///
/// 600 random tickets over 12 balls, so every match count occurs, then the first 100 again as
/// duplicates; each with reversed numbers, tabs, double spaces and CRLF. Then [`FIXTURE_TAIL`].
fn fixture() -> Vec<u8> {
    let mut random = random::Random(222);
    let mut numbers: Vec<[u8; 5]> = (0..600).map(|_| random.numbers(12)).collect();
    numbers.extend_from_within(..100);
    let mut bytes = Vec::new();
    for &[a, b, c, d, e] in &numbers {
        writeln!(bytes, " {e}\t{d}  {c} {b} {a}\r").expect("writing to a Vec cannot fail");
    }
    bytes.extend_from_slice(FIXTURE_TAIL);
    bytes
}

/// Picks for the fixture: random ones over its 12 balls, plus every ball outside it once.
fn fixture_picks() -> Vec<[u8; 5]> {
    let mut random = random::Random(333);
    let mut picks: Vec<_> = (0..1000).map(|_| random.numbers(12)).collect();
    picks.push([86, 87, 88, 89, 90]);
    picks
}

/// A player file of [`MANY_VALID`] random tickets over all 90 balls, about 1.4 MB, so that it
/// spans several 512 KiB chunks, with the lines of [`MANY_INVALID`] in place.
fn many_players() -> Vec<u8> {
    let mut random = random::Random(0x1077_0bad);
    let mut invalid = MANY_INVALID.iter().peekable();
    let mut bytes = Vec::new();
    let (mut line, mut valid) = (0, 0);
    while valid < MANY_VALID {
        line += 1;
        if let Some((_, contents)) = invalid.next_if(|&&(number, _)| number == line) {
            writeln!(bytes, "{contents}").expect("writing to a Vec cannot fail");
        } else {
            let [n1, n2, n3, n4, n5] = random.numbers(90);
            writeln!(bytes, "{n1} {n2} {n3} {n4} {n5}").expect("writing to a Vec cannot fail");
            valid += 1;
        }
    }
    bytes
}

/// Run the `lotto` binary on `input` with `options` and `picks` on stdin; returns (status,
/// stdout, stderr).
fn run_binary(
    input: &path::Path,
    options: &[&str],
    picks: &[u8],
) -> (process::ExitStatus, String, String) {
    let mut child = command(process::Stdio::piped())
        .arg(input)
        .args(options)
        .spawn()
        .expect("the lotto binary should start");
    let mut stdin = child.stdin.take().expect("stdin is piped");
    // A process that exits early closes its stdin; the exit status reports why.
    if let Err(error) = stdin.write_all(picks) {
        assert_eq!(
            error.kind(),
            io::ErrorKind::BrokenPipe,
            "writing picks: {error}"
        );
    }
    drop(stdin);
    let output = child
        .wait_with_output()
        .expect("the lotto process should finish");
    (
        output.status,
        String::from_utf8(output.stdout).expect("stdout is ASCII"),
        String::from_utf8(output.stderr).expect("stderr is UTF-8"),
    )
}

/// Expected stdout for the valid picks `picks`: READY, then the oracle's answer to each.
fn expected_stdout(oracle: &Oracle, picks: &[[u8; 5]]) -> String {
    let mut expected = String::from("READY\n");
    for &pick in picks {
        let [e2, e3, e4, e5] = oracle.query(pick);
        writeln!(expected, "{e2} {e3} {e4} {e5}").expect("writing to a String cannot fail");
    }
    expected
}

/// Run the binary on `input` with `options`, and assert the oracle's answer to every pick and
/// the file's line counts. Returns the warnings of the log.
fn assert_answers(
    input: &path::Path,
    options: &[&str],
    oracle: &Oracle,
    picks: &[[u8; 5]],
) -> Vec<String> {
    let mut requests = Vec::new();
    for &[n1, n2, n3, n4, n5] in picks {
        writeln!(requests, "{n1} {n2} {n3} {n4} {n5}").expect("writing to a Vec cannot fail");
    }
    let expected = expected_stdout(oracle, picks);

    let directory = tempfile::tempdir().expect("a temporary directory should be available");
    let log = directory.path().join("log.txt");
    let log_name = log.to_str().expect("the path is UTF-8");
    let mut arguments = options.to_vec();
    arguments.extend(["--log-level", "info", "--log-name", log_name]);

    let (status, stdout, stderr) = run_binary(input, &arguments, &requests);

    assert!(status.success(), "{options:?}: {status}, stderr: {stderr}");
    assert_eq!(stdout, expected, "{options:?}");
    let counts = format!("valid={} invalid={}", oracle.valid(), oracle.invalid);
    let contents = fs::read_to_string(&log).expect("the log file should be readable text");
    assert!(contents.contains(&counts), "{options:?}: {contents}");
    logged(&log)
}

/// Every worker count answers the fixture's draws as the oracle does.
#[test]
fn answers_match_the_oracle_on_the_fixture() -> eyre::Result<()> {
    let bytes = fixture();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, &bytes)?;
    let oracle = Oracle::from_bytes(&bytes);

    assert_eq!(
        (oracle.valid(), oracle.invalid),
        (FIXTURE_VALID, FIXTURE_INVALID)
    );
    for threads in ["1", "2", "7", "16"] {
        assert_answers(&path, &["--threads", threads], &oracle, &fixture_picks());
    }
    Ok(())
}

/// A file of the shortest valid lines puts the most tickets a chunk can hold into its fixed
/// region, so a smaller bound fails here while ordinary files stay far below it.
#[test]
fn shortest_lines_fill_a_chunk() -> eyre::Result<()> {
    const PLAYERS: usize = 60_000;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, b"1 2 3 4 5\n".repeat(PLAYERS))?;

    let (status, stdout, stderr) = run_binary(&path, &["--threads", "1"], b"1 2 3 4 5\n");

    assert!(status.success(), "{status}, stderr: {stderr}");
    assert_eq!(stdout, format!("READY\n0 0 0 {PLAYERS}\n"));
    Ok(())
}

/// The warnings in the log file `path`, each without its timestamp.
fn logged(path: &path::Path) -> Vec<String> {
    fs::read_to_string(path)
        .expect("the log file should be readable text")
        .lines()
        .filter_map(|line| {
            line.split_once("WARN ")
                .map(|(_, warning)| warning.to_owned())
        })
        .collect()
}

/// The warning about `source` line `number`, with `contents` not clipped.
fn warning(source: &str, number: u64, contents: &str) -> String {
    format!(
        "lotto::input: invalid line, ignored source=\"{source}\" line={number} clipped=false \
         contents={contents}"
    )
}

/// With logging on, the binary prints READY, answers only valid picks, logs the invalid one and
/// the file's line counts, and keeps it all off stderr.
#[test]
fn binary_serves_the_protocol() -> eyre::Result<()> {
    let bytes = fixture();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, &bytes)?;
    let expected = expected_stdout(&Oracle::from_bytes(&bytes), &PROTOCOL_ANSWERED);
    let log = directory.path().join("log.txt");
    let log_name = log.to_str().expect("the path is UTF-8");

    let (status, stdout, stderr) = run_binary(
        &path,
        &["--log-level", "info", "--log-name", log_name],
        PROTOCOL_PICKS,
    );

    assert!(status.success(), "{status}, stderr: {stderr}");
    assert_eq!(stdout, expected);
    let contents = fs::read_to_string(&log).expect("the log file should be readable text");
    let counts = format!("valid={FIXTURE_VALID} invalid={FIXTURE_INVALID}");
    assert!(contents.contains(&counts), "{contents}");
    assert!(!stderr.contains("invalid line"), "{stderr}");
    let warnings = logged(&log);
    assert_eq!(warnings.len(), 7, "{warnings:?}");
    assert_eq!(
        warnings.last(),
        Some(&warning("draw", 3, "1 1 2 3 4")),
        "{warnings:?}"
    );
    Ok(())
}

/// Each invalid player line reaches the log once, in file order, with its true line number across
/// chunks and blank lines, and with its bytes escaped and clipped. Nothing else is in the log.
#[test]
fn binary_logs_each_invalid_line() -> eyre::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    let mut players = b"1 2 3 4 5\n\n1 1 2 3 4\n".to_vec();
    // Past the first 512 KiB chunk, and with no invalid line in the second until its end.
    players.extend(b"1 2 3 4 5\n".repeat(60_000));
    players.extend(b"85 21 67 93 2\n\xff\tx\n");
    players.extend(b"x".repeat(300));
    fs::write(&path, &players)?;
    let log = directory.path().join("custom.log");
    let log_name = log.to_str().expect("the path is UTF-8");

    let (status, stdout, stderr) = run_binary(
        &path,
        &["--log-level", "info", "--log-name", log_name],
        b"1 2 3 4 5\n5 5 5 5 5\n\n1 2\n",
    );

    assert!(status.success(), "{status}, stderr: {stderr}");
    assert_eq!(stdout, "READY\n0 0 0 60001\n");
    let clipped = format!(
        "lotto::input: invalid line, ignored source=\"players\" line=60006 clipped=true \
         contents={}",
        "x".repeat(256)
    );
    assert_eq!(
        logged(&log),
        [
            warning("players", 3, "1 1 2 3 4"),
            warning("players", 60_004, "85 21 67 93 2"),
            warning("players", 60_005, "\\xff\\tx"),
            clipped,
            warning("draw", 2, "5 5 5 5 5"),
            warning("draw", 4, "1 2"),
        ]
    );
    Ok(())
}

/// Kill and reap the child even if an interactive protocol assertion fails.
struct ChildGuard(process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        // `kill` succeeds on a child that has exited; a second panic while unwinding would abort.
        let reaped = self.0.kill().and_then(|()| self.0.wait());
        assert!(
            thread::panicking() || reaped.is_ok(),
            "the child should be reaped: {reaped:?}"
        );
    }
}

/// The binary with stdin and stdout piped and stderr to `stderr`. It runs from Cargo's scratch
/// directory for tests, where the default log file lands.
fn command(stderr: process::Stdio) -> process::Command {
    let mut command = process::Command::new(env!("CARGO_BIN_EXE_lotto"));
    command
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .stdin(process::Stdio::piped())
        .stdout(process::Stdio::piped())
        .stderr(stderr);
    command
}

/// Read `stdout` on its own thread. The returned function gives the next line, and a timeout
/// bounds a process that stopped answering; [`ChildGuard`] closes the pipe on failure.
fn line_reader(stdout: process::ChildStdout) -> (impl Fn() -> String, thread::JoinHandle<()>) {
    let (sender, receiver) = sync::mpsc::channel();
    let reader = thread::spawn(move || {
        for line in io::BufReader::new(stdout).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let read_line = move || {
        receiver
            .recv_timeout(time::Duration::from_secs(30))
            .expect("protocol line should arrive without closing stdin")
            .expect("stdout should be readable")
    };
    (read_line, reader)
}

/// READY and each answer arrive while the input stream remains open.
#[test]
fn binary_flushes_each_protocol_line() -> eyre::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, b"1 2 3 4 5\n1 2 3 6 7\n")?;

    let mut child = ChildGuard(command(process::Stdio::null()).arg(&path).spawn()?);
    let mut stdin = child.0.stdin.take().expect("stdin is piped");
    let (read_line, reader) = line_reader(child.0.stdout.take().expect("stdout is piped"));

    assert_eq!(read_line(), "READY");
    writeln!(stdin, "1 2 3 4 5")?;
    assert_eq!(read_line(), "0 1 0 1");
    writeln!(stdin, "86 87 88 89 90")?;
    assert_eq!(read_line(), "0 0 0 0");
    drop(stdin);
    assert!(child.0.wait()?.success());
    reader.join().expect("stdout reader should not panic");
    Ok(())
}

/// A read that ends inside a line still flushes the answers it completed: the process must not
/// wait for the rest of that line, which the caller may send only after it has those answers.
#[test]
fn binary_flushes_answers_before_waiting_for_a_partial_line() -> eyre::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, b"1 2 3 4 5\n1 2 3 6 7\n")?;
    let mut child = ChildGuard(command(process::Stdio::null()).arg(&path).spawn()?);
    let mut stdin = child.0.stdin.take().expect("stdin is piped");
    let (read_line, reader) = line_reader(child.0.stdout.take().expect("stdout is piped"));

    assert_eq!(read_line(), "READY");
    stdin.write_all(b"1 2 3 4 5\n1 2 3 6 7\n86 87")?;
    assert_eq!(read_line(), "0 1 0 1");
    assert_eq!(read_line(), "0 1 0 1");
    stdin.write_all(b" 88 89 90\n")?;
    assert_eq!(read_line(), "0 0 0 0");
    drop(stdin);

    assert!(child.0.wait()?.success());
    reader.join().expect("stdout reader should not panic");
    Ok(())
}

/// A caller that never reads stderr cannot stall the answers, however many invalid picks it
/// sends: each warning goes to the log file.
#[test]
fn binary_answers_while_stderr_is_not_read() -> eyre::Result<()> {
    const INVALID_PICKS: usize = 3000;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, b"1 2 3 4 5\n1 2 3 6 7\n")?;
    let log = directory.path().join("log.txt");
    let mut child = ChildGuard(
        command(process::Stdio::piped())
            .current_dir(directory.path())
            .arg(&path)
            .arg("--log-level=warn")
            .spawn()?,
    );
    let mut stdin = child.0.stdin.take().expect("stdin is piped");
    let (read_line, reader) = line_reader(child.0.stdout.take().expect("stdout is piped"));

    // One warning per pick would fill a 64 KiB pipe; these bytes still fit in stdin's own pipe.
    stdin.write_all(&b"1 1 1 1 1\n".repeat(INVALID_PICKS))?;
    writeln!(stdin, "1 2 3 4 5")?;

    assert_eq!(read_line(), "READY");
    assert_eq!(read_line(), "0 1 0 1");
    drop(stdin);
    let stderr = io::read_to_string(child.0.stderr.take().expect("stderr is piped"))?;
    assert!(child.0.wait()?.success());
    reader.join().expect("stdout reader should not panic");
    assert_eq!(logged(&log).len(), INVALID_PICKS, "{stderr}");
    Ok(())
}

/// A pipe has no length to size the chunks by, so `/dev/stdin` as the player file must fail
/// before READY instead of loading no players and answering every draw with zeros.
#[test]
fn binary_rejects_a_pipe_as_player_file() {
    let (status, stdout, stderr) = run_binary(path::Path::new("/dev/stdin"), &[], b"1 2 3 4 5\n");

    assert_eq!(status.code(), Some(1), "{stderr}");
    assert_eq!(stdout, "");
    assert!(stderr.contains("not a regular file"), "{stderr}");
}

/// The binary agrees with the oracle on a file of several chunks over all 90 balls, for 1,000
/// random draws and the first 100 players' own tickets, and logs each invalid line with its
/// number.
#[test]
fn answers_match_the_oracle_on_many_players() -> eyre::Result<()> {
    let bytes = many_players();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("players.txt");
    fs::write(&path, &bytes)?;
    let oracle = Oracle::from_bytes(&bytes);
    let mut random = random::Random(0xdeca_fbad);
    let mut picks: Vec<_> = (0..1000).map(|_| random.numbers(90)).collect();
    picks.extend_from_slice(&oracle.first);

    let warnings = assert_answers(&path, &[], &oracle, &picks);

    assert!(
        bytes.len() > 2 * 512 * 1024,
        "{} bytes fit two chunks",
        bytes.len()
    );
    assert_eq!((oracle.valid(), oracle.invalid), (MANY_VALID, 6));
    assert_eq!(oracle.first.len(), 100);
    let expected: Vec<_> = MANY_INVALID
        .iter()
        .map(|&(number, contents)| warning("players", number, contents))
        .collect();
    assert_eq!(warnings, expected);
    Ok(())
}

/// The oracle's own parser follows the documented line grammar.
#[test]
fn oracle_parser_follows_the_grammar() {
    assert_eq!(parse(b"+1 02 003 90 4"), Parsed::Ticket([1, 2, 3, 4, 90]));
    assert_eq!(parse(b" \t\r"), Parsed::Blank);
    for line in [
        &b"1 2 3 4"[..],
        b"1 2 3 4 91",
        b"1 2 3 4 0",
        b"1 2 3 4 4",
        b"1 2 3 4 +",
        b"++1 2 3 4 5",
        b"1 2 3 4 5+",
    ] {
        assert_eq!(
            parse(line),
            Parsed::Invalid,
            "{:?}",
            String::from_utf8_lossy(line)
        );
    }
    assert_eq!(parse(b"5 4 3 2 1\r"), Parsed::Ticket([1, 2, 3, 4, 5]));
}
