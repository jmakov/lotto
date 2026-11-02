//! Fuzz the player-file and draw parsers with arbitrary bytes.
//!
//! Any input must be accepted without panicking, and every way of reading it must agree: the
//! whole parse, vectorized where the build allows, the scalar parse, the file chunks with their
//! overhang, and the draw reader, reading the input whole and in short pieces, must each count or
//! deliver the lines that splitting the input at `\n` and parsing each line on its own gives.
//!
//! Each input runs twice, as it is and rendered as canonical lines, which is what reaches the fast
//! path.
//!
//! Run from this directory with `cargo +nightly fuzz run parse -- -max_total_time=60`.
#![no_main]
use std::io;
use std::io::Write;

use lotto::{input, ticket};

/// A reader that returns at most `chunk` bytes per read, so that lines span reads as they can
/// in a pipe; one read of the whole input never splits a line.
struct Chunked<'a> {
    /// The bytes not read yet.
    data: &'a [u8],
    /// The most bytes one read returns; at least 1.
    chunk: usize,
}

impl io::Read for Chunked<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let length = self.chunk.min(buffer.len()).min(self.data.len());
        let (head, tail) = self.data.split_at(length);
        buffer[..length].copy_from_slice(head);
        self.data = tail;
        Ok(length)
    }
}

/// Bytes in place of a number: the neighbours of the ASCII digits, signs, and other whitespace.
const NEAR_MISSES: [u8; 16] = *b"/:;.,+-\r\t\x0b\x0c \n\x00\x7fx";
/// Byte values from here on render as [`NEAR_MISSES`].
const FIRST_NEAR_MISS: u8 = 240;

/// Render `data` as lines of five one- or two-digit numbers, most of them valid.
///
/// The top byte values become [`NEAR_MISSES`], and the last line may be short: each leaves the
/// fast path in input that is otherwise well-formed.
fn canonical_lines(data: &[u8]) -> Vec<u8> {
    let mut lines = Vec::with_capacity(data.len() * 3);
    for line in data.chunks(5) {
        for (index, &byte) in line.iter().enumerate() {
            if index > 0 {
                lines.push(b' ');
            }
            match byte.checked_sub(FIRST_NEAR_MISS) {
                Some(index) => lines.push(NEAR_MISSES[usize::from(index)]),
                None => write!(lines, "{}", byte % 100).expect("writing to a Vec cannot fail"),
            }
        }
        lines.push(b'\n');
    }
    lines
}

/// The lines of `data` split at `\n`, each parsed on its own.
fn split_lines(data: &[u8]) -> Vec<ticket::Line> {
    data.split_inclusive(|&byte| byte == b'\n')
        .map(|line| ticket::parse_line(line.strip_suffix(b"\n").unwrap_or(line)))
        .collect()
}

/// What a parse counts, in a form independent of the order of the ranks.
fn summary(mut counts: input::RankCounts) -> (Vec<u32>, input::Tally) {
    counts.ranks.sort_unstable();
    (counts.ranks, counts.tally)
}

/// What the `chunks` equal file chunks of `data` count between them.
fn chunked_counts(data: &[u8], chunks: u64) -> (Vec<u32>, input::Tally) {
    let length = data.len() as u64;
    let mut buffer = vec![0; input::BUFFER_BYTES];
    let mut counts = input::RankCounts::default();
    for chunk in 0..chunks {
        let owned = input::chunk_lines(
            |destination, offset| {
                let offset = usize::try_from(offset).expect("offsets are within the data");
                destination.copy_from_slice(&data[offset..offset + destination.len()]);
                Ok(())
            },
            length,
            length * chunk / chunks..length * (chunk + 1) / chunks,
            &mut buffer,
        )
        .expect("reading from memory cannot fail");
        input::parse_lines(owned, &mut counts);
    }
    summary(counts)
}

/// The lines of `reader` as the draw reader delivers them.
fn draw_lines(reader: impl io::Read) -> Vec<ticket::Line> {
    let mut draws = input::Draws::new(reader);
    let mut lines = Vec::new();
    while draws
        .visit_read(|_, line| {
            lines.push(line);
            Ok(())
        })
        .expect("reading from memory cannot fail")
    {}
    lines
}

/// Run every comparison on one input.
fn check(data: &[u8]) {
    let expected = split_lines(data);
    let mut per_line = input::RankCounts::default();
    for &line in &expected {
        per_line.observe(line);
    }
    let expected_counts = summary(per_line);
    let mut whole = input::RankCounts::default();
    input::parse_lines(data, &mut whole);
    let invalid = whole.tally.invalid;
    let lines = whole.tally.valid + invalid + whole.tally.blank;
    assert_eq!(lines, expected.len() as u64, "the whole parse lost a line");
    assert_eq!(summary(whole), expected_counts, "the whole parse and parse_line disagree");
    let mut positions = Vec::new();
    input::visit_invalid_lines(data, invalid, |position, _| positions.push(position));
    let invalid_positions = (0_u64..)
        .zip(&expected)
        .filter(|&(_, &line)| line == ticket::Line::Invalid)
        .map(|(position, _)| position);
    assert!(positions.into_iter().eq(invalid_positions), "the scan of invalid lines disagrees");
    let mut scalar = input::RankCounts::default();
    input::parse_lines_scalar(data, &mut scalar);
    assert_eq!(summary(scalar), expected_counts, "the scalar parse and parse_line disagree");

    // The first byte picks the chunk count and the read size, so the fuzzer can steer where
    // chunks and reads end.
    let steer = data.first().map_or(1, |&byte| u64::from(byte % 64) + 1);
    assert_eq!(chunked_counts(data, steer), expected_counts, "file chunks disagree");
    assert_eq!(draw_lines(data), expected, "the draw reader disagrees");
    let chunk = usize::try_from(steer).expect("at most 64");
    assert_eq!(
        draw_lines(Chunked { data, chunk }),
        expected,
        "the draw reader disagrees on short reads"
    );
}

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    check(data);
    check(&canonical_lines(data));
});
