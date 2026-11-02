//! Bounded-memory reads of the player file and of the draw stream.
//!
//! The player file may be larger than memory. It is read in fixed-size chunks that workers take
//! in parallel, each through one reusable buffer, and [`crate::engine`] counts the ranks of a
//! batch of chunks before the next batch is read. A line belongs to the chunk holding its first
//! byte. A chunk is read with the byte before it, which tells whether a line starts at the
//! chunk's first byte, and with an overhang after it, which completes its last line unless that
//! line is longer than [`ticket::MAX_LINE_BYTES`] and invalid anyway. The draw stream is read
//! line by line under the same limit, so no input, however long its lines, grows a buffer.

use std::io::BufRead;
use std::os::unix::fs::FileExt;
use std::{fs, io, iter, num, path};

use eyre::WrapErr;

use crate::ticket;

/// Bytes of one file chunk in the program's load. Workers take chunks as they become free, so
/// fast workers simply take more of them. On the benchmark machine, ten million players loaded
/// fastest at this size: 0.9 ms slower at 1 MiB, 18 ms slower at 10 MiB and 3 ms slower at
/// 128 KiB (the `load_chunk` benchmarks).
pub(crate) const CHUNK_BYTES: num::NonZeroUsize =
    num::NonZeroUsize::new(512 * 1024).expect("512 KiB is not zero");
/// The shortest valid player line, `1 2 3 4 5` plus its newline.
pub(crate) const MIN_TICKET_BYTES: usize = 10;
/// Bytes a parsing task's buffer holds for the program's 512 KiB chunks: a chunk, the byte before
/// it, and the longest valid last line with its newline.
pub const BUFFER_BYTES: usize = buffer_bytes(CHUNK_BYTES.get());
/// Bytes read from the draw stream at a time; one pipe write of draws fits in one read.
const DRAW_BUFFER_BYTES: usize = 64 * 1024;
/// Bytes of an invalid line that its warning quotes. A wrong line of a few numbers fits, and a
/// line up to the size of a chunk cannot fill the log.
const QUOTED_LINE_BYTES: usize = 256;

/// Bytes a parsing task's buffer holds for chunks of `chunk_bytes`: a chunk, the byte before it,
/// and the longest valid last line with its newline.
const fn buffer_bytes(chunk_bytes: usize) -> usize {
    chunk_bytes + 1 + ticket::MAX_LINE_BYTES + 1
}

/// An opened player file, read in ranges of a fixed number of bytes.
#[derive(Debug)]
pub(crate) struct Source {
    /// The player file; workers read it at offsets, so they share no cursor.
    file: fs::File,
    /// File length in bytes, fixed for the whole load.
    length: u64,
    /// Bytes of every chunk but the last, which may be shorter.
    chunk_bytes: usize,
}

impl Source {
    /// Open `path` to be read in chunks of `chunk_bytes`, [`CHUNK_BYTES`] in the program.
    ///
    /// # Errors
    ///
    /// Returns an error for a path that cannot be opened or inspected or is not a regular file.
    pub(crate) fn open(path: &path::Path, chunk_bytes: num::NonZeroUsize) -> eyre::Result<Self> {
        let file = fs::File::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
        let metadata = file
            .metadata()
            .wrap_err_with(|| format!("inspecting {}", path.display()))?;
        // Chunks are sized by the file's length and read at offsets. A pipe reports length 0, so
        // it would pass for an empty file and every draw would be answered with zeros.
        eyre::ensure!(
            metadata.is_file(),
            "{} is not a regular file",
            path.display()
        );
        Ok(Self {
            file,
            length: metadata.len(),
            chunk_bytes: chunk_bytes.get(),
        })
    }

    /// Number of chunks covering the file; the last one may be shorter.
    pub(crate) fn chunks(&self) -> usize {
        usize::try_from(self.length.div_ceil(self.chunk_bytes as u64))
            .expect("a file's chunk count should fit in usize")
    }

    /// Upper bound on valid lines starting inside one chunk, which sizes per-chunk regions.
    pub(crate) fn chunk_tickets(&self) -> usize {
        self.chunk_bytes / MIN_TICKET_BYTES + 1
    }

    /// Bytes a buffer for [`Self::read_chunk`] holds; see [`buffer_bytes`].
    pub(crate) fn buffer_bytes(&self) -> usize {
        buffer_bytes(self.chunk_bytes)
    }

    /// Read the lines starting inside chunk `index` into `buffer`, which holds
    /// [`Self::buffer_bytes`], and return them; see [`chunk_lines`].
    ///
    /// # Errors
    ///
    /// Returns an error for a failed read, including a file that shrank since it was opened.
    pub(crate) fn read_chunk<'b>(
        &self,
        index: usize,
        buffer: &'b mut [u8],
    ) -> eyre::Result<&'b [u8]> {
        let start = index as u64 * self.chunk_bytes as u64;
        chunk_lines(
            |destination, offset| self.file.read_exact_at(destination, offset),
            self.length,
            start..(start + self.chunk_bytes as u64).min(self.length),
            buffer,
        )
    }
}

/// Read the lines starting in `range` of a `length`-byte file into `buffer` and return them.
///
/// `read_exact_at(destination, offset)` fills `destination` from the file at `offset`; it is
/// injected so that the chunk logic can be tested without a filesystem. The read covers the byte
/// before the range and an overhang of [`ticket::MAX_LINE_BYTES`] + 1 bytes after it, so every
/// returned line is complete and ends with its newline, except the last line of the file, and
/// a last line longer than the limit, which is cut and invalid by its length either way.
///
/// # Errors
///
/// Returns an error for a failed read.
///
/// # Panics
///
/// If `buffer` is shorter than the range plus the byte before it and the overhang.
pub fn chunk_lines(
    read_exact_at: impl FnOnce(&mut [u8], u64) -> io::Result<()>,
    length: u64,
    range: std::ops::Range<u64>,
    buffer: &mut [u8],
) -> eyre::Result<&[u8]> {
    if range.is_empty() {
        return Ok(&[]);
    }
    let from = range.start.saturating_sub(1);
    let to = (range.end + ticket::MAX_LINE_BYTES as u64 + 1).min(length);
    let bytes = &mut buffer[..usize::try_from(to - from).expect("a read fits in memory")];
    read_exact_at(bytes, from).wrap_err_with(|| format!("reading player file at byte {from}"))?;
    let bytes = &*bytes;
    // Where the range ends in `bytes`.
    let last = usize::try_from(range.end - from).expect("the range fits in the buffer");
    // A line starts at the file's first byte and after every newline. Past the file's start, the
    // first owned line therefore follows the first newline in the byte before the range or in any
    // range byte but the last.
    let start = if range.start == 0 {
        0
    } else {
        match memchr::memchr(b'\n', &bytes[..last - 1]) {
            Some(newline) => newline + 1,
            None => return Ok(&[]),
        }
    };
    // The last owned line holds the range's last byte and ends at the first newline from there,
    // or where the buffer ends: at the end of the file, or because the line is longer than the
    // overhang and therefore invalid.
    let end =
        memchr::memchr(b'\n', &bytes[last - 1..]).map_or(bytes.len(), |newline| last + newline);
    Ok(&bytes[start..end])
}

/// Parse the lines of `bytes` into `counts`.
///
/// Every line ends with a newline except possibly the last. Lines in the file's usual format
/// take the `simd` module's path on a CPU with AVX-512 and those of
/// `parse_lines_without_avx512` on any other; any other line goes through
/// [`ticket::parse_line`]. The ranks reach `counts` in an order the counting does not depend on.
pub fn parse_lines(bytes: &[u8], counts: &mut RankCounts) {
    #[cfg(target_arch = "x86_64")]
    if let Some(cpu) = crate::simd::Avx512::detect() {
        crate::simd::parse_lines(cpu, bytes, counts);
        return;
    }
    parse_lines_without_avx512(bytes, counts);
}

/// [`parse_lines`] on a CPU without AVX-512: the vectorized path of the `decode` module where the
/// build has SSE4.1, and the fast path of [`ticket::parse_canonical`] otherwise.
pub(crate) fn parse_lines_without_avx512(bytes: &[u8], counts: &mut RankCounts) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    crate::decode::parse_lines(bytes, counts);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1")))]
    parse_lines_scalar(bytes, counts);
}

/// Parse the lines of `bytes` into `counts` one line at a time, in file order; the reference
/// for the vectorized path.
pub fn parse_lines_scalar(bytes: &[u8], counts: &mut RankCounts) {
    for (line, _) in lines(bytes) {
        counts.observe(line);
    }
}

/// Call `visit` with the zero-based position and the bytes of each invalid line of `bytes`, the
/// lines of a chunk, which hold `invalid` of them; the scan ends at the last one.
///
/// The parsers drop a line's bytes and position for speed, so a chunk with an invalid line is
/// scanned again, line by line. A clean chunk never comes here.
pub fn visit_invalid_lines(bytes: &[u8], invalid: u64, mut visit: impl FnMut(u64, &[u8])) {
    let invalid_lines = (0..)
        .zip(lines(bytes))
        .filter(|(_, (line, _))| *line == ticket::Line::Invalid);
    // More than `usize::MAX` can only mean all of them.
    for (position, (_, contents)) in
        invalid_lines.take(usize::try_from(invalid).unwrap_or(usize::MAX))
    {
        visit(position, contents);
    }
}

/// The lines of `bytes`, each parsed and with its bytes, the newline left out; see
/// [`first_line`].
fn lines(mut bytes: &[u8]) -> impl Iterator<Item = (ticket::Line, &[u8])> {
    iter::from_fn(move || {
        if bytes.is_empty() {
            return None;
        }
        let (line, contents, used) = first_line(bytes);
        bytes = bytes.get(used..).unwrap_or_default();
        Some((line, contents))
    })
}

/// Parse the first line of `bytes`, which must not be empty, through the fast path of
/// [`ticket::parse_canonical`] when the line is in the file's usual format.
///
/// Returns the parse, the line's bytes without its newline, and its length with the newline: one
/// more than the bytes left for a last line without one.
#[inline]
pub(crate) fn first_line(bytes: &[u8]) -> (ticket::Line, &[u8], usize) {
    if let Some((pick, used)) = ticket::parse_canonical(bytes) {
        let line = &bytes[..used];
        return (
            ticket::Line::Ticket(pick),
            line.strip_suffix(b"\n").unwrap_or(line),
            used,
        );
    }
    let end = memchr::memchr(b'\n', bytes).unwrap_or(bytes.len());
    (ticket::parse_line(&bytes[..end]), &bytes[..end], end + 1)
}

/// Warn about the invalid line `number`, one-based, of `source`, quoting its first
/// [`QUOTED_LINE_BYTES`] bytes.
///
/// The bytes come from outside the program, so anything but printable ASCII is escaped, and a
/// line cannot forge another line of the log.
#[cold]
#[inline(never)]
pub(crate) fn warn_invalid_line(source: &'static str, number: u64, line: &[u8]) {
    tracing::warn!(
        source,
        line = number,
        clipped = line.len() > QUOTED_LINE_BYTES,
        contents = %line[..line.len().min(QUOTED_LINE_BYTES)].escape_ascii(),
        "invalid line, ignored"
    );
}

/// Line counts of one file chunk.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Highest number on any valid ticket; counters above it stay zero.
    pub balls: usize,
    /// Valid players.
    pub valid: u64,
    /// Rejected nonblank lines.
    pub invalid: u64,
    /// Lines with no tokens.
    pub blank: u64,
}

impl Tally {
    /// Add the counts of `other`.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.balls = self.balls.max(other.balls);
        self.valid += other.valid;
        self.invalid += other.invalid;
        self.blank += other.blank;
    }

    /// Lines counted, which gives the line number of an invalid line in the next chunk.
    pub(crate) const fn lines(&self) -> u64 {
        self.valid + self.invalid + self.blank
    }
}

/// What one worker collects from its current chunk.
#[derive(Debug, Default)]
pub struct RankCounts {
    /// Quint ranks of valid tickets, in file order on the scalar path.
    pub ranks: Vec<u32>,
    /// The chunk's line counts.
    pub tally: Tally,
}

impl RankCounts {
    /// Reserve room for `tickets` ranks so that staging never reallocates inside a worker.
    pub(crate) fn with_capacity(tickets: usize) -> Self {
        Self {
            ranks: Vec::with_capacity(tickets),
            ..Self::default()
        }
    }

    /// Stage the quint ranks of valid tickets ranked together, whose highest number is `highest`.
    ///
    /// Only the x86-64 decoders rank tickets in groups, so other targets leave this out.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) fn observe_ranks(&mut self, ranks: &[u32], highest: u8) {
        self.tally.balls = self.tally.balls.max(usize::from(highest));
        self.tally.valid += ranks.len() as u64;
        self.ranks.extend_from_slice(ranks);
    }

    /// Record one parsed line: stage a ticket's quint rank, or count a malformed line.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "quint ranks are below C(90, 5) = 43,949,268"
    )]
    pub fn observe(&mut self, line: ticket::Line) {
        match line {
            ticket::Line::Ticket(ticket) => {
                self.tally.balls = self.tally.balls.max(usize::from(ticket.numbers()[4]));
                self.tally.valid += 1;
                self.ranks.push(ticket.quint_rank() as u32);
            }
            ticket::Line::Invalid => self.tally.invalid += 1,
            ticket::Line::Blank => self.tally.blank += 1,
        }
    }
}

/// Parse draw line `number`, warning if it is invalid.
fn parse_draw(number: u64, line: &[u8]) -> ticket::Line {
    let parsed = ticket::parse_line(line);
    if parsed == ticket::Line::Invalid {
        warn_invalid_line("draw", number, line);
    }
    parsed
}

/// The draw stream, read in bounded memory.
///
/// Lines complete in the read buffer are parsed in place. A line that spans reads is collected
/// up to [`ticket::MAX_LINE_BYTES`] + 1 bytes, enough to tell that it is too long to be valid,
/// and the rest of it is dropped.
pub struct Draws<R> {
    /// The stream, with its read buffer.
    reader: io::BufReader<R>,
    /// The bytes read so far of a line that spans reads.
    line: Vec<u8>,
    /// Lines read so far, including invalid and blank ones.
    number: u64,
}

impl<R: io::Read> Draws<R> {
    /// Start at the first line of `reader`, allocating the buffers now rather than at the first
    /// read.
    pub fn new(reader: R) -> Self {
        Self {
            reader: io::BufReader::with_capacity(DRAW_BUFFER_BYTES, reader),
            line: Vec::with_capacity(ticket::MAX_LINE_BYTES + 1),
            number: 0,
        }
    }

    /// Do one read and visit the lines it completes, in order.
    ///
    /// `visit` receives a one-based line number and its parse, including blanks. Returns `false`
    /// at the end of the stream, after visiting a last line that has no newline. The caller sees
    /// each read end, so it can flush what it produced from one read before the next read blocks.
    ///
    /// # Errors
    ///
    /// I/O errors and visitor errors propagate with read context.
    pub fn visit_read(
        &mut self,
        mut visit: impl FnMut(u64, ticket::Line) -> eyre::Result<()>,
    ) -> eyre::Result<bool> {
        let buffer = loop {
            match self.reader.fill_buf() {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                result => break result.wrap_err("reading the draws")?,
            }
        };
        if buffer.is_empty() {
            if !self.line.is_empty() {
                self.number += 1;
                visit(self.number, parse_draw(self.number, &self.line))?;
                self.line.clear();
            }
            return Ok(false);
        }
        let length = buffer.len();
        let mut rest = buffer;
        // A line that began in an earlier read ends at the first newline, or goes on.
        if !self.line.is_empty() {
            let Some(end) = memchr::memchr(b'\n', rest) else {
                collect(&mut self.line, rest);
                self.reader.consume(length);
                return Ok(true);
            };
            collect(&mut self.line, &rest[..end]);
            self.number += 1;
            visit(self.number, parse_draw(self.number, &self.line))?;
            self.line.clear();
            rest = &rest[end + 1..];
        }
        while !rest.is_empty() {
            // A line in the usual format is parsed without searching for its end first. It must
            // end with a newline here, as the buffer may hold only the start of a line.
            if let Some((pick, used)) = ticket::parse_canonical(rest)
                && rest[used - 1] == b'\n'
            {
                self.number += 1;
                visit(self.number, ticket::Line::Ticket(pick))?;
                rest = &rest[used..];
                continue;
            }
            let Some(end) = memchr::memchr(b'\n', rest) else {
                collect(&mut self.line, rest);
                break;
            };
            self.number += 1;
            visit(self.number, parse_draw(self.number, &rest[..end]))?;
            rest = &rest[end + 1..];
        }
        self.reader.consume(length);
        Ok(true)
    }
}

/// Keep `bytes` of a line that spans reads in `line`, up to one byte more than a valid line can
/// have, which is enough to reject it.
fn collect(line: &mut Vec<u8>, bytes: &[u8]) {
    let room = (ticket::MAX_LINE_BYTES + 1).saturating_sub(line.len());
    line.extend_from_slice(&bytes[..room.min(bytes.len())]);
}

/// Regression checks for the module invariants.
#[cfg(test)]
mod tests {
    /// Lines of every kind, including a very long one and an unterminated last line.
    fn mixed_lines() -> Vec<u8> {
        [
            &b"1 2 3 4 5\r\n \n1 2 3 4 5 6\n90 89 88 87 86\n"[..],
            &[b'7'; 100],
            b"\n+01 2 3 4 5\n1 2 3 4 5",
        ]
        .concat()
    }

    /// What a parse of `data` counts.
    fn counted(data: &[u8]) -> crate::testing::Counted {
        let mut counts = super::RankCounts::default();
        super::parse_lines(data, &mut counts);
        crate::testing::summary(counts)
    }

    /// The lines of `data` split at `\n`, each parsed on its own.
    fn split_lines(data: &[u8]) -> Vec<crate::ticket::Line> {
        crate::testing::pieces(data)
            .map(crate::ticket::parse_line)
            .collect()
    }

    /// What `chunks` equal chunks of `data` count between them, in the form of [`counted`].
    fn chunked(data: &[u8], chunks: usize) -> eyre::Result<crate::testing::Counted> {
        let length = data.len() as u64;
        let mut buffer = vec![0; super::BUFFER_BYTES];
        let mut counts = super::RankCounts::default();
        for chunk in 0..chunks {
            let range =
                length * chunk as u64 / chunks as u64..length * (chunk as u64 + 1) / chunks as u64;
            let owned = super::chunk_lines(
                |destination, offset| {
                    let offset = usize::try_from(offset).unwrap();
                    destination.copy_from_slice(&data[offset..offset + destination.len()]);
                    Ok(())
                },
                length,
                range,
                &mut buffer,
            )?;
            super::parse_lines(owned, &mut counts);
        }
        Ok(crate::testing::summary(counts))
    }

    /// The lines of `reader` as the draw reader delivers them.
    fn draw_lines(reader: impl std::io::Read) -> eyre::Result<Vec<(u64, crate::ticket::Line)>> {
        let mut draws = super::Draws::new(reader);
        let mut lines = Vec::new();
        while draws.visit_read(|number, line| {
            lines.push((number, line));
            Ok(())
        })? {}
        Ok(lines)
    }

    /// Every split of the file into chunks counts each line once, as the whole parse and the
    /// split parse do.
    #[test]
    fn chunks_count_every_line_once() -> eyre::Result<()> {
        let data = mixed_lines();
        let expected = crate::testing::reference(&data);

        assert_eq!(counted(&data), expected);
        for chunks in 1..=data.len() + 2 {
            assert_eq!(chunked(&data, chunks)?, expected, "{chunks} chunks");
        }
        Ok(())
    }

    /// A line longer than the limit is invalid, counted once by the chunk it starts in and
    /// skipped by the chunks it runs through, whether it is cut by the overhang or read whole.
    #[test]
    fn overlong_lines_are_invalid_once() -> eyre::Result<()> {
        let long = vec![b'1'; crate::ticket::MAX_LINE_BYTES + 1];
        let data = [&b"1 2 3 4 5\n"[..], &long, b"\n1 2 3 4 5\n"].concat();
        let pick = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        let ticket = crate::ticket::Line::Ticket(pick);
        let rank = u32::try_from(pick.quint_rank())?;
        let tally = super::Tally {
            balls: 5,
            valid: 2,
            invalid: 1,
            blank: 0,
        };
        let expected = (vec![rank, rank], tally);

        assert_eq!(counted(&data), expected);
        for chunks in [1, 2, 3, 7, 50, 300] {
            assert_eq!(chunked(&data, chunks)?, expected, "{chunks} chunks");
        }
        assert_eq!(
            draw_lines(crate::testing::Chunked::new(&data, 64))?,
            [(1, ticket), (2, crate::ticket::Line::Invalid), (3, ticket)]
        );
        Ok(())
    }

    /// A read failure propagates.
    #[test]
    fn read_errors_propagate() {
        let mut buffer = vec![0; super::BUFFER_BYTES];

        let failed = super::chunk_lines(
            |_, _| Err(std::io::Error::other("injected")),
            1,
            0..1,
            &mut buffer,
        );

        assert!(failed.is_err());
    }

    /// The draw reader numbers every line, whatever the reads look like.
    #[test]
    fn draws_are_numbered_across_reads() -> eyre::Result<()> {
        let data = mixed_lines();
        let expected: Vec<_> = (1..).zip(split_lines(&data)).collect();

        for chunk in [1, 2, 7, 64, 4096] {
            assert_eq!(
                draw_lines(crate::testing::Chunked::new(&data, chunk))?,
                expected,
                "{chunk}-byte reads"
            );
        }
        Ok(())
    }

    proptest::proptest! {
        #![proptest_config(crate::testing::proptest_config())]

        /// The parse counts every line, which is what numbers the invalid lines of the next
        /// chunk, and the second scan finds exactly the invalid lines, at their positions.
        #[test]
        fn the_scan_finds_each_invalid_line(
            lines in proptest::collection::vec(crate::testing::any_line(), 0..40),
            terminated in proptest::bool::ANY,
        ) {
            let data = crate::testing::join(&lines, terminated);
            let mut counts = super::RankCounts::default();
            super::parse_lines(&data, &mut counts);
            let mut scalar = super::RankCounts::default();
            super::parse_lines_scalar(&data, &mut scalar);
            let pieces: Vec<&[u8]> = crate::testing::pieces(&data).collect();
            let expected: Vec<(u64, &[u8])> = (0_u64..)
                .zip(pieces.iter().copied())
                .filter(|&(_, line)| {
                    crate::ticket::parse_line(line) == crate::ticket::Line::Invalid
                })
                .collect();
            let mut found = Vec::new();

            super::visit_invalid_lines(&data, counts.tally.invalid, |position, line| {
                found.push((position, line.to_vec()));
            });

            let found: Vec<(u64, &[u8])> =
                found.iter().map(|(position, line)| (*position, &line[..])).collect();
            proptest::prop_assert_eq!(found, expected);
            proptest::prop_assert_eq!(counts.tally.lines(), pieces.len() as u64);
            proptest::prop_assert_eq!(scalar.tally.lines(), pieces.len() as u64);
        }

        /// Any split of the data into chunks counts the same as the whole parse, which counts
        /// each line as parsing it on its own does; the draw reader, reading in short pieces,
        /// delivers each line in order with its number.
        #[test]
        fn chunks_and_draws_match_the_split_lines(
            lines in proptest::collection::vec(crate::testing::any_line(), 0..40),
            terminated in proptest::bool::ANY,
            chunks in 1_usize..=12,
            piece in 1_usize..=96,
        ) {
            let data = crate::testing::join(&lines, terminated);
            let expected = crate::testing::reference(&data);

            proptest::prop_assert_eq!(&counted(&data), &expected);
            proptest::prop_assert_eq!(&chunked(&data, chunks).unwrap(), &expected);
            let numbered: Vec<_> = (1..).zip(split_lines(&data)).collect();
            let draws = draw_lines(crate::testing::Chunked::new(&data, piece)).unwrap();
            proptest::prop_assert_eq!(draws, numbered);
        }
    }
}
