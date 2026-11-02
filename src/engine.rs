//! Dense subset counts turn the interactive phase into 26 reads and a small inversion.
//!
//! The quint table, by far the largest, holds 8-bit counters: a counter that reaches [`u8::MAX`]
//! stays there and the rest of its count lives in a small map, so a ticket repeated more than
//! 255 times costs one map entry instead of a wider table. Every byte of the table is zeroed and
//! read five times for the derivation, and each ticket updates one counter, so the width of the
//! counters sets the load's memory traffic.

use std::{collections, fmt, iter, mem, num, ops, path, thread};

use eyre::WrapErr;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::{combinatorics, input, streaming, ticket};

/// Quints counted by one owner: 128 KiB of counters, which stay in cache while it counts. Sizes
/// from 2^15 to 2^17 reached READY within about 2 ms of each other on the benchmark machine. A
/// power of two keeps the owner of a rank a cheap division.
pub(crate) const OWNER_SIZE: usize = 1 << 17;
/// Owners covering all quints. Each counts its own slice of the table, so counting needs no
/// atomics; a shared table of atomic counters reached READY 53 ms later on the benchmark machine.
const OWNERS: usize = combinatorics::SIZES[5].div_ceil(OWNER_SIZE);
/// File bytes parsed between two counting passes, so the ranks staged at a time stay bounded
/// however large the file is: 512 chunks of the program's 512 KiB.
const BATCH_BYTES: usize = 256 * 1024 * 1024;

/// Keeps [`Count`] to the two widths the program chooses between.
mod sealed {
    /// Implemented for [`u32`] and [`u64`] only.
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// The width of the pair, triple and quad counters: [`u32`] for up to about 1.4 billion valid
/// players, [`u64`] for more.
///
/// The width sets each table's size and so the memory a draw touches: on the benchmark machine,
/// 64-bit tables made `query` 20% and `query_large` 41% slower. [`load`] therefore takes 32 bits
/// whenever the number of valid players allows it.
pub trait Count:
    Copy
    + Default
    + Eq
    + fmt::Debug
    + Send
    + Sync
    + ops::AddAssign
    + ops::DivAssign
    + iter::Sum
    + From<u8>
    + sealed::Sealed
{
    /// Most valid players whose counts these counters hold exactly. More players stop a load into
    /// this width with an error, so a count is never silently wrong.
    const MAX_PLAYERS: u64;

    /// `count`, which the player limit keeps within this width.
    fn from_count(count: u64) -> Self;

    /// The counter as a signed 64-bit integer, for the draw's inversion.
    fn signed(self) -> i64;
}

/// The derivation in [`crate::streaming`] sums up to three times a count.
impl Count for u32 {
    const MAX_PLAYERS: u64 = u32::MAX as u64 / 3;

    fn from_count(count: u64) -> Self {
        Self::try_from(count).expect("the player limit keeps every count within u32")
    }

    #[inline]
    fn signed(self) -> i64 {
        i64::from(self)
    }
}

/// [`Engine::query`] works in signed 64-bit arithmetic, and its largest term reaches 30 times
/// the player count when every player holds the drawn ticket. The derivation's sums, up to three
/// times a count, would allow more.
impl Count for u64 {
    const MAX_PLAYERS: u64 = i64::MAX.cast_unsigned() / 30;

    fn from_count(count: u64) -> Self {
        count
    }

    #[inline]
    fn signed(self) -> i64 {
        self.cast_signed()
    }
}

/// Most load workers by default. READY climbs past this on a large machine, from starting the
/// threads and spreading them apart, while the evaluator's CPU count is unknown; see the README.
const MAX_DEFAULT_WORKERS: usize = 32;

/// One load worker per CPU this process may use, up to `MAX_DEFAULT_WORKERS`.
#[must_use]
pub fn default_workers() -> usize {
    thread::available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_DEFAULT_WORKERS)
}

/// What parsing one file chunk left for counting.
#[derive(Debug)]
struct ParsedChunk<'a> {
    /// The chunk's quint ranks, grouped by owner so each owner reads one contiguous slice.
    ranks: &'a [u32],
    /// Owner `o` holds `ranks[owners[o]..owners[o + 1]]`.
    owners: [u32; OWNERS + 1],
    /// The chunk's line counts.
    tally: input::Tally,
}

/// Stable counting sort of one chunk's ranks by owner into `region`; returns group bounds.
///
/// # Panics
///
/// If `ranks` exceeds `region`, which the [`input::Source::chunk_tickets`] bound rules out.
fn group_by_owner(ranks: &[u32], region: &mut [u32]) -> [u32; OWNERS + 1] {
    assert!(
        ranks.len() <= region.len(),
        "{} ranks exceed the {}-ticket chunk bound",
        ranks.len(),
        region.len()
    );
    let mut bounds = [0; OWNERS + 1];
    for &rank in ranks {
        bounds[rank as usize / OWNER_SIZE + 1] += 1;
    }
    for owner in 0..OWNERS {
        bounds[owner + 1] += bounds[owner];
    }
    let mut cursors = bounds;
    for &rank in ranks {
        let cursor = &mut cursors[rank as usize / OWNER_SIZE];
        region[*cursor as usize] = rank;
        *cursor += 1;
    }
    bounds
}

/// Add the staged ranks of `chunks` to `quints`, counting each wrap of a counter in `wraps`.
///
/// Each owner counts its own slice of the table from every chunk's group for it, so counting
/// needs no atomics and an owner's writes stay in cache. A counter wraps after 256 players; the
/// wrap is noted and the count goes on modulo 256, which keeps the loop a plain increment:
/// checking for saturation before the store instead cost 16 ms of READY on the benchmark machine.
/// [`saturate`] turns the wraps into the stored form once the file is counted.
fn count_by_owner(
    quints: &mut [u8],
    wraps: &mut collections::HashMap<u32, u64>,
    chunks: &[ParsedChunk],
) {
    let wrapped: Vec<Vec<u32>> = quints
        .par_chunks_mut(OWNER_SIZE)
        .enumerate()
        .map(|(owner, counters)| {
            let base = owner * OWNER_SIZE;
            let mut wrapped = Vec::new();
            for chunk in chunks {
                let group = chunk.owners[owner] as usize..chunk.owners[owner + 1] as usize;
                for &rank in &chunk.ranks[group] {
                    let counter = &mut counters[rank as usize - base];
                    *counter = counter.wrapping_add(1);
                    if *counter == 0 {
                        wrapped.push(rank);
                    }
                }
            }
            wrapped
        })
        .collect();
    for rank in wrapped.into_iter().flatten() {
        *wraps.entry(rank).or_default() += 1;
    }
}

/// Turn counters that wrapped into the stored form: the counter holds [`u8::MAX`] and the
/// returned map holds the rest of the count, so a query consults the map only for a counter at
/// the maximum.
fn saturate(
    quints: &mut [u8],
    mut wraps: collections::HashMap<u32, u64>,
) -> collections::HashMap<u32, u64> {
    // In place: a second map would double the largest table the load can hold in this map.
    for (&rank, wrapped) in &mut wraps {
        let counter = &mut quints[rank as usize];
        *wrapped = u64::from(*counter) + (*wrapped << 8) - u64::from(u8::MAX);
        *counter = u8::MAX;
    }
    wraps
}

/// Warn about the invalid lines of `chunks`, in file order, the first being chunk `first` of
/// `source`, which follows `lines_before` lines.
///
/// The parsers keep no bytes of an invalid line, so each chunk with one is read and scanned again.
/// With warnings off, which is the default, that work is skipped.
///
/// # Errors
///
/// Returns an error for a failed read.
fn warn_invalid_lines(
    source: &input::Source,
    first: usize,
    mut lines_before: u64,
    chunks: &[ParsedChunk],
) -> eyre::Result<()> {
    if !tracing::enabled!(tracing::Level::WARN) {
        return Ok(());
    }
    let mut buffer = Vec::new();
    for (index, chunk) in (first..).zip(chunks) {
        if chunk.tally.invalid > 0 {
            buffer.resize(source.buffer_bytes(), 0);
            let lines = source.read_chunk(index, &mut buffer)?;
            input::visit_invalid_lines(lines, chunk.tally.invalid, |position, line| {
                input::warn_invalid_line("players", lines_before + position + 1, line);
            });
        }
        lines_before += chunk.tally.lines();
    }
    Ok(())
}

/// An engine for a file, with the counter width [`load`] chose from its number of valid players.
#[derive(Debug)]
pub enum Loaded {
    /// 32-bit counters, for up to [`Count::MAX_PLAYERS`] of [`u32`], about 1.4 billion, valid
    /// players.
    Narrow(Engine<u32>),
    /// 64-bit counters, for more players.
    Wide(Engine<u64>),
}

/// Load the file at `path` on `threads` workers, with the narrowest counters its valid players
/// allow.
///
/// Counting the quints does not depend on the width, which only matters to the tables derived
/// afterwards. So the width is chosen once the whole file is counted, from the exact number of
/// valid players, and the load stops only past the 64-bit limit.
///
/// # Arguments
///
/// * `path` - a regular file of player lines.
/// * `threads` - load workers, at least 1.
///
/// # Returns
///
/// The engine, in the variant of its counter width. The caller matches once and runs the draws
/// on the concrete engine, so a draw pays no branch on the width.
///
/// # Errors
///
/// Returns an error for I/O failures, failure to create the workers, or more valid players than
/// 64-bit counters hold.
pub fn load(path: impl AsRef<path::Path>, threads: usize) -> eyre::Result<Loaded> {
    load_in_chunks(path, threads, input::CHUNK_BYTES)
}

/// [`load`] reading the file in chunks of another size than the program's 512 KiB, for the
/// benchmarks that compare chunk sizes.
///
/// # Arguments
///
/// * `path` - a regular file of player lines.
/// * `threads` - load workers, at least 1.
/// * `chunk_bytes` - bytes of every chunk but the last. Each parsing task holds a buffer of a
///   chunk plus 4 KiB, and a batch of chunks stays 256 MiB of file, so the staged ranks keep
///   their bound up to chunks of that size.
///
/// # Returns
///
/// The engine, as [`load`] returns it.
///
/// # Errors
///
/// As [`load`].
pub fn load_in_chunks(
    path: impl AsRef<path::Path>,
    threads: usize,
    chunk_bytes: num::NonZeroUsize,
) -> eyre::Result<Loaded> {
    let source = input::Source::open(path.as_ref(), chunk_bytes)?;
    on_workers(threads, || {
        Ok(by_width(count(
            &source,
            BATCH_BYTES / chunk_bytes,
            u64::MAX_PLAYERS,
        )?))
    })
}

/// Derive the tables of `counted` at the narrowest width that holds its valid players.
fn by_width(counted: Counted) -> Loaded {
    if counted.tally.valid <= u32::MAX_PLAYERS {
        Loaded::Narrow(counted.into_engine())
    } else {
        Loaded::Wide(counted.into_engine())
    }
}

/// Run `load` on a new pool of `threads` Rayon workers.
fn on_workers<T: Send>(
    threads: usize,
    load: impl FnOnce() -> eyre::Result<T> + Send,
) -> eyre::Result<T> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .wrap_err("starting the load workers")?
        .install(load)
}

/// What counting a file leaves for the derivation, the same for both counter widths.
#[derive(Debug)]
struct Counted {
    /// Counts of every five-ball subset, by colex rank, modulo 256.
    quints: Vec<u8>,
    /// How often each quint counter wrapped past [`u8::MAX`], by rank.
    wraps: collections::HashMap<u32, u64>,
    /// The counted lines and the highest number on a valid ticket.
    tally: input::Tally,
}

impl Counted {
    /// Derive the quad, triple and pair tables at width `C`.
    fn into_engine<C: Count>(self) -> Engine<C> {
        let Self {
            mut quints,
            wraps,
            tally,
        } = self;
        let overflow = saturate(&mut quints, wraps);
        let mut quads = streaming::derive(&quints, tally.balls, 5, C::from(1));
        // The derivation saw the saturated counters only; each overflow entry adds the rest of
        // its count to the five quads of its quint. The smaller tables then inherit it.
        for (&rank, &extra) in &overflow {
            let subset = combinatorics::unrank(rank as usize, 5)
                .expect("overflow ranks come from valid tickets");
            let numbers = std::array::from_fn(|index| subset[index] + 1);
            let ticket = ticket::Ticket::validated(numbers).expect("unranked balls are a ticket");
            for quad in ticket.ranks().quads {
                quads[quad] += C::from_count(extra);
            }
        }
        let triples = streaming::derive(&quads, tally.balls, 4, C::from(2));
        let pairs = streaming::derive(&triples, tally.balls, 3, C::from(3));
        Engine {
            tables: [pairs, triples, quads],
            quints,
            overflow,
            valid: tally.valid,
            invalid: tally.invalid,
        }
    }
}

/// Parse `source` and count its quints on the current Rayon pool, `batch` chunks at a time,
/// rejecting more than `limit` valid players.
fn count(source: &input::Source, batch: usize, limit: u64) -> eyre::Result<Counted> {
    let batch = batch.clamp(1, source.chunks().max(1));
    // One fixed-size region per chunk of a batch receives the chunk's ranks grouped by owner,
    // so workers never grow a buffer, and a batch bounds the staged ranks for any file size.
    let mut staging = vec![0; batch * source.chunk_tickets()];
    let mut quints = vec![0; combinatorics::SIZES[5]];
    let mut wraps = collections::HashMap::new();
    let mut tally = input::Tally::default();
    for first in (0..source.chunks()).step_by(batch) {
        let last = (first + batch).min(source.chunks());
        let chunks: Vec<_> = (first..last)
            .into_par_iter()
            .zip(staging.par_chunks_mut(source.chunk_tickets()))
            .map_init(
                // Each parsing task's scratch state, reused for the chunks of that task; a
                // worker can run several tasks. The read buffer is written now, so that its
                // pages are not faulted in inside the first read: about 1 ms of READY on
                // the benchmark machine.
                #[expect(
                    clippy::slow_vector_initialization,
                    reason = "a zeroed allocation would leave the pages untouched"
                )]
                || {
                    let mut buffer = Vec::with_capacity(source.buffer_bytes());
                    buffer.resize(source.buffer_bytes(), 0);
                    (
                        buffer,
                        input::RankCounts::with_capacity(source.chunk_tickets()),
                    )
                },
                |(buffer, counts), (index, region)| {
                    let lines = source.read_chunk(index, buffer)?;
                    input::parse_lines(lines, counts);
                    let owners = group_by_owner(&counts.ranks, region);
                    counts.ranks.clear();
                    Ok(ParsedChunk {
                        ranks: region,
                        owners,
                        tally: mem::take(&mut counts.tally),
                    })
                },
            )
            .collect::<eyre::Result<_>>()?;
        let lines_before = tally.lines();
        for chunk in &chunks {
            tally.merge(&chunk.tally);
        }
        eyre::ensure!(tally.valid <= limit, "more than {limit} valid players");
        // Side by side: scanning before the count instead cost 0.4 ms more of READY on the
        // benchmark machine.
        let (warned, ()) = rayon::join(
            || warn_invalid_lines(source, first, lines_before, &chunks),
            || count_by_owner(&mut quints, &mut wraps, &chunks),
        );
        warned?;
    }
    Ok(Counted {
        quints,
        wraps,
        tally,
    })
}

/// Immutable counts after loading; `query` can be called concurrently without locks.
#[derive(Debug, PartialEq, Eq)]
pub struct Engine<C: Count> {
    /// Counts of every subset of size two through four, by colex rank.
    tables: [Vec<C>; 3],
    /// Counts of every five-ball subset, by colex rank, saturating at [`u8::MAX`].
    quints: Vec<u8>,
    /// The count above [`u8::MAX`] of every saturated quint counter, by rank.
    overflow: collections::HashMap<u32, u64>,
    /// Valid player lines; duplicate tickets remain distinct players.
    valid: u64,
    /// Malformed nonblank player lines omitted from every count.
    invalid: u64,
}

impl<C: Count> Engine<C> {
    /// Load an immutable file on `threads` workers into counters of width `C`, then derive the
    /// smaller tables.
    ///
    /// Returns an engine with file diagnostics available through accessors. Blank lines are
    /// ignored. The program loads through [`load`], which picks the width; this form is for
    /// benchmarks and tests that need one width.
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failures, failure to create the workers, or more than
    /// [`Count::MAX_PLAYERS`] valid players.
    pub fn load(path: impl AsRef<path::Path>, threads: usize) -> eyre::Result<Self> {
        let source = input::Source::open(path.as_ref(), input::CHUNK_BYTES)?;
        on_workers(threads, || {
            Ok(count(&source, BATCH_BYTES / input::CHUNK_BYTES, C::MAX_PLAYERS)?.into_engine())
        })
    }

    /// Build the tables from validated tickets, for tests that have no file.
    #[cfg(test)]
    pub(crate) fn from_tickets(tickets: impl IntoIterator<Item = ticket::Ticket>) -> Self {
        let mut quints = vec![0_u8; combinatorics::SIZES[5]];
        let mut wraps = collections::HashMap::new();
        let mut tally = input::Tally::default();
        for ticket in tickets {
            tally.valid += 1;
            tally.balls = tally.balls.max(usize::from(ticket.numbers()[4]));
            let rank = ticket.quint_rank();
            quints[rank] = quints[rank].wrapping_add(1);
            if quints[rank] == 0 {
                let rank = u32::try_from(rank).expect("quint ranks are below C(90, 5)");
                *wraps.entry(rank).or_default() += 1;
            }
        }
        Counted {
            quints,
            wraps,
            tally,
        }
        .into_engine()
    }

    /// The number of players holding the quint of `rank`.
    ///
    /// The check for a saturated counter costs about 1.6 ns per draw on the benchmark machine;
    /// taking the map lookup out of line cost 1.5 ns more, so it stays inline.
    #[inline]
    fn quint_count(&self, rank: usize) -> u64 {
        let counter = u64::from(self.quints[rank]);
        if counter == u64::from(u8::MAX) {
            let extra = u32::try_from(rank)
                .ok()
                .and_then(|rank| self.overflow.get(&rank));
            counter + extra.copied().unwrap_or(0)
        } else {
            counter
        }
    }

    /// Return exact winner counts ordered by two, three, four, then five matches.
    ///
    /// `pick` must already be validated. Each query performs 26 counter reads, plus one map lookup
    /// when the quint's counter is saturated. Intermediates use signed 64-bit arithmetic because
    /// subtraction precedes cancellation in the inversion; [`Count::MAX_PLAYERS`] keeps every
    /// count and every intermediate within `i64`.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "each inversion yields a count of players, so it is never negative"
    )]
    pub fn query(&self, pick: ticket::Ticket) -> [u64; 4] {
        let ranks = pick.ranks();
        let a2: i64 = ranks
            .pairs
            .into_iter()
            .map(|i| self.tables[0][i].signed())
            .sum();
        let a3: i64 = ranks
            .triples
            .into_iter()
            .map(|i| self.tables[1][i].signed())
            .sum();
        let a4: i64 = ranks
            .quads
            .into_iter()
            .map(|i| self.tables[2][i].signed())
            .sum();
        let a5 = self.quint_count(ranks.quint).cast_signed();
        [
            (a2 - 3 * a3 + 6 * a4 - 10 * a5) as u64,
            (a3 - 4 * a4 + 10 * a5) as u64,
            (a4 - 5 * a5) as u64,
            a5 as u64,
        ]
    }

    /// Return the number of valid player lines, counting identical tickets separately.
    #[must_use]
    pub fn valid_count(&self) -> u64 {
        self.valid
    }

    /// Return the number of rejected nonblank file lines for stderr diagnostics.
    #[must_use]
    pub fn invalid_count(&self) -> u64 {
        self.invalid
    }

    /// The count tables, for independent validation: pairs, triples and quads, then the quint
    /// counters and their overflow.
    #[cfg(test)]
    pub(crate) fn tables(&self) -> ([&[C]; 3], &[u8], &collections::HashMap<u32, u64>) {
        (
            self.tables.each_ref().map(Vec::as_slice),
            &self.quints,
            &self.overflow,
        )
    }
}

/// Regression checks for the module invariants.
#[cfg(test)]
mod tests {
    /// The valid players choose the width: the 32-bit limit itself keeps 32-bit counters, one
    /// player more takes 64-bit ones, and both answer exactly. Debug builds also check that the
    /// 32-bit derivation does not overflow at the limit.
    #[test]
    fn player_count_chooses_the_counter_width() -> eyre::Result<()> {
        let limit = <u32 as super::Count>::MAX_PLAYERS;
        let first = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        let second = crate::ticket::Ticket::new([1, 2, 3, 4, 6])?;

        let at_limit = super::by_width(counted(&[(first, limit - 1), (second, 1)]));
        let over_limit = super::by_width(counted(&[(first, limit), (second, 1)]));

        let super::Loaded::Narrow(narrow) = at_limit else {
            panic!("{limit} players should take 32-bit counters");
        };
        let super::Loaded::Wide(wide) = over_limit else {
            panic!("{} players should take 64-bit counters", limit + 1);
        };
        assert_eq!(narrow.query(first), [0, 0, 1, limit - 1]);
        assert_eq!(wide.query(first), [0, 0, 1, limit]);
        Ok(())
    }

    /// A file of usual size loads into 32-bit counters, the same tables as the build in memory.
    #[test]
    fn load_takes_narrow_counters_for_a_usual_file() -> eyre::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("players.txt");
        std::fs::write(&path, "1 2 3 4 5\n1 2 3 4 6\n")?;
        let tickets = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]].map(crate::ticket::Ticket::new);
        let reference = super::Engine::<u32>::from_tickets(
            tickets.into_iter().collect::<eyre::Result<Vec<_>>>()?,
        );

        let loaded = super::load(&path, 2)?;

        let super::Loaded::Narrow(engine) = loaded else {
            panic!("a two-line file should take 32-bit counters");
        };
        assert!(engine.tables() == reference.tables(), "the tables differ");
        Ok(())
    }

    /// Counting the chunks one batch at a time gives the same tables as counting them all at
    /// once, on a file of several chunks.
    #[test]
    fn batches_count_every_chunk() -> eyre::Result<()> {
        // About four chunks of tickets over 20 balls, with many duplicates.
        let mut state = 0x2545_f491_u64;
        let mut file = Vec::new();
        while file.len() < 1 << 21 {
            let numbers = crate::testing::random_numbers(&mut state, 20);
            let line = numbers.map(|number| number.to_string()).join(" ");
            file.extend_from_slice(line.as_bytes());
            file.push(b'\n');
        }
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("players.txt");
        std::fs::write(&path, &file)?;
        let source = crate::input::Source::open(&path, crate::input::CHUNK_BYTES)?;
        assert!(source.chunks() >= 4);

        let batched =
            super::count(&source, 1, <u32 as super::Count>::MAX_PLAYERS)?.into_engine::<u32>();
        let at_once = super::count(&source, usize::MAX, <u32 as super::Count>::MAX_PLAYERS)?
            .into_engine::<u32>();

        assert_eq!(batched.valid_count(), at_once.valid_count());
        assert_eq!(batched.invalid_count(), 0);
        assert!(batched.tables() == at_once.tables(), "the tables differ");
        Ok(())
    }

    /// Every chunk size counts the same quints and lines and warns about the same invalid lines,
    /// sizes above the program's included: the warnings read those chunks again.
    #[test]
    fn chunk_sizes_count_alike() -> eyre::Result<()> {
        // About three default chunks of tickets over 20 balls; every 1,000th has a sixth number.
        let mut state = 0x9e37_79b9_u64;
        let mut file = Vec::new();
        for index in 1..=120_000 {
            let numbers = crate::testing::random_numbers(&mut state, 20);
            let line = numbers.map(|number| number.to_string()).join(" ");
            file.extend_from_slice(line.as_bytes());
            if index % 1_000 == 0 {
                file.extend_from_slice(b" 6");
            }
            file.push(b'\n');
        }
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("players.txt");
        std::fs::write(&path, &file)?;
        let count = |chunk_bytes: usize| -> eyre::Result<_> {
            let chunk_bytes = std::num::NonZeroUsize::new(chunk_bytes).unwrap();
            let source = crate::input::Source::open(&path, chunk_bytes)?;
            let log = directory.path().join(format!("{chunk_bytes}.log"));
            let subscriber = tracing_subscriber::fmt()
                .without_time()
                .with_writer(std::fs::File::create(&log)?)
                .finish();
            // One worker, the thread that holds the subscriber, so the warnings run where they
            // are enabled.
            let counted = super::on_workers(1, || {
                tracing::subscriber::with_default(subscriber, || {
                    super::count(&source, usize::MAX, <u32 as super::Count>::MAX_PLAYERS)
                })
            })?;
            let log = std::fs::read_to_string(log)?;
            Ok((counted.quints, counted.wraps, counted.tally, log))
        };

        let expected = count(crate::input::CHUNK_BYTES.get())?;

        assert_eq!(expected.2.invalid, 120);
        assert_eq!(expected.3.lines().count(), 120);
        for chunk_bytes in [1_000, crate::input::CHUNK_BYTES.get() * 3 / 2] {
            assert!(
                count(chunk_bytes)? == expected,
                "{chunk_bytes}-byte chunks differ"
            );
        }
        Ok(())
    }

    /// Wraps add up across batches: a repeated ticket wraps its counter in every batch, and
    /// another one wraps only in the last batch.
    #[test]
    fn wraps_accumulate_across_batches() -> eyre::Result<()> {
        // Nearly three chunks of `first` at 10 bytes a line; the third chunk ends with enough of
        // `second` to wrap once.
        const FIRST_PLAYERS: usize = 150_000;
        const SECOND_PLAYERS: usize = 300;
        let first = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        let second = crate::ticket::Ticket::new([1, 2, 3, 4, 6])?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("players.txt");
        let mut file = b"1 2 3 4 5\n".repeat(FIRST_PLAYERS);
        file.extend_from_slice(&b"1 2 3 4 6\n".repeat(SECOND_PLAYERS));
        std::fs::write(&path, &file)?;
        let source = crate::input::Source::open(&path, crate::input::CHUNK_BYTES)?;
        assert!(source.chunks() >= 3);
        let reference = super::Engine::<u32>::from_tickets(
            std::iter::repeat_n(first, FIRST_PLAYERS)
                .chain(std::iter::repeat_n(second, SECOND_PLAYERS)),
        );

        let batched =
            super::count(&source, 1, <u32 as super::Count>::MAX_PLAYERS)?.into_engine::<u32>();

        assert_eq!(
            batched.query(first),
            [0, 0, SECOND_PLAYERS as u64, FIRST_PLAYERS as u64]
        );
        assert!(batched.tables() == reference.tables(), "the tables differ");
        Ok(())
    }

    /// One valid player over the limit is rejected, as soon as its batch is counted; the limit
    /// itself is allowed.
    #[test]
    fn players_over_the_limit_are_rejected() -> eyre::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("players.txt");
        std::fs::write(&path, b"1 2 3 4 5\n".repeat(12))?;
        let source = crate::input::Source::open(&path, crate::input::CHUNK_BYTES)?;

        let at_limit = super::count(&source, 1, 12);
        let over_limit = super::count(&source, 1, 11);

        assert_eq!(at_limit?.tally.valid, 12);
        assert!(over_limit.is_err());
        Ok(())
    }

    /// Grouping keeps every rank, inside its owner's slice and in input order within it.
    #[test]
    fn group_by_owner_is_a_stable_partition() {
        let owner_size = u32::try_from(super::OWNER_SIZE).unwrap();
        let last_rank = u32::try_from(crate::combinatorics::SIZES[5] - 1).unwrap();
        let ranks = [
            last_rank,
            3,
            owner_size,
            2,
            2,
            owner_size - 1,
            5 * owner_size + 7,
            owner_size + 1,
            last_rank - 1,
        ];
        let mut region = [0; 12];

        let bounds = super::group_by_owner(&ranks, &mut region);

        assert_eq!(bounds[super::OWNERS] as usize, ranks.len());
        for owner in 0..super::OWNERS {
            let group = bounds[owner] as usize..bounds[owner + 1] as usize;
            let expected: Vec<_> = ranks
                .iter()
                .copied()
                .filter(|&rank| rank as usize / super::OWNER_SIZE == owner)
                .collect();
            assert_eq!(region[group], expected, "owner {owner}");
        }
    }

    proptest::proptest! {
        // The tests' unoptimized derivation makes 256 cases slow.
        #![proptest_config(proptest::test_runner::Config {
            cases: 64,
            ..crate::testing::proptest_config()
        })]

        /// The tables match a popcount oracle on random populations. Few distinct tickets are drawn
        /// many times; few balls make every match count common.
        #[test]
        fn tables_match_the_popcount_oracle(
            (pool, players, picks) in proptest::strategy::Strategy::prop_flat_map(
                proptest::prop_oneof![15 => 5_u8..=20, 1 => proptest::strategy::Just(90)],
                |balls| {
                    (
                        proptest::collection::vec(crate::testing::ticket(balls), 1..12),
                        proptest::collection::vec(
                            proptest::arbitrary::any::<proptest::sample::Index>(),
                            0..200,
                        ),
                        proptest::collection::vec(crate::testing::ticket(balls), 1..8),
                    )
                },
            )
        ) {
            let players: Vec<_> = players.iter().map(|index| *index.get(&pool)).collect();
            let population: Vec<_> = players.iter().map(|&player| (player, 1)).collect();

            let tables = super::Engine::<u32>::from_tickets(players.iter().copied());

            for pick in picks.into_iter().chain(pool.iter().copied()) {
                proptest::prop_assert_eq!(tables.query(pick), expected_winners(pick, &population));
            }
        }
    }

    /// The quint counters and wraps that counting `population`, tickets with their player counts,
    /// would leave: no test can hold such a file.
    fn counted(population: &[(crate::ticket::Ticket, u64)]) -> super::Counted {
        let mut quints = vec![0_u8; crate::combinatorics::SIZES[5]];
        let mut wraps = std::collections::HashMap::new();
        let mut tally = crate::input::Tally::default();
        for &(ticket, count) in population {
            let rank = ticket.quint_rank();
            quints[rank] = u8::try_from(count % 256).unwrap();
            if count >= 256 {
                wraps.insert(u32::try_from(rank).unwrap(), count / 256);
            }
            tally.balls = tally.balls.max(usize::from(ticket.numbers()[4]));
            tally.valid += count;
        }
        super::Counted {
            quints,
            wraps,
            tally,
        }
    }

    /// The winners of `pick` in `population`, counted ticket by ticket.
    fn expected_winners(
        pick: crate::ticket::Ticket,
        population: &[(crate::ticket::Ticket, u64)],
    ) -> [u64; 4] {
        let mut expected = [0; 4];
        for &(ticket, count) in population {
            let matched = (pick.mask() & ticket.mask()).count_ones();
            if (2..=5).contains(&matched) {
                expected[matched as usize - 2] += count;
            }
        }
        expected
    }

    /// Counts past u32 are exact in every table: one ticket holds more than `u32::MAX` players,
    /// and tickets sharing its numbers add up in the counters they have in common.
    #[test]
    fn counts_above_u32_are_exact() -> eyre::Result<()> {
        let first = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        let second = crate::ticket::Ticket::new([1, 2, 3, 4, 6])?;
        let third = crate::ticket::Ticket::new([1, 2, 3, 7, 8])?;
        let fourth = crate::ticket::Ticket::new([1, 2, 9, 10, 11])?;
        let misses = crate::ticket::Ticket::new([20, 21, 22, 23, 24])?;
        for count in [1 << 32, (1 << 32) + 300, 5_000_000_000] {
            let population = [(first, count), (second, 3_000_000_000), (third, 256)];

            let engine = counted(&population).into_engine::<u64>();

            assert_eq!(engine.quint_count(first.ranks().quint), count);
            for pick in [first, second, third, fourth, misses] {
                assert_eq!(engine.query(pick), expected_winners(pick, &population));
            }
        }
        Ok(())
    }

    /// At the player limit a draw stays exact, in every category and for the draw that drives its
    /// sums highest, the one every player holds. Debug builds also check that no sum overflows.
    #[test]
    fn player_limit_fits_the_draw() -> eyre::Result<()> {
        let limit = <u64 as super::Count>::MAX_PLAYERS;
        let first = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        let second = crate::ticket::Ticket::new([1, 2, 3, 4, 6])?;
        let third = crate::ticket::Ticket::new([1, 2, 3, 7, 8])?;
        let fourth = crate::ticket::Ticket::new([1, 2, 9, 10, 11])?;

        let everyone = counted(&[(first, limit)]).into_engine::<u64>();
        let mixed = counted(&[(first, limit - 300), (second, 300)]).into_engine::<u64>();

        assert_eq!(everyone.query(first), [0, 0, 0, limit]);
        assert_eq!(everyone.query(second), [0, 0, limit, 0]);
        assert_eq!(everyone.query(third), [0, limit, 0, 0]);
        assert_eq!(everyone.query(fourth), [limit, 0, 0, 0]);
        assert_eq!(mixed.query(first), [0, 0, 300, limit - 300]);
        assert_eq!(mixed.query(second), [0, 0, limit - 300, 300]);
        assert_eq!(mixed.query(third), [0, limit, 0, 0]);
        assert_eq!(mixed.query(fourth), [limit, 0, 0, 0]);
        Ok(())
    }

    /// Large mixed duplicates preserve every subset contribution, at both counter widths.
    #[test]
    fn duplicates_are_exact() -> eyre::Result<()> {
        duplicates_are_exact_at::<u32>()?;
        duplicates_are_exact_at::<u64>()
    }

    /// [`duplicates_are_exact`] for counters of width `C`.
    fn duplicates_are_exact_at<C: super::Count>() -> eyre::Result<()> {
        let ticket = crate::ticket::Ticket::new([1, 2, 3, 4, 5])?;
        for count in [15, 16, 255, 256, 70_000, 1_000_000] {
            let engine = super::Engine::<C>::from_tickets(std::iter::repeat_n(ticket, count));
            assert_eq!(engine.query(ticket), [0, 0, 0, count as u64]);
            assert_eq!(engine.quint_count(ticket.ranks().quint), count as u64);
        }
        let population = [
            (ticket, 1_000_000_usize),
            (crate::ticket::Ticket::new([1, 2, 3, 4, 6])?, 70_000),
            (crate::ticket::Ticket::new([1, 2, 3, 7, 8])?, 256),
            (crate::ticket::Ticket::new([1, 2, 9, 10, 11])?, 255),
            (crate::ticket::Ticket::new([86, 87, 88, 89, 90])?, 70_000),
        ];
        let engine = super::Engine::<C>::from_tickets(
            population
                .iter()
                .flat_map(|&(ticket, count)| std::iter::repeat_n(ticket, count)),
        );
        let mut direct: [Vec<i64>; 3] =
            std::array::from_fn(|i| vec![0; crate::combinatorics::SIZES[i + 2]]);
        for &(ticket, count) in &population {
            let ranks = ticket.ranks();
            let count = i64::try_from(count)?;
            for rank in ranks.pairs {
                direct[0][rank] += count;
            }
            for rank in ranks.triples {
                direct[1][rank] += count;
            }
            for rank in ranks.quads {
                direct[2][rank] += count;
            }
        }
        for (actual, expected) in engine.tables().0.iter().zip(&direct) {
            assert!(
                actual
                    .iter()
                    .map(|&count| count.signed())
                    .eq(expected.iter().copied())
            );
        }
        let population = population.map(|(ticket, count)| (ticket, count as u64));
        for &(pick, _) in &population {
            assert_eq!(engine.query(pick), expected_winners(pick, &population));
        }
        Ok(())
    }

    /// Duplicate-heavy narrow-range draws exercise every term of the inversion against popcount,
    /// at both counter widths.
    #[test]
    fn differential_counts() -> eyre::Result<()> {
        differential_counts_at::<u32>()?;
        differential_counts_at::<u64>()
    }

    /// [`differential_counts`] for counters of width `C`.
    fn differential_counts_at<C: super::Count>() -> eyre::Result<()> {
        let tickets: Vec<_> = crate::testing::draws(12)?
            .into_iter()
            .filter(|ticket| ticket.numbers().iter().sum::<u8>() % 3 == 0)
            .flat_map(|ticket| [ticket, ticket])
            .collect();
        let engine = super::Engine::<C>::from_tickets(tickets.iter().copied());
        let population: Vec<_> = tickets.iter().map(|&ticket| (ticket, 1)).collect();
        for &pick in &tickets {
            assert_eq!(engine.query(pick), expected_winners(pick, &population));
        }
        assert_eq!(engine.valid_count(), tickets.len() as u64);
        assert_eq!(engine.invalid_count(), 0);
        assert!(engine.tables().0.iter().all(|table| {
            table
                .iter()
                .all(|&value| value.signed() <= tickets.len().try_into().unwrap())
        }));
        Ok(())
    }
}
