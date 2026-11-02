//! Shared test support: the property-test configuration, ticket and line generators, the
//! line-by-line reference that every parse path is compared with, and a reader of short chunks.

use std::io;

use proptest::strategy::Strategy;

/// Property tests run a fixed number of cases from a fixed seed, so every run checks the same
/// inputs, and they never write failure files into the source tree.
pub(crate) fn proptest_config() -> proptest::test_runner::Config {
    proptest::test_runner::Config {
        cases: 256,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x1077_0000),
        failure_persistence: None,
        ..proptest::test_runner::Config::default()
    }
}

/// Any valid ticket over the numbers `1..=balls`, of which there must be at least five.
pub(crate) fn ticket(balls: u8) -> impl Strategy<Value = crate::ticket::Ticket> {
    proptest::sample::subsequence((1..=balls).collect::<Vec<_>>(), 5).prop_map(|numbers| {
        let numbers: [u8; 5] = numbers
            .try_into()
            .expect("the subsequence has five numbers");
        crate::ticket::Ticket::new(numbers).expect("the numbers are distinct and in range")
    })
}

/// Every five-number draw over `1..=balls`, in lexicographic order, enumerated without the
/// production rank functions.
///
/// `balls` is a small test domain in 5..=12.
pub(crate) fn draws(balls: u8) -> eyre::Result<Vec<crate::ticket::Ticket>> {
    let mut draws = Vec::new();
    for a in 1..balls {
        for b in a + 1..balls {
            for c in b + 1..balls {
                for d in c + 1..balls {
                    for e in d + 1..=balls {
                        draws.push(crate::ticket::Ticket::new([a, b, c, d, e])?);
                    }
                }
            }
        }
    }
    Ok(draws)
}

/// Advance `state` one `SplitMix64` step and return its output: a seeded, deterministic stream
/// for tests that need more random values than a property test's cases.
pub(crate) fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Five distinct numbers in `1..=balls` from [`split_mix`] steps of `state`, in the order drawn,
/// so a ticket built from them still has to sort them. `balls` is at least 5.
pub(crate) fn random_numbers(state: &mut u64, balls: u8) -> [u8; 5] {
    let mut numbers = [0; 5];
    let mut index = 0;
    while index < numbers.len() {
        let number = u8::try_from(split_mix(state) % u64::from(balls))
            .expect("the remainder is below `balls`")
            + 1;
        if !numbers[..index].contains(&number) {
            numbers[index] = number;
            index += 1;
        }
    }
    numbers
}

/// One number token: mostly a plain 1..=90, sometimes zero, too large, signed or zero-padded.
fn token() -> impl Strategy<Value = Vec<u8>> {
    let value =
        proptest::prop_oneof![8 => 1_u16..=90, 1 => proptest::strategy::Just(0), 1 => 91_u16..=999];
    let zeros = proptest::prop_oneof![9 => proptest::strategy::Just(0_usize), 1 => 1_usize..=3];
    (value, zeros, proptest::bool::weighted(0.05)).prop_map(|(value, zeros, plus)| {
        let sign = if plus { "+" } else { "" };
        format!("{sign}{}{value}", "0".repeat(zeros)).into_bytes()
    })
}

/// Whitespace between tokens: mostly one space.
fn separator() -> impl Strategy<Value = &'static [u8]> {
    proptest::prop_oneof![
        8 => proptest::strategy::Just(&b" "[..]),
        1 => proptest::strategy::Just(&b"  "[..]),
        1 => proptest::strategy::Just(&b"\t"[..]),
    ]
}

/// A line close to the canonical format, without its newline: usually five tokens, sometimes
/// another count, leading whitespace, a CR, or one byte replaced by any byte but a newline.
fn near_canonical_line() -> impl Strategy<Value = Vec<u8>> {
    let count = proptest::prop_oneof![8 => proptest::strategy::Just(5_usize), 1 => 0_usize..=7];
    count
        .prop_flat_map(|count| {
            (
                proptest::collection::vec((token(), separator()), count),
                proptest::bool::weighted(0.05),
                proptest::bool::weighted(0.1),
                proptest::option::weighted(
                    0.05,
                    (
                        proptest::arbitrary::any::<proptest::sample::Index>(),
                        0_u8..=254,
                    ),
                ),
            )
        })
        .prop_map(|(tokens, leading, carriage_return, replacement)| {
            let mut line = Vec::new();
            if leading {
                line.push(b' ');
            }
            for (index, (token, separator)) in tokens.iter().enumerate() {
                if index > 0 {
                    line.extend_from_slice(separator);
                }
                line.extend_from_slice(token);
            }
            if carriage_return {
                line.push(b'\r');
            }
            if let Some((position, byte)) = replacement
                && !line.is_empty()
            {
                // 254 values plus a skip over the newline keep every byte but `\n` reachable.
                let byte = if byte >= b'\n' { byte + 1 } else { byte };
                let position = position.index(line.len());
                line[position] = byte;
            }
            line
        })
}

/// Any line without its newline: mostly near-canonical, sometimes arbitrary bytes.
pub(crate) fn any_line() -> impl Strategy<Value = Vec<u8>> {
    let arbitrary = proptest::collection::vec(
        proptest::num::u8::ANY.prop_filter("a line holds no newline", |&byte| byte != b'\n'),
        0..32,
    );
    proptest::prop_oneof![4 => near_canonical_line(), 1 => arbitrary]
}

/// A line in the block kernel's format: five tokens of one or two digits joined by single spaces.
///
/// Values are mostly 1..=90 and sometimes 0, 91..=99 or zero-padded, which the format admits
/// and the later validation rejects or reads as the plain number; repeated numbers occur too.
#[cfg(target_arch = "x86_64")]
fn kernel_line() -> impl Strategy<Value = Vec<u8>> {
    let token = proptest::prop_oneof![
        16 => (1_u8..=90).prop_map(|value| value.to_string()),
        1 => (0_u8..=9).prop_map(|value| format!("0{value}")),
        1 => (91_u8..=99).prop_map(|value| value.to_string()),
        1 => proptest::strategy::Just("0".to_owned()),
    ];
    proptest::collection::vec(token, 5).prop_map(|tokens| tokens.join(" ").into_bytes())
}

/// Lines for the block kernel: mostly in its format, so that most 64-byte blocks take the
/// vector path, and sometimes any line, which stops the kernel and makes it resume after it.
#[cfg(target_arch = "x86_64")]
pub(crate) fn kernel_run_line() -> impl Strategy<Value = Vec<u8>> {
    proptest::prop_oneof![19 => kernel_line(), 1 => any_line()]
}

/// Bytes close to the block kernel's format: canonical lines with bytes replaced, inserted or
/// removed, from the digits, the separators, their ASCII neighbours and other whitespace.
///
/// Unlike [`any_line`], an edit can also add or remove a newline or the final one, so lines merge,
/// split and shift across block boundaries.
#[cfg(target_arch = "x86_64")]
pub(crate) fn adversarial_bytes() -> impl Strategy<Value = Vec<u8>> {
    let byte = proptest::sample::select(b"0123456789  \n\n\r\t\x0b\x0c+-/:x\0\x7f\xff".to_vec());
    let edit = (
        proptest::arbitrary::any::<proptest::sample::Index>(),
        byte,
        0_u8..3,
    );
    (
        proptest::collection::vec(kernel_line(), 0..24),
        proptest::collection::vec(edit, 0..8),
        proptest::bool::ANY,
    )
        .prop_map(|(lines, edits, terminated)| {
            let mut data = join(&lines, terminated);
            for (position, byte, kind) in edits {
                if data.is_empty() {
                    break;
                }
                let position = position.index(data.len());
                match kind {
                    0 => data[position] = byte,
                    1 => data.insert(position, byte),
                    _ => {
                        data.remove(position);
                    }
                }
            }
            data
        })
}

/// Join `lines` with newlines, ending with one when `terminated` is set.
pub(crate) fn join(lines: &[Vec<u8>], terminated: bool) -> Vec<u8> {
    let mut data = lines.join(&b'\n');
    if terminated && !lines.is_empty() {
        data.push(b'\n');
    }
    data
}

/// The lines of `data` split at `\n`, without it; a final newline ends the last line rather than
/// starting an empty one.
pub(crate) fn pieces(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    data.split_inclusive(|&byte| byte == b'\n')
        .map(|piece| piece.strip_suffix(b"\n").unwrap_or(piece))
}

/// What a parse counts: the quint ranks, sorted, and every line count.
pub(crate) type Counted = (Vec<u32>, crate::input::Tally);

/// The counts of `counts` in a form independent of the order of the ranks, which the vector paths
/// stage in groups.
pub(crate) fn summary(mut counts: crate::input::RankCounts) -> Counted {
    counts.ranks.sort_unstable();
    (counts.ranks, counts.tally)
}

/// What `data` counts when each of its [`pieces`] is parsed on its own: the reference for every
/// parse path.
pub(crate) fn reference(data: &[u8]) -> Counted {
    let mut counts = crate::input::RankCounts::default();
    for line in pieces(data) {
        counts.observe(crate::ticket::parse_line(line));
    }
    summary(counts)
}

/// A reader that returns at most `chunk` bytes per read, so that a stream reader must carry
/// partial lines from one read into the next.
pub(crate) struct Chunked<'a> {
    /// The bytes not read yet.
    data: &'a [u8],
    /// The most bytes one read returns; at least 1, so that a read of a nonempty buffer before
    /// the end makes progress.
    chunk: usize,
}

impl<'a> Chunked<'a> {
    /// Read `data` in pieces of at most `chunk` bytes, which must be at least 1.
    pub(crate) fn new(data: &'a [u8], chunk: usize) -> Self {
        assert!(chunk > 0, "a chunk must hold at least one byte");
        Self { data, chunk }
    }
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
