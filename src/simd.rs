//! AVX-512 fast path of the load: whole 64-byte blocks of player lines at once.
//!
//! [`crate::input::parse_lines`] hands it a chunk's lines on a CPU that has the features; any other
//! CPU takes the `decode` module or the portable parsers. Nearly every line of a well-formed file
//! has the canonical shape `n n n n n\n`: five one- or two-digit numbers separated by single
//! spaces. One pass over 64 bytes handles about five lines:
//!
//! * Byte masks mark digits, spaces and newlines. Any other byte, a separator without digits on
//!   both sides, or a run of three digits marks the block as noncanonical.
//! * The separators, packed in order, must read four spaces and a newline, repeated, so every
//!   line holds exactly five numbers.
//! * A number ends at a digit followed by a separator. Its value is that digit plus ten times
//!   the preceding digit, computed in all 64 lanes at once; `vpcompressb` then packs the values
//!   of the number ends, giving five bytes per ticket.
//!
//! A noncanonical block (CRLF endings, extra spaces, signs, garbage) is reparsed line by line
//! with the portable parsers in [`crate::ticket`], starting at the first line it touches, so
//! counts never depend on which path ran. The first line of the input, an unterminated last
//! line and the lines of its last 65 bytes also go through them: a block reads two bytes before
//! it and one byte after it, which the input does not hold there.
//!
//! [`rank_lines`] then validates, sorts and ranks the decoded numbers 16 lines at a time, instead
//! of running a sorting network and table lookups per ticket.
//!
//! Every x86-64 build compiles this module, and [`Avx512::detect`] asks the CPU at run time, so a
//! build made without these features still takes this path on a CPU that has them. The `unsafe`
//! blocks outside the tests are the 64-byte loads and stores, which take raw pointers, and the one
//! call into the target-feature code.

use std::arch::x86_64;

use crate::{input, ticket};

/// Bytes classified per vector step.
const BLOCK: usize = 64;

/// Input bytes per kernel call; bounds the stack buffer of decoded numbers.
///
/// Each call ends at the last complete line and the next call starts there, so the partial
/// line at a batch end is decoded twice: about 32 of 4,096 bytes, under 1% extra work.
const BATCH: usize = 64 * BLOCK;

/// Decoded numbers per batch: at most one number per two bytes, plus one full vector store.
const NUMBERS: usize = BATCH / 2 + BLOCK;

/// Lines validated, sorted and ranked together by [`rank_lines`], one per 32-bit vector lane.
const GROUP: usize = 16;

/// Most lines in the decoded numbers of a batch, so most ranks of a [`rank_lines`] call.
const RANKS: usize = NUMBERS / 5;

/// The inverse of 3 modulo 2^32: a multiple of 3 times this is the exact quotient.
///
/// The `decode` module's binomials divide by it too; it lives here because this module is
/// compiled on every x86-64 build and `decode` only on those with SSE4.1.
pub(crate) const INVERSE_OF_3: u32 = 0xAAAA_AAAB;
/// The inverse of 5 modulo 2^32; see [`INVERSE_OF_3`].
pub(crate) const INVERSE_OF_5: u32 = 0xCCCC_CCCD;

/// Load 64 bytes into a vector.
#[inline]
#[expect(unsafe_code, reason = "the load intrinsic takes a raw pointer")]
#[target_feature(enable = "avx512f")]
fn load(bytes: &[u8; BLOCK]) -> x86_64::__m512i {
    // SAFETY: the array holds 64 readable bytes, and an unaligned load has no alignment
    // requirement.
    unsafe { x86_64::_mm512_loadu_si512(bytes.as_ptr().cast()) }
}

/// Load the 64 bytes of `data` starting at `at`, which must lie inside it.
#[inline]
#[target_feature(enable = "avx512f")]
fn load_at(data: &[u8], at: usize) -> x86_64::__m512i {
    load(
        data[at..at + BLOCK]
            .try_into()
            .expect("the range is one block long"),
    )
}

/// Store a vector as 64 bytes.
#[inline]
#[expect(unsafe_code, reason = "the store intrinsic takes a raw pointer")]
#[target_feature(enable = "avx512f")]
fn store_bytes(vector: x86_64::__m512i) -> [u8; BLOCK] {
    let mut bytes = [0; BLOCK];
    // SAFETY: the array holds 64 writable bytes, and an unaligned store has no alignment
    // requirement.
    unsafe { x86_64::_mm512_storeu_si512(bytes.as_mut_ptr().cast(), vector) };
    bytes
}

/// Store a vector as sixteen 32-bit lanes.
#[inline]
#[expect(unsafe_code, reason = "the store intrinsic takes a raw pointer")]
#[target_feature(enable = "avx512f")]
fn store_lanes(vector: x86_64::__m512i) -> [u32; GROUP] {
    let mut lanes = [0; GROUP];
    // SAFETY: the array holds 64 writable bytes, and an unaligned store has no alignment
    // requirement.
    unsafe { x86_64::_mm512_storeu_si512(lanes.as_mut_ptr().cast(), vector) };
    lanes
}

/// Byte permutation gathering field `k / 16` of line `k % 16` from 16 five-byte lines, for
/// fields 0 to 3 (`last` false) or field 4 into each 16-byte lane (`last` true).
///
/// Lines bytes 0..64 come from the first table of `vpermt2b` and bytes 64..80 from the
/// second, which holds bytes 16..80; index bit 6 selects the second table.
#[expect(
    clippy::cast_possible_truncation,
    reason = "table indices are below 128"
)]
const fn transpose_index(last: bool) -> [u8; BLOCK] {
    let mut index = [0; BLOCK];
    let mut lane = 0;
    while lane < BLOCK {
        let field = if last { 4 } else { lane / GROUP };
        let source = 5 * (lane % GROUP) + field;
        index[lane] = if source < BLOCK {
            source
        } else {
            BLOCK + source - GROUP
        } as u8;
        lane += 1;
    }
    index
}

/// Per-line totals of a [`rank_lines`] call.
#[derive(Debug)]
struct Ranked {
    /// Lines consumed from the front of the input, a multiple of 16; the caller handles the rest.
    lines: usize,
    /// Valid tickets among them, whose ranks fill the front of the output.
    valid: usize,
    /// Rejected lines among them: a number outside 1..=90 or repeated.
    invalid: u64,
    /// Largest one-based number of a valid ticket, or 0 if none was valid.
    highest: u8,
}

/// One group's numbers rearranged by [`transpose_index`], made zero-based so that 0 wraps to
/// 255: `first` holds fields 0 to 3 in its four 16-byte lanes and `last` field 4 in every lane.
struct Fields {
    /// Fields 0 to 3, zero-based, one field per 16-byte lane and one line per byte.
    first: x86_64::__m512i,
    /// Field 4, zero-based, repeated in every 16-byte lane.
    last: x86_64::__m512i,
    /// Lines whose five numbers all lie in 1..=90.
    in_range: u16,
}

/// Transpose one group of 16 lines into [`Fields`].
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "each lane's 16 line bits are kept"
)]
fn transpose(group: &[[u8; 5]; GROUP]) -> Fields {
    const FIRST: [u8; BLOCK] = transpose_index(false);
    const LAST: [u8; BLOCK] = transpose_index(true);
    let bytes = group.as_flattened();
    let (head, tail) = (load_at(bytes, 0), load_at(bytes, GROUP));
    let one = x86_64::_mm512_set1_epi8(1);
    let balls = x86_64::_mm512_set1_epi8(crate::combinatorics::BALLS as i8);
    let first = x86_64::_mm512_sub_epi8(
        x86_64::_mm512_permutex2var_epi8(head, load(&FIRST), tail),
        one,
    );
    let last = x86_64::_mm512_sub_epi8(
        x86_64::_mm512_permutex2var_epi8(head, load(&LAST), tail),
        one,
    );
    let fields = x86_64::_mm512_cmplt_epu8_mask(first, balls);
    Fields {
        first,
        last,
        in_range: (fields & fields >> 16 & fields >> 32 & fields >> 48) as u16
            & x86_64::_mm512_cmplt_epu8_mask(last, balls) as u16,
    }
}

/// Validate, sort and rank whole groups of 16 decoded lines, writing valid ranks from the front of
/// `ranks`; the lines after the last whole group are left to the caller.
///
/// `lines` holds five numbers per line as written, and `ranks` has room for one rank per line of
/// a whole group; its bytes past the valid ranks are overwritten with no meaning. Ranks come in
/// line order and equal [`ticket::Ticket::quint_rank`] of the validated ticket.
///
/// One step transposes 16 lines to one 32-bit vector per field, checks the range, sorts across
/// lanes with the nine-comparator network of [`ticket::Ticket::validated`], rejects repeats, and
/// computes the colex rank `a + C(b,2) + C(c,3) + C(d,4) + C(e,5)` arithmetically.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[expect(
    clippy::many_single_char_names,
    reason = "a to e are the sorted fields, named as in the rank formula"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the highest number of a valid ticket is at most 90"
)]
fn rank_lines(lines: &[[u8; 5]], ranks: &mut [u32]) -> Ranked {
    let one = x86_64::_mm512_set1_epi32(1);
    let third = x86_64::_mm512_set1_epi32(INVERSE_OF_3.cast_signed());
    let fifth = x86_64::_mm512_set1_epi32(INVERSE_OF_5.cast_signed());
    let minus = |n, k: i32| x86_64::_mm512_sub_epi32(n, x86_64::_mm512_set1_epi32(k));
    let times = |a, b| x86_64::_mm512_mullo_epi32(a, b);
    // C(n,2) = n(n-1)/2; C(n,3) = n(n-1)(n-2)/6; C(n,4) = C(n,3)(n-3)/4; C(n,5) = C(n,4)(n-4)/5.
    // Every product of a valid lane stays below 2^28.
    let choose2 = |n| x86_64::_mm512_srli_epi32::<1>(times(n, minus(n, 1)));
    let choose3 = |n| {
        let product = times(times(n, minus(n, 1)), minus(n, 2));
        times(x86_64::_mm512_srli_epi32::<1>(product), third)
    };
    let choose4 = |n| x86_64::_mm512_srli_epi32::<2>(times(choose3(n), minus(n, 3)));
    let choose5 = |n| times(times(choose4(n), minus(n, 4)), fifth);
    let widen = |bytes: x86_64::__m128i| x86_64::_mm512_cvtepu8_epi32(bytes);
    let mut top = x86_64::_mm512_setzero_si512();
    let mut written = 0;
    let (groups, rest) = lines.as_chunks::<GROUP>();
    for group in groups {
        let Fields {
            first,
            last,
            in_range: mut valid,
        } = transpose(group);
        let mut numbers = [
            widen(x86_64::_mm512_extracti32x4_epi32::<0>(first)),
            widen(x86_64::_mm512_extracti32x4_epi32::<1>(first)),
            widen(x86_64::_mm512_extracti32x4_epi32::<2>(first)),
            widen(x86_64::_mm512_extracti32x4_epi32::<3>(first)),
            widen(x86_64::_mm512_extracti32x4_epi32::<0>(last)),
        ];
        for (left, right) in ticket::SORTING_NETWORK {
            let (a, b) = (numbers[left], numbers[right]);
            numbers[left] = x86_64::_mm512_min_epu32(a, b);
            numbers[right] = x86_64::_mm512_max_epu32(a, b);
        }
        let [a, b, c, d, e] = numbers;
        valid &= !(x86_64::_mm512_cmpeq_epi32_mask(a, b)
            | x86_64::_mm512_cmpeq_epi32_mask(b, c)
            | x86_64::_mm512_cmpeq_epi32_mask(c, d)
            | x86_64::_mm512_cmpeq_epi32_mask(d, e));
        let rank = x86_64::_mm512_add_epi32(
            x86_64::_mm512_add_epi32(a, choose2(b)),
            x86_64::_mm512_add_epi32(choose3(c), x86_64::_mm512_add_epi32(choose4(d), choose5(e))),
        );
        let packed = store_lanes(x86_64::_mm512_maskz_compress_epi32(valid, rank));
        ranks[written..written + GROUP].copy_from_slice(&packed);
        written += valid.count_ones() as usize;
        top = x86_64::_mm512_mask_max_epu32(top, valid, top, x86_64::_mm512_add_epi32(e, one));
    }
    let whole = lines.len() - rest.len();
    Ranked {
        lines: whole,
        valid: written,
        invalid: (whole - written) as u64,
        highest: x86_64::_mm512_reduce_max_epu32(top) as u8,
    }
}

/// Separators of consecutive canonical lines; a window starting at `phase` (0..5) is expected.
const SEPARATORS: [u8; 2 * BLOCK] = {
    let mut pattern = [b' '; 2 * BLOCK];
    let mut index = 4;
    while index < pattern.len() {
        pattern[index] = b'\n';
        index += 5;
    }
    pattern
};

/// Result of one kernel call over whole canonical blocks.
struct Blocks {
    /// Start of the first line not fully decoded; parsing resumes here.
    resume: usize,
    /// Count of decoded numbers belonging to complete lines (a multiple of five).
    numbers: usize,
    /// Start of a noncanonical block, which must be parsed line by line.
    noncanonical: Option<usize>,
}

/// Proof that this CPU has every feature the kernels use; only [`Self::detect`] makes one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx512(());

impl Avx512 {
    /// Ask the CPU for the kernels' features; `None` means this path must not run here.
    ///
    /// The standard library caches the answers after the first call and folds a feature the build
    /// enables to `true`, so one check per chunk costs nothing measurable.
    #[must_use]
    pub(crate) fn detect() -> Option<Self> {
        let available = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("lzcnt");
        available.then_some(Self(()))
    }
}

/// Parse the lines of `bytes` into `counts`; see [`input::parse_lines`].
///
/// Canonical lines are decoded and ranked in blocks and every other line goes through the portable
/// parsers, so the counts equal [`input::parse_lines_scalar`]'s. The ranks reach `counts` in
/// groups, not in file order, which the counting does not need.
pub(crate) fn parse_lines(_cpu: Avx512, bytes: &[u8], counts: &mut input::RankCounts) {
    #[expect(
        unsafe_code,
        reason = "calling a target-feature function needs the feature at the call site"
    )]
    // SAFETY: only `Avx512::detect` makes the proof in `_cpu`, after finding these features.
    unsafe {
        parse_blocks(bytes, counts);
    }
}

/// [`parse_lines`] with the target features on, so that the kernels inline into it.
///
/// Returns how many lines the block kernel decoded, which the tests check: the portable parsers
/// give the same counts, so the counts alone cannot show that the kernel ran.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,popcnt,lzcnt")]
fn parse_blocks(bytes: &[u8], counts: &mut input::RankCounts) -> usize {
    // Complete lines end here. What follows is an unterminated last line, which only the portable
    // parsers read; the kernel sees its first bytes as lookahead.
    let end = memchr::memrchr(b'\n', bytes).map_or(0, |newline| newline + 1);
    let mut decoded = 0;
    // Each kernel call overwrites the prefix it reports, so one buffer serves every batch.
    let mut numbers = [0; NUMBERS];
    // Likewise for the ranks of a batch, which are staged all at once.
    let mut ranks = [0; RANKS];
    // The block kernel resumes only at line starts at or after this offset.
    let mut portable_until = 0;
    let mut position = 0;
    while position < bytes.len() {
        // The kernel reads two bytes before each block to see whether it starts mid-number.
        if position >= 2 && position >= portable_until && position < end {
            let limit = end.min(position + BATCH);
            let blocks = decode_blocks(bytes, position, limit, &mut numbers);
            let lines = numbers[..blocks.numbers].as_chunks::<5>().0;
            stage(lines, &mut ranks, counts);
            decoded += lines.len();
            portable_until = match blocks.noncanonical {
                Some(block) => block + BLOCK,
                // No whole block fits before `end` or the lookahead runs out.
                None if blocks.resume == position => end,
                None => 0,
            };
            position = blocks.resume;
            continue;
        }
        let (line, _, used) = input::first_line(&bytes[position..]);
        counts.observe(line);
        position += used;
    }
    decoded
}

/// Count decoded lines: whole groups through [`rank_lines`], the rest one by one.
///
/// `lines` holds five numbers per line as written in the file, at most [`RANKS`] lines.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn stage(lines: &[[u8; 5]], ranks: &mut [u32; RANKS], counts: &mut input::RankCounts) {
    let ranked = rank_lines(lines, ranks);
    counts.observe_ranks(&ranks[..ranked.valid], ranked.highest);
    counts.tally.invalid += ranked.invalid;
    for &fields in &lines[ranked.lines..] {
        counts.observe(ticket::Line::from_numbers(fields));
    }
}

/// Decode whole canonical blocks of `data[position..limit]`, starting at a line start.
///
/// Requires `position >= 2`. Stops before the first noncanonical block, at `limit`, or when
/// fewer than 65 bytes of `data` remain (each block also inspects the byte after it).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,popcnt,lzcnt")]
fn decode_blocks(
    data: &[u8],
    mut position: usize,
    limit: usize,
    numbers: &mut [u8; NUMBERS],
) -> Blocks {
    let zero = x86_64::_mm512_set1_epi8(b'0'.cast_signed());
    let nine = x86_64::_mm512_set1_epi8(9);
    let space = x86_64::_mm512_set1_epi8(b' '.cast_signed());
    let newline = x86_64::_mm512_set1_epi8(b'\n'.cast_signed());
    let mut result = Blocks {
        resume: position,
        numbers: 0,
        noncanonical: None,
    };
    let mut decoded = 0;
    // Position of the next separator within the four-spaces-and-newline cycle.
    let mut phase = 0;
    while position + BLOCK <= limit && position + BLOCK < data.len() {
        let bytes = load_at(data, position);
        // Subtracting '0' maps digits to 0..=9 and every other byte above 9 (unsigned).
        let values = x86_64::_mm512_sub_epi8(bytes, zero);
        let previous = x86_64::_mm512_sub_epi8(load_at(data, position - 1), zero);
        let digit = x86_64::_mm512_cmple_epu8_mask(values, nine);
        let digit_before = x86_64::_mm512_cmple_epu8_mask(previous, nine);
        let digit_before_previous = x86_64::_mm512_cmple_epu8_mask(
            x86_64::_mm512_sub_epi8(load_at(data, position - 2), zero),
            nine,
        );
        let digit_after = x86_64::_mm512_cmple_epu8_mask(
            x86_64::_mm512_sub_epi8(load_at(data, position + 1), zero),
            nine,
        );
        let spaces = x86_64::_mm512_cmpeq_epi8_mask(bytes, space);
        let newlines = x86_64::_mm512_cmpeq_epi8_mask(bytes, newline);
        let separators = spaces | newlines;
        let malformed = !(digit | separators)
            | (spaces & !(digit_before & digit_after))
            | (newlines & !digit_before)
            | (digit & digit_before & digit_before_previous);
        let count = separators.count_ones();
        let expected = load_at(&SEPARATORS, phase);
        let packed = x86_64::_mm512_maskz_compress_epi8(separators, bytes);
        let matched = x86_64::_mm512_cmpeq_epi8_mask(packed, expected);
        let required = u64::MAX.unbounded_shr(64 - count);
        if malformed != 0 || matched & required != required {
            result.noncanonical = Some(position);
            break;
        }
        // value = units + 10 * tens, as additions: tens * 2 + tens * 8.
        let ends = digit & !digit_after;
        let tens = x86_64::_mm512_maskz_mov_epi8(digit_before, previous);
        let twice = x86_64::_mm512_add_epi8(tens, tens);
        let four_times = x86_64::_mm512_add_epi8(twice, twice);
        let value = x86_64::_mm512_add_epi8(
            x86_64::_mm512_add_epi8(values, twice),
            x86_64::_mm512_add_epi8(four_times, four_times),
        );
        let stored = store_bytes(x86_64::_mm512_maskz_compress_epi8(ends, value));
        numbers[decoded..decoded + BLOCK].copy_from_slice(&stored);
        // Every canonical block contains a newline: a canonical line is at most 15 bytes.
        let last_newline = 63 - newlines.leading_zeros() as usize;
        result.resume = position + last_newline + 1;
        result.numbers = decoded + (ends & (u64::MAX >> (63 - last_newline))).count_ones() as usize;
        decoded += ends.count_ones() as usize;
        phase = (phase + count as usize) % 5;
        position += BLOCK;
    }
    result
}

/// The block path must agree with `parse_line` on every input, which pins the kernel to the
/// portable parser; the rest of the checks show that the kernel really runs.
#[cfg(test)]
#[expect(unsafe_code, reason = "the kernels are target-feature functions")]
mod tests {
    /// Every path a CPU can take must count `data` as [`crate::testing::reference`] does.
    fn agree(data: &[u8]) {
        let summary = crate::testing::summary;
        let expected = crate::testing::reference(data);
        let mut scalar = crate::input::RankCounts::default();
        crate::input::parse_lines_scalar(data, &mut scalar);
        let mut fallback = crate::input::RankCounts::default();
        crate::input::parse_lines_without_avx512(data, &mut fallback);
        // On a CPU with AVX-512 this is the block path.
        let mut dispatched = crate::input::RankCounts::default();
        crate::input::parse_lines(data, &mut dispatched);

        assert_eq!(summary(fallback), expected, "fallback path on {data:?}");
        assert_eq!(summary(dispatched), expected, "dispatch on {data:?}");
        assert_eq!(
            summary(scalar),
            expected,
            "scalar path on {:?}",
            String::from_utf8_lossy(data)
        );
    }

    /// Compare all of `data` from its first and its second line: linear, for inputs too long for
    /// [`compare`].
    fn compare_whole(data: &[u8]) {
        let second = memchr::memchr(b'\n', data).map_or(data.len(), |newline| newline + 1);
        for begin in [0, second] {
            agree(&data[begin..]);
        }
    }

    /// Compare slices of `data` from several starts, ending at each line end and just before it,
    /// so that blocks start and stop at every offset and the last line is also unterminated.
    fn compare(data: &[u8]) {
        for begin in [0, 1, 2, 3, 17] {
            let Some(rest) = data.get(begin..) else {
                continue;
            };
            agree(rest);
            for newline in memchr::memchr_iter(b'\n', rest) {
                agree(&rest[..newline]);
                agree(&rest[..=newline]);
            }
        }
    }

    /// Small deterministic generator; quality only needs to spread shapes and values.
    fn generator(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// A file of `lines` canonical lines with every digit-width shape.
    fn canonical_file(lines: usize) -> Vec<u8> {
        let mut next = generator(0x91c2_6fe0_549a_873b);
        let mut canonical = Vec::new();
        for line in 0..lines {
            let numbers: Vec<String> = (0..5)
                .map(|field| {
                    if line & (1 << field) == 0 {
                        (next() % 10).to_string()
                    } else {
                        (10 + next() % 90).to_string()
                    }
                })
                .collect();
            canonical.extend_from_slice(numbers.join(" ").as_bytes());
            canonical.push(b'\n');
        }
        canonical
    }

    /// `canonical` with `line` and its newline inserted at the first line start from `offset`.
    fn spliced(canonical: &[u8], offset: usize, line: &[u8]) -> Vec<u8> {
        let split = canonical[offset..]
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(canonical.len(), |n| offset + n + 1);
        let mut data = canonical[..split].to_vec();
        data.extend_from_slice(line);
        data.push(b'\n');
        data.extend_from_slice(&canonical[split..split + 700]);
        data
    }

    /// Canonical files with every digit-width shape, then single malformed lines spliced in.
    #[test]
    fn kernel_matches_parse_line() {
        let canonical = canonical_file(400);

        compare(&canonical);
        for malformed in [
            "1 2 3 4 4",
            "1 2 3 4 5 6",
            "1 2 3 4",
            "1 2 3 4 93",
            "0 2 3 4 5",
            "00 2 3 4 5",
            "01 02 03 04 05",
            "123 2 3 4 5",
            "1 2 3 4 5\r",
            " 1  2\t3 4 5",
            "+1 2 3 4 5",
            "",
            " ",
            "\t\r",
        ] {
            for offset in [0, 37, 64, 127, 128, 641] {
                compare(&spliced(&canonical, offset, malformed.as_bytes()));
            }
        }
    }

    /// Lines around the length limit are valid or invalid by their length alone, wherever they
    /// fall among the blocks.
    #[test]
    fn long_lines_match_parse_line() {
        let canonical = canonical_file(100);
        let limit = crate::ticket::MAX_LINE_BYTES;
        for length in [limit - 1, limit, limit + 1, 3 * limit] {
            let spaces = vec![b' '; length - 9];
            let padded = [b"1 2 3 4 5".as_slice(), &spaces].concat();
            let zeros = vec![b'0'; length - 8];
            let zero_padded = [zeros.as_slice(), b"1 2 3 4 5"].concat();
            for line in [padded, zero_padded, vec![b'7'; length]] {
                for offset in [0, 37, 64, 127] {
                    compare_whole(&spliced(&canonical, offset, &line));
                }
            }
        }
    }

    /// CRLF endings make every block noncanonical; the counts stay right.
    #[test]
    fn crlf_files_match_parse_line() {
        let data = String::from_utf8(canonical_file(200))
            .unwrap()
            .replace('\n', "\r\n");

        compare_whole(data.as_bytes());
    }

    /// A mix of canonical, malformed and blank lines, long enough to reach the kernels.
    #[test]
    fn mixed_lines_match_parse_line() {
        let data =
            b"12 34 56 78 90\n1 23 4 56 7\n5 6 5 7 8\n\n1 2 3 4 :\n 1  2 3 4 5\r\n".repeat(40);

        compare_whole(&data);
    }

    /// Canonical lines stay on the block kernel. The portable parsers give the same answers, so
    /// the comparisons here cannot see a mask that rejects too much and loses the kernel's speed.
    #[test]
    fn canonical_lines_stay_on_the_kernel() {
        if super::Avx512::detect().is_none() {
            return;
        }
        // One- and two-digit fields; the kernel needs two bytes before its first block.
        let data = b"12 34 56 78 90\n1 23 4 56 7\n".repeat(200);
        let start = 15;
        let limit = start + super::BATCH;
        let mut numbers = [0; super::NUMBERS];

        // SAFETY: the test returned above on a CPU without the kernel's features.
        let blocks = unsafe { super::decode_blocks(&data, start, limit, &mut numbers) };

        assert_eq!(blocks.noncanonical, None);
        assert!(blocks.resume + 15 > limit, "stopped at {}", blocks.resume);
        assert_eq!(blocks.numbers % 5, 0);
    }

    /// Lines that `parse_lines` leaves to the portable parsers in `data`.
    fn portable_lines(_cpu: super::Avx512, data: &[u8]) -> usize {
        let mut counts = crate::input::RankCounts::default();
        // SAFETY: only `Avx512::detect` makes the proof in `_cpu`, after finding the features.
        let decoded = unsafe { super::parse_blocks(data, &mut counts) };
        memchr::memchr_iter(b'\n', data).count() - decoded
    }

    /// A whole file takes the kernel from its second line to its last 65 bytes, across batches,
    /// which hold at most 8 lines of 10 bytes or more.
    #[test]
    fn canonical_files_take_the_kernel() {
        let Some(cpu) = super::Avx512::detect() else {
            return;
        };
        let data = canonical_file(4000);

        let portable = portable_lines(cpu, &data);

        assert!(portable <= 9, "{portable}");
    }

    /// The kernel comes back after a noncanonical line: besides the usual lines at the two ends
    /// of the file, only those around the block holding it are left out, at most 92 bytes from
    /// the partial line before the block to the partial line after it, so at most 13 lines.
    #[test]
    fn kernel_resumes_after_a_noncanonical_line() {
        let Some(cpu) = super::Avx512::detect() else {
            return;
        };
        let data = spliced(&canonical_file(400), 2000, b"1 2 3 4 5\r");

        let portable = portable_lines(cpu, &data);

        assert!(portable <= 9 + 13, "{portable}");
    }

    /// Vector ranking agrees with per-line validation on random, invalid and boundary lines.
    #[test]
    fn rank_lines_matches_scalar_validation() {
        if super::Avx512::detect().is_none() {
            return;
        }
        let mut next = generator(0x5851_f42d_4c95_7f2d);
        let mut lines: Vec<[u8; 5]> = (0..5000)
            .map(|line| {
                let mut fields = [0; 5];
                for field in &mut fields {
                    *field = match line % 7 {
                        // Every byte value, including 0, 91..=255 and repeats.
                        0 => next().to_le_bytes()[0],
                        // Narrow ranges make repeats common.
                        1 => 1 + u8::try_from(next() % 6).unwrap(),
                        _ => 1 + u8::try_from(next() % 90).unwrap(),
                    };
                }
                fields
            })
            .collect();
        lines.extend([
            [1, 2, 3, 4, 5],
            [90, 89, 88, 87, 86],
            [5, 4, 3, 2, 1],
            [1, 90, 45, 2, 89],
        ]);
        for length in [0, 15, 16, 17, 100, lines.len()] {
            let mut ranks = vec![7; length];
            // SAFETY: the test returned above on a CPU without the kernel's features.
            let ranked = unsafe { super::rank_lines(&lines[..length], &mut ranks) };
            let expected: Vec<_> = lines[..ranked.lines]
                .iter()
                .filter_map(|&fields| crate::ticket::Ticket::validated(fields))
                .collect();
            assert_eq!(ranked.lines, length / 16 * 16);
            assert_eq!(ranked.valid, expected.len());
            assert_eq!(ranked.invalid, (ranked.lines - expected.len()) as u64);
            let top = expected.iter().map(|ticket| ticket.numbers()[4]).max();
            assert_eq!(ranked.highest, top.unwrap_or(0));
            let expected_ranks = expected
                .iter()
                .map(|ticket| u32::try_from(ticket.quint_rank()).unwrap());
            assert!(ranks[..ranked.valid].iter().copied().eq(expected_ranks));
        }
    }

    /// Canonical text with a few random byte edits exercises every rejection mask.
    #[test]
    fn kernel_matches_parse_line_after_random_edits() {
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        let alphabet = b"0123456789 \n\r\t+x";
        let canonical: Vec<u8> = (0..60)
            .map(|line: u64| format!("{} 7 {} 33 {}\n", line % 9 + 1, line + 10, 89 - line))
            .flat_map(String::into_bytes)
            .collect();
        for _ in 0..400 {
            let mut data = canonical.clone();
            for _ in 0..=next() % 3 {
                let at = usize::try_from(next()).unwrap() % data.len();
                data[at] = alphabet[usize::try_from(next()).unwrap() % alphabet.len()];
            }
            data.push(b'\n');
            compare(&data);
        }
    }

    proptest::proptest! {
        #![proptest_config(crate::testing::proptest_config())]

        /// The kernel agrees with `parse_line` on runs of mostly canonical lines, at every line
        /// end, so that vector blocks start and stop at every offset.
        #[test]
        fn kernel_matches_parse_line_on_generated_lines(
            lines in proptest::collection::vec(crate::testing::kernel_run_line(), 0..64),
        ) {
            compare(&crate::testing::join(&lines, true));
        }

        /// The same over several 4,096-byte batches, where a partial line is decoded twice,
        /// compared over the whole input only, with or without a final newline.
        #[test]
        fn kernel_matches_parse_line_across_batches(
            lines in proptest::collection::vec(crate::testing::kernel_run_line(), 300..900),
            terminated in proptest::bool::ANY,
        ) {
            compare_whole(&crate::testing::join(&lines, terminated));
        }

        /// The kernel agrees with `parse_line` on canonical text after arbitrary small damage,
        /// which moves separators and newlines to where the masks must catch them.
        #[test]
        fn kernel_matches_parse_line_on_adversarial_bytes(
            data in crate::testing::adversarial_bytes(),
        ) {
            compare(&data);
        }
    }
}
