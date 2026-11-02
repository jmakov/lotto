//! Vectorized fast path for the file's usual lines.
//!
//! Nearly every line of a real player file is five numbers of one or two digits, separated by
//! single spaces. The scalar parser spends about 100 instructions per such line on the digits,
//! the sorting network and the rank, with a data-dependent branch per number. Here SSE2 decodes
//! one line per 16-byte load, without branches on the data, into five numbers, and SSE4.1
//! validates, sorts and ranks four tickets at a time in 32-bit lanes. Each binomial of the rank
//! is computed from the previous one with one multiplication and one exact division, so no
//! table is gathered. Anything the fast path cannot take, a line in another format, a line
//! shorter than the 16 bytes it loads, or a ticket that fails validation, goes through
//! [`crate::ticket::parse_line`] and [`crate::ticket::Ticket::validated`], which define the
//! results; the tests check that both paths agree on every input.
//!
//! The module is compiled only with SSE4.1 enabled, which the native build does on a CPU that has
//! it; the portable build and other architectures use the scalar path alone. The three `unsafe`
//! blocks outside the tests are the 16-byte load and store, which take raw pointers, and the one
//! call into the target-feature code.

use std::arch::x86_64::{
    __m128i, _mm_add_epi8, _mm_add_epi32, _mm_and_si128, _mm_cmpeq_epi8, _mm_cmpeq_epi32,
    _mm_cvtepu8_epi32, _mm_cvtsi32_si128, _mm_extract_epi32, _mm_loadu_si128, _mm_max_epu32,
    _mm_min_epu8, _mm_min_epu32, _mm_movemask_epi8, _mm_mullo_epi32, _mm_or_si128, _mm_set1_epi8,
    _mm_set1_epi32, _mm_slli_epi16, _mm_srli_epi32, _mm_srli_si128, _mm_storeu_si128, _mm_sub_epi8,
    _mm_sub_epi32,
};

use crate::{input, simd, ticket};

/// Tickets validated, sorted and ranked together: one per 32-bit lane of a vector.
const LANES: usize = 4;

/// Load 16 bytes into a vector.
#[inline]
#[expect(unsafe_code, reason = "the load intrinsic takes a raw pointer")]
#[target_feature(enable = "sse4.1")]
fn load(bytes: &[u8; 16]) -> __m128i {
    // SAFETY: the array holds 16 readable bytes, and an unaligned load has no alignment
    // requirement.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

/// Store a vector as 16 bytes.
#[inline]
#[expect(unsafe_code, reason = "the store intrinsic takes a raw pointer")]
#[target_feature(enable = "sse4.1")]
fn store(vector: __m128i) -> [u8; 16] {
    let mut bytes = [0; 16];
    // SAFETY: the array holds 16 writable bytes, and an unaligned store has no alignment
    // requirement.
    unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), vector) };
    bytes
}

/// The 16 lanes of a byte comparison as a bit mask.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mask(comparison: __m128i) -> u32 {
    _mm_movemask_epi8(comparison).cast_unsigned()
}

/// Decode the first line of `bytes` if it is in the file's usual format, from one 16-byte load.
///
/// Returns the five numbers as written, each at most 99, and the bytes used including the
/// newline. Returns `None` for fewer than 16 bytes, which the scalar path handles, and for any
/// other format. The decode has no branch on the data: every byte becomes a digit value, the
/// two-digit numbers are formed by adding nine times the first digit to the second, and the
/// token starts are read off a bit mask.
#[inline]
#[target_feature(enable = "sse4.1")]
fn decode_line(bytes: &[u8]) -> Option<([u8; 5], usize)> {
    let block = load(bytes.first_chunk()?);
    let digits = _mm_sub_epi8(block, _mm_set1_epi8(b'0'.cast_signed()));
    let is_digit = _mm_cmpeq_epi8(_mm_min_epu8(digits, _mm_set1_epi8(9)), digits);
    let digit = mask(is_digit);
    let newline = mask(_mm_cmpeq_epi8(block, _mm_set1_epi8(b'\n'.cast_signed())));
    let space = mask(_mm_cmpeq_epi8(block, _mm_set1_epi8(b' '.cast_signed())));
    let length = newline.trailing_zeros();
    if length >= 16 {
        return None;
    }
    let line = (1 << length) - 1;
    let digit = digit & line;
    let separators = !digit & line;
    let mut starts = digit & !(digit << 1);
    // Five tokens of one or two digits, single spaces between them, nothing else.
    if digit & 1 == 0
        || separators != space & line
        || separators & (separators << 1) != 0
        || separators.count_ones() != 4
        || starts.count_ones() != 5
        || digit & (digit >> 1) & (digit >> 2) != 0
    {
        return None;
    }
    // Where a digit is followed by a digit, the value is ten times the first plus the second;
    // the digit shifts stay inside their byte because a digit is below 16.
    let next_digits = _mm_srli_si128::<1>(digits);
    let two_digits = _mm_and_si128(is_digit, _mm_srli_si128::<1>(is_digit));
    let eight_times = _mm_and_si128(
        _mm_slli_epi16::<3>(digits),
        _mm_set1_epi8(0xF8_u8.cast_signed()),
    );
    let rest = _mm_add_epi8(_mm_add_epi8(eight_times, digits), next_digits);
    let values = store(_mm_add_epi8(digits, _mm_and_si128(two_digits, rest)));
    let mut numbers = [0; 5];
    for number in &mut numbers {
        *number = values[starts.trailing_zeros() as usize];
        starts &= starts - 1;
    }
    Some((numbers, length as usize + 1))
}

/// C(x, 2) for each lane, from x.
#[inline]
#[target_feature(enable = "sse4.1")]
fn choose_2(x: __m128i) -> __m128i {
    _mm_srli_epi32::<1>(_mm_mullo_epi32(x, _mm_sub_epi32(x, _mm_set1_epi32(1))))
}

/// C(x, 3) for each lane, from C(x, 2): the product with x - 2 is a multiple of 3.
#[inline]
#[target_feature(enable = "sse4.1")]
fn choose_3(x: __m128i, choose_2: __m128i) -> __m128i {
    _mm_mullo_epi32(
        _mm_mullo_epi32(choose_2, _mm_sub_epi32(x, _mm_set1_epi32(2))),
        _mm_set1_epi32(simd::INVERSE_OF_3.cast_signed()),
    )
}

/// C(x, 4) for each lane, from C(x, 3): the product with x - 3 is a multiple of 4.
#[inline]
#[target_feature(enable = "sse4.1")]
fn choose_4(x: __m128i, choose_3: __m128i) -> __m128i {
    _mm_srli_epi32::<2>(_mm_mullo_epi32(
        choose_3,
        _mm_sub_epi32(x, _mm_set1_epi32(3)),
    ))
}

/// C(x, 5) for each lane, from C(x, 4): the product with x - 4 is a multiple of 5.
#[inline]
#[target_feature(enable = "sse4.1")]
fn choose_5(x: __m128i, choose_4: __m128i) -> __m128i {
    _mm_mullo_epi32(
        _mm_mullo_epi32(choose_4, _mm_sub_epi32(x, _mm_set1_epi32(4))),
        _mm_set1_epi32(simd::INVERSE_OF_5.cast_signed()),
    )
}

/// Validate, sort and rank the four tickets of `numbers`, one per lane.
///
/// `numbers[k][lane]` is the k-th number of the lane's ticket as written. Returns the colex
/// quint ranks and the highest number of the four tickets, or `None` if any lane is not a
/// ticket, which the caller then settles lane by lane with the scalar validation.
#[target_feature(enable = "sse4.1")]
#[expect(
    clippy::many_single_char_names,
    reason = "a to e are the sorted balls, named as in the rank formula"
)]
fn rank_lanes(numbers: &[[u8; LANES]; 5]) -> Option<([u32; LANES], u8)> {
    let mut balls = numbers.map(|lane_numbers| {
        let packed = u32::from_le_bytes(lane_numbers).cast_signed();
        _mm_sub_epi32(
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed)),
            _mm_set1_epi32(1),
        )
    });
    // Zero-based balls are at most 89; a zero wrapped to the maximum and fails too.
    let limit = _mm_set1_epi32(89);
    let in_range = balls.iter().fold(_mm_set1_epi32(-1), |all, &ball| {
        _mm_and_si128(all, _mm_cmpeq_epi32(_mm_min_epu32(ball, limit), ball))
    });
    if mask(in_range) != 0xFFFF {
        return None;
    }
    for (left, right) in ticket::SORTING_NETWORK {
        let low = _mm_min_epu32(balls[left], balls[right]);
        let high = _mm_max_epu32(balls[left], balls[right]);
        balls[left] = low;
        balls[right] = high;
    }
    let repeated = balls.windows(2).fold(_mm_set1_epi32(0), |any, pair| {
        _mm_or_si128(any, _mm_cmpeq_epi32(pair[0], pair[1]))
    });
    if mask(repeated) != 0 {
        return None;
    }
    let [a, b, c, d, e] = balls;
    let c2_c = choose_2(c);
    let c2_d = choose_2(d);
    let c3_d = choose_3(d, c2_d);
    let c2_e = choose_2(e);
    let c3_e = choose_3(e, c2_e);
    let c4_e = choose_4(e, c3_e);
    let rank = _mm_add_epi32(
        _mm_add_epi32(a, choose_2(b)),
        _mm_add_epi32(
            _mm_add_epi32(choose_3(c, c2_c), choose_4(d, c3_d)),
            choose_5(e, c4_e),
        ),
    );
    let ranks = [
        _mm_extract_epi32::<0>(rank),
        _mm_extract_epi32::<1>(rank),
        _mm_extract_epi32::<2>(rank),
        _mm_extract_epi32::<3>(rank),
    ]
    .map(i32::cast_unsigned);
    let highest = store(_mm_add_epi32(e, _mm_set1_epi32(1)));
    let highest = highest[0].max(highest[4]).max(highest[8]).max(highest[12]);
    Some((ranks, highest))
}

/// Decoded tickets waiting for a full set of lanes.
#[derive(Default)]
struct Batch {
    /// `numbers[k][lane]` is the k-th number, as written, of the ticket in `lane`.
    numbers: [[u8; LANES]; 5],
    /// Lanes filled so far.
    filled: usize,
}

impl Batch {
    /// Add a decoded line; a full set of lanes is ranked at once.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn push(&mut self, numbers: [u8; 5], counts: &mut input::RankCounts) {
        for (slot, number) in self.numbers.iter_mut().zip(numbers) {
            slot[self.filled] = number;
        }
        self.filled += 1;
        if self.filled == LANES {
            self.flush(counts);
        }
    }

    /// Rank the filled lanes: all at once when they are full and all valid, otherwise one by
    /// one through the scalar validation, which also counts the invalid ones.
    #[target_feature(enable = "sse4.1")]
    fn flush(&mut self, counts: &mut input::RankCounts) {
        if self.filled == LANES
            && let Some((ranks, highest)) = rank_lanes(&self.numbers)
        {
            counts.observe_ranks(&ranks, highest);
        } else {
            for lane in 0..self.filled {
                let numbers = std::array::from_fn(|k| self.numbers[k][lane]);
                counts.observe(ticket::Line::from_numbers(numbers));
            }
        }
        self.filled = 0;
    }
}

/// Parse the lines of `bytes` into `counts`; see [`input::parse_lines`].
///
/// Lines in the usual format are decoded and ranked in lanes; the ranks reach `counts` in
/// groups, not in file order, which the counting does not need. Every other line goes through
/// the scalar parser at once.
pub(crate) fn parse_lines(bytes: &[u8], counts: &mut input::RankCounts) {
    #[expect(
        unsafe_code,
        reason = "calling a target-feature function needs the feature at the call site"
    )]
    // SAFETY: this module is compiled only when SSE4.1 is enabled for the whole build, so the
    // feature is available wherever this runs.
    unsafe {
        parse_lanes(bytes, counts);
    }
}

/// [`parse_lines`] with the target feature on, so that the kernels inline into it.
#[target_feature(enable = "sse4.1")]
fn parse_lanes(mut bytes: &[u8], counts: &mut input::RankCounts) {
    let mut batch = Batch::default();
    while !bytes.is_empty() {
        if let Some((numbers, used)) = decode_line(bytes) {
            batch.push(numbers, counts);
            bytes = &bytes[used..];
            continue;
        }
        let end = memchr::memchr(b'\n', bytes).unwrap_or(bytes.len());
        counts.observe(ticket::parse_line(&bytes[..end]));
        bytes = bytes.get(end + 1..).unwrap_or_default();
    }
    batch.flush(counts);
}

/// Regression checks for the module invariants.
#[cfg(test)]
#[expect(unsafe_code, reason = "the kernels are target-feature functions")]
mod tests {
    /// The lane arithmetic reproduces every binomial the rank uses, for every ball.
    #[test]
    fn lane_binomials_match_the_table() {
        for x in (4..crate::combinatorics::BALLS).step_by(4) {
            let balls: [u8; 4] =
                std::array::from_fn(|lane| u8::try_from((x + lane).min(89)).unwrap());
            // One lane per ball, with the other four numbers fixed below it, is a valid ticket
            // whose rank is dominated by the ball's C(x, 5) term; check the whole rank instead.
            let numbers = [[1; 4], [2; 4], [3; 4], [4; 4], balls.map(|ball| ball + 1)];

            // SAFETY: the module only exists in builds with SSE4.1 enabled.
            let (ranks, highest) = unsafe { super::rank_lanes(&numbers) }.expect("valid tickets");

            for (lane, ball) in balls.into_iter().enumerate() {
                let ticket = crate::ticket::Ticket::new([1, 2, 3, 4, ball + 1]).unwrap();
                assert_eq!(ranks[lane] as usize, ticket.quint_rank(), "ball {ball}");
            }
            assert_eq!(highest, balls.into_iter().max().unwrap() + 1);
        }
    }

    /// Invalid lanes are refused as a group: a repeat, a zero, a number past 90.
    #[test]
    fn invalid_lanes_are_refused() {
        for bad in [
            [1, 1, 3, 4, 5],
            [0, 2, 3, 4, 5],
            [1, 2, 3, 4, 91],
            [99, 2, 3, 4, 5],
        ] {
            let mut numbers = [[1; 4], [2; 4], [3; 4], [4; 4], [5; 4]];
            for (k, number) in bad.into_iter().enumerate() {
                numbers[k][2] = number;
            }

            // SAFETY: the module only exists in builds with SSE4.1 enabled.
            assert!(unsafe { super::rank_lanes(&numbers) }.is_none(), "{bad:?}");
        }
    }

    /// The line decoder reads the usual format exactly, and nothing else.
    #[test]
    fn decode_line_reads_the_usual_format_only() {
        let padded = |line: &[u8]| [line, &[b'x'; 16]].concat();
        // SAFETY: the module only exists in builds with SSE4.1 enabled.
        let decode = |bytes: &[u8]| unsafe { super::decode_line(bytes) };

        assert_eq!(decode(&padded(b"1 2 3 4 5\n")), Some(([1, 2, 3, 4, 5], 10)));
        assert_eq!(
            decode(&padded(b"90 9 10 1 45\n8 8")),
            Some(([90, 9, 10, 1, 45], 13))
        );
        assert_eq!(
            decode(&padded(b"05 00 99 7 70\n")),
            Some(([5, 0, 99, 7, 70], 14))
        );
        assert_eq!(decode(b"1 2 3 4 5\n"), None, "too short to load");
        for line in [
            &b"1 2 3 4 5\r\n"[..],
            b" 1 2 3 4 5\n",
            b"1  2 3 4 5\n",
            b"1 2 3 4 5 \n",
            b"1 2 3 4 5 6\n",
            b"1 2 3 4\n",
            b"001 2 3 4 5\n",
            b"1 2 3 4 5:\n",
            b"+1 2 3 4 5\n",
            b"1\t2 3 4 5\n",
            b"1 2 3 4 5",
        ] {
            assert_eq!(
                decode(&padded(line)),
                None,
                "{:?}",
                String::from_utf8_lossy(line)
            );
        }
    }

    proptest::proptest! {
        #![proptest_config(crate::testing::proptest_config())]

        /// The lane path and the scalar path count the same ranks and the same lines, with the
        /// highest ball, for any mix of lines.
        #[test]
        fn lanes_agree_with_the_scalar_parser(
            lines in proptest::collection::vec(crate::testing::any_line(), 0..40),
            terminated in proptest::bool::ANY,
        ) {
            let data = crate::testing::join(&lines, terminated);
            let mut scalar = crate::input::RankCounts::default();
            crate::input::parse_lines_scalar(&data, &mut scalar);
            let mut lanes = crate::input::RankCounts::default();

            super::parse_lines(&data, &mut lanes);

            proptest::prop_assert_eq!(
                crate::testing::summary(lanes),
                crate::testing::summary(scalar)
            );
        }
    }
}
