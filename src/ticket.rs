//! A validated ticket makes out-of-range table indices unrepresentable.

use eyre::OptionExt;

use crate::combinatorics;

/// Five sorted, distinct, zero-based balls; construction always validates the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket([u8; 5]);

/// Distinguish harmless whitespace from malformed tickets without allocating an error per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// A validated ticket ready for counting.
    Ticket(Ticket),
    /// No tokens: ignored by both file loading and the interactive protocol.
    Blank,
    /// Tokens are malformed, repeated, out of range, or not exactly five in number.
    Invalid,
}

impl Line {
    /// Classify five numbers as written: a ticket if they validate, otherwise invalid.
    #[inline]
    #[must_use]
    pub(crate) fn from_numbers(numbers: [u8; 5]) -> Self {
        Ticket::validated(numbers).map_or(Self::Invalid, Self::Ticket)
    }
}

/// Dense indices for the 26 subset counters touched by one draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Ranks {
    /// Ten two-ball indices, each below 4,005.
    pub(crate) pairs: [usize; 10],
    /// Ten three-ball indices, each below 117,480.
    pub(crate) triples: [usize; 10],
    /// Five four-ball indices, each below 2,555,190.
    pub(crate) quads: [usize; 5],
    /// The unique five-ball index, below 43,949,268.
    pub(crate) quint: usize,
}

/// Comparators of an optimal sorting network for five values: after exchanging each pair into
/// order, in this order, the values are sorted.
pub(crate) const SORTING_NETWORK: [(usize, usize); 9] = [
    (0, 1),
    (3, 4),
    (2, 4),
    (2, 3),
    (1, 4),
    (0, 3),
    (0, 2),
    (1, 3),
    (1, 2),
];

impl Ticket {
    /// Validate one-based `numbers` and sort them using a fixed nine-comparator network.
    ///
    /// # Errors
    ///
    /// Returns an error for repeated balls or numbers outside 1..=90.
    pub fn new(numbers: [u8; 5]) -> eyre::Result<Self> {
        Self::validated(numbers).ok_or_eyre("expected five distinct integers in 1..=90")
    }

    /// Keep malformed-row handling allocation-free while enforcing all ticket invariants.
    #[inline]
    pub(crate) fn validated(mut numbers: [u8; 5]) -> Option<Self> {
        if numbers.iter().any(|n| !(1..=90).contains(n)) {
            return None;
        }
        for (left, right) in SORTING_NETWORK {
            let a = numbers[left];
            let b = numbers[right];
            numbers[left] = a.min(b);
            numbers[right] = a.max(b);
        }
        if numbers.windows(2).any(|pair| pair[0] == pair[1]) {
            return None;
        }
        Some(Self(numbers.map(|n| n - 1)))
    }

    /// Return sorted one-based numbers for display or transport; no allocation is required.
    #[must_use]
    pub fn numbers(self) -> [u8; 5] {
        self.0.map(|n| n + 1)
    }

    /// Encode a ticket as a bit set, for the tests' independent AND/popcount oracle.
    #[cfg(test)]
    pub(crate) fn mask(self) -> u128 {
        self.0.iter().fold(0, |mask, &n| mask | (1_u128 << n))
    }

    /// Compute all subset indices with precomputed binomial terms and no heap allocation.
    ///
    /// Returns indices whose bounds follow from ticket validation, in no semantic order.
    ///
    /// Always inlined: with a copy of the loaders and the draw loop for each counter width, LLVM
    /// stopped inlining it into the draw, which returned the 26 indices through memory and made
    /// `query` 22.3 to 39.6 ns on the benchmark machine.
    #[inline(always)]
    #[expect(
        clippy::many_single_char_names,
        reason = "a to e are the sorted numbers, named as in the rank formula"
    )]
    pub(crate) fn ranks(self) -> Ranks {
        let quint = self.quint_rank();
        let [a, b, c, d, e] = self.0.map(usize::from);
        let table = &combinatorics::BINOMIAL;
        let [b2, c2, d2, e2] = [table[b][2], table[c][2], table[d][2], table[e][2]];
        let [c3, d3, e3] = [table[c][3], table[d][3], table[e][3]];
        let [d4, e4] = [table[d][4], table[e][4]];
        Ranks {
            pairs: [
                a + b2,
                a + c2,
                b + c2,
                a + d2,
                b + d2,
                c + d2,
                a + e2,
                b + e2,
                c + e2,
                d + e2,
            ],
            triples: [
                a + b2 + c3,
                a + b2 + d3,
                a + c2 + d3,
                b + c2 + d3,
                a + b2 + e3,
                a + c2 + e3,
                b + c2 + e3,
                a + d2 + e3,
                b + d2 + e3,
                c + d2 + e3,
            ],
            quads: [
                a + b2 + c3 + d4,
                a + b2 + c3 + e4,
                a + b2 + d3 + e4,
                a + c2 + d3 + e4,
                b + c2 + d3 + e4,
            ],
            quint,
        }
    }

    /// Compute only the five-ball index, the one value the loading phase needs per player.
    ///
    /// Loading calls this once per file line; going through [`Ticket::ranks`] instead was
    /// measured 10 ns slower per line because the 26-index function is not inlined there.
    #[inline]
    #[expect(
        clippy::many_single_char_names,
        reason = "a to e are the sorted numbers, named as in the rank formula"
    )]
    pub(crate) fn quint_rank(self) -> usize {
        let [a, b, c, d, e] = self.0.map(usize::from);
        let table = &combinatorics::BINOMIAL;
        a + table[b][2] + table[c][3] + table[d][4] + table[e][5]
    }
}

/// Most bytes in a line, without its newline. Longer lines are invalid and are never kept whole,
/// so the memory for a line is bounded whatever the input holds. A valid line needs at most 15
/// bytes; the limit leaves room for padding, leading zeros and signs.
pub const MAX_LINE_BYTES: usize = 4096;

/// Parse one line, without its newline: five numbers separated by ASCII whitespace, each digits
/// with an optional plus sign and leading zeros, in 1..=90 and all distinct.
///
/// Returns `Blank` for whitespace only and `Invalid` for anything else that is not a ticket,
/// including a line longer than [`MAX_LINE_BYTES`]. Invalid bytes and numbers of any length are
/// rejected without panicking. A line in the file's usual format takes the fast path of
/// [`parse_canonical`].
#[must_use]
pub fn parse_line(line: &[u8]) -> Line {
    match parse_canonical(line) {
        Some((ticket, used)) if used == line.len() => Line::Ticket(ticket),
        _ => parse_tokens(line),
    }
}

/// Parse one line, without its newline, token by token; see [`parse_line`] for the grammar.
fn parse_tokens(line: &[u8]) -> Line {
    if line.len() > MAX_LINE_BYTES {
        return Line::Invalid;
    }
    let mut numbers = [0; 5];
    let mut count = 0;
    for token in line
        .split(u8::is_ascii_whitespace)
        .filter(|token| !token.is_empty())
    {
        let Some(number) = parse_number(token).filter(|_| count < numbers.len()) else {
            return Line::Invalid;
        };
        numbers[count] = number;
        count += 1;
    }
    match count {
        0 => Line::Blank,
        5 => Line::from_numbers(numbers),
        _ => Line::Invalid,
    }
}

/// Parse a token as a number in 1..=90: an optional plus sign, then digits, leading zeros allowed.
fn parse_number(token: &[u8]) -> Option<u8> {
    let digits = token.strip_prefix(b"+").unwrap_or(token);
    if digits.is_empty() {
        return None;
    }
    let mut value = 0_u8;
    for &digit in digits {
        if !digit.is_ascii_digit() {
            return None;
        }
        // Past 255 the token is out of range anyway.
        value = value.checked_mul(10)?.checked_add(digit - b'0')?;
    }
    (1..=90).contains(&value).then_some(value)
}

/// Parse the first line of `bytes` if it is in the file's usual format: five numbers of one or two
/// digits, separated by single spaces, ending with a newline or with the bytes. Returns the ticket
/// and the bytes used, newline included.
///
/// Anything else, including an invalid ticket in that format, returns `None` and is left to the
/// general tokenizer. This format is nearly every line of a real file; the fast path saves about
/// 15 ms of READY on the benchmark machine.
#[inline]
#[must_use]
pub fn parse_canonical(bytes: &[u8]) -> Option<(Ticket, usize)> {
    let mut numbers = [0; 5];
    let mut offset = 0;
    for (index, number) in numbers.iter_mut().enumerate() {
        let first = bytes.get(offset)?.wrapping_sub(b'0');
        if first > 9 {
            return None;
        }
        offset += 1;
        let second = bytes.get(offset).map_or(10, |byte| byte.wrapping_sub(b'0'));
        if second <= 9 {
            *number = first * 10 + second;
            offset += 1;
        } else {
            *number = first;
        }
        match bytes.get(offset) {
            Some(b' ') if index < 4 => offset += 1,
            Some(b'\n') if index == 4 => offset += 1,
            None if index == 4 => {}
            _ => return None,
        }
    }
    Some((Ticket::validated(numbers)?, offset))
}

/// Regression checks for the module invariants.
#[cfg(test)]
mod tests {
    /// Every width of number takes the fast path, which stops after the first line; lines of any
    /// other format fall through to the full parser.
    #[test]
    fn canonical_accepts_its_format_only() {
        let canonical = super::parse_canonical;
        let parsed = |numbers, used| Some((super::Ticket::new(numbers).unwrap(), used));

        assert_eq!(canonical(b"1 2 3 4 5\n"), parsed([1, 2, 3, 4, 5], 10));
        assert_eq!(canonical(b"1 2 3 4 5"), parsed([1, 2, 3, 4, 5], 9));
        assert_eq!(
            canonical(b"90 9 10 1 45\n8 8"),
            parsed([90, 9, 10, 1, 45], 13)
        );
        assert_eq!(canonical(b"90 9 10 1 45"), parsed([90, 9, 10, 1, 45], 12));
        for line in [
            &b"1 2 3 4 5\r\n"[..],
            b" 1 2 3 4 5\n",
            b"1  2 3 4 5\n",
            b"1 2 3 4 5 6\n",
            b"1 2 3 4 91\n",
            b"1 2 3 4 0\n",
            b"1 1 2 3 4\n",
            b"001 2 3 4 5\n",
            b"1 2 3 4 5:\n",
            b"+1 2 3 4 5\n",
            b": 2 3 4 5\n",
            b"/ 2 3 4 5\n",
            b"1/ 2 3 4 5\n",
        ] {
            assert_eq!(canonical(line), None, "{:?}", String::from_utf8_lossy(line));
        }
    }

    /// Cover malformed file rows and deviations that must take the robust path.
    #[test]
    fn parser_edges() {
        for line in [
            "1 1 3 4 5",
            "5 4 3 2 5",
            "1 2 3 4 5 6",
            "1 2 3 4",
            "1 2 3 4 93",
            "1 2 3 4 -7",
            "1 2 3 4 100",
            "0 2 3 4 5",
            "1 2 3 4 +",
            "1 2 3 4 99999999999999999",
        ] {
            assert_eq!(
                super::parse_line(line.as_bytes()),
                super::Line::Invalid,
                "{line}"
            );
        }
        let expected =
            super::Line::Ticket(super::Ticket::new([1, 2, 3, 4, 5]).expect("valid fixture"));
        for line in [
            "1 2 3 4 5",
            "5 4 3 2 1\r\n",
            "  1\t2  3 4 5  ",
            "+01 002 3 4 5",
        ] {
            assert_eq!(super::parse_line(line.as_bytes()), expected);
        }
        assert_eq!(super::parse_line(b"\xff"), super::Line::Invalid);
        assert_eq!(super::parse_line(b" \t\r\n"), super::Line::Blank);
    }

    /// Exhaust all five-ball permutations to guard the fixed sorting network.
    #[test]
    fn sorting_network() {
        for a in 1..=5 {
            for b in 1..=5 {
                for c in 1..=5 {
                    for d in 1..=5 {
                        for e in 1..=5 {
                            if let Ok(ticket) = super::Ticket::new([a, b, c, d, e]) {
                                assert_eq!(ticket.numbers(), [1, 2, 3, 4, 5]);
                            }
                        }
                    }
                }
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(crate::testing::proptest_config())]

        /// The fast path accepts a line only as the general tokenizer reads it, and uses exactly
        /// that line, whether a newline or the end of the bytes follows.
        #[test]
        fn canonical_agrees_with_the_tokenizer(line in crate::testing::any_line()) {
            let mut bytes = line.clone();
            bytes.extend_from_slice(b"\n1 2");

            let terminated = super::parse_canonical(&bytes);
            let unterminated = super::parse_canonical(&line);

            proptest::prop_assert_eq!(
                terminated.map(|(ticket, used)| (ticket, used - 1)),
                unterminated
            );
            if let Some((ticket, used)) = unterminated {
                proptest::prop_assert_eq!(used, line.len());
                proptest::prop_assert_eq!(super::parse_tokens(&line), super::Line::Ticket(ticket));
            }
        }

        /// No line shorter than `1 2 3 4 5` is a ticket, so a chunk's length bounds its tickets by
        /// [`crate::input::Source::chunk_tickets`].
        #[test]
        fn short_lines_are_never_tickets(
            line in proptest::collection::vec(
                proptest::sample::select(b"0123456789 \t,;\r-+".to_vec()),
                0..crate::input::MIN_TICKET_BYTES - 1,
            )
        ) {
            proptest::prop_assert!(!matches!(super::parse_line(&line), super::Line::Ticket(_)));
        }

        /// Each of the 26 hand-written indices is the checked rank of a subset of the ticket.
        #[test]
        fn ranks_match_the_checked_rank(ticket in crate::testing::ticket(90)) {
            let balls = ticket.numbers().map(|number| number - 1);
            let expected = |size: u32| {
                let mut ranks: Vec<_> = (0_u32..1 << balls.len())
                    .filter(|subset| subset.count_ones() == size)
                    .map(|subset| {
                        let members: Vec<_> = (0..balls.len())
                            .filter(|&index| subset >> index & 1 == 1)
                            .map(|index| balls[index])
                            .collect();
                        crate::combinatorics::rank(&members).unwrap()
                    })
                    .collect();
                ranks.sort_unstable();
                ranks
            };
            let sorted = |ranks: &[usize]| {
                let mut ranks = ranks.to_vec();
                ranks.sort_unstable();
                ranks
            };

            let ranks = ticket.ranks();

            proptest::prop_assert_eq!(sorted(&ranks.pairs), expected(2));
            proptest::prop_assert_eq!(sorted(&ranks.triples), expected(3));
            proptest::prop_assert_eq!(sorted(&ranks.quads), expected(4));
            proptest::prop_assert_eq!(vec![ranks.quint], expected(5));
            proptest::prop_assert_eq!(ticket.quint_rank(), ranks.quint);
        }
    }
}
