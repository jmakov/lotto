//! Derive each count table from the next larger one.
//!
//! A k-subset's count is the sum of the counts of the (k + 1)-subsets that contain it, divided by
//! how many of those each player was counted in. Colex order keeps the sums cheap: the subsets
//! with the same maximum are contiguous, so the work is adding whole slices.
//!
//! Take the k-subsets with maximum `d`. A (k + 1)-subset containing one of them either adds a
//! ball `m` above `d`, or keeps `d` as its maximum and adds a smaller ball. The first kind, for
//! one `m`, is a contiguous run of the (k + 1)-subset table: the run of subsets with maximum `m`,
//! holding the subsets whose remaining k balls end in `d`. It adds onto the output group element
//! by element. The second kind is the same derivation one size down: the (k + 1)-subsets with
//! maximum `d`, each without `d`, form the k-subset table over the balls below `d`, and deriving
//! it gives the (k - 1)-subsets below `d`, which, with `d` put back, are the output group.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{combinatorics, engine};

/// Derive the table of (`size` - 1)-subsets from the table of `size`-subsets.
///
/// The first C(`balls`, `size`) entries of `input` hold the counts; the unused suffix of the
/// output stays zero. A player with `size` - 1 of a subset's numbers is in 6 - `size` of the
/// `size`-subsets on a five-number ticket, so `divisor` is 6 - `size`. Every sum is `divisor`
/// times a count, which the load's player limit keeps within the counter width `C`.
///
/// Each output group, the subsets with one maximum, is a disjoint slice of the table, and so
/// is each of its parts by the maximum of the subsets' other balls, which is how the kept
/// derivation lays the group out. The parts are the parallel tasks: a few thousand of them for
/// the quads, against 87 groups of very unequal size, which kept the largest one on the
/// critical path and cost 1.5 ms of READY on the benchmark machine. Each task adds the runs of the
/// larger balls that end in its ball, restricted to its part, then keeps its ball through
/// [`derive_group`] one size down. The pairs, 4,005 counters, are not split.
pub(crate) fn derive<T, C>(input: &[T], balls: usize, size: usize, divisor: C) -> Vec<C>
where
    T: Copy + Sync,
    C: engine::Count + From<T>,
{
    let choose = &combinatorics::BINOMIAL;
    let mut output = vec![C::default(); combinatorics::SIZES[size - 1]];
    let mut rest = &mut output[..choose[balls][size - 1]];
    let mut tasks = Vec::new();
    for maximum in size - 2..balls {
        let (mut group, tail) =
            rest.split_at_mut(choose[maximum + 1][size - 1] - choose[maximum][size - 1]);
        rest = tail;
        if size == 3 || maximum + 1 < size {
            tasks.push((maximum, None, group));
            continue;
        }
        // The group's parts: the subsets whose other balls have maximum `other`, laid out by
        // the rank of those balls, so part `other` spans C(other, size - 2)..C(other + 1, size - 2).
        for (other, row) in choose.iter().enumerate().take(maximum).skip(size - 3) {
            let (part, tail) = group.split_at_mut(row[size - 3]);
            tasks.push((maximum, Some(other), part));
            group = tail;
        }
    }
    tasks.into_par_iter().for_each(|(maximum, other, sums)| {
        match other {
            None => derive_group(input, balls, size, maximum, sums),
            Some(other) => {
                let base = choose[maximum][size - 1];
                let part = base + choose[other][size - 2]..base + choose[other + 1][size - 2];
                for above in maximum + 1..balls {
                    add(sums, &group(input, above, size)[part.clone()]);
                }
                derive_group(group(input, maximum, size), maximum, size - 1, other, sums);
            }
        }
        // The quads, by far the longest output, divide by 1 and skip the divisions.
        if divisor != C::from(1) {
            for sum in sums {
                *sum /= divisor;
            }
        }
    });
    output
}

/// Add the derivation of `input`, the `size`-subsets over `balls` balls, into `output`, the
/// C(`maximum`, `size` - 2) counters of the (`size` - 1)-subsets with maximum `maximum`.
///
/// This is one output group of [`derive()`]: the whole task of a group that is not split into
/// parts, and, one size down, the kept derivation of each part of a split group. The runs of the
/// larger balls that end in `maximum` are added, and the run of `maximum` itself is derived into
/// the group.
fn derive_group<T, C>(input: &[T], balls: usize, size: usize, maximum: usize, output: &mut [C])
where
    T: Copy,
    C: engine::Count + From<T>,
{
    let choose = &combinatorics::BINOMIAL;
    for above in maximum + 1..balls {
        add(
            output,
            &group(input, above, size)[choose[maximum][size - 1]..choose[maximum + 1][size - 1]],
        );
    }
    // The first output group, holding the lowest balls only, has no input subset to keep.
    if maximum + 1 >= size {
        derive_into(group(input, maximum, size), maximum, size - 1, output);
    }
}

/// Add the derivation of `input`, the `size`-subsets over `balls` balls, into `output`, the
/// C(`balls`, `size` - 1) counters of the (`size` - 1)-subsets.
///
/// This is the serial form of [`derive()`], arranged by input group instead of output group so
/// that dropping a maximum is one addition of the whole run: the run's element `r` is the subset
/// of rank `r` with the maximum added. Keeping it is the derivation one size down, into the
/// output group of that maximum. The 1-subsets end the recursion: the empty subset is in all of
/// them.
fn derive_into<T, C>(input: &[T], balls: usize, size: usize, output: &mut [C])
where
    T: Copy,
    C: engine::Count + From<T>,
{
    if size == 1 {
        output[0] += input.iter().map(|&count| C::from(count)).sum::<C>();
        return;
    }
    let choose = &combinatorics::BINOMIAL;
    for maximum in size - 1..balls {
        let run = group(input, maximum, size);
        add(output, run);
        derive_into(
            run,
            maximum,
            size - 1,
            &mut output[choose[maximum][size - 1]..choose[maximum + 1][size - 1]],
        );
    }
}

/// Add `counts` to `sums` element by element; the longer slice's tail is left alone.
fn add<T, C>(sums: &mut [C], counts: &[T])
where
    T: Copy,
    C: engine::Count + From<T>,
{
    for (sum, &count) in sums.iter_mut().zip(counts) {
        *sum += C::from(count);
    }
}

/// The `size`-subsets with maximum `maximum`: a colex-contiguous run whose element `r` is the
/// subset of rank `r` among those below `maximum`, with `maximum` added.
fn group<T>(table: &[T], maximum: usize, size: usize) -> &[T] {
    &table[combinatorics::BINOMIAL[maximum][size]..combinatorics::BINOMIAL[maximum + 1][size]]
}

/// Independent scatter counting checks the streaming identity and its divisions.
#[cfg(test)]
mod tests {
    /// Random narrow-range tickets, including duplicates, populate every match category.
    #[test]
    fn streaming_matches_direct_counts() -> eyre::Result<()> {
        for balls in [14, 30] {
            let mut direct: [Vec<u64>; 4] =
                std::array::from_fn(|i| vec![0; crate::combinatorics::BINOMIAL[balls][i + 2]]);
            let mut state = 0x1234_5678_u64;
            for _ in 0..3000 {
                let numbers = crate::testing::random_numbers(&mut state, u8::try_from(balls)?);
                let ranks = crate::ticket::Ticket::new(numbers)?.ranks();
                // Two identical players make missed multiplicities observable.
                for rank in ranks.pairs {
                    direct[0][rank] += 2;
                }
                for rank in ranks.triples {
                    direct[1][rank] += 2;
                }
                for rank in ranks.quads {
                    direct[2][rank] += 2;
                }
                direct[3][ranks.quint] += 2;
            }
            for size in (3..=5).rev() {
                let divisor = u8::try_from(6 - size)?;
                let result = super::derive(&direct[size - 2], balls, size, u64::from(divisor));
                let narrow: Vec<u32> = direct[size - 2]
                    .iter()
                    .map(|&count| u32::try_from(count))
                    .collect::<Result<_, _>>()?;
                let narrow = super::derive(&narrow, balls, size, u32::from(divisor));
                assert_eq!(&result[..direct[size - 3].len()], &direct[size - 3]);
                assert!(result[direct[size - 3].len()..].iter().all(|&x| x == 0));
                assert!(
                    result
                        .iter()
                        .copied()
                        .eq(narrow.iter().map(|&x| u64::from(x)))
                );
            }
        }
        Ok(())
    }

    /// Whether every quad holding `limit` players derives triples and pairs of `limit`; debug
    /// builds panic instead if a sum before a division overflows.
    fn fits_the_sums<C: crate::engine::Count>(limit: C) -> bool {
        let quads = vec![limit; crate::combinatorics::BINOMIAL[5][4]];

        let triples = super::derive(&quads, 5, 4, C::from(2));
        let pairs = super::derive(&triples, 5, 3, C::from(3));

        triples[..10]
            .iter()
            .chain(&pairs[..10])
            .all(|&count| count == limit)
    }

    /// The sums before each division reach 2 and 3 times a count without overflowing, at each
    /// width's player limit and at the largest count 64-bit sums can take.
    #[test]
    fn largest_population_fits_the_sums() {
        use crate::engine::Count;

        let narrow = u32::try_from(u32::MAX_PLAYERS).unwrap();

        assert!(fits_the_sums(narrow));
        assert!(fits_the_sums(u64::MAX_PLAYERS));
        assert!(fits_the_sums(u64::MAX / 3));
    }

    /// Without a valid player there is nothing to derive, for the tables of any size.
    #[test]
    fn no_players_derive_to_zeros() {
        let quints = vec![0_u8; crate::combinatorics::SIZES[5]];

        let quads = super::derive(&quints, 0, 5, 1_u32);
        let triples = super::derive(&quads, 0, 4, 2);
        let pairs = super::derive(&triples, 0, 3, 3);

        assert!(pairs.iter().chain(&triples).chain(&quads).all(|&x| x == 0));
    }

    /// Over the full 90 balls and from 8-bit counters, the parallel groups add up to the serial
    /// recursion.
    #[test]
    fn parallel_groups_match_the_serial_recursion() {
        let balls = crate::combinatorics::BALLS;
        let mut quints = vec![0_u8; crate::combinatorics::SIZES[5]];
        let mut state = 0x9e37_79b9_u64;
        for _ in 0..50_000 {
            let rank =
                usize::try_from(crate::testing::split_mix(&mut state)).unwrap() % quints.len();
            quints[rank] = quints[rank].wrapping_add(1);
        }

        let parallel = super::derive(&quints, balls, 5, 1_u32);
        let mut serial = vec![0; crate::combinatorics::SIZES[4]];
        super::derive_into(&quints, balls, 5, &mut serial);

        assert!(parallel == serial, "the tables differ");
    }
}
