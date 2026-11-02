//! Colexicographic ranks put every unordered subset in a dense, collision-free table.

/// Number of available lottery balls; tickets use zero-based indices internally.
pub(crate) const BALLS: usize = 90;
/// The highest zero-based ball index, as a subset stores it.
const LAST_BALL: u8 = 89;
// Keeps the two in step at compile time.
const _: () = assert!(LAST_BALL as usize + 1 == BALLS);
/// Number of distinct balls on a ticket.
pub(crate) const WIDTH: usize = 5;
/// Dense table lengths, indexed by subset size (including the empty subset).
pub const SIZES: [usize; WIDTH + 1] = BINOMIAL[BALLS];
/// Pascal's triangle, evaluated at compile time; entries with k > n are zero.
pub(crate) const BINOMIAL: [[usize; WIDTH + 1]; BALLS + 1] = binomial_table();

/// Build the small lookup table without any runtime initialization or synchronization.
const fn binomial_table() -> [[usize; WIDTH + 1]; BALLS + 1] {
    let mut table = [[0; WIDTH + 1]; BALLS + 1];
    let mut n = 0;
    while n <= BALLS {
        table[n][0] = 1;
        let mut k = 1;
        while k <= WIDTH && k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}

/// Rank a strictly ascending, zero-based subset of at most five balls.
///
/// Returns its dense colex index. This checked form is the tests' reference for
/// [`crate::ticket::Ticket::ranks`], whose indices are hand-written for speed.
///
/// # Errors
///
/// Returns an error for more than five balls or unsorted, duplicate, or out-of-range indices.
#[cfg(test)]
pub(crate) fn rank(subset: &[u8]) -> eyre::Result<usize> {
    eyre::ensure!(subset.len() <= WIDTH, "a subset has at most five balls");
    eyre::ensure!(
        subset.iter().all(|&n| usize::from(n) < BALLS),
        "ball outside 0..90"
    );
    eyre::ensure!(
        subset.windows(2).all(|p| p[0] < p[1]),
        "subset must be strictly ascending"
    );
    Ok(subset
        .iter()
        .enumerate()
        .map(|(i, &n)| BINOMIAL[usize::from(n)][i + 1])
        .sum())
}

/// Invert a colex index into `size` ascending zero-based ball indices.
///
/// Empty subsets are allowed.
///
/// # Errors
///
/// Returns an error if `size > 5` or `index` is outside that table.
pub fn unrank(mut index: usize, size: usize) -> eyre::Result<Vec<u8>> {
    eyre::ensure!(
        size <= WIDTH && index < SIZES[size],
        "colex index or size out of range"
    );
    let mut subset = vec![0; size];
    let mut n = LAST_BALL;
    for k in (1..=size).rev() {
        while BINOMIAL[usize::from(n)][k] > index {
            n -= 1;
        }
        subset[k - 1] = n;
        index -= BINOMIAL[usize::from(n)][k];
        n = n.saturating_sub(1);
    }
    Ok(subset)
}

/// Regression checks for the module invariants.
#[cfg(test)]
mod tests {
    /// Exhaustive small tables and spread-out quint ranks catch gaps and wrong ordering.
    #[test]
    fn rank_bijection() -> eyre::Result<()> {
        for size in 0..=4 {
            for index in 0..super::SIZES[size] {
                assert_eq!(super::rank(&super::unrank(index, size)?)?, index);
            }
        }
        for index in (0..super::SIZES[5])
            .step_by(997)
            .chain([super::SIZES[5] - 1])
        {
            assert_eq!(super::rank(&super::unrank(index, 5)?)?, index);
        }
        assert!(super::rank(&[90]).is_err());
        assert!(super::rank(&[1, 1]).is_err());
        assert!(super::unrank(0, 6).is_err());
        Ok(())
    }
}
