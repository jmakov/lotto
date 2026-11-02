//! Global counting laws complement the per-draw brute-force oracle.
//!
//! Exhausting a small game's draws checks the total contribution of every player to each
//! match category. Relabeling balls and reordering players must preserve individual answers,
//! including when repeated tickets overflow the packed five-number counters.

/// Select an asymmetric population from `draws`, including sparse quint-counter overflows.
///
/// `draws` contains all draws of a domain with at least five balls.
fn players(draws: &[crate::ticket::Ticket]) -> Vec<crate::ticket::Ticket> {
    let mut players: Vec<_> = draws.iter().step_by(7).copied().collect();
    players.extend(std::iter::repeat_n(draws[0], 40));
    players.extend(std::iter::repeat_n(draws[draws.len() / 2], 16));
    players
}

/// Compute the small binomial coefficients of the independent combinatorial oracle.
///
/// `n` and `k` are at most 12; impossible choices return zero.
fn choose(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    (0..k).fold(1, |count, index| count * (n - index) / (index + 1))
}

/// Return answers for `draws` over the population `players`.
fn answers(
    players: &[crate::ticket::Ticket],
    draws: &[crate::ticket::Ticket],
) -> eyre::Result<Vec<[u64; 4]>> {
    let narrow = crate::engine::Engine::<u32>::from_tickets(players.iter().copied());
    let wide = crate::engine::Engine::<u64>::from_tickets(players.iter().copied());

    assert_eq!(narrow.valid_count(), u64::try_from(players.len())?);
    let answers: Vec<_> = draws.iter().map(|&draw| narrow.query(draw)).collect();
    assert!(
        draws
            .iter()
            .map(|&draw| wide.query(draw))
            .eq(answers.iter().copied()),
        "the counter widths disagree"
    );
    Ok(answers)
}

/// Every player contributes C(5,m) * C(n-5,5-m) winners across all possible draws.
#[test]
fn exhaustive_draws_conserve_each_match_category() -> eyre::Result<()> {
    for balls in [10, 12] {
        let draws = crate::testing::draws(balls)?;
        let players = players(&draws);
        let population = u64::try_from(players.len())?;

        let totals = answers(&players, &draws)?
            .into_iter()
            .fold([0_u64; 4], |totals, answer| {
                std::array::from_fn(|index| totals[index] + answer[index])
            });

        for (index, total) in totals.into_iter().enumerate() {
            let matched = u64::try_from(index)? + 2;
            let expected =
                population * choose(5, matched) * choose(u64::from(balls) - 5, 5 - matched);
            assert_eq!(total, expected, "balls={balls}, matches={matched}");
        }
    }
    Ok(())
}

/// Map all 90 labels bijectively, moving a small fixture across the full production domain.
///
/// Multiplication by 37 permutes residues modulo 90 because the two numbers are coprime.
/// `ticket` is already validated; its mapped labels remain distinct and in range.
fn relabel(ticket: crate::ticket::Ticket) -> eyre::Result<crate::ticket::Ticket> {
    let numbers = ticket.numbers().map(|number| {
        let residue = (u16::from(number - 1) * 37 + 11) % 90;
        u8::try_from(residue + 1).expect("a residue modulo 90 plus one fits in u8")
    });
    crate::ticket::Ticket::new(numbers)
}

/// Player order and ball labels cannot change the number of matches in any category.
#[test]
fn answers_survive_player_reordering_and_ball_relabeling() -> eyre::Result<()> {
    let draws = crate::testing::draws(10)?;
    let mut players = players(&draws);
    let expected = answers(&players, &draws)?;

    players.reverse();
    assert_eq!(
        answers(&players, &draws)?,
        expected,
        "reversed player order"
    );
    let players = players
        .into_iter()
        .map(relabel)
        .collect::<eyre::Result<Vec<_>>>()?;
    let draws = draws
        .into_iter()
        .map(relabel)
        .collect::<eyre::Result<Vec<_>>>()?;

    assert_eq!(answers(&players, &draws)?, expected, "permuted ball labels");
    Ok(())
}
