//! Answering a draw allocates nothing.
//!
//! The draw is the latency-critical step, so [`Engine::query`](crate::engine::Engine::query) must
//! not reach the allocator. Test builds count each thread's allocations through their global
//! allocator, so tests running on other threads cannot disturb the count.

use std::alloc::GlobalAlloc;
use std::{alloc, cell};

std::thread_local! {
    /// Allocations made by this thread. The `const` initializer and the absence of a destructor
    /// mean that using it never allocates, which an allocator's counter must not.
    static ALLOCATIONS: cell::Cell<u64> = const { cell::Cell::new(0) };
}

/// The system allocator, counting allocations in [`ALLOCATIONS`]. The default `alloc_zeroed` and
/// `realloc` go through `alloc`, so they are counted too.
struct Counting;

#[expect(unsafe_code, reason = "implementing an allocator is unsafe by nature")]
// SAFETY: every call forwards to `System` with the caller's arguments. The counter touches only
// a thread-local integer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: alloc::Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which is `System`'s too.
        unsafe { alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: alloc::Layout) {
        // SAFETY: `pointer` came from `alloc` above, so from `System`, with this `layout`.
        unsafe { alloc::System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// 10,000 random players plus one holding the last pick, and 1,000 picks.
fn players_and_picks() -> eyre::Result<(Vec<crate::ticket::Ticket>, Vec<crate::ticket::Ticket>)> {
    let mut state = 7;
    let mut players: Vec<_> = (0..10_000)
        .map(|_| crate::ticket::Ticket::new(crate::testing::random_numbers(&mut state, 90)))
        .collect::<eyre::Result<_>>()?;
    let picks: Vec<_> = (0..1_000)
        .map(|_| crate::ticket::Ticket::new(crate::testing::random_numbers(&mut state, 90)))
        .collect::<eyre::Result<_>>()?;
    players.push(*picks.last().expect("there are 1,000 picks"));
    Ok((players, picks))
}

/// Draw every pick on `engine`; return the summed answers and this thread's allocations.
fn draw(engine: &crate::engine::Engine<u32>, picks: &[crate::ticket::Ticket]) -> ([u64; 4], u64) {
    let before = ALLOCATIONS.get();
    let mut totals = [0; 4];
    for &pick in picks {
        let answer = std::hint::black_box(engine.query(std::hint::black_box(pick)));
        totals
            .iter_mut()
            .zip(answer)
            .for_each(|(total, count)| *total += count);
    }
    (totals, ALLOCATIONS.get() - before)
}

/// Draws allocate nothing.
#[test]
fn draws_do_not_allocate() -> eyre::Result<()> {
    let (players, picks) = players_and_picks()?;
    let engine = crate::engine::Engine::<u32>::from_tickets(players.iter().copied());

    let (totals, allocations) = draw(&engine, &picks);

    assert!(
        totals[3] >= 1,
        "the last pick should match its own ticket: {totals:?}"
    );
    assert_eq!(allocations, 0);
    Ok(())
}
