//! The stdin/stdout protocol: READY, then one answer line per draw.
//!
//! The protocol output is deliberately isolated from logging and file loading: stdout carries
//! only READY and the answers, and diagnostics go through `tracing`.

use std::io;
use std::io::Write;

use eyre::WrapErr;

use crate::{engine, input, ticket};

/// Bytes of the longest answer: four 20-digit counts, three spaces and a newline.
const ANSWER_BYTES: usize = 4 * 20 + 3 + 1;

/// Write `counts` as one answer line at the end of `buffer`; returns the line.
///
/// Digits come out last to first, so the line is built backwards from the buffer's end.
fn format_answer(counts: [u64; 4], buffer: &mut [u8; ANSWER_BYTES]) -> &[u8] {
    let mut start = buffer.len() - 1;
    buffer[start] = b'\n';
    for (index, mut count) in counts.into_iter().rev().enumerate() {
        if index > 0 {
            start -= 1;
            buffer[start] = b' ';
        }
        loop {
            start -= 1;
            buffer[start] = b"0123456789"[(count % 10) as usize];
            count /= 10;
            if count == 0 {
                break;
            }
        }
    }
    &buffer[start..]
}

/// Print READY to `output`, then answer the draws on `input`.
///
/// An invalid draw gets no answer, and a warning from [`input::Draws`] if warnings are on: the
/// protocol has no error line, and `0 0 0 0` would be a false result. Later lines are still parsed
/// correctly, but a caller that pairs answers with draws by position has to send valid draws.
///
/// Answers are flushed once per read of `input`, before the next read can block: a caller that
/// sends one draw at a time gets each answer at once, and draws sent in one write share a write.
/// The writer's 8 KiB buffer bounds how long an answer waits inside a large read. The draw
/// reader is created before READY, because the first draw may already be waiting: allocating
/// the read buffer after it added 70 us to the first answer on the benchmark machine.
///
/// The function is generic over the counter width, so each width gets its own copy and a draw
/// pays no branch on it.
///
/// # Errors
///
/// Returns an error when reading a draw or writing an answer fails.
pub fn serve<C: engine::Count>(
    engine: &engine::Engine<C>,
    input: impl io::Read,
    output: impl io::Write,
) -> eyre::Result<()> {
    tracing::info!(
        valid = engine.valid_count(),
        invalid = engine.invalid_count(),
        "player lines loaded"
    );
    let mut output = io::BufWriter::new(output);
    let mut draws = input::Draws::new(input);
    output.write_all(b"READY\n").wrap_err("writing READY")?;
    output.flush().wrap_err("flushing READY")?;
    let mut answer = [0; ANSWER_BYTES];
    let mut open = true;
    while open {
        open = draws.visit_read(|_, line| {
            if let ticket::Line::Ticket(pick) = line {
                let answer = format_answer(engine.query(pick), &mut answer);
                output.write_all(answer).wrap_err("writing winner counts")?;
            }
            Ok(())
        })?;
        output.flush().wrap_err("flushing winner counts")?;
    }
    Ok(())
}

/// Regression checks for the module invariants.
#[cfg(test)]
mod tests {
    proptest::proptest! {
        #![proptest_config(crate::testing::proptest_config())]

        /// An answer reads as `format!` writes it, from one digit up to the longest line.
        #[test]
        fn answers_match_the_standard_formatter(
            counts in proptest::array::uniform4(proptest::prop_oneof![
                proptest::strategy::Just(0_u64),
                proptest::strategy::Just(u64::MAX),
                0_u64..1000,
                proptest::arbitrary::any::<u64>(),
            ])
        ) {
            let [e2, e3, e4, e5] = counts;
            let expected = format!("{e2} {e3} {e4} {e5}\n");
            let mut buffer = [0; super::ANSWER_BYTES];

            let line = super::format_answer(counts, &mut buffer);

            proptest::prop_assert_eq!(line, expected.as_bytes());
        }
    }

    /// The buffer holds the longest answer exactly.
    #[test]
    fn longest_answer_fills_the_buffer() {
        let mut buffer = [0; super::ANSWER_BYTES];

        let line = super::format_answer([u64::MAX; 4], &mut buffer);

        assert_eq!(line.len(), super::ANSWER_BYTES);
    }

    /// Served in process: READY first, one answer per valid pick, blanks and invalid picks
    /// skipped, and a last pick without a newline answered, at both counter widths.
    #[test]
    fn serve_answers_each_valid_pick() -> eyre::Result<()> {
        serve_answers_each_valid_pick_at::<u32>()?;
        serve_answers_each_valid_pick_at::<u64>()
    }

    /// [`serve_answers_each_valid_pick`] for counters of width `C`.
    fn serve_answers_each_valid_pick_at<C: crate::engine::Count>() -> eyre::Result<()> {
        let players = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [10, 20, 30, 40, 50]]
            .map(crate::ticket::Ticket::new);
        let engine = crate::engine::Engine::<C>::from_tickets(
            players.into_iter().collect::<eyre::Result<Vec<_>>>()?,
        );
        let picks = &b"1 2 3 4 5\n\n1 1 2 3 4\n 5 4 3 2 1\r\n10 20 30 40 50"[..];
        let mut output = Vec::new();

        super::serve(&engine, picks, &mut output)?;

        assert_eq!(output, b"READY\n0 0 1 1\n0 0 1 1\n0 0 0 1\n");
        Ok(())
    }
}
