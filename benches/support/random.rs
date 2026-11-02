//! The pseudo-random numbers behind the players and picks of the benchmarks and the end-to-end
//! tests. `shared.rs` and `tests/end_to_end.rs` include this file with `#[path]`.

/// Deterministic `SplitMix64` stream, so every run generates the same players and picks.
pub(crate) struct Random(pub(crate) u64);

impl Random {
    /// The next pseudo-random 64-bit value.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Five distinct numbers in `1..=balls`, in drawing order.
    pub(crate) fn numbers(&mut self, balls: u8) -> [u8; 5] {
        let mut numbers = [0; 5];
        for index in 0..numbers.len() {
            numbers[index] = loop {
                let number = u8::try_from(self.next_u64() % u64::from(balls))
                    .expect("the remainder is below `balls`")
                    + 1;
                if !numbers[..index].contains(&number) {
                    break number;
                }
            };
        }
        numbers
    }
}
