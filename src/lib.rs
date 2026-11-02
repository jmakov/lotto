//! Report the winners of a Hungarian lottery draw as soon as possible after it.
//!
//! The `lotto` binary loads the players, prints READY, then answers each draw read from stdin
//! with the number of players matching exactly 2, 3, 4 and 5 numbers. All work that does not
//! depend on the draw happens in the load, which builds exact counts of every subset of 2 to 5
//! numbers ([`engine`]); a draw then adds up 26 of them. See docs/DESIGN.md for the architecture.
//!
//! This library holds everything but the command line, so that the benchmarks and the fuzz
//! target use the same code as the binary: [`input`] reads the file and the draws, [`ticket`]
//! validates and ranks tickets, [`engine`] builds and queries the tables, [`streaming`] derives
//! the smaller tables, `simd` is the fast path of the load on a CPU with AVX-512, `decode` the one
//! on x86-64 with SSE4.1, and [`protocol`] serves the draws.

#[cfg(test)]
mod allocation;
pub mod combinatorics;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
mod decode;
pub mod engine;
pub mod input;
#[cfg(test)]
mod invariants;
pub mod protocol;
#[cfg(target_arch = "x86_64")]
mod simd;
pub mod streaming;
#[cfg(test)]
pub(crate) mod testing;
pub mod ticket;
