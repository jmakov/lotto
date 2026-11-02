# lotto: Hungarian lottery winner counts

Load the five numbers of every player and then report how many players match exactly 2, 3, 4 and 5 numbers of each draw.
All the work goes into the load, so each draw is answered with 26 table reads. With ten million
players, a fresh process is ready after 35 ms and answers a draw in 22 ns, or in about 8 µs through
stdin/stdout ([Performance](#performance)). The file is read in fixed-size pieces, so it may be
larger than memory.

## Usage

### Dependencies

Ubuntu:

```sh
sudo apt-get install -y build-essential curl git
```

Fedora:

```sh
sudo dnf install -y gcc gcc-c++ glibc-devel make curl git
```
Rust dependencies:
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal
. "$HOME/.cargo/env"
rustup toolchain install

# Development: tests, checks, fuzzing
rustup toolchain install nightly --profile minimal --component rustfmt
cargo install --locked --version 0.9.150 cargo-nextest
cargo install --locked --version 0.20.2 cargo-deny
cargo install --locked --version 0.13.2 cargo-fuzz
```

### Build and run

```sh
cargo build --release

# run interactive (Ctrl+D to exit - signals end of input)
./target/release/lotto input.txt

# pass multiple predefined queries
printf '1 2 3 4 5\n4 79 13 80 56\n' | ./target/release/lotto input.txt
# Output:
# READY
# 225152 8219 95 0
# 224073 8154 97 1
```

`lotto --help`:

| Option | Effect |
|---|---|
| `--threads COUNT` | Load workers. Default: the CPUs available, at most 32, because more cost more to start than they save. An explicit count has no limit. |
| `--log-level LEVEL` | `off` (default), `error`, `warn`, `info`, `debug` or `trace`. |
| `--log-name PATH` | Log file, used unless the level is `off`. Default: `log.txt`. |
| `-h`, `--help` | Print help. |
| `-V`, `--version` | Print the version. |

Each log level includes the ones before it:

| `--log-level` | Adds to the log |
|---|---|
| `off` | Nothing; no file is created. |
| `error` | The fatal error that ends a run, which stderr gets at every level. |
| `warn` | Each invalid line, player or draw, with its source, line number and contents. |
| `info` | The number of valid and invalid player lines. |
| `debug`, `trace` | Nothing more at present. |

## Performance

Test data:

| Property | Value |
|---|---|
| Size | 145 MB, one player per line |
| Player lines | 10,000,000 |
| Valid tickets | 9,999,994 |
| Invalid lines | 6: two repeat a number, one has six numbers, one has four, one has 93, one has -7 |
| Distinct tickets | 8,945,152: some players picked the same five numbers (at most 6 players alike), and each of them counts |
| Numbers | Each of 1 to 90 is on 553,692 to 556,999 tickets |

Machine and build: a two-socket AMD server with 512 hardware threads, Fedora 44, Rust 1.99.1,
`cargo build --release` (`target-cpu=native`, so the AVX-512 decoder), 32 load workers (the default
on this machine).

| Measurement | Result |
|---|---|
| Start until `READY` | 34.7 ms |
| Start until `READY`, file not cached | 53.3 ms |
| One draw in process | 21.7 ns |
| One draw through stdin/stdout | p50 8.1 µs, p99 10.6 µs, p99.9 14.7 µs, max 164 µs |
| 1,000 draws sent in one write | 97.9 µs, 98 ns per draw |

## Development

### Tests

```sh
cargo nextest run

# fuzz parsers
cd fuzz && cargo +nightly fuzz run parse -- -max_total_time=10
```


### Benchmarks

```sh
cargo bench

# load time with the file read in chunks of 128 KiB to 10 MiB
cargo bench --bench lotto -- load_chunk

# p50, p99, p99.9 and max of 1,000,000 draws through the running app
LOTTO_BENCH_PERCENTILES=1000000 cargo bench --bench percentiles

# open HTML report of the last run
xdg-open target/criterion/report/index.html
```

## Design summary

Based on the following insights:

- the ticket submission period might be long, but we want to answer the query fast
- for 90 numbers (balls) and ticket length of 5, we can keep all the combinations in memory and count them
  - by doing that, the query takes about 22 ns, but doesn't scale for more numbers (balls) or tickets of longer lengths
- the naive approach (see [docs/DESIGN.md](docs/DESIGN.md)) doesn't scale with file size, but we can use an LLM to explore
  algorithmic optimizations, which resulted in a draw time of O(1), independent of the number of players
- there's a limit on how many tickets we can process. The limit is the integer type we use for the pair, triple and
  quad counters (currently `u64`).
  - based on the number of valid players we use `u32` or `u64` counters (`u32` counters are measurably faster)

The players are known minutes before the draw, but the answer is needed immediately after it. So
the load builds exact counts for every subset of 2, 3, 4 and 5 numbers that any player holds, and a
draw only adds up 26 of those counts and solves a small set of equations.
[docs/DESIGN.md](docs/DESIGN.md) explains the design and the theory behind it.

Supports:

- reading files larger than memory
- vectorization with fallback (in case CPU doesn't support AVX-512)

### Computational complexity

The asymptotic run time and memory, in terms of $n$ valid players, $q$ draws, $b = 90$ numbers,
$k = 5$ numbers per ticket and machine words of $w = 64$ bits.

With the task's $b = 90$ and $k = 5$:

| | Count tables (this solution) | Bit matrix (naive) |
|---|---|---|
| Load time | $O(n)$ | $O(n)$ |
| Draw time | $O(1)$ | $O(n)$ |
| Memory | $O(1)$: tables of 52 to 62 MiB | $O(n)$ |
| Load, then $q$ draws | $O(n + q)$ | $O(n + n q)$ |

With $b$ and $k$ as variables too:

| | Count tables (this solution) | Bit matrix (naive) |
|---|---|---|
| Load time | $O(n k \log k + k \binom{b}{k})$ | $O(n (k + b / w))$ |
| Draw time | $O(2^k)$ | $O(k n / w)$ |
| Memory | $O(\binom{b}{k})$ counters | $O(n b)$ bits |
| Load, then $q$ draws | $O(n k \log k + k \binom{b}{k} + q 2^k)$ | $O(n (k + b / w) + q k n / w)$ |

### Scaling

A draw already takes the same time for any number of players. More players only lengthen the load,
which reads the whole file after sales close.

**More players**

1. Each server keeps its own table with one counter per possible ticket (about 44 MB) and
   adds 1 for each ticket it sells: the same single increment the load makes now. A cancelled
   ticket subtracts 1.
2. The tables are [histograms](https://en.wikipedia.org/wiki/Histogram), so they add up: a
   coordinator sums the servers' tables element by element. Servers send what changed during the
   sales period, so little is left to send at the cutoff.
3. At the cutoff, the coordinator derives the smaller tables once (230 million reads for any number
   of players), prints `READY` and answers draws as now.

The time from the cutoff to `READY` then depends on the number of sales servers, not on the number
of players, and every machine's memory stays fixed. Alternatively, each server derives its own
tables and answers the draw, and the coordinator sums the four winner counts, which add up just as
the counters do. That moves no tables but adds a network round trip to every draw.

**Faster answers**

A draw takes 22 ns in process and about 8 µs through stdin/stdout, so the pipe, not the
calculation, sets the answer time. Sending many draws in one write already brings it to 98 ns per
draw. Further:

- Calling the program as a library takes the 22 ns: `lotto::engine::load` once, then
  `Engine::query` for each draw.
- The tables are read-only after the load, so any number of threads can answer draws at once,
  without locks.

**Faster load from a file**

If sales wrote each ticket's position in the table (a 4-byte number) instead of its text (14.5 bytes
a player on average), the file would shrink to under a third and need no parsing.

**More numbers or longer tickets**

The tables grow with the number of possible tickets: 14 million for 6 numbers out of 49 is fine,
but 8 trillion for 5 out of 1,000 is too many. Past that point, a hash map keeps counters only
for the subsets some player holds, at most 26 a player for 5-number tickets. Memory then grows with
the players again, but a draw still makes 26 lookups.