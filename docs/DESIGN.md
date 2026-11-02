# Architecture and design

## The big picture

**The problem.** A file lists millions of players, each with a ticket of five different numbers
from 1 to 90. Draws then arrive one at a time, each also five numbers. For every draw the program
must say how many players matched exactly 2, 3, 4 and 5 of the numbers.

**The obvious solution** compares every ticket with the draw. It works, but every draw then
visits every player, so more players mean slower answers.

**Words used below:**

| Word | Meaning |
|---|---|
| Ball | One of the 90 numbers. The word avoids confusion with counts, which are numbers too. |
| Subset | A group of balls, in no particular order. |
| Pair, triple, quad, quint | A subset of 2, 3, 4 or 5 balls. |
| Table | A long list of counters, one for every possible subset of one size. |
| Rank | The position of a subset in its table. |

### The insight: count every possible ticket, not every player

A ticket is five different balls out of 90, and their order does not matter: `5 4 3 2 1` is the
same ticket as `1 2 3 4 5`. A ticket is therefore a
[combination](https://en.wikipedia.org/wiki/Combination), not a permutation, and only 43,949,268
different tickets exist. Counted with their order there would be 5,273,912,160, 120 times as many,
one for each order of five balls; sorting each ticket's numbers removes that factor.

The groups a draw can share with a ticket are just as few: 4,005 possible pairs (`1 2`, `1 3` and
so on up to `89 90`), 117,480 triples and 2,555,190 quads. One counter for every possible pair,
triple, quad and quint, 46,625,943 counters in all, fits in about 52 MiB. That number is set by the
90 balls alone, so ten million players and ten billion use the same number of counters. It is also
where the approach stops working: with 1,000 balls there would be over 8 trillion possible tickets,
far too many counters to hold.

The number of counters is fixed, but each counter still needs enough bits to hold its count, which
can reach the number of players, for instance when everybody plays the same ticket. 32-bit
counters are exact for up to about 1.4 billion players, a third of what 32 bits can count, because
deriving the smaller tables adds up to three times a count. A file with more players gets 64-bit
counters for the pairs, triples and quads, about 62 MiB in all. Those
hold about 307 quadrillion players, a file of at least 3 exabytes, so in practice the number of
players is unlimited. The program still checks the limit, so a count can never wrap around and be
silently wrong. [Data structures](#data-structures) gives the details.

So the program does not keep the players. It keeps, for every possible group, a counter of how
many players hold it, and fills the counters while it loads the file, before the first draw. A
draw contains 26 groups: 10 pairs, 10 triples, 5 quads and the draw itself. Their 26 counters, with
a little arithmetic, give the exact number of winners in each category. The player file is never
read again, and a draw takes the same time for any number of players.

Each counter sits at a fixed position worked out from its balls, called its rank. The rank acts as
a perfect hash with no empty slots: each possible group gets its own position in the table for its
size, and the table has exactly one position per possible group. So neither loading nor a draw ever
searches for a counter. [How does a subset find its counter?](#how-does-a-subset-find-its-counter)
gives the formula.

### How it works

In outline, the program does this:

```text
load:  for each valid ticket in the file:
           for each subset s of the ticket with 2 to 5 balls:        # 26 subsets
               count[size(s)][rank(s)] += 1

draw:  for k in 2, 3, 4, 5:
           A_k = sum of count[k][rank(s)] over the subsets s of the draw with k balls
       print  A_2 - 3 A_3 + 6 A_4 - 10 A_5                           # exactly 2 matches
              A_3 - 4 A_4 + 10 A_5                                   # exactly 3
              A_4 - 5 A_5                                            # exactly 4
              A_5                                                    # exactly 5
```

`count` is a [histogram](https://en.wikipedia.org/wiki/Histogram) over every possible subset. Like
any histogram, it can be filled in any order and in any number of pieces, each piece adding to the
same arrays, with nothing to merge at the end. Had two pieces been counted into separate arrays,
adding those element by element would give the same result. The subtractions are there because a
player appears in `A_2` once for every pair of the draw on their ticket; [The theory](#the-theory)
shows that they remove the repeats exactly.

The program computes the same counters faster, in four stages:

1. **Load** ([Loading the file](#loading-the-file)). Read the file in pieces of 512 KiB, several at
   once on different CPU cores. Each valid line adds 1 to a single counter, the one of its quint.
   All pieces add to the same quint table, so after the last piece it holds the whole file.
2. **Derive** ([Deriving the smaller tables](#deriving-the-smaller-tables)). From the finished quint
   table, compute the quad table, then the triples from the quads and the pairs from the triples.
   The counters come out exactly as the outline's load would leave them, and this step takes the
   same time for any number of players.
3. **READY.** Print `READY`, which tells the caller that draws can be sent.
4. **Answer** ([Answering a draw](#answering-a-draw)). For each draw, compute the ranks of its 26
   subsets, read their counters and print four numbers, as in the outline.

| Stage | Code |
|---|---|
| Load | `load` and `count` in `engine.rs`: reading in `input.rs`, parsing and ranking in `simd.rs`, `decode.rs` and `ticket.rs`, staging in `group_by_owner`, counting in `count_by_owner` |
| Derive | `by_width` in `engine.rs` picks the counter width, then `Counted::into_engine` calls `derive` in `streaming.rs` |
| READY and the draw loop | `serve` in `protocol.rs`, called from `main.rs` |
| Answer | `Ticket::ranks` in `ticket.rs`, then `Engine::query` in `engine.rs` |

**Cost.** Loading takes time in proportion to the file's length but a fixed amount of memory: the
file is read in fixed-size pieces, so a file larger than memory loads at the same speed per line.
Deriving reads at most about 230 million counters, however many players there are. A draw makes 26
reads and a few multiplications. The tables take about 52 MiB, or 62 MiB with 64-bit counters, for
any number of players; the only thing that can grow is a small map for tickets held by more than 255
players. [README.md](../README.md#computational-complexity) gives the complexity in full.

### An example

This example fills the counters the simple way, as in the outline. The program's load and
derivation end with the same counters.

Take six players: P1 holds `1 2 3 4 5`, P2 `1 2 3 4 9`, P3 `1 2 3 8 9`, P4 `1 2 7 8 9`, P5
`1 6 7 8 9` and P6 `6 7 8 9 10`.

**Filling the tables.** All counters start at zero. Then every valid player line adds 1 to the
counter of each subset on its ticket. P3's ticket `1 2 3 8 9` holds 26 of them:

| Size | Subsets of `1 2 3 8 9` |
|---|---|
| Pairs | `1 2`, `1 3`, `1 8`, `1 9`, `2 3`, `2 8`, `2 9`, `3 8`, `3 9`, `8 9` |
| Triples | `1 2 3`, `1 2 8`, `1 2 9`, `1 3 8`, `1 3 9`, `1 8 9`, `2 3 8`, `2 3 9`, `2 8 9`, `3 8 9` |
| Quads | `1 2 3 8`, `1 2 3 9`, `1 2 8 9`, `1 3 8 9`, `2 3 8 9` |
| Quint | `1 2 3 8 9` |

Once every line has been added, a counter says how many players hold its subset. The order of the
lines does not matter, and two players with the same ticket add to the same counters twice. Each
player adds 1 to ten pair counters, so all the pair counters together add up to 10 times the number
of valid players. The six tickets make 60 additions to the pair table, 60 to the triple table, 30
to the quad table and 6 to the quint table.

**Reading the tables.** A draw lists its own subsets the same way and reads only their counters.
For the draw `1 2 3 4 5`, these are the pair counters it reads. The program stores only the number;
the middle column shows where it came from.

| Pair of the draw | Players holding it | Counter |
|---|---|---:|
| `1 2` | P1, P2, P3, P4 | 4 |
| `1 3` | P1, P2, P3 | 3 |
| `1 4` | P1, P2 | 2 |
| `1 5` | P1 | 1 |
| `2 3` | P1, P2, P3 | 3 |
| `2 4` | P1, P2 | 2 |
| `2 5` | P1 | 1 |
| `3 4` | P1, P2 | 2 |
| `3 5` | P1 | 1 |
| `4 5` | P1 | 1 |
| Total, called $A_2$ | | 20 |

Adding up its 10 triple counters, its 5 quad counters and its quint counter the same way gives
$A_3 = 15$, $A_4 = 6$ and $A_5 = 1$. That is 26 reads, however many players there are.

P5 and P6 appear nowhere in the table. P5 shares only the ball 1 with the draw and P6 shares none,
so every subset on their tickets holds a ball the draw lacks. Their additions went to counters such
as `1 6` and `6 7`, which this draw never reads. A player who matches fewer than two balls therefore
costs a draw nothing.

The four formulas then give $20 - 45 + 36 - 10 = 1$ player with 2 matches, $15 - 24 + 10 = 1$ with
3, $6 - 5 = 1$ with 4 and 1 with 5: P4, P3, P2 and P1.

## The theory

The big picture rests on a few facts about counting subsets: why the counters of a draw's subsets
hold everything the answer needs, why the four formulas count every player exactly once, and how a
subset finds its counter. This section explains each one by asking the question it answers.

### How many subsets are there?

The number of ways to choose $k$ things out of $n$ is written $\binom{n}{k}$ and called a
[binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient). These numbers form
[Pascal's triangle](https://en.wikipedia.org/wiki/Pascal%27s_triangle). Row $n$, entry $k$, both
counted from 0, is $\binom{n}{k}$:

```text
row 0                 1
row 1               1   1
row 2             1   2   1
row 3           1   3   3   1
row 4         1   4   6   4   1
row 5       1   5  10  10   5   1
row 6     1   6  15  20  15   6   1
```

Row 5 says that five balls contain 1 empty subset, 5 single balls, 10 pairs, 10 triples, 5 quads and
1 quint. That is where the 26 lookups come from: $10 + 10 + 5 + 1 = 26$.
[Why do subset counts give the answer?](#why-do-subset-counts-give-the-answer) shows why the single
balls and the empty subset are not needed.

Each entry is the sum of the two above it, which is
[Pascal's rule](https://en.wikipedia.org/wiki/Pascal%27s_rule):
$\binom{n + 1}{k} = \binom{n}{k} + \binom{n}{k - 1}$. The reason is simple. Line up $n + 1$ balls
and look at the last one. A subset of size $k$ either leaves it out, and then all $k$ balls come
from the other $n$, or holds it, and then the remaining $k - 1$ do. This rule comes back when the
subsets are numbered.

### Why does a draw not need the tickets?

In the example, the draw's pair counters add up to $A_2 = 20$. Picture a grid with a row for each
player, a column for each of the draw's 10 pairs, and a 1 wherever the player holds the pair. A
pair counter is the sum of one column, so $A_2$ is the sum of the whole grid. Summing the rows
instead gives how many of the draw's pairs each player holds: 10 for P1, 6 for P2, 3 for P3, 1 for
P4 and 0 for P5 and P6, again 20 in total. Both add up the same grid, a technique called
[double counting](https://en.wikipedia.org/wiki/Double_counting_(proof_technique)). Filling the
tables works out the column sum of every possible pair while loading, before any draw is known, and
a draw picks out its own 10. The same holds for the triples, the quads and the quint.

### Why do subset counts give the answer?

The totals $A_2$ to $A_5$ are not the answer yet, because a player can appear in them many times.
P3 holds `1 2 3 8 9` and shares `1 2 3` with the draw `1 2 3 4 5`. That ticket holds three of the
draw's pairs (`1 2`, `1 3`, `2 3`) and one of its triples (`1 2 3`). P3 wins once, in the 3-match
category, yet adds 3 to $A_2$ and 1 to $A_3$.

That pattern is the same for every player. A ticket's overlap with the draw is a set of $m$ balls,
and a subset of the draw is on the ticket exactly when it lies inside that overlap. So a player with
$m$ matches adds $\binom{m}{k}$ to $A_k$, which is a value read straight off the triangle:

| Matches $m$ | Adds to $A_2$ | $A_3$ | $A_4$ | $A_5$ |
|---:|---:|---:|---:|---:|
| 5 | 10 | 10 | 5 | 1 |
| 4 | 6 | 4 | 1 | 0 |
| 3 | 3 | 1 | 0 | 0 |
| 2 | 1 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 |

Now call $E_m$ the number of players with exactly $m$ matches, which is the answer we want. Reading
the table column by column gives four equations:

$$\begin{array}{rcrcrcrcr} A_5 & = & & & & & & & E_5 \cr A_4 & = & & & & & E_4 & + & 5 E_5 \cr A_3 & = & & & E_3 & + & 4 E_4 & + & 10 E_5 \cr A_2 & = & E_2 & + & 3 E_3 & + & 6 E_4 & + & 10 E_5 \end{array}$$

These are easy to solve from the top down. $A_5$ is $E_5$ outright. Knowing $E_5$, the second line
gives $E_4$, and so on. Players with 0 or 1 match add nothing to any total, so they cannot disturb
the result, and the task does not ask about them. That is why the single balls and the empty
subset are not needed.

The six players of the example have one of each match count with the draw `1 2 3 4 5`:

| Player | Ticket | Matches | Adds to $A_2$ | $A_3$ | $A_4$ | $A_5$ |
|---|---|---:|---:|---:|---:|---:|
| P1 | `1 2 3 4 5` | 5 | 10 | 10 | 5 | 1 |
| P2 | `1 2 3 4 9` | 4 | 6 | 4 | 1 | |
| P3 | `1 2 3 8 9` | 3 | 3 | 1 | | |
| P4 | `1 2 7 8 9` | 2 | 1 | | | |
| P5 | `1 6 7 8 9` | 1 | | | | |
| P6 | `6 7 8 9 10` | 0 | | | | |
| Total | | | 20 | 15 | 6 | 1 |

Solving from the top gives $E_5 = 1$, then $E_4 = 6 - 5 \cdot 1 = 1$, then
$E_3 = 15 - 4 \cdot 1 - 10 \cdot 1 = 1$, and finally
$E_2 = 20 - 3 \cdot 1 - 6 \cdot 1 - 10 \cdot 1 = 1$. That is one player in each category, as built.

### Why is the result always exact?

This section is the proof, and nothing later depends on it. Solving the equations in general gives
the same numbers with alternating signs:

$$E_m = A_m - \binom{m + 1}{m} A_{m + 1} + \binom{m + 2}{m} A_{m + 2} - \dots$$

This is a form of the
[inclusion-exclusion principle](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle),
also known as [binomial inversion](https://en.wikipedia.org/wiki/Binomial_transform): count too
much, take away the excess, put back what was taken away twice.

It is exact because the signed entries of any row of the triangle below the first add up to zero,
as in $1 - 3 + 3 - 1 = 0$. Follow a player with $m$ matches through the formula for $E_j$. That
player contributes $\binom{k}{j} \binom{m}{k}$ to the term for size $k$. This product equals
$\binom{m}{j} \binom{m - j}{k - j}$, since choosing $k$ of the $m$ and then $j$ of those $k$ is the
same as choosing the $j$ first and then the other $k - j$. So the player's total contribution is:

$$\sum_k (-1)^{k - j} \binom{k}{j} \binom{m}{k} = \binom{m}{j} (1 - 1)^{m - j}$$

That is 1 when $m = j$ and 0 otherwise. Each player therefore cancels out of every category except
their own. In the example, the player with 3 matches enters $E_2$ as 3 from $A_2$ and
$-3 \cdot 1$ from $A_3$, which adds up to 0.

### How does a subset find its counter?

Each table needs a way to go from a subset straight to its counter, with no searching. So every
subset gets a number, its rank, with no gaps and no two subsets sharing one. Then the rank can be
used directly as a position in the table. In programming terms the rank is a
[minimal perfect hash](https://en.wikipedia.org/wiki/Perfect_hash_function): every subset gets a
different position, and every position belongs to a subset.

Pascal's rule gives this numbering. Count the balls from 0 here, so they run from 0 to 89, and
decide them one at a time from the highest down. At each ball, the subsets split into those that
leave it out and those that hold it, as in the rule. Give the subsets that leave ball $x$ out the
low numbers and those that hold it the numbers after them. A subset that holds ball $x$, with $k$
balls still to place, therefore skips past all $\binom{x}{k}$ subsets that leave it out. A subset
that leaves the ball out skips nothing. So the rank is one addition per ball held:

$$\mathrm{rank} = \binom{x_1}{1} + \binom{x_2}{2} + \dots + \binom{x_k}{k} \quad \text{for } x_1 < x_2 < \dots < x_k$$

Here is the whole numbering for the subsets of 3 balls out of five:

```text
rank     0    1    2    3    4    5    6    7    8    9
subset  012  013  023  123  014  024  124  034  134  234
        |-- leave out 4 -|  |--------- hold 4 ---------|
```

Take `124`. Its balls in rising order are 1, 2 and 4, so its rank is
$\binom{1}{1} + \binom{2}{2} + \binom{4}{3} = 1 + 1 + 4 = 6$. An entry past the edge of the triangle
counts as zero, so `012` gets $\binom{0}{1} + \binom{1}{2} + \binom{2}{3} = 0$.

This order is called [colexicographic
order](https://en.wikipedia.org/wiki/Lexicographic_order#Colexicographic_order), *colex* for short:
subsets sort by their largest ball first, then by the next largest, and so on. The formula is the
[combinatorial number system](https://en.wikipedia.org/wiki/Combinatorial_number_system), which
guarantees that every position in a table belongs to exactly one subset. The code keeps the part of
the triangle the formula needs, rows 0 to 90 and columns 0 to 5, in a table called `BINOMIAL`, built
with Pascal's rule at compile time.

A draw needs 26 ranks, but they share most of their terms. In the sorted draw $a < b < c < d < e$,
only 14 terms ever occur:

$$\begin{array}{lccccc} \text{column 1} & \binom{a}{1} & \binom{b}{1} & \binom{c}{1} & \binom{d}{1} & \cr \text{column 2} & & \binom{b}{2} & \binom{c}{2} & \binom{d}{2} & \binom{e}{2} \cr \text{column 3} & & & \binom{c}{3} & \binom{d}{3} & \binom{e}{3} \cr \text{column 4} & & & & \binom{d}{4} & \binom{e}{4} \cr \text{column 5} & & & & & \binom{e}{5} \end{array}$$

Column 1 is the balls themselves, since $\binom{x}{1} = x$. So a draw reads ten entries of
`BINOMIAL` once each, and `Ticket::ranks` builds all 26 ranks by adding them up.

### How large is each table, and how is it laid out?

A table has one counter per possible subset, so its length is entry $k$ of row 90 of the triangle:
4,005 pairs, 117,480 triples, 2,555,190 quads and 43,949,268 quints.

The rank formula never mentions 90. A subset gets the same rank however many balls exist, which
gives each table a useful layout. The subsets with the same largest ball sit next to each other, in
a block called a *run*. The run of ball $x$ starts at rank $\binom{x}{k}$ and holds
$\binom{x}{k - 1}$ subsets. In the numbering above the runs are `012`, then `013 023 123`, then the
six subsets holding ball 4. Run lengths are a column of the triangle, and adding them up gives the
table's size, a result called the
[hockey-stick identity](https://en.wikipedia.org/wiki/Hockey-stick_identity). For the quints:

$$\binom{4}{4} + \binom{5}{4} + \dots + \binom{89}{4} = \binom{90}{5}$$

A run is also a complete smaller table once its largest ball is removed. The run of ball 4 above,
without the 4, reads `01 02 12 03 13 23`, which is all the pairs of balls 0 to 3 in rank order.
[Deriving the smaller tables](#deriving-the-smaller-tables) relies on this: it can add whole blocks
of one table to another, because each block is itself a table.

## Data structures

**A ticket** (`ticket.rs`) is `Ticket([u8; 5])`: five different balls, sorted, counted from 0. The
only way to make one is `Ticket::validated`, which checks the range, sorts the five numbers and
rejects repeats, so the rest of the program can trust every ticket. Sorting uses a fixed
[sorting network](https://en.wikipedia.org/wiki/Sorting_network) of nine comparisons. Its steps are
the same for every ticket, so several tickets can be sorted at once.

**A rank** (`ticket.rs`, from the binomial table in `combinatorics.rs`) is computed with the formula
above. The sample file's first player holds `4 79 13 80 56`. Sorted and counted from 0 that is `3 12
55 78 79`, and its rank among the 43,949,268 quints is:

$$\begin{aligned} & \binom{3}{1} + \binom{12}{2} + \binom{55}{3} + \binom{78}{4} + \binom{79}{5} \cr & \quad = 3 + 66 + 26{,}235 + 1{,}426{,}425 + 22{,}537{,}515 = 23{,}990{,}244 \end{aligned}$$

Its pair $\lbrace 3, 12 \rbrace$ has rank $3 + \binom{12}{2} = 69$, and its quad
$\lbrace 3, 12, 55, 78 \rbrace$ has rank $3 + 66 + 26{,}235 + 1{,}426{,}425 = 1{,}452{,}729$.

Ranks have one more useful property: subsets close together in a table share their largest balls.
So cutting the quint table into equal slices by rank also groups tickets by their largest balls,
and the load can treat each slice on its own.

**The count tables** (`engine.rs`) are four plain lists indexed by rank. Entry $r$ of a table holds
how many players' tickets contain the subset of rank $r$.

| Subset size $k$ | Entries $\binom{90}{k}$ | Counter | Memory | Mean count | Highest count |
|---|---|---|---|---:|---:|
| 5 (quints) | 43,949,268 | 8 bits | 41.9 MiB | 0.23 | 6 |
| 4 (quads) | 2,555,190 | 32 bits | 9.7 MiB | 20 | 44 |
| 3 (triples) | 117,480 | 32 bits | 0.4 MiB | 851 | 983 |
| 2 (pairs) | 4,005 | 32 bits | 16 KiB | 24,969 | 25,523 |

The last two columns are the counters of the sample file, `10m-v2.txt`.

**Why 8 bits for the quints and 32 for the rest.** An 8-bit counter takes a quarter of the memory of
a 32-bit one but only reaches 255. Saving memory matters only in a large table, and the largest
table happens to have the smallest counts. Ten million tickets are ten million quints spread over 44
million counters, so four counters in five are zero and the highest is 6. The same tickets hold 100
million pairs spread over 4,005 counters, so every pair counter is near 25,000.

A narrow counter still needs a way out, because any counter could reach the number of players if
everybody played one ticket. For the quints that way out is the overflow map below. It works because
going past 255 takes that many players with the very same ticket. No counter in the sample file
does, and even at the limit of 1.4 billion players the mean is 33.

The same way out fails for the smaller subsets. Every triple and pair counter in this file is above
255, so the map would hold the whole table, and a draw would make 20 map lookups instead of reading
an array. Sixteen bits would hold this file's triples and pairs, but the pairs outgrow them near 26
million players. The quads would fit in 8 bits here, but their mean reaches 255 near 130 million
players, and narrowing them could save at most 7.3 MiB and touches about 4% of the derivation's
reads, so that was not tried. The three smaller tables stay at 32 bits for a file of usual size.

**Why very many players get 64-bit counters.** Deriving the smaller tables adds up to three times
a count (see [Deriving the smaller tables](#deriving-the-smaller-tables)), so 32-bit counters hold
every count only up to `u32::MAX / 3`, about 1.4 billion, valid players. The width matters only to
the pair, triple and quad tables, which are derived after the whole file is counted, so the load
picks it then, from the exact number of valid players. Past the limit those tables get 64-bit
counters. That doubles them to about 20 MiB, and made a draw 20% slower, and 41% slower for draws
spread over the whole table, which is why 32 bits are kept whenever they suffice. The width is
chosen once: the program holds one copy of the draw loop for each width, so a draw never checks
which one it has.

64-bit counters have limits of their own:

- **The draw.** A draw solves its equations in signed 64-bit numbers, and the largest term reaches
  30 times the player count when every player holds the drawn ticket. That allows `i64::MAX / 30`,
  about 307 quadrillion, valid players.
- **The quint counters.** These stay 8 bits at both widths. Past about 11 billion players,
  $255 \binom{90}{5}$, the average quint counter reaches 255, and the overflow map below grows
  toward one entry per quint, about 1 GiB (derived, not measured). The counts stay exact; only the
  memory grows. Such a file holds at least 110 GB.

**The overflow map.** A ticket held by more than 255 players keeps its counter at the maximum, and
the rest of its count sits in a small map from rank to count. A draw consults the map only when it
reads a counter at the maximum. During counting, counters simply wrap from 255 back to 0 and each
wrap is noted, which is faster than checking first. The notes become the map's counts once the file
is done. The overflow is added to the quads of its quint before the triples and pairs are derived,
so the smaller tables are exact too.

**The staging area** is where ranks wait between being parsed and being counted. Counting directly
would make all workers write all over one 42 MiB table, so the workers would slow each other down.
Instead the quint table is cut into 336 slices of $2^{17}$ counters, 128 KiB each (the last one is
shorter), and each slice has one *owner*, the only task allowed to write it. A rank's owner is its
rank divided by $2^{17}$. Each chunk of the file gets a fixed region of 52,429 entries of type `u32`
(`chunk_tickets`), the most valid lines a 512 KiB chunk can hold. A worker fills it with the chunk's
ranks grouped by owner, and a 337-entry list says where each owner's group starts. Regions are
allocated for one batch of 512 chunks at a time, 107 MB at most, and reused for the next batch.

**Parsing buffers.** Each active parsing task has a 516 KiB read buffer, a chunk plus the byte
before it and 4,097 bytes past its end, and one temporary list, `RankCounts`, holding the chunk's
ranks. A task reuses its buffers for its chunks, and a worker can run several tasks. The draw reader
keeps a 64 KiB read buffer and one partial line of at most 4,097 bytes.

**Memory allocation** goes through [jemalloc](https://jemalloc.net), a replacement for the system's
default allocator. It took READY from 39.4 ms to 35.2 ms at 32 workers compared with
glibc's allocator. Draws do not allocate memory, so it does not affect them.

## Loading the file

The load has to be fast, and must not need the whole file in memory. So the file is cut into chunks
of 512 KiB that several CPU cores handle at once. The workers are a
[Rayon](https://docs.rs/rayon) thread pool, sized to the available CPUs, at most 32, or to the
`--threads` option. Each batch of 512 chunks goes through four steps:

1. **Read a chunk** (`input.rs`). A worker reads its chunk with one read at an explicit position in
   the file (`pread`), so workers never share a file position. Lines rarely end exactly at a chunk's
   end, so a line belongs to the chunk holding its first byte. To tell, the worker also reads the
   byte before the chunk: if it is not a newline, the chunk starts mid-line, and the worker skips to
   the first newline. It also reads 4,097 bytes past the end, which completes the chunk's last line.
   A line longer than 4,096 bytes is cut there and rejected for its length, the same rule as for any
   line. Nothing is kept between chunks.
2. **Decode and rank** (`simd.rs`, `decode.rs`, `ticket.rs`). This turns text into quint ranks, and
   it is the busiest step, so it uses
   [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions, which
   process many bytes in one CPU step. The program checks at run time what the CPU offers:
   - With [AVX-512](https://en.wikipedia.org/wiki/AVX-512) (its F, BW, VBMI and VBMI2 parts), POPCNT
     and LZCNT, `simd.rs` handles 64 bytes, about five lines, per step, and validates, sorts and
     ranks 16 tickets at a time.
   - Otherwise, with SSE4.1 in the build, `decode.rs` works line by line.
     [SSE2](https://en.wikipedia.org/wiki/SSE2) code loads 16 bytes of a line and marks its digits,
     spaces and newline in a mask. A few checks of that mask reject any other format, and the five
     numbers come out of it with no `if` per digit or per number, because a wrong guess about an
     `if` costs a modern CPU time. Four tickets at a time are then validated, sorted and ranked
     together.

   Division is slow, so both decoders avoid it in the rank: dividing by 2 or 4 is a bit shift, and
   dividing exactly by 3 or 5 is a multiplication with a
   [modular inverse](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse).

   Both fast paths handle only the file's usual format: five numbers of one or two digits with
   single spaces. Anything else, and a line too close to either end of the chunk for a vector load,
   goes through the plain parser, which also accepts extra whitespace, CRLF line ends, leading zeros
   and plus signs. A decoded group that holds an invalid ticket is checked again one ticket at a
   time with `Ticket::validated`. The plain parser defines what is correct, and the tests and the
   fuzz target check that all paths agree on every input. Outside the tests, the two decoders hold
   the crate's only `unsafe` blocks: their vector loads and stores and their calls into
   target-feature code. Each valid ticket ends as one quint rank.
3. **Stage.** The worker groups the chunk's ranks by owner into the chunk's staging region with a
   [counting sort](https://en.wikipedia.org/wiki/Counting_sort), two passes over the ranks, and
   notes where each group starts.
4. **Count** (`engine.rs`). Once the batch is parsed, each owner adds its groups from every chunk to
   its slice of the quint table. A slice is 128 KiB, small enough to stay in the
   [CPU cache](https://en.wikipedia.org/wiki/CPU_cache), which is much faster than main memory. No
   two owners write the same counter, so they never have to wait for each other. Each batch adds to
   the counts the batches before it left, so after the last batch the table holds the whole file.
   A counter that wraps past 255 is noted; checking for the maximum before each store instead cost
   16 ms of READY.

The line counts of each chunk are added up as batches finish. The load stops with an error as soon
as the valid players pass the limit of 64-bit counters, about 307 quadrillion, so a count is never
silently wrong. Past the 32-bit limit it goes on, and the tables are derived with 64-bit counters
(see [Data structures](#data-structures)).

**Invalid lines.** An invalid line is ignored. With `--log-level warn` or more, a warning in the log
file also gives its line number and contents. That takes extra work, because the parsers keep
neither of those: a line that fails validation in a fast path has lost its bytes by then, and chunks
are parsed in no particular order. So while the owners count a batch, one worker reads each chunk
that had an invalid line again and scans it up to the last such line. The line counts of the chunks
before it give the line numbers, and the warnings come out in file order. With logging off, the
default, none of this runs, and READY stayed within noise of a build with logging compiled out
(35.2 ms for both on the sample file). `warn` adds about 0.3 ms for the sample's 6 invalid lines;
scanning on the loading thread before counting added 0.7 ms.

## Deriving the smaller tables

The load only counts quints. The other tables could be filled the same way, with 26 additions per
player, but that work would grow with the number of players. Instead `streaming.rs` computes each
smaller table from the next larger one, so this stage takes the same time for any number of players:
the quads from the quints, the triples from the quads and the pairs from the triples.

**The arithmetic.** How many players hold the quad `1 2 3 4`? Every such player holds exactly one
quint that contains it, so the answer is the sum of the counts of the quints `1 2 3 4 x`, over every
other ball `x`. For triples the sum counts each player twice, because a ticket holding `1 2 3` holds
two quads containing it, so the sum is divided by 2. For pairs, each player is counted three times,
so the sum is divided by 3. In general, for a smaller subset built from subsets of size $k$, the
divisor is $6 - k$.

**Why it is fast.** Done one counter at a time, those sums would jump around the table. The rank
order turns them into additions of whole blocks, which a CPU does very quickly. Take the output
subsets with largest ball $d$, which are one block of the output. The balls that can be added to
them are of two kinds:

- **A ball above the largest.** Adding a ball $m$ above $d$ gives subsets that lie in the run of $m$
  in the larger table, next to each other, and in the same order as the output block. So that run
  gives one block addition. In the five-ball example, the pairs with largest ball 2 (`02 12`, ranks
  1 and 2) get the triples `023 123` from the run of 3 and `024 124` from the run of 4, each a block
  of two entries.
- **A ball below the largest.** The subsets that keep $d$ as their largest ball are the run of $d$.
  With $d$ removed, that run is a complete smaller table, as shown earlier, so this is the same
  derivation one size smaller. In the example, the pairs `02 12` also get the triple `012`: removing
  the 2 leaves the pair `01`, deriving it one size down spreads its count over the singles `0` and
  `1`, and putting the 2 back gives `02` and `12`.

The second kind repeats on smaller and smaller tables until only single balls are left, each of
which contains the empty subset.

**The cost.** Every quint counter is read five times in total, once per quad it contains, so
deriving all three tables reads about 230 million counters when every number occurs in the file.
The work does not depend on the number of players.

**Using several cores.** The output blocks do not overlap, so they can be computed side by side.
They are split further by the second-largest ball into parallel tasks, a few thousand for the quads.
With one task per block the whole stage waited for the largest block, 5% of the work, which cost 1.5
ms. Inside a task the work is sequential, and adding a run is one loop that the compiler turns into
SIMD code.

## Answering a draw

A draw looks up the 26 counters of its subsets, adds them into the totals $A_2$ to $A_5$, and solves
the equations from [The theory](#why-do-subset-counts-give-the-answer):

$$\begin{aligned} E_5 &= A_5 \cr E_4 &= A_4 - 5 A_5 \cr E_3 &= A_3 - 4 A_4 + 10 A_5 \cr E_2 &= A_2 - 3 A_3 + 6 A_4 - 10 A_5 \end{aligned}$$

That takes no search and no hashing, except one map lookup when the quint counter is at its maximum.
The arithmetic uses signed 64-bit numbers, because the subtractions happen before everything cancels
out.

For example, a player who shares exactly three numbers with the draw appears once in the draw's
triples and three times in its pairs, and not at all in its quads or quint. So $E_3$ gains 1 and
$E_2$ gains $3 - 3 \cdot 1 = 0$. Every player lands in exactly one category this way.

## The protocol loop

`serve` in `protocol.rs`, called from `main.rs`, prints READY and then reads stdin into one 64 KiB
buffer. Every complete line in a read is parsed and answered on the spot. Answers collect in an
8 KiB buffer, which is written when it fills and after every read, before the next read can wait for
input. So a caller sending one draw at a time gets each answer at once, while draws sent in one
write share a few writes, which saves system calls.

Answers are formatted into a fixed 84-byte buffer without Rust's general formatting code,
`core::fmt`, which does more than four whole numbers need. A draw allocates no memory.

An invalid draw gets no answer. With warnings on, its warning is written to the log file on this
thread, which a file cannot stall the way a full stderr pipe can.

## Compared with scanning a bit matrix

The obvious solution keeps one row of 90 bits per player, with a one for each number on the ticket.
For a draw it selects the five columns of the drawn numbers and adds them up row by row. Each row's
sum is that player's match count:

```text
ball          1  2  3  4  5  ...  90
player 1      1  0  1  0  0  ...   0
player 2      0  1  1  0  1  ...   0
...
draw mask     1  1  0  0  1  ...   0    selects 5 of the 90 columns
```

That solution is correct and far shorter: it needs no ranks, no equations and no derivation. Its
cost is that every draw visits every player. The count tables do that work once, during the load, so
a draw no longer depends on how many players there are:

| | Bit matrix | Count tables |
|---|---|---|
| Load | Grows with the file | Grows with the file, plus a fixed table build |
| Data held for 10 million players | 107 MiB, growing with the players | 52 MiB, fixed |
| Draw | Grows with the players | Fixed |
| Data read per draw | 6.25 MB, five columns of one bit per player | 26 counters, 101 bytes |
| Draw time, in process | 16 to 24 ms | 22 ns |

[Computational complexity](../README.md#computational-complexity) in the README gives the
asymptotic run time and memory of both solutions.
