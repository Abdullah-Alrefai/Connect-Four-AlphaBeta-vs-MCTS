# Alpha-Beta Pruning vs. Monte Carlo Tree Search at Connect Four

## Setup
- **Code.** The program (`main.py`) is built on AIMA Python, fourth edition (`aima/games.py`). It uses `ConnectFour()` with default settings. In AIMA this board is 7 rows by 6 columns, with 4 in a row needed to win.
- **Alpha-beta agent.** Adapted from `alpha_beta_cutoff_search`:
  - It uses iterative deepening: it runs the search with cutoff d = 1, 2, 3, …
  - Each search checks `time.perf_counter()` and stops as soon as the 15-second deadline passes.
  - The agent returns the move from the deepest search that finished.
  - The first legal move is kept as a fallback, so a legal move is always returned.
  - AIMA's cutoff test (`depth > d`) and move order are unchanged. A cutoff of d looks d + 1 moves ahead.
  - Deepening stops early once the search already reaches every empty square.
- **MCTS agent.** Adapted from `monte_carlo_tree_search`:
  - It runs simulations until the 15-second deadline instead of a fixed N.
  - It returns the move of the most-visited child.
  - AIMA's selection (UCB), expansion, random playout and backpropagation are unchanged.
- **Games.** 30 games were played:
  - Alpha-beta moved first in games 1–15, and MCTS moved first in games 16–30.
  - Each game's random seed equals its game number and is set with `random.seed(seed)`.
- **Machine and limit.** Every game ran on the same laptop with the same 15-second limit per move.
  - Hardware: Intel i7-1260P, 15.7 GB RAM, Windows 11.
  - Software: Python 3.13.5, numpy 2.1.3.
  - To shorten the total run time, 6 games were played at the same time, in separate processes.
- **Memory cleanup.** Python's automatic garbage collection was turned off during moves and run between moves, outside the timed part. Testing showed a cleanup pause could push an MCTS move to 15.15 s, or land inside the opponent's turn.

## Heuristic evaluation function
AIMA's `ConnectFour` utility is 0 for every unfinished position. So alpha-beta scores positions at the search cutoff with the function `evaluate(state, player)`. Every possible line of 4 squares on the board (69 lines: horizontal, vertical, and both diagonals) is checked.

| Feature | Score | Why it indicates a favorable or unfavorable position |
|---|---|---|
| Finished game | utility × 1,000,000 | An actual win (+) or loss (−) matters more than any other feature. A draw scores 0. |
| Line with 3 of my pieces and 1 empty square | +50 each | One move from winning, so it is an immediate threat the opponent must answer. |
| Line with 2 of my pieces and 2 empty squares | +10 each | A line that can still become a threat later. |
| Same patterns for the opponent | −50 / −10 each | The opponent's threats are dangers for me. |
| Line holding pieces of both players | 0 | Blocked: neither player can win with this line anymore. |
| Piece in a center column (columns 3–4) | +3 mine, −3 opponent's | Center squares belong to more lines of 4 than edge squares, so they give more ways to win. |

## Results

**Totals (30 games)**
- Alpha-beta wins: **9**
- MCTS wins: **21**
- Draws: **0**

**Results by which method moved first**

| Moved first | Games | Alpha-beta wins | MCTS wins | Draws |
|---|---|---|---|---|
| Alpha-beta | 15 | 9 | 6 | 0 |
| MCTS | 15 | 0 | 15 | 0 |

**Move time and search effort**

| Method | Moves | Mean move time | Search effort |
|---|---|---|---|
| Alpha-beta | 462 | 10.618 s | Mean deepest completed cutoff d = 10.75 per move (about 11.75 moves ahead) |
| MCTS | 468 | 15.000 s | Mean of 160,007 simulations per move |

- **Why alpha-beta's mean time is below 15 s.** Late in a game, its search reached every empty square before time ran out, so it stopped early.
- **The same late-game moves raise its mean depth.** Mid-game searches were shallower than the 10.75 average.
- **MCTS always used the full 15 s,** as required.

**Game record (seeds for reproducing the MCTS trials)**

| Game | Seed | Moved first | Winner |
|---|---|---|---|
| 1 | 1 | alpha-beta | MCTS |
| 2 | 2 | alpha-beta | alpha-beta |
| 3 | 3 | alpha-beta | MCTS |
| 4 | 4 | alpha-beta | alpha-beta |
| 5 | 5 | alpha-beta | MCTS |
| 6 | 6 | alpha-beta | alpha-beta |
| 7 | 7 | alpha-beta | alpha-beta |
| 8 | 8 | alpha-beta | MCTS |
| 9 | 9 | alpha-beta | MCTS |
| 10 | 10 | alpha-beta | alpha-beta |
| 11 | 11 | alpha-beta | alpha-beta |
| 12 | 12 | alpha-beta | alpha-beta |
| 13 | 13 | alpha-beta | alpha-beta |
| 14 | 14 | alpha-beta | alpha-beta |
| 15 | 15 | alpha-beta | MCTS |
| 16–30 | 16–30 | MCTS | MCTS (all 15) |

## Which method won more often
**MCTS won more often: 21 games to 9, with no draws.**
- When MCTS moved first, it won all 15 games.
- When alpha-beta moved first, alpha-beta won 9 of 15.

Alpha-beta won only when it had the first-move advantage, and even then MCTS won 6 of those 15 games.

## Limitations
- **The alpha-beta evaluation function.**
  - Alpha-beta can only be as good as its hand-made heuristic. The weights (50, 10, 3) were chosen by reasoning, not tuned.
  - The function does not recognize deeper Connect Four ideas. For example, it doesn't consider which player can actually fill a threat square first, given that pieces drop from the bottom.
  - A won or lost position always scores ±1,000,000, however many moves away it is. So alpha-beta does not prefer a quicker win, and once it sees a forced loss, every move looks equally bad.
  - A different heuristic could change the result.
- **Random MCTS simulations.**
  - MCTS judges moves by purely random playouts.
  - Random play is a noisy and sometimes misleading estimate of a position's value, especially for short tactical threats that a sensible player would block.
  - Its strength here comes from running about 160,000 simulations per move. On a slower machine, or with a shorter time limit, it would get fewer simulations and its play would likely weaken.
- **Move ordering.**
  - Alpha-beta tried moves in AIMA's default order, row by row across the board, not best moves first.
  - Pruning works best when strong moves are searched first. With this order, fewer branches were pruned, so the search reached less depth in 15 seconds than it could have.
  - Ordering by center columns first, or by the previous iteration's best move, would likely let alpha-beta search deeper.
- **Number of games.**
  - 30 games, 15 per first player, is a small sample.
  - The two halves differ sharply (alpha-beta won 9 of the 15 games it started, and none of the 15 MCTS started), so the first-move effect is large and is estimated from only 15 games each.
  - Alpha-beta's play is deterministic, and the only randomness comes from MCTS. The games may therefore follow similar lines and be less independent than 30 separate trials.
- **Running 6 games at once.**
  - Games shared the laptop's CPU, so each search may have been somewhat slower than it would be on an idle machine.
  - Both players in a game ran under the same conditions, and every game used the same setup.
  - Because MCTS stops at a time limit rather than a fixed count, rerunning a seed reproduces the random number sequence. The exact number of simulations, and possibly the moves, can still differ with machine load.
