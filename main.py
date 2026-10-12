"""COSC 5600 Homework 05: alpha-beta pruning vs. Monte Carlo tree search at Connect Four.

Starting points (AIMA Python, fourth edition, aima/games.py):
alpha_beta_cutoff_search, monte_carlo_tree_search and ConnectFour.
"""

import gc
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean

import numpy as np
from aima.games import ConnectFour
from aima.utils import MCT_Node, ucb

TIME_LIMIT = 15      # maximum seconds each agent may spend choosing a move
NUM_GAMES = 30       # alpha-beta moves first in games 1-15, MCTS in games 16-30
PARALLEL_GAMES = 6   # games played at the same time

AB = 'alpha-beta'
MCTS = 'MCTS'

# AIMA's default ConnectFour: squares are (row, column), rows 1-7 (7 is the bottom), columns 1-6.
game = ConnectFour()


# ______________________________________________________________________________
# Heuristic evaluation function


def lines_of_four(game):
    """Every horizontal, vertical and diagonal line of k squares on the board."""
    lines = []
    for x in range(1, game.h + 1):
        for y in range(1, game.v + 1):
            for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
                line = [(x + i * dx, y + i * dy) for i in range(game.k)]
                if all(1 <= r <= game.h and 1 <= c <= game.v for r, c in line):
                    lines.append(line)
    return lines


LINES = lines_of_four(game)
CENTER_COLUMNS = (3, 4)


def evaluate(state, player):
    """Score state from player's point of view.
    Terminal states: AIMA's utility (+1 win, -1 loss, 0 draw) scaled to +-1,000,000.
    Nonterminal states:
      +50 / +10 for each line of four holding 3 / 2 of player's pieces and no opponent pieces,
      -50 / -10 for the same patterns of the opponent (lines holding both colors count 0),
      +3 / -3 for each of player's / the opponent's pieces in the center columns."""
    if game.terminal_test(state):
        return game.utility(state, player) * 1_000_000
    opponent = 'O' if player == 'X' else 'X'
    board = state.board
    score = 0
    for line in LINES:
        pieces = [board.get(square) for square in line]
        mine = pieces.count(player)
        theirs = pieces.count(opponent)
        if theirs == 0:
            if mine == 3:
                score += 50
            elif mine == 2:
                score += 10
        elif mine == 0:
            if theirs == 3:
                score -= 50
            elif theirs == 2:
                score -= 10
    for (_, y), piece in board.items():
        if y in CENTER_COLUMNS:
            score += 3 if piece == player else -3
    return score


# ______________________________________________________________________________
# Alpha-beta agent (adapted from alpha_beta_cutoff_search)


class SearchTimeout(Exception):
    """Raised when the move deadline passes in the middle of a search."""


def alpha_beta_time_search(state, game, d, deadline, cutoff_test=None, eval_fn=None):
    """AIMA's alpha_beta_cutoff_search with a deadline: raises SearchTimeout
    as soon as time.perf_counter() reaches the deadline."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_time_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: evaluate(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alpha_beta_player(state, deadline):
    """Iterative deepening: search with d = 1, 2, 3, ... until the deadline.
    Return the move from the deepest search that finished, and that depth d.
    The first legal move is kept as a fallback, so a legal move is always returned."""
    best_move = game.actions(state)[0]
    completed_depth = 0
    d = 1
    while True:
        try:
            best_move = alpha_beta_time_search(state, game, d, deadline)
        except SearchTimeout:
            break
        completed_depth = d
        # A search with cutoff d looks d + 1 moves ahead; stop once that covers every empty square.
        if d + 1 >= len(state.moves):
            break
        d += 1
    return best_move, completed_depth


# ______________________________________________________________________________
# MCTS agent (adapted from monte_carlo_tree_search)


def monte_carlo_time_search(state, game, deadline):
    """AIMA's monte_carlo_tree_search, running simulations until the deadline
    instead of N iterations. Return the move of the most-visited child and the
    number of simulations run."""
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)
    simulations = 0

    # Run at least one simulation so the root has children to choose from.
    while simulations == 0 or time.perf_counter() < deadline:
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)
        simulations += 1

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state), simulations


# ______________________________________________________________________________
# Playing the games


def play_game(game_number, seed, alphabeta_first, time_limit=TIME_LIMIT):
    """Play one game of alpha-beta vs. MCTS and return its result and search statistics."""
    random.seed(seed)
    # Python's automatic garbage collection can pause for ~0.6 s while freeing a large MCTS tree,
    # pushing a move past the time limit or into the opponent's turn. Collect between moves instead.
    gc.disable()
    method_of = {'X': AB, 'O': MCTS} if alphabeta_first else {'X': MCTS, 'O': AB}
    stats = {AB: {'times': [], 'depths': []},
             MCTS: {'times': [], 'simulations': []}}
    state = game.initial
    while not game.terminal_test(state):
        method = method_of[game.to_move(state)]
        start = time.perf_counter()
        deadline = start + time_limit
        if method == AB:
            move, depth = alpha_beta_player(state, deadline)
            stats[AB]['depths'].append(depth)
        else:
            move, simulations = monte_carlo_time_search(state, game, deadline)
            stats[MCTS]['simulations'].append(simulations)
        stats[method]['times'].append(time.perf_counter() - start)
        assert move in game.actions(state)
        state = game.result(state, move)
        gc.collect()  # untimed: free this move's search tree before the next move starts

    outcome = game.utility(state, 'X')
    winner = method_of['X'] if outcome > 0 else method_of['O'] if outcome < 0 else 'draw'
    return {'game': game_number, 'seed': seed, 'first': method_of['X'],
            'winner': winner, 'stats': stats}


def main():
    # (game number, seed, alpha-beta moves first?)
    games = [(n, n, n <= NUM_GAMES // 2) for n in range(1, NUM_GAMES + 1)]
    print(f"Playing {NUM_GAMES} games, {TIME_LIMIT} s per move, {PARALLEL_GAMES} games at a time.",
          flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=PARALLEL_GAMES) as pool:
        futures = [pool.submit(play_game, *g) for g in games]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"[{len(results)}/{NUM_GAMES} finished] Game {r['game']} | seed {r['seed']} | "
                  f"first: {r['first']} | winner: {r['winner']}", flush=True)
    results.sort(key=lambda r: r['game'])

    print("\nGame record (seed = random seed used for that game)")
    print(f"{'Game':>4}  {'Seed':>4}  {'Moved first':<11}  Winner")
    for r in results:
        print(f"{r['game']:>4}  {r['seed']:>4}  {r['first']:<11}  {r['winner']}")

    def wins(method, group):
        return sum(r['winner'] == method for r in group)

    ab_wins, mcts_wins, draws = wins(AB, results), wins(MCTS, results), wins('draw', results)
    print("\nTotals")
    print(f"  Alpha-beta wins: {ab_wins}")
    print(f"  MCTS wins:       {mcts_wins}")
    print(f"  Draws:           {draws}")

    print("\nResults by which method moved first")
    print(f"{'Moved first':<11}  {'Games':>5}  {'Alpha-beta wins':>15}  {'MCTS wins':>9}  {'Draws':>5}")
    for first in (AB, MCTS):
        group = [r for r in results if r['first'] == first]
        print(f"{first:<11}  {len(group):>5}  {wins(AB, group):>15}  "
              f"{wins(MCTS, group):>9}  {wins('draw', group):>5}")

    ab_times = [t for r in results for t in r['stats'][AB]['times']]
    ab_depths = [d for r in results for d in r['stats'][AB]['depths']]
    mcts_times = [t for r in results for t in r['stats'][MCTS]['times']]
    mcts_sims = [s for r in results for s in r['stats'][MCTS]['simulations']]
    print("\nMove time and search effort")
    print(f"  Alpha-beta: mean move time {mean(ab_times):.3f} s over {len(ab_times)} moves; "
          f"mean deepest completed depth d = {mean(ab_depths):.2f} (AIMA cutoff d looks d + 1 moves ahead)")
    print(f"  MCTS:       mean move time {mean(mcts_times):.3f} s over {len(mcts_times)} moves; "
          f"mean simulations per move = {mean(mcts_sims):,.0f}")

    if ab_wins > mcts_wins:
        print(f"\nAlpha-beta won more often ({ab_wins} to {mcts_wins}).")
    elif mcts_wins > ab_wins:
        print(f"\nMCTS won more often ({mcts_wins} to {ab_wins}).")
    else:
        print(f"\nNeither method won more often ({ab_wins} to {mcts_wins}).")

    print(f"\nALL {NUM_GAMES} GAMES FINISHED", flush=True)


if __name__ == "__main__":
    main()
