# Connect Four: Alpha-Beta vs. Monte Carlo Tree Search

Two Connect Four AIs play 30 games against each other, with 15 seconds per move:

- **Alpha-beta:** looks ahead as deep as it can in time and scores boards with a custom heuristic
- **Monte Carlo tree search (MCTS):** plays thousands of random games to pick the best move

**Result:** MCTS won 21 games and alpha-beta won 9. The full write-up is in [report.md](report.md), and the raw results are in `results.txt`.

## Run it

This uses the [AIMA Python](https://github.com/aimacode/aima-python) code from the textbook.

1. Download AIMA Python into this folder:

   ```bash
   git clone https://github.com/aimacode/aima-python.git
   ```

2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run it (PowerShell):

   ```powershell
   $env:PYTHONPATH = "aima-python"; python main.py
   ```
