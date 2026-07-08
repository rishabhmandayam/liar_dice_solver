# Liar's Dice Solver

This repository implements a **Liar's Dice Solver** based on **Counterfactual Regret Minimization (CFR)** in Python.

---

## Getting Started

### 1. Setup Python Virtual Environment using `uv`

We use [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package and environment management.

To create and activate the virtual environment, run:

```bash
# Create the virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

---

## Python Implementation

The Python version implements the core rules, allows interactive play against the bot, and trains game strategies.

### Commands

Run `python3 main.py` with one of the following commands:

* **Train a Specific Configuration**:

    ```bash
    python3 main.py train <p1_dice> <p2_dice> --iter <iterations>
    ```

    *Example*: `python3 main.py train 1 1 --iter 10000`

* **Train Subgames in Parallel**:
    Trains all combinations up to `<max_dice>` in parallel using multiple processes:

    ```bash
    python3 main.py train-batch <max_dice> --iter <iterations>
    ```

    *Example*: `python3 main.py train-batch 2 --iter 10000`

* **Play Against the Solver Bot**:
    Starts an interactive terminal game against the bot using the pre-trained strategy CSV:

    ```bash
    python3 main.py play <p1_dice> <p2_dice>
    ```

    *Example*: `python3 main.py play 1 1`

### Running Tests

To run the Python game unit tests:

```bash
python3 -m unittest test_game.py
```

---

## Repository Structure

* [game.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/game.py): Liar's Dice rule implementation and state management.
* [cfr.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/cfr.py): Vanilla CFR trainer (External Sampling MCCFR).
* [main.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/main.py): CLI interface to train and play games.
* [utils.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/utils.py): Serialization helpers for CSV strategy tables.
* [test_game.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/test_game.py): Unit tests for validating the game mechanics.
