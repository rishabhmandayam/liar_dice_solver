import csv
import os
from typing import Dict, Tuple

def get_strategy_filename(n_dice_p1: int, n_dice_p2: int) -> str:
    """Returns the filename for the strategy table based on dice counts."""
    return f"strategy_{n_dice_p1}v{n_dice_p2}.csv"

def save_strategy(strategy: Dict[str, Dict[str, float]], n_dice_p1: int, n_dice_p2: int):
    """
    Saves the strategy table to a CSV file.
    Format: InfoSet, Action, Probability
    """
    filename = get_strategy_filename(n_dice_p1, n_dice_p2)
    print(f"Saving strategy to {filename}...")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["InfoSet", "Action", "Probability"])
        
        for info_set, actions in strategy.items():
            for action_str, prob in actions.items():
                writer.writerow([info_set, action_str, prob])
    
    print("Save complete.")

def load_strategy(n_dice_p1: int, n_dice_p2: int) -> Dict[str, Dict[str, float]]:
    """
    Loads the strategy table from a CSV file.
    Returns a dictionary mapping InfoSet -> {Action -> Probability}
    """
    filename = get_strategy_filename(n_dice_p1, n_dice_p2)
    strategy = {}
    
    if not os.path.exists(filename):
        print(f"Warning: Strategy file {filename} not found.")
        return strategy

    print(f"Loading strategy from {filename}...")
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            info_set = row["InfoSet"]
            action = row["Action"]
            prob = float(row["Probability"])
            
            if info_set not in strategy:
                strategy[info_set] = {}
            strategy[info_set][action] = prob
            
    print("Load complete.")
    return strategy

def action_to_str(action: Tuple[int, int]) -> str:
    """Converts action tuple to string format 'Q-F' or 'Challenge'."""
    if action == (-1, -1):
        return "Challenge"
    return f"{action[0]}-{action[1]}"

def str_to_action(action_str: str) -> Tuple[int, int]:
    """Converts string format back to action tuple."""
    if action_str == "Challenge":
        return (-1, -1)
    parts = action_str.split('-')
    return (int(parts[0]), int(parts[1]))
