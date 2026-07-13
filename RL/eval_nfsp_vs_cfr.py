import os
import sys
import random
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

# Add parent directory to sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, id_to_action
from utils import load_strategy, str_to_action, action_to_str
from RL.models import SLPolicy, forward_sl

def play_match(p1_dice: int, p2_dice: int, cfr_strategy: dict, sl_params: dict, nfsp_player: int) -> float:
    """
    Simulates a single game between NFSP and CFR.
    Returns NFSP's payoff (+1.0 for win, -1.0 for loss).
    """
    game = GameState(p1_dice, p2_dice)
    
    while True:
        player = game.current_player
        
        if player == nfsp_player:
            # NFSP Agent Turn
            state = game.get_information_tensor()
            action_mask = game.get_action_mask()
            
            # Select best action from Average Policy (SL) model
            logits = forward_sl(sl_params, state)
            logits = logits + (1.0 - action_mask) * -1e9
            action_id = int(jnp.argmax(logits))
            action = id_to_action(action_id)
        else:
            # CFR Agent Turn
            info_set = game.get_information_set()
            if info_set in cfr_strategy:
                action_probs = cfr_strategy[info_set]
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action_str = random.choices(actions, weights=probs)[0]
                action = str_to_action(action_str)
            else:
                # Fallback to random valid action
                action = random.choice(game.get_valid_actions())
                
        is_terminal = game.apply_action(action)
        
        if is_terminal:
            payoff = game.get_payoff()
            # payoff is for the challenger (game.current_player)
            if game.current_player == nfsp_player:
                return payoff
            else:
                return -payoff

def main():
    parser = argparse.ArgumentParser(description="Evaluate NFSP Bot vs CFR Bot")
    parser.add_argument("--model-path", type=str, required=True, help="Path to loaded SL policy msgpack parameters")
    parser.add_argument("--p1", type=int, default=1, help="P1 dice count")
    parser.add_argument("--p2", type=int, default=1, help="P2 dice count")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of games to simulate per seating position")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        return
        
    print(f"Loading CFR strategy for {args.p1}v{args.p2}...")
    cfr_strategy = load_strategy(args.p1, args.p2)
    if not cfr_strategy:
        print(f"No CFR strategy found for {args.p1}v{args.p2}. Please train the CFR bot first.")
        return
        
    print(f"Loading NFSP policy parameters from {args.model_path}...")
    dummy_game = GameState(args.p1, args.p2)
    state_dim = dummy_game.get_information_tensor().shape[0]
    total_dice = args.p1 + args.p2
    action_dim = 1 + (total_dice * 6)
    
    sl_policy = SLPolicy(action_dim=action_dim)
    key = jax.random.PRNGKey(0)
    dummy_state = jnp.zeros((state_dim,))
    empty_params = sl_policy.init(key, dummy_state)
    
    with open(args.model_path, "rb") as f:
        sl_params = serialization.from_bytes(empty_params, f.read())
        
    print(f"Loaded successfully. Evaluating {args.episodes * 2} games (balanced positions)...")
    
    # Position A: NFSP is Player 0 (goes first), CFR is Player 1
    nfsp_p0_wins = 0
    for _ in range(args.episodes):
        payoff = play_match(args.p1, args.p2, cfr_strategy, sl_params, nfsp_player=0)
        if payoff > 0:
            nfsp_p0_wins += 1
            
    # Position B: CFR is Player 0 (goes first), NFSP is Player 1
    nfsp_p1_wins = 0
    for _ in range(args.episodes):
        payoff = play_match(args.p1, args.p2, cfr_strategy, sl_params, nfsp_player=1)
        if payoff > 0:
            nfsp_p1_wins += 1
            
    p0_rate = (nfsp_p0_wins / args.episodes) * 100
    p1_rate = (nfsp_p1_wins / args.episodes) * 100
    avg_rate = ((nfsp_p0_wins + nfsp_p1_wins) / (args.episodes * 2)) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"NFSP as Player 1 (first turn) Win Rate: {p0_rate:.2f}% ({nfsp_p0_wins}/{args.episodes})")
    print(f"NFSP as Player 2 (second turn) Win Rate: {p1_rate:.2f}% ({nfsp_p1_wins}/{args.episodes})")
    print(f"Overall NFSP Win Rate: {avg_rate:.2f}% ({nfsp_p0_wins + nfsp_p1_wins}/{args.episodes * 2})")

if __name__ == "__main__":
    main()
