import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from game import GameState
from cfr import CFRTrainer
from utils import load_strategy, save_strategy, action_to_str, str_to_action

def train_wrapper(args: Tuple[int, int, int]):
    """Wrapper for parallel execution."""
    p1, p2, iterations = args
    trainer = CFRTrainer(p1, p2)
    trainer.train(iterations)
    strategy = trainer.get_final_strategy()
    save_strategy(strategy, p1, p2)
    return f"Finished training {p1}v{p2}"

def train(p1: int, p2: int, iterations: int):
    train_wrapper((p1, p2, iterations))

def train_batch(max_dice: int, iterations: int):
    """Trains all subgames up to max_dice v max_dice in parallel."""
    configs = []
    for i in range(1, max_dice + 1):
        for j in range(1, max_dice + 1):
            configs.append((i, j, iterations))
    
    print(f"Starting batch training for {len(configs)} configurations using parallel processes...")
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(train_wrapper, configs)
        for res in results:
            print(res)
            
    print(f"Batch training complete in {time.time() - start_time:.2f}s")

def play(p1_dice: int, p2_dice: int):
    strategy = load_strategy(p1_dice, p2_dice)
    if not strategy:
        print("No strategy found. Please train first.")
        return

    print(f"Starting game {p1_dice}v{p2_dice} against Bot!")
    game = GameState(p1_dice, p2_dice)
    
    # Randomly decide who starts
    # game.current_player is 0 (P1) or 1 (P2). 
    # Let's say User is P1 (0) and Bot is P2 (1).
    
    print(f"You have {game.hand_p1}. Bot has {game.dice_p2} dice.")
    
    while True:
        print(f"\nCurrent Bid: {game.current_bid if game.current_bid else 'None'}")
        print(f"History: {[action_to_str(a) for a in game.history]}")
        
        if game.current_player == 0:
            # User turn
            valid_actions = game.get_valid_actions()
            print("Valid actions:")
            for i, a in enumerate(valid_actions):
                print(f"{i}: {action_to_str(a)}")
            
            while True:
                try:
                    choice = int(input("Enter action index: "))
                    if 0 <= choice < len(valid_actions):
                        action = valid_actions[choice]
                        break
                except ValueError:
                    pass
                print("Invalid choice.")
                
            print(f"You chose: {action_to_str(action)}")
            is_terminal = game.apply_action(action)
            
        else:
            # Bot turn
            info_set = game.get_information_set()
            # InfoSet for bot (P2) uses its own hand. GameState handles this.
            
            # Get strategy for this info set
            if info_set in strategy:
                action_probs = strategy[info_set]
                # Sample action
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action_str = random.choices(actions, weights=probs)[0]
                action = str_to_action(action_str)
            else:
                # Fallback: Random valid action
                print("Bot encountered unknown state. Playing random.")
                valid_actions = game.get_valid_actions()
                action = random.choice(valid_actions)
            
            print(f"Bot chooses: {action_to_str(action)}")
            is_terminal = game.apply_action(action)

        if is_terminal:
            # Game Over
            # get_payoff returns payoff for the CHALLENGER (current_player).
            # The last move was 'Challenge' by current_player.
            payoff = game.get_payoff()
            
            print("\n--- Game Over ---")
            print(f"Your hand: {game.hand_p1}")
            print(f"Bot's hand: {game.hand_p2}")
            
            # If User (0) challenged:
            if game.current_player == 0:
                if payoff > 0:
                    print("You challenged and WON! Bot was lying.")
                else:
                    print("You challenged and LOST! Bot was truthful.")
            else:
                # Bot (1) challenged
                if payoff > 0:
                    print("Bot challenged and WON! You were lying.")
                else:
                    print("Bot challenged and LOST! You were truthful.")
            break

def main():
    parser = argparse.ArgumentParser(description="Liar's Dice Bot")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the bot')
    train_parser.add_argument('p1', type=int, help='P1 dice count')
    train_parser.add_argument('p2', type=int, help='P2 dice count')
    train_parser.add_argument('--iter', type=int, default=10000, help='Iterations')

    # Train Batch command
    batch_parser = subparsers.add_parser('train-batch', help='Train all subgames up to N dice')
    batch_parser.add_argument('max_dice', type=int, help='Max dice count')
    batch_parser.add_argument('--iter', type=int, default=10000, help='Iterations')

    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the bot')
    play_parser.add_argument('p1', type=int, help='P1 dice count')
    play_parser.add_argument('p2', type=int, help='P2 dice count')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.p1, args.p2, args.iter)
    elif args.command == 'train-batch':
        train_batch(args.max_dice, args.iter)
    elif args.command == 'play':
        play(args.p1, args.p2)

if __name__ == "__main__":
    main()
