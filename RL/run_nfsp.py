import os
import sys
import argparse
import random
import datetime
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

# Add parent directory to sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, id_to_action, action_to_id
from utils import action_to_str, str_to_action
from RL.buffer import Memory
from RL.models import DQN, SLPolicy, forward_rl, forward_sl
from RL.train import train_rl_step, train_sl_step

class NFSPAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, 
                 memory_capacity: int = 100_000, eta: float = 0.1, 
                 epsilon_start: float = 0.1, epsilon_end: float = 0.01, 
                 epsilon_decay: int = 20000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_counter = 0

        # Memory buffers
        self.memory = Memory(state_dim, action_dim, capacity=memory_capacity)

        # Networks
        self.dqn = DQN(action_dim=action_dim)
        self.sl_policy = SLPolicy(action_dim=action_dim)

        # Optimizer
        self.tx = optax.adam(learning_rate=lr)

        # Keys
        self.key = jax.random.PRNGKey(random.randint(0, 1000000))
        self.key, key_rl, key_sl = jax.random.split(self.key, 3)

        # Init parameters
        dummy_state = jnp.zeros((state_dim,))
        self.rl_params = self.dqn.init(key_rl, dummy_state)
        self.target_rl_params = self.dqn.init(key_rl, dummy_state)
        self.sl_params = self.sl_policy.init(key_sl, dummy_state)

        # Opt states
        self.rl_opt_state = self.tx.init(self.rl_params)
        self.sl_opt_state = self.tx.init(self.sl_params)

    def select_action(self, state, action_mask, is_evaluation=False):
        self.step_counter += 1
        
        # Decaying epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay
        )

        if is_evaluation:
            # Always play average policy during evaluation
            logits = forward_sl(self.sl_params, state)
            # Mask invalid actions
            logits = logits + (1.0 - action_mask) * -1e9
            action = int(jnp.argmax(logits))
            return action, "SL"

        # Roll mixture decision
        use_rl = (random.random() < self.eta)
        
        if use_rl:
            # Play DQN best response
            if random.random() < self.epsilon:
                # Explore: select a random valid action
                valid_indices = np.where(action_mask == 1.0)[0]
                action = int(np.random.choice(valid_indices))
            else:
                # Exploit: argmax Q-values
                q_values = forward_rl(self.rl_params, state)
                q_values = q_values + (1.0 - action_mask) * -1e9
                action = int(jnp.argmax(q_values))
            return action, "RL"
        else:
            # Play SL average policy
            logits = forward_sl(self.sl_params, state)
            logits = logits + (1.0 - action_mask) * -1e9
            probs = jax.nn.softmax(logits)
            
            probs = np.array(probs)
            probs[action_mask == 0.0] = 0.0
            sum_probs = probs.sum()
            if sum_probs > 0:
                probs /= sum_probs
            else:
                probs = action_mask / action_mask.sum()
            
            action = int(np.random.choice(self.action_dim, p=probs))
            return action, "SL"

    def update_rl(self, batch_size: int) -> float:
        if self.memory.size_rl < batch_size:
            return 0.0
        batch = self.memory.sample_rl(batch_size)
        batch = tuple(jnp.array(x) for x in batch)
        self.rl_params, self.rl_opt_state, loss = train_rl_step(
            self.rl_params, self.target_rl_params, self.rl_opt_state, self.tx, batch
        )
        return float(loss)

    def update_sl(self, batch_size: int) -> float:
        if self.memory.size_sl < batch_size:
            return 0.0
        batch = self.memory.sample_sl(batch_size)
        batch = tuple(jnp.array(x) for x in batch)
        self.sl_params, self.sl_opt_state, loss = train_sl_step(
            self.sl_params, self.sl_opt_state, self.tx, batch
        )
        return float(loss)

    def sync_target(self):
        self.target_rl_params = self.rl_params

def play_episode(agent: NFSPAgent, p1_dice: int, p2_dice: int):
    game = GameState(p1_dice, p2_dice)
    
    # Track the last (state, action) for each player
    prev_states = [None, None]
    prev_actions = [None, None]
    
    while True:
        player = game.current_player
        state = game.get_information_tensor()
        action_mask = game.get_action_mask()
        
        # Select action
        action_id, policy_used = agent.select_action(state, action_mask)
        
        # If policy used is RL, store state-action pair in SL buffer
        if policy_used == "RL":
            agent.memory.add_sl(state, action_id)
            
        action = id_to_action(action_id)
        
        # If this player had a previous state and action, add transition to RL buffer
        if prev_states[player] is not None:
            agent.memory.add_rl(
                prev_states[player],
                prev_actions[player],
                0.0,
                state,
                False,
                action_mask
            )
            
        # Store current state and action as previous
        prev_states[player] = state
        prev_actions[player] = action_id
        
        # Apply action
        is_terminal = game.apply_action(action)
        
        if is_terminal:
            payoff = game.get_payoff()
            
            # The player who just challenged gets payoff
            agent.memory.add_rl(
                prev_states[player],
                prev_actions[player],
                payoff,
                np.zeros(agent.state_dim),
                True,
                np.zeros(agent.action_dim)
            )
            
            # The opponent gets -payoff
            opponent = 1 - player
            if prev_states[opponent] is not None:
                agent.memory.add_rl(
                    prev_states[opponent],
                    prev_actions[opponent],
                    -payoff,
                    np.zeros(agent.state_dim),
                    True,
                    np.zeros(agent.action_dim)
                )
            break

def train(p1_dice: int, p2_dice: int, episodes: int, lr: float, eta: float,
          batch_size: int, train_every: int, sync_every: int, save_path: str):
    
    # Calculate state and action dimensions
    dummy_game = GameState(p1_dice, p2_dice)
    state_dim = dummy_game.get_information_tensor().shape[0]
    total_dice = p1_dice + p2_dice
    action_dim = 1 + (total_dice * 6) # max_actions in game.py
    
    print(f"Initializing NFSPAgent: state_dim={state_dim}, action_dim={action_dim}")
    agent = NFSPAgent(state_dim=state_dim, action_dim=action_dim, lr=lr, eta=eta)
    
    print(f"Starting training for {episodes} episodes...")
    for ep in range(1, episodes + 1):
        play_episode(agent, p1_dice, p2_dice)
        
        # Run updates
        if ep % train_every == 0:
            rl_loss = agent.update_rl(batch_size)
            sl_loss = agent.update_sl(batch_size)
            
        # Sync target network
        if ep % sync_every == 0:
            agent.sync_target()
            
        # Logging progress
        if ep % 1000 == 0 or ep == episodes:
            print(f"Episode {ep}/{episodes} | RL Loss: {rl_loss:.5f} | SL Loss: {sl_loss:.5f} | Epsilon: {agent.epsilon:.3f}")
            
    # Serialize SL Policy parameters using Flax Native Msgpack
    print(f"Serializing SL policy parameters...")
    serialized_bytes = serialization.to_bytes(agent.sl_params)
    
    # Resolve the timestamped directory inside the base directory
    base_dir = os.path.dirname(os.path.abspath(save_path))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{p1_dice}v{p2_dice}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save policy parameters
    policy_file = os.path.join(run_dir, "nfsp_policy.msgpack")
    with open(policy_file, "wb") as f:
        f.write(serialized_bytes)
    print(f"Successfully saved SL policy parameters to: {policy_file}")

    # Save hyperparameters as a YAML file
    hyperparams = {
        "p1_dice": p1_dice,
        "p2_dice": p2_dice,
        "episodes": episodes,
        "lr": lr,
        "eta": eta,
        "batch_size": batch_size,
        "train_every": train_every,
        "sync_every": sync_every,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    yaml_file = os.path.join(run_dir, "model_export.yaml")
    with open(yaml_file, "w") as f:
        yaml.safe_dump(hyperparams, f, default_flow_style=False)
    print(f"Successfully saved hyperparameters to: {yaml_file}")

def play(p1_dice: int, p2_dice: int, model_path: str):
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Please train first.")
        return
        
    # Reconstruct dimensions
    dummy_game = GameState(p1_dice, p2_dice)
    state_dim = dummy_game.get_information_tensor().shape[0]
    total_dice = p1_dice + p2_dice
    action_dim = 1 + (total_dice * 6)
    
    # Load parameters
    sl_policy = SLPolicy(action_dim=action_dim)
    key = jax.random.PRNGKey(0)
    dummy_state = jnp.zeros((state_dim,))
    empty_params = sl_policy.init(key, dummy_state)
    
    with open(model_path, "rb") as f:
        sl_params = serialization.from_bytes(empty_params, f.read())
        
    print(f"Successfully loaded SL policy from {model_path}")
    print(f"Starting game against NFSP Bot! You have {p1_dice} dice. Bot has {p2_dice} dice.")
    
    game = GameState(p1_dice, p2_dice)
    print(f"You rolled: {game.hand_p1}. Bot has rolled {p2_dice} dice.")
    
    while True:
        print(f"\nCurrent Bid: {game.current_bid if game.current_bid else 'None'}")
        print(f"History: {[action_to_str(a) for a in game.history]}")
        
        if game.current_player == 0:
            # Human turn
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
            # NFSP Bot turn
            state = game.get_information_tensor()
            action_mask = game.get_action_mask()
            
            # Select best action from Average Policy (SL) model
            logits = forward_sl(sl_params, state)
            logits = logits + (1.0 - action_mask) * -1e9
            action_id = int(jnp.argmax(logits))
            
            action = id_to_action(action_id)
            print(f"NFSP Bot chooses: {action_to_str(action)}")
            is_terminal = game.apply_action(action)
            
        if is_terminal:
            payoff = game.get_payoff()
            print("\n--- Game Over ---")
            print(f"Your hand: {game.hand_p1}")
            print(f"Bot's hand: {game.hand_p2}")
            
            # Game terminal state was triggered by the player who just acted.
            # If current_player is 0, the human made the final action (Challenge).
            if game.current_player == 0:
                if payoff > 0:
                    print("You challenged and WON! Bot was lying.")
                else:
                    print("You challenged and LOST! Bot was telling the truth.")
            else:
                if payoff > 0:
                    print("Bot challenged and WON! You were lying.")
                else:
                    print("Bot challenged and LOST! You were telling the truth.")
            break

def main():
    parser = argparse.ArgumentParser(description="NFSP Trainer & Player for Liar's Dice")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the NFSP model")
    train_parser.add_argument("--p1", type=int, default=1, help="P1 dice count")
    train_parser.add_argument("--p2", type=int, default=1, help="P2 dice count")
    train_parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--eta", type=float, default=0.1, help="NFSP anticipatory parameter (mix ratio)")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for training updates")
    train_parser.add_argument("--train-every", type=int, default=1, help="Episodes between network updates")
    train_parser.add_argument("--sync-every", type=int, default=100, help="Episodes between target network updates")
    train_parser.add_argument("--save-path", type=str, default="artifacts/nfsp_policy.msgpack", help="Where to save SL policy parameters")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play against a trained NFSP bot")
    play_parser.add_argument("--p1", type=int, default=1, help="P1 dice count")
    play_parser.add_argument("--p2", type=int, default=1, help="P2 dice count")
    play_parser.add_argument("--model-path", type=str, default="artifacts/nfsp_policy.msgpack", help="Path to loaded SL policy parameters")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            p1_dice=args.p1,
            p2_dice=args.p2,
            episodes=args.episodes,
            lr=args.lr,
            eta=args.eta,
            batch_size=args.batch_size,
            train_every=args.train_every,
            sync_every=args.sync_every,
            save_path=args.save_path
        )
    elif args.command == "play":
        play(
            p1_dice=args.p1,
            p2_dice=args.p2,
            model_path=args.model_path
        )

if __name__ == "__main__":
    main()
