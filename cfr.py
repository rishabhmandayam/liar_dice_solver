import random
from typing import Dict, List, Tuple
from game import GameState
from utils import action_to_str

class CFRNode:
    def __init__(self, num_actions: int):
        self.regret_sum = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions
        self.num_actions = num_actions

    def get_strategy(self, realization_weight: float) -> List[float]:
        """
        Returns the current strategy using Regret Matching.
        Also updates strategy_sum for the average strategy.
        """
        strategy = [0.0] * self.num_actions
        normalizing_sum = 0.0
        
        for i in range(self.num_actions):
            strategy[i] = max(self.regret_sum[i], 0)
            normalizing_sum += strategy[i]

        for i in range(self.num_actions):
            if normalizing_sum > 0:
                strategy[i] /= normalizing_sum
            else:
                strategy[i] = 1.0 / self.num_actions
            
            self.strategy_sum[i] += realization_weight * strategy[i]
            
        return strategy

    def get_average_strategy(self) -> List[float]:
        avg_strategy = [0.0] * self.num_actions
        normalizing_sum = sum(self.strategy_sum)
        
        for i in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum
            else:
                avg_strategy[i] = 1.0 / self.num_actions
        
        return avg_strategy

class CFRTrainer:
    def __init__(self, n_dice_p1: int, n_dice_p2: int):
        self.n_dice_p1 = n_dice_p1
        self.n_dice_p2 = n_dice_p2
        self.nodes: Dict[str, CFRNode] = {} # Map InfoSet -> CFRNode

    def train(self, iterations: int):
        """Runs MCCFR for a specified number of iterations."""
        print(f"Starting training for {self.n_dice_p1}v{self.n_dice_p2} with {iterations} iterations...")
        for i in range(iterations):
            if i % 1000 == 0:
                print(f"Iteration {i}/{iterations}")
            
            game = GameState(self.n_dice_p1, self.n_dice_p2)
            self.cfr(game, 1.0, 1.0)

    def cfr(self, game: GameState, p0_weight: float, p1_weight: float) -> float:
        """
        Recursive CFR function.
        Returns the utility for the current player.
        """
        # Get valid actions first to check if terminal
        # Note: In our game logic, we check terminal status via return value of apply_action
        # But here we need to know if we CAN move.
        # Actually, apply_action returns True if the move *caused* termination (Challenge).
        # So we are at a node. If the previous move was a challenge, we wouldn't be here (handled in loop).
        # Wait, the structure of recursion is:
        # 1. Check if terminal (not possible at start of function with current GameState structure, 
        #    because GameState doesn't store "Game Over" flag, it just returns it from apply_action).
        #    BUT, we can check if the last action was a challenge? No, GameState history stores it.
        #    Let's rely on the loop below.
        
        # Actually, standard CFR recursion:
        # We are at a node. We need to choose an action.
        
        player = game.current_player
        valid_actions = game.get_valid_actions()
        
        # If no valid actions (shouldn't happen in Liar's Dice unless limits reached), return 0
        if not valid_actions:
            return 0.0

        info_set = game.get_information_set()
        node = self.get_node(info_set, len(valid_actions))
        
        strategy = node.get_strategy(p0_weight if player == 0 else p1_weight)
        util = [0.0] * len(valid_actions)
        node_util = 0.0
        
        for i, action in enumerate(valid_actions):
            # Clone game state for recursion (or backtrack, but cloning is safer for Python prototype)
            # Since GameState is small, we can just create a new one and copy fields.
            # Or better, make GameState copyable.
            # For now, let's just manually copy relevant fields.
            next_game = GameState(game.dice_p1, game.dice_p2)
            next_game.hand_p1 = game.hand_p1[:]
            next_game.hand_p2 = game.hand_p2[:]
            next_game.current_bid = game.current_bid
            next_game.history = game.history[:]
            next_game.current_player = game.current_player
            
            is_terminal = next_game.apply_action(action)
            
            if is_terminal:
                # If terminal, get payoff for the player who made the move (current_player)
                # get_payoff returns value for the challenger.
                # Since we just made the move 'action', we are the challenger.
                # The payoff is for US.
                util[i] = next_game.get_payoff()
            else:
                # Recursion
                if player == 0:
                    util[i] = -self.cfr(next_game, p0_weight * strategy[i], p1_weight)
                else:
                    util[i] = -self.cfr(next_game, p0_weight, p1_weight * strategy[i])
            
            node_util += strategy[i] * util[i]
            
        # Regret Update
        for i in range(len(valid_actions)):
            regret = util[i] - node_util
            if player == 0:
                node.regret_sum[i] += p1_weight * regret
            else:
                node.regret_sum[i] += p0_weight * regret
                
        return node_util

    def get_node(self, info_set: str, num_actions: int) -> CFRNode:
        if info_set not in self.nodes:
            self.nodes[info_set] = CFRNode(num_actions)
        return self.nodes[info_set]

    def get_final_strategy(self) -> Dict[str, Dict[str, float]]:
        """
        Converts the learned nodes into a clean dictionary for export.
        """
        strategy_table = {}
        # We need to know the actions for each info_set to map back to strings.
        # This is a bit tricky since CFRNode only stores counts.
        # We can reconstruct valid actions from the info_set string?
        # InfoSet: "Hand|Bid|Count"
        # We can parse this, create a dummy GameState, and get_valid_actions.
        
        for info_set, node in self.nodes.items():
            avg_strat = node.get_average_strategy()
            
            # Reconstruct actions
            parts = info_set.split('|')
            # hand_str = parts[0]
            bid_str = parts[1]
            # count_str = parts[2]
            
            # Create dummy game to get actions
            # We only need current_bid to determine valid actions
            dummy_game = GameState(self.n_dice_p1, self.n_dice_p2)
            if bid_str != "None":
                b_parts = bid_str.split('-')
                dummy_game.current_bid = (int(b_parts[0]), int(b_parts[1]))
            else:
                dummy_game.current_bid = None
                
            valid_actions = dummy_game.get_valid_actions()
            
            action_probs = {}
            for i, action in enumerate(valid_actions):
                if avg_strat[i] > 0.001: # Filter negligible probabilities
                    action_probs[action_to_str(action)] = avg_strat[i]
            
            strategy_table[info_set] = action_probs
            
        return strategy_table
