
import random
from typing import List, Tuple, Optional
import jax.numpy as jnp

# Constants
DICE_FACES = 6

def action_to_id(action: Tuple[int, int], max_dice: int) -> int:
        """
        map (quantity, face) -> a singlular integer
        """
        if action == (-1, -1):
            return 0
        q,f = action
        return (q - 1) * 6 + f

def id_to_action(action_id: int) -> Tuple[int, int]:
    """
    map id -> (quantity, face)
    """
    if action_id == 0:
        return (-1, -1)
    q = ((action_id - 1) // 6) + 1
    f = ((action_id - 1) % 6) + 1
    return q, f
class GameState:
    def __init__(self, dice_p1: int, dice_p2: int):
        self.dice_p1 = dice_p1
        self.dice_p2 = dice_p2
        self.hand_p1 = []
        self.hand_p2 = []
        self.current_bid = None  # (quantity, face)
        self.history = [] # List of bids
        self.current_player = 0 # 0 for P1, 1 for P2
        self.roll_dice()

    def roll_dice(self):
        self.hand_p1 = sorted([random.randint(1, DICE_FACES) for _ in range(self.dice_p1)])
        self.hand_p2 = sorted([random.randint(1, DICE_FACES) for _ in range(self.dice_p2)])


    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of valid actions.
        Action format: (quantity, face)
        Special action: (-1, -1) represents 'Challenge' (Liar)
        """
        actions = []
        # If no bid has been made, any valid bid is allowed.
        if self.current_bid is None:
            total_dice = self.dice_p1 + self.dice_p2
            for q in range(1, total_dice + 1):
                for f in range(1, DICE_FACES + 1):
                    actions.append((q, f))
            return actions

        curr_q, curr_f = self.current_bid
        total_dice = self.dice_p1 + self.dice_p2

        # 1. Challenge is always valid after the first bid
        actions.append((-1, -1))

        # 2. Raise face (same quantity, higher face)
        for f in range(curr_f + 1, DICE_FACES + 1):
            actions.append((curr_q, f))

        # 3. Raise quantity (higher quantity, any face)
        for q in range(curr_q + 1, total_dice + 1):
            for f in range(1, DICE_FACES + 1):
                actions.append((q, f))
        return actions

    def get_action_mask(self) -> jnp.ndarray:
        """
        Returns a binary mask for valid actions.
        Action format: (quantity, face)
        Special action: (-1, -1) represents 'Challenge' (Liar)
        """
        total_dice = self.dice_p1 + self.dice_p2
        max_actions = 1 + (total_dice * DICE_FACES)
        mask = jnp.zeros((max_actions,), dtype=jnp.float32)

        for act in self.get_valid_actions():
            act_id = action_to_id(act, total_dice)
            mask = mask.at[act_id].set(1.0)
        return mask

    def apply_action(self, action: Tuple[int, int]):
        """
        Applies an action and transitions the state.
        Returns True if the game is over (terminal state), False otherwise.
        """
        if action == (-1, -1):
            return True # Terminal state (Challenge)

        self.current_bid = action
        self.history.append(action)
        self.current_player = 1 - self.current_player
        return False

    def get_payoff(self) -> float:
        """
        Returns the payoff for the player who made the LAST move (the challenger).
        Note: The 'current_player' is the one who just challenged.

        If Challenger wins, payoff is +1.
        If Challenger loses (Bidder wins), payoff is -1.

        The actual game rule: Loser loses a die.
        For CFR, we usually model win/loss as +1/-1 per hand.
        The outer loop handles dice removal.
        """
        if self.current_bid is None:
            return 0.0 # Should not happen

        bid_q, bid_f = self.current_bid

        # Count actual dice matching the bid
        count = 0
        all_dice = self.hand_p1 + self.hand_p2
        for d in all_dice:
            if d == bid_f:
                count += 1

        bidder_wins = (count >= bid_q)

        if bidder_wins:
            return -1.0
        else:
            return 1.0

    def get_information_tensor(self)->jnp.ndarray:
        """
        returns information set 
        """
        total_dice = self.dice_p1 + self.dice_p2
        max_actions = 1 + (total_dice * DICE_FACES)

        my_hand = self.hand_p1 if self.current_player == 0 else self.hand_p2
        hand_counts = jnp.zeros(DICE_FACES, dtype=jnp.float32)
        for die in my_hand:
            hand_counts = hand_counts.at[die - 1].add(1.0)
        
        bid_tensor = jnp.zeros(max_actions, dtype=jnp.float32)
        if self.current_bid is not None:
            bid_id = action_to_id(self.current_bid, total_dice)
            bid_tensor = bid_tensor.at[bid_id].set(1.0)
        
        meta_features = jnp.array([
            float(self.dice_p1),
            float(self.dice_p2),
            float(len(self.history))
        ], dtype=jnp.float32)
        
        return jnp.concatenate([hand_counts, bid_tensor, meta_features])

    def get_information_set(self) -> str:
        """
        Returns a string representation of the information set for the current player.
        Abstraction: (MyHand, CurrentBid, BidCount)
        """
        my_hand = self.hand_p1 if self.current_player == 0 else self.hand_p2
        hand_str = "".join(map(str, my_hand))

        bid_str = "None"
        if self.current_bid:
            bid_str = f"{self.current_bid[0]}-{self.current_bid[1]}"

        # Bid count is the number of bids made so far
        count_str = str(len(self.history))

        return f"{hand_str}|{bid_str}|{count_str}"
