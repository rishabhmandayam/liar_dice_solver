import random
from typing import List, Tuple, Optional

# Constants
DICE_FACES = 6

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
            return 0 # Should not happen

        bid_q, bid_f = self.current_bid
        
        # Count actual dice matching the bid
        # Note: 1s are NOT wild as per requirements
        count = 0
        all_dice = self.hand_p1 + self.hand_p2
        for d in all_dice:
            if d == bid_f:
                count += 1
        
        # Bidder wins if count >= bid_q
        bidder_wins = (count >= bid_q)
        
        # The player who called 'Challenge' is self.current_player
        # The player who made the bid is 1 - self.current_player
        
        if bidder_wins:
            # Bidder was correct. Challenger (current_player) loses.
            return -1.0
        else:
            # Bidder was lying. Challenger (current_player) wins.
            return 1.0

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
