import unittest
from game import GameState

class TestGameState(unittest.TestCase):
    def test_initial_state(self):
        game = GameState(2, 2)
        self.assertEqual(len(game.hand_p1), 2)
        self.assertEqual(len(game.hand_p2), 2)
        self.assertIsNone(game.current_bid)
        self.assertEqual(game.current_player, 0)

    def test_valid_actions_start(self):
        game = GameState(1, 1)
        actions = game.get_valid_actions()
        # Total dice = 2. Max quantity = 2. Faces = 6.
        # Valid actions: (1,1)..(1,6), (2,1)..(2,6) -> 12 actions.
        self.assertEqual(len(actions), 12)
        self.assertIn((1, 1), actions)
        self.assertIn((2, 6), actions)

    def test_valid_actions_after_bid(self):
        game = GameState(1, 1)
        game.apply_action((1, 3)) # Bid 1 3s
        actions = game.get_valid_actions()
        
        # Valid:
        # Challenge (-1, -1)
        # Raise Face: (1, 4), (1, 5), (1, 6)
        # Raise Quantity: (2, 1)..(2, 6)
        
        self.assertIn((-1, -1), actions)
        self.assertIn((1, 4), actions)
        self.assertIn((2, 1), actions)
        self.assertNotIn((1, 2), actions) # Lower face, same quantity
        self.assertNotIn((1, 3), actions) # Same bid

    def test_challenge_logic(self):
        game = GameState(1, 1)
        # Mock hands for deterministic test
        game.hand_p1 = [2]
        game.hand_p2 = [5]
        
        # P1 bids 1 2s (True)
        game.apply_action((1, 2))
        
        # P2 challenges
        is_terminal = game.apply_action((-1, -1))
        self.assertTrue(is_terminal)
        
        # P2 challenged. P1 was truthful (there is one 2).
        # Bidder (P1) wins. Challenger (P2) loses.
        # get_payoff returns payoff for Challenger (P2). Should be -1.
        self.assertEqual(game.get_payoff(), -1.0)

    def test_bluff_logic(self):
        game = GameState(1, 1)
        game.hand_p1 = [2]
        game.hand_p2 = [5]
        
        # P1 bids 1 6s (Lie)
        game.apply_action((1, 6))
        
        # P2 challenges
        game.apply_action((-1, -1))
        
        # P1 lied. Challenger (P2) wins. Payoff +1.
        self.assertEqual(game.get_payoff(), 1.0)

if __name__ == '__main__':
    unittest.main()
