import unittest
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys

# Add repository root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game import GameState
from RL.run_nfsp import NFSPAgent

class TestNFSP(unittest.TestCase):
    def setUp(self):
        self.p1_dice = 1
        self.p2_dice = 1
        dummy_game = GameState(self.p1_dice, self.p2_dice)
        self.state_dim = dummy_game.get_information_tensor().shape[0]
        self.action_dim = dummy_game.get_action_mask().shape[0]
        self.agent = NFSPAgent(state_dim=self.state_dim, action_dim=self.action_dim, lr=1e-3, memory_capacity=100)

    def test_action_masking_dqn(self):
        state = np.zeros(self.state_dim)
        action_mask = np.zeros(self.action_dim)
        # Only actions 3 and 5 are valid
        action_mask[3] = 1.0
        action_mask[5] = 1.0
        
        self.agent.eta = 1.0
        self.agent.epsilon = 0.0 # Exploit to test DQN policy
        
        for _ in range(50):
            action, policy = self.agent.select_action(state, action_mask)
            self.assertEqual(policy, "RL")
            self.assertTrue(action in [3, 5])

    def test_action_masking_sl(self):
        state = np.zeros(self.state_dim)
        action_mask = np.zeros(self.action_dim)
        # Only actions 2 and 7 are valid
        action_mask[2] = 1.0
        action_mask[7] = 1.0
        
        self.agent.eta = 0.0 # Force SL average policy selection
        
        for _ in range(50):
            action, policy = self.agent.select_action(state, action_mask)
            self.assertEqual(policy, "SL")
            self.assertTrue(action in [2, 7])

    def test_training_updates(self):
        # Populate memory with fake transitions
        state = np.random.randn(self.state_dim).astype(np.float32)
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        for _ in range(16):
            self.agent.memory.add_rl(state, 1, 1.0, next_state, False, np.ones(self.action_dim, dtype=np.float32))
            self.agent.memory.add_sl(state, 2)
            
        # Run updates
        rl_loss = self.agent.update_rl(batch_size=4)
        sl_loss = self.agent.update_sl(batch_size=4)
        
        self.assertIsInstance(rl_loss, float)
        self.assertIsInstance(sl_loss, float)
        
    def test_msgpack_serialization(self):
        from flax import serialization
        serialized = serialization.to_bytes(self.agent.sl_params)
        
        # Load back
        from RL.models import SLPolicy
        sl_policy = SLPolicy(action_dim=self.action_dim)
        empty_params = sl_policy.init(jax.random.PRNGKey(0), jnp.zeros((self.state_dim,)))
        
        loaded_params = serialization.from_bytes(empty_params, serialized)
        self.assertEqual(loaded_params["params"].keys(), self.agent.sl_params["params"].keys())

if __name__ == '__main__':
    unittest.main()
