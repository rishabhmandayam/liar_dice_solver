import numpy as np 
import typing


class Memory():
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.action_dim = action_dim
        self.ptr_rl = 0
        self.size_rl = 0

        #Rl Allocations
        self.rl_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rl_actions = np.zeros(capacity, dtype=np.int32)
        self.rl_rewards = np.zeros(capacity, dtype=np.float32)
        self.rl_next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rl_dones = np.zeros(capacity, dtype=np.bool_)
        self.rl_next_masks = np.zeros((capacity, action_dim), dtype=np.float32)

        #SL Allocations
        self.ptr_sl = 0
        self.size_sl = 0
        self.sl_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.sl_actions = np.zeros(capacity, dtype=np.int32)
    
    def add_rl(self, state, action, reward, next_state, done, next_action_mask):
        idx = self.ptr_rl % self.capacity
        self.rl_states[idx] = state
        self.rl_actions[idx] = action
        self.rl_rewards[idx] = reward
        self.rl_next_states[idx] = next_state
        self.rl_dones[idx] = done
        self.rl_next_masks[idx] = next_action_mask

        self.ptr_rl += 1
        self.size_rl = min(self.size_rl + 1, self.capacity)

    def add_sl(self, state, action):
        idx = self.ptr_sl % self.capacity
        self.sl_states[idx] = state
        self.sl_actions[idx] = action

        self.ptr_sl += 1
        self.size_sl = min(self.size_sl + 1, self.capacity)

    
    def sample_rl(self, batch_size: int):
        idxs = np.random.choice(self.size_rl, batch_size, replace=False)
        return (
            self.rl_states[idxs],
            self.rl_actions[idxs],
            self.rl_rewards[idxs],
            self.rl_next_states[idxs],
            self.rl_dones[idxs],
            self.rl_next_masks[idxs]
        )
    
    def sample_sl(self, batch_size: int):
        idxs = np.random.choice(self.size_sl, batch_size, replace=False)

        return (
            self.sl_states[idxs],
            self.sl_actions[idxs]
        )






    