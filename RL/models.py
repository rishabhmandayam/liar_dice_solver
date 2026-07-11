import flax.linen as nn
import jax.numpy as jnp

class DQN(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class SLPolicy(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

def forward_rl(params, state):
    """
    Forward pass for RL Q-network (DQN).
    params: Flax parameter dict
    state: 1D array representing game state info tensor
    """
    # Extract action_dim dynamically from the final dense layer's bias parameters
    action_dim = params["params"]["Dense_2"]["bias"].shape[0]
    return DQN(action_dim=action_dim).apply(params, state)

def forward_sl(params, state):
    """
    Forward pass for SL Policy network.
    params: Flax parameter dict
    state: 1D array representing game state info tensor
    """
    action_dim = params["params"]["Dense_2"]["bias"].shape[0]
    return SLPolicy(action_dim=action_dim).apply(params, state)
