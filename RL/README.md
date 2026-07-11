# Reinforcement Learning (RL) & Neural Fictitious Self-Play (NFSP) in Liar's Dice

This directory contains the core implementation of the Reinforcement Learning (RL) and Supervised Learning (SL) components for a Neural Fictitious Self-Play (NFSP) agent to solve Liar's Dice.

---

## 1. How Neural Fictitious Self-Play (NFSP) Works

NFSP is a deep reinforcement learning algorithm designed to learn Nash equilibria in imperfect information games (like Liar's Dice) through self-play. It combines two types of learning:

1. **Reinforcement Learning (DQN / Best Response)**:
   - The agent learns a **Best Response** policy ($\Pi^{RL}$) to the historic behavior of other players.
   - It trains using a Deep Q-Network (DQN) to maximize expected return (payoff) against the current opponents.
   - Experiences $(s, a, r, s', done)$ are stored in the **RL Buffer**.

2. **Supervised Learning (Average Policy)**:
   - The agent learns an **Average Policy** ($\Pi^{SL}$) by copying its own historical best responses.
   - It trains using behavioral cloning (supervised learning) on the states and actions chosen by the RL policy.
   - Experiences $(s, a)$ are stored in the **SL Buffer**.

### Action Selection Strategy
During gameplay / self-play simulation, the agent selects actions using a mixture of both policies:
- With probability $\eta$ (anticipatory parameter, e.g., $0.1$), the agent plays according to the **RL Best Response policy** (and stores this decision in the SL buffer to train the average policy).
- With probability $1 - \eta$, the agent plays according to the **SL Average Policy** (behavioral cloning network).

---

## 2. Understanding the Components

### A. Memory Buffer ([buffer.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/RL/buffer.py))
The `Memory` class manages two separate replay buffers in memory using NumPy arrays:
- **RL Buffer**: Stores transitions for Deep Q-Learning:
  - `rl_states`: The information tensors representing the game state from the player's perspective.
  - `rl_actions`: Integer IDs representing the chosen action.
  - `rl_rewards`: Payoffs/rewards received after transitions.
  - `rl_next_states`: The information tensors of the next states.
  - `rl_dones`: Boolean indicators indicating whether the game reached a terminal state.
- **SL Buffer**: Stores states and actions chosen by the RL policy to train the average policy network:
  - `sl_states`: The information tensors.
  - `sl_actions`: Integer IDs of chosen actions.

> [!WARNING]
> **Known Typos/Bugs in [buffer.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/RL/buffer.py)**:
> 1. **Line 24**: `np.zeroes` is used instead of `np.zeros`.
> 2. **Line 48 (`sample_rl`)**: `self.rl_size` is used, but the class attribute is named `self.size_rl`.
> 3. **Line 58 (`sample_sl`)**: `self.sl_size` is used, but the class attribute is named `self.size_sl`.
> 4. **Imports**: Unused imports (`from jax.lax import dtype`, `from jax._src import abstract_arrays`, and `from .. import game`) are present.

### B. Training Steps ([train.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/RL/train.py))
The training script implements single training updates for both the RL and SL networks using **JAX** and **Optax**:

1. **`train_rl_step(...)`**:
   - Takes the current RL network parameters, target network parameters, optimizer state, Optax optimizer, and a batch from the RL buffer.
   - Computes the TD loss:
     $$\mathcal{L}_{RL} = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s'; \theta^{target}) - Q(s, a; \theta) \right)^2 \right]$$
   - Computes gradients using `jax.value_and_grad` and updates the parameters using the Optax optimizer.

2. **`train_sl_step(...)`**:
   - Takes the current SL network parameters, optimizer state, Optax optimizer, and a batch from the SL buffer.
   - Computes the cross-entropy loss between the predicted logits and target actions:
     $$\mathcal{L}_{SL} = -\mathbb{E} \left[ \log \Pi^{SL}(a | s) \right]$$
   - Computes gradients and updates the parameters to align the average policy with the historical best responses.

---

## 3. Game Representations ([game.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/game.py))
To feed state information to the RL/SL networks, `game.py` provides the state and action representations:

- **State Representation**:
  - [GameState.get_information_tensor()](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/game.py#L133-L156) returns a 1D `jnp.ndarray` representing the information set.
  - The vector contains:
    1. **Hand Counts**: One-hot/count representation of dice in the current player's hand (length 6).
    2. **Current Bid**: One-hot representation of the active bid (length `max_actions`).
    3. **Meta Features**: A 3-dimensional vector `[dice_p1, dice_p2, len(history)]` representing game context.
  - **State Dimension**: $6 + (1 + 6 \times \text{total\_dice}) + 3$. For a 1v1 game with 2 dice total, the state dimension is $6 + 13 + 3 = 22$.

- **Action Masking**:
  - [GameState.get_action_mask()](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/game.py#L74-L87) returns a binary mask vector of length `max_actions`. Invalid actions are marked as `0.0` and valid actions as `1.0`.
  - When evaluating action probabilities or Q-values, invalid actions should be masked (e.g., set Q-values of invalid actions to $-\infty$, or apply the mask to SL logits).

---

## 4. The Missing `forward_rl` and `forward_sl` Functions

To run the training steps in [train.py](file:///Users/rishabhmandayam/Documents/GitHub/liar_dice_solver/RL/train.py), we need to define how inputs are passed through the RL and SL models. `forward_rl` and `forward_sl` represent the forward passes of these models.

### Example Network Architectures
Here is an example of how you can define these networks and functions using **Flax** or **Haiku** (or simple manual JAX parameter computations).

#### Option A: Simple JAX Multi-Layer Perceptron (Manual Parameters)
If you want to keep the implementation dependency-free:

```python
import jax
import jax.numpy as jnp

def forward_rl(params, state):
    # params is a dictionary of weights and biases: {'w1': ..., 'b1': ..., 'w2': ..., 'b2': ...}
    x = state
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return x # Outputs Q-values for all actions

def forward_sl(params, state):
    # params is a dictionary of weights and biases: {'w1': ..., 'b1': ..., 'w2': ..., 'b2': ...}
    x = state
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return x # Outputs logits for all actions
```

#### Option B: Using Flax (Recommended for JAX Projects)
```python
import flax.linen as nn

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

# Instantiation and forward function mapping:
# dqn = DQN(action_dim=max_actions)
# forward_rl = lambda params, state: dqn.apply(params, state)
```

---

## 5. Integrating and Running NFSP

To run training, you would implement an outer loop in a runner script (e.g. `RL/run_nfsp.py`) that:
1. Initializes game parameters (`n_dice_p1`, `n_dice_p2`).
2. Creates the `Memory` buffer with the calculated state dimension.
3. Initializes parameters `rl_params`, `target_rl_params`, `sl_params`, and their corresponding Optax optimizers.
4. Executes self-play matches:
   - At each step, determines player's strategy.
   - Collects actions and transitions.
   - Pushes transitions to the `Memory` buffer.
5. Periodically samples batches from the buffer and calls `train_rl_step` and `train_sl_step` to update model parameters.
