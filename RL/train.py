import jax
import jax.numpy as jnp
import optax
from .models import forward_rl, forward_sl

@jax.jit(static_argnums=3)
def train_rl_step(rl_params, target_rl_params, opt_state, tx, batch):
    states, actions, rewards, next_states, dones, next_masks = batch

    def loss_fn(params):
        q_values = jax.vmap(forward_rl, in_axes=(None, 0))(params, states)
        chosen_q = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(axis=-1)

        next_q = jax.vmap(forward_rl, in_axes=(None, 0))(target_rl_params, next_states)
        # Mask out invalid next actions
        next_q_masked = next_q + (1.0 - next_masks) * -1e9
        max_next_q = jnp.max(next_q_masked, axis=-1)

        target_q = rewards + 0.99 * max_next_q * (1.0 - dones)

        return jnp.mean((target_q - chosen_q) ** 2)

    l,g = jax.value_and_grad(loss_fn)(rl_params)
    updates, next_opt_state = tx.update(g, opt_state)
    next_rl_params = optax.apply_updates(rl_params, updates)

    return next_rl_params, next_opt_state, l

@jax.jit(static_argnums=2)
def train_sl_step(sl_params, opt_state, tx, batch):
    states, target_actions = batch
    
    def loss_fn(params):
        logits = jax.vmap(forward_sl, in_axes=(None, 0))(params, states)
        
        one_hot_targets = jax.nn.one_hot(target_actions, logits.shape[-1])
        return -jnp.mean(jnp.sum(one_hot_targets * jax.nn.log_softmax(logits), axis=-1))
    
    l, g = jax.value_and_grad(loss_fn)(sl_params)
    updates, next_opt_state = tx.update(g, opt_state)
    next_sl_params = optax.apply_updates(sl_params, updates)

    return next_sl_params, next_opt_state, l
    