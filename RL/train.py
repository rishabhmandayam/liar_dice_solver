import jax 
import jax.numpy as jnp
import optax

#TODO define forward_rll
def train_rl_step(rl_params, target_rl_params, opt_state, tx, batch):
    
    states, actions, rewards, next_states, dones = batch

    def loss(params):
        q_values = jax.vmap(forward_rl, in_axes=(None, 0))(params, states)