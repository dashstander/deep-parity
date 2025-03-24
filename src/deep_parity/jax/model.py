import equinox as eqx
from functools import partial
import jax
from jax.nn import relu


class Perceptron(eqx.Module):
    linear: eqx.Module
    unembed: eqx.Module

    def __init__(self, n: int, model_dim: int, key):
        linear_key, unembed_key = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_features=n, out_features=model_dim, key=linear_key)
        self.unembed = eqx.nn.Linear(in_features=model_dim, out_features=2, use_bias=False, key=unembed_key)
    
    @partial(jax.vmap, in_axes=(None, 0))
    def __call__(self, x):
        preactivations = self.linear(x)
        outputs = self.unembed(relu(preactivations))
        return outputs
