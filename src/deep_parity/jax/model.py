import equinox as eqx
import jax
from jax.nn import relu


class Perceptron(eqx.Module):
    n: eqx.AbstractVar[int]
    model_dim: eqx.AbstractVar[int]
    linear: eqx.Module
    unembed: eqx.Module

    def __init__(self, n: int, model_dim: int, key):
        linear_key, unembed_key = jax.random.split(key)
        self.n = n
        self.model_dim = model_dim
        self.linear = eqx.nn.Linear(in_features=n, out_features=model_dim, key=linear_key)
        self.unembed = eqx.nn.Linear(in_features=model_dim, out_features=1, key=unembed_key)
    
    def __call__(self, x):
        preactivations = self.linear(x)
        outputs = self.unembed(relu(preactivations))
        return outputs
