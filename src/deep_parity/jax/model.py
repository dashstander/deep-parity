import equinox as eqx
from functools import partial
import jax
from jax.nn import relu
import jax.numpy as jnp
import math


class Perceptron(eqx.Module):
    linear: eqx.Module
    unembed: eqx.Module

    def __init__(self, n: int, model_dim: int, key, use_bias=True):
        linear_key, unembed_key = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_features=n, out_features=model_dim, use_bias=use_bias, key=linear_key)
        self.unembed = eqx.nn.Linear(in_features=model_dim, out_features=2, use_bias=False, key=unembed_key)
    
    @partial(jax.vmap, in_axes=(None, 0))
    def __call__(self, x):
        preactivations = self.linear(x)
        outputs = self.unembed(relu(preactivations))
        return outputs
    

class SnPerceptron(eqx.Module):
    embed: eqx.Module
    linear: eqx.Module
    unembed: eqx.Module

    def __init__(self, n: int, embed_dim: int, model_dim: int, key, use_bias=True):
        embed_key, linear_key, unembed_key = jax.random.split(key, 3)
        n_group = math.factorial(n)
        self.embed = eqx.nn.Embedding(n_group, embedding_size=embed_dim, key=embed_key)
        self.linear = eqx.nn.Linear(in_features=(2 * embed_dim), out_features=model_dim, use_bias=use_bias, key=linear_key)
        self.unembed = eqx.nn.Linear(in_features=model_dim, out_features=n_group, use_bias=False, key=unembed_key)
    
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def __call__(self, sigma, tau):
        embed_sigma = self.embed(sigma)
        embed_tau = self.embed(tau)
        preactivations = self.linear(jnp.concat([embed_sigma, embed_tau], axis=0))
        outputs = self.unembed(relu(preactivations))
        return outputs
