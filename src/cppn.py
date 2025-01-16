from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int
    nonlin: str = 'tanh' # use tanh or relu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.d_hidden)(x)
            x = getattr(nn, self.nonlin)(x)
        x = nn.Dense(3)(x)
        return jax.nn.sigmoid(x)

    def generate_image(self, params, img_size=128):
        x = y = jnp.linspace(-1, 1, img_size)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        d = jnp.sqrt(x**2 + y**2)
        xyd = jnp.stack([x, y, d], axis=-1)
        rgb = jax.vmap(jax.vmap(partial(self.apply, params)))(xyd)
        return rgb


import evosax
class FlattenCPPNParameters():
    def __init__(self, cppn):
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((3,))))
        self.n_params = self.param_reshaper.total_params
    
    def init(self, rng, x):
        params = self.cppn.init(rng, x)
        return self.param_reshaper.flatten_single(params)

    def generate_image(self, params, img_size=128):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, img_size=img_size)
    