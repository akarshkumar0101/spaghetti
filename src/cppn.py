from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int
    nonlin: str = 'tanh' # use tanh or relu
    residual: bool = False
    layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        intermediate_features = [x]
        for i_layer in range(self.n_layers):
            xp = nn.Dense(self.d_hidden)(x)
            xp = getattr(nn, self.nonlin)(xp)
            if self.layernorm:
                xp = nn.LayerNorm()(xp)
            x = (x + xp) if (self.residual and i_layer>0) else xp
            intermediate_features.append(x)
        x = nn.Dense(3)(x)
        intermediate_features.append(x)
        rgb = jax.nn.sigmoid(x)
        return rgb, intermediate_features

    def generate_image(self, params, img_size=128, intermediate_features=False):
        x = y = jnp.linspace(-1, 1, img_size)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        d = jnp.sqrt(x**2 + y**2)
        xyd = jnp.stack([x, y, d], axis=-1)
        rgb, features = jax.vmap(jax.vmap(partial(self.apply, params)))(xyd)
        if intermediate_features:
            return rgb, features
        else:
            return rgb

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int
    nonlin: str = 'tanh' # use tanh or relu

    @nn.compact
    def __call__(self, x):
        intermediate_features = [x]
        for i_layer in range(self.n_layers):
            x = nn.Dense(self.d_hidden)(x)
            x = getattr(nn, self.nonlin)(x)
            intermediate_features.append(x)
        x = nn.Dense(3)(x)
        intermediate_features.append(x)
        rgb = jax.nn.sigmoid(x)
        return rgb, intermediate_features

    def generate_image(self, params, img_size=128, intermediate_features=False):
        x = y = jnp.linspace(-1, 1, img_size)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        d = jnp.sqrt(x**2 + y**2)
        xyd = jnp.stack([x, y, d], axis=-1)
        rgb, features = jax.vmap(jax.vmap(partial(self.apply, params)))(xyd)
        if intermediate_features:
            return rgb, features
        else:
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

    def generate_image(self, params, img_size=128, intermediate_features=False):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, img_size=img_size, intermediate_features=intermediate_features)
    