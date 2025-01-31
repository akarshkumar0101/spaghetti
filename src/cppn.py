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

def hsv2rgb(hsv):
    h, s, v = hsv
    h = h * 360.

    c = v * s
    x = c * (1 - jnp.abs((h / 60) % 2 - 1))
    m = v - c
    rgbp1, c1 = jnp.stack([c, x, 0], axis=-1), (0 <= h)*(h<60)
    rgbp2, c2 = jnp.stack([x, c, 0], axis=-1), (60 <= h)*(h<120)
    rgbp3, c3 = jnp.stack([0, c, x], axis=-1), (120 <= h)*(h<180)
    rgbp4, c4 = jnp.stack([0, x, c], axis=-1), (180 <= h)*(h<240)
    rgbp5, c5 = jnp.stack([x, 0, c], axis=-1), (240 <= h)*(h<300)
    rgbp6, c6 = jnp.stack([c, 0, x], axis=-1), (300 <= h)*(h<360)
    rgbp = rgbp1 * c1 + rgbp2 * c2 + rgbp3 * c3 + rgbp4 * c4 + rgbp5 * c5 + rgbp6 * c6
    rgb = rgbp + m
    return rgb.clip(0., 1.)

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int
    nonlin: str = 'tanh' # use tanh or relu
    hsv: bool = False

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

        if self.hsv:
            rgb = jax.vmap(jax.vmap(hsv2rgb))(rgb)
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
    