from functools import partial
import jax
import jax.numpy as jnp
from jax.random import split
import flax
import flax.linen as nn

from einops import rearrange

import evosax

from color import hsv2rgb

identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu

activation_fn_map = dict(identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int
    nonlins: str = "relu" # "sin,tanh,sigmoid,gaussian,relu"

    inputs: str = "x,y,d" # "x,y,d,xabs,yabs"

    @nn.compact
    def __call__(self, x):
        nonlins = self.nonlins.split(",")
        features = [x]
        for i_layer in range(self.n_layers):
            x = nn.Dense(self.d_hidden)(x)

            x = rearrange(x, "(n k) -> n k", n=len(nonlins))
            x = [activation_fn_map[nonlins[i]](x[i]) for i in range(len(nonlins))]
            x = rearrange(x, "n k -> (n k)")
            # x = jax.nn.relu(x)

            features.append(x)
        x = nn.Dense(3)(x)
        features.append(x)
        h, s, v = jax.nn.tanh(x) # CHANGED THIS TO TANH
        return (h, s, v), features

    def generate_image(self, params, img_size=256, return_features=False):
        inputs = {}
        x = y = jnp.linspace(-1, 1, img_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['xabs'], inputs['yabs'] = jnp.abs(inputs['x']), jnp.abs(inputs['y'])
        inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        inputs = jnp.stack(inputs, axis=-1)
        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)
        if return_features:
            return rgb, features
        else:
            return rgb

class FlattenCPPNParameters():
    def __init__(self, cppn):
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        d_in = len(self.cppn.inputs.split(","))
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((d_in,))))
        self.n_params = self.param_reshaper.total_params
    
    def init(self, rng):
        d_in = len(self.cppn.inputs.split(","))
        params = self.cppn.init(rng, jnp.zeros((d_in,)))
        return self.param_reshaper.flatten_single(params)

    def generate_image(self, params, img_size=256, return_features=False):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, img_size=img_size, return_features=return_features)