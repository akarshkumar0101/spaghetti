import jax
import jax.numpy as jnp
from flax import linen as nn

class CPPN(nn.Module):
    n_layers: int
    d_hidden: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.d_hidden)(x)
            x = nn.relu(x)
        x = nn.Dense(3)(x)
        return x

    def generate_image(self, params):
        x = y = jnp.linspace(-1, 1, 256)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        d = jnp.sqrt(x**2 + y**2)
        xyd = jnp.stack([x, y, d], axis=-1)
        rgb = jax.vmap(jax.vmap(partial(self.apply, params)))(xyd)
        rgb = jax.nn.sigmoid(rgb)
        return rgb

