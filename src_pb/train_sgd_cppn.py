import os, sys, glob, pickle, copy, time
from functools import partial
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
from einop import einop

import jax
import jax.numpy as jnp
from jax.random import split

import flax
import flax.linen as nn
from flax.training.train_state import TrainState

import optax

from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")

group = parser.add_argument_group("data")
group.add_argument("--img_file", type=str, default="../skull.jpg", help="path of image file")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("data")
group.add_argument("--n_layers", type=int, default=10, help="number of layers")
group.add_argument("--d_hidden", type=int, default=25, help="number of hidden units")
group.add_argument("--nonlins", type=str, default="identity,sin,cos,gaussian,sigmoid", help="nonlinearity to use")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=50000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    print(args)

    target_img = jnp.array(plt.imread(args.img_file)/255.)

    cppn = CPPN(args.n_layers, args.d_hidden, nonlins=args.nonlins)
    cppn = FlattenCPPNParameters(cppn)

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    def loss_fn(params, target_img):
        img = cppn.generate_image(params, img_size=256)
        return jnp.mean((img - target_img)**2)

    @jax.jit
    def train_step(state, _):
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_img)
        state = state.apply_gradients(grads=grad)
        return state, loss

    tx = optax.adam(learning_rate=args.lr)
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    gen_img_fn = jax.jit(partial(cppn.generate_image, img_size=256))
    losses, imgs_train = [], [gen_img_fn(state.params)]
    pbar = tqdm(range(args.n_iters//100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        img = gen_img_fn(state.params)
        losses.append(loss)
        imgs_train.append(img)
    losses = np.array(jnp.concatenate(losses))
    imgs_train = np.array(jnp.stack(imgs_train))
    params = state.params

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "losses", losses)
        util.save_pkl(args.save_dir, "imgs_train", imgs_train)
        util.save_pkl(args.save_dir, "params", params)

if __name__ == '__main__':
    main(parse_args())



