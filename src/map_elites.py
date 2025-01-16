import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from einops import rearrange, repeat

from clip import CLIP
from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=10000000, help="")
group.add_argument("--pop_size", type=int, default=6800, help="")
group.add_argument("--bs", type=int, default=32, help="")
group.add_argument("--sigma", type=float, default=0.5, help="mutation rate")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    with open('../nounlist.txt', 'r') as f:
        nouns = f.read().strip().split('\n')
    
    nouns = nouns[:args.pop_size]
    nouns = [f"an image of a {noun}" for noun in nouns]
    # assert args.pop_size % args.bs == 0

    clip = CLIP()
    z_txt = clip.embed_txt(nouns)

    cppn = CPPN(n_layers=4, d_hidden=16, nonlin='tanh')
    cppn = FlattenCPPNParameters(cppn)

    rng = jax.random.PRNGKey(args.seed)

    def get_pheno(params):
        img = cppn.generate_image(params)
        z_img = clip.embed_img(img)
        return dict(params=params, img=img, z_img=z_img)
    
    def get_niche(archive, pheno):
        return jnp.argmax(pheno['z_img'] @ archive['z_txt'].T)
    
    def calc_quality(archive, nid, pheno):
        return archive['z_txt'][nid] @ pheno['z_img']
    
    def mutate(rng, params):
        noise = jax.random.normal(rng, params.shape)
        return params + noise * args.sigma

    # def mutate(rng, params):
    #     rng, _rng = split(rng)
    #     mask = jax.random.uniform(rng, params.shape) < args.sigma
    #     noise = jax.random.normal(_rng, params.shape)
    #     return noise * mask + params * (1-mask)
    
    def place_in_archive_all(archive, pheno): # place into all niches that it beats
        z_img = pheno['z_img']
        scores = z_img@archive['z_txt'].T # (N, )
        replace = scores > archive['quality'] # (N, )

        pheno_repeat = jax.tree.map(lambda x: repeat(x, "... -> N ...", N=args.pop_size), pheno)
        archive['pheno'] = jax.tree.map(lambda x, y: jnp.where(replace.reshape((args.pop_size, )+(1,)*(x.ndim-1)), x, y), pheno_repeat, archive['pheno'])
        archive['quality'] = jnp.where(replace, scores, archive['quality'])
        data = dict(n_replacements=replace.sum())
        return archive, data

    def place_in_archive_best(archive, pheno): # only place into one niche
        archive = jax.tree.map(lambda x: x, archive) # copy

        nid = get_niche(archive, pheno)
        current_pheno = jax.tree.map(lambda x: x[nid], archive['pheno'])

        current_quality = archive['quality'][nid]
        quality = calc_quality(archive, nid, pheno)
        replace = quality > current_quality
        new_pheno = jax.tree.map(lambda x, y: jnp.where(replace, x, y), pheno, current_pheno)
        new_quality = jnp.where(replace, quality, current_quality)
        archive['pheno'] = jax.tree.map(lambda x, y: x.at[nid].set(y), archive['pheno'], new_pheno)
        archive['quality'] = archive['quality'].at[nid].set(new_quality)
        data = dict(pheno=pheno, nid=nid, replace=replace)
        return archive, data
    
    place_in_archive = place_in_archive_all

    @jax.jit
    def do_iter(rng, archive):
        p = (archive['quality'] > -jnp.inf).astype(jnp.float32)
        p = p / p.sum()

        rng, _rng = split(rng)
        idx_parent = jax.random.choice(_rng, args.pop_size, shape=(), p=p)
        params_parent = archive['pheno']['params'][idx_parent]
        rng, _rng = split(rng)
        params_child = mutate(_rng, params_parent)

        pheno_child = get_pheno(params_child)
        archive, data = place_in_archive(archive, pheno_child)
        data['nid_parent'] = idx_parent
        return archive, data

    pheno = get_pheno(jax.random.normal(rng, (cppn.n_params, )))

    archive = dict(
        z_txt=z_txt,
        pheno=jax.tree.map(lambda x: repeat(x, "... -> N ...", N=args.pop_size), pheno),
        quality=jnp.full((args.pop_size,), -jnp.inf),
    )
    print('archive shape: ', jax.tree.map(lambda x: x.shape, archive))
    
    archive, di = place_in_archive(archive, pheno)

    data = []

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        archive, di = do_iter(rng, archive)
        # del di['pheno']
        data.append(di)

        if i_iter % (10000) == 0 or i_iter == args.n_iters-1:
            percent_archive_filled = (archive['quality'] > -jnp.inf).mean()
            avg_quality = (archive['quality'] * (archive['quality'] > -jnp.inf)).mean() / percent_archive_filled
            pbar.set_postfix(percent_archive_filled=percent_archive_filled, avg_quality=avg_quality)

        if i_iter % (args.n_iters//20) == 0 or i_iter == args.n_iters-1:
            util.save_pkl(args.save_dir, 'archive', jax.tree.map(lambda x: np.array(x), archive))
            util.save_pkl(args.save_dir, 'data', jax.tree.map(lambda x: np.array(x), data))

if __name__ == '__main__':
    main(parse_args())



