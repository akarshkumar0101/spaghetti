
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

import clip
from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="")
group.add_argument("--pop_size", type=int, default=4096, help="")
group.add_argument("--bs", type=int, default=32, help="")
group.add_argument("--sigma", type=float, default=.1, help="mutation rate")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):

    # load nounfile.txt as list of strings
    with open('../nounlist.txt', 'r') as f:
        nouns = f.read().strip().split('\n')
    
    np.random.seed(args.seed)
    np.random.shuffle(nouns)

    nouns = nouns[:args.pop_size]
    assert args.pop_size % args.bs == 0

    fm = clip.CLIP()
    z_txt = fm.embed_txt(nouns)
    cppn = CPPN(n_layers=4, d_hidden=16)
    cppn = FlattenCPPNParameters(cppn)

    rng = jax.random.PRNGKey(args.seed)
    x = jnp.zeros((3,))

    params = jax.random.normal(rng, (args.pop_size, cppn.n_params))

    def get_pheno(params):
        img = cppn.generate_image(params)
        z_img = fm.embed_img(img)
        return dict(params=params, img=img, z_img=z_img)
    
    def scan_fn(_, params):
        return None, get_pheno(params)
    
    _, archive = jax.lax.scan(scan_fn, None, params)
    print(jax.tree.map(lambda x: x.shape, archive))

    # @jax.jit
    # def do_iter(rng, archive):
    #     rng, _rng = split(rng)
    #     idx_parent = jax.random.randint(_rng, (args.bs), 0, args.pop_size)

    #     archive_parent = jax.tree.map(lambda x: x[idx_parent], archive)
    #     print('archive parent', jax.tree.map(lambda x: x.shape, archive_parent))

    #     params_parent = archive_parent['params']
    #     rng, _rng = split(rng)
    #     params_children = params_parent + args.sigma * jax.random.normal(_rng, params_parent.shape)

    #     img = jax.vmap(cppn.generate_image)(params_children)
    #     z_img = jax.vmap(fm.embed_img)(img)
    #     scores_children = z_img @ z_txt.T  # (bs, pop_size)

    #     def scan_fn(archive, pc_score):
    #         print('hey', archive['params'].shape, archive['scores'].shape)
    #         print('hey2', pc_score[0].shape, pc_score[1].shape)

    #         pc, score = pc_score
    #         replace_mask = (score > archive['scores'])
    #         print('replace mask', replace_mask.shape)

    #         pc = repeat(pc, "... -> N ...", N=args.pop_size)

    #         params = jnp.where(replace_mask[:, None], pc, archive['params'])
    #         scores = jnp.where(replace_mask, score, archive['scores'])
    #         archive = dict(params=params, scores=scores)
    #         print('new archive', jax.tree.map(lambda x: x.shape, archive))
    #         return archive, None
        
    #     print('children set', jax.tree.map(lambda x: x.shape, (params_children, scores_children)))
    #     archive, _ = jax.lax.scan(scan_fn, archive, (params_children, scores_children))
    #     return archive
    
    @jax.jit
    def do_iter(rng, archive):
        rng, _rng = split(rng)
        idx_parent = jax.random.randint(_rng, (), 0, args.pop_size)
        params_parent = archive['params'][idx_parent]
        rng, _rng = split(rng)
        params_children = params_parent + args.sigma * jax.random.normal(_rng, params_parent.shape)

        pheno = get_pheno(params_children)

        print('archive', jax.tree.map(lambda x: x.shape, archive))
        print('pheno', jax.tree.map(lambda x: x.shape, pheno))

        scores_sitting = (archive['z_img'] * z_txt).sum(axis=-1) # (pop_size)
        score_child = (pheno['z_img'] * z_txt).sum(axis=-1) # (pop_size)
        
        pheno = jax.tree.map(lambda x: repeat(x, "... -> N ...", N=args.pop_size), pheno)
        print('pheno', jax.tree.map(lambda x: x.shape, pheno))

        replace_mask = score_child > scores_sitting # (pop_size)
        print('replace mask', replace_mask.shape)

        pheno = jax.tree.map(lambda x: rearrange(x, "N ... -> ... N"), pheno)
        archive = jax.tree.map(lambda x: rearrange(x, "N ... -> ... N"), archive)
        print('archive', jax.tree.map(lambda x: x.shape, archive))
        print('pheno', jax.tree.map(lambda x: x.shape, pheno))
        archive = jax.tree.map(lambda x, y: jnp.where(replace_mask, x, y), pheno, archive)
        archive = jax.tree.map(lambda x: rearrange(x, "... N -> N ..."), archive)
        print('archive', jax.tree.map(lambda x: x.shape, archive))

        n_transfers = replace_mask.sum()
        return archive, n_transfers
        

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        archive, n_transfers = do_iter(rng, archive)
        pbar.set_postfix(n_transfers=n_transfers)
        if i_iter%1000 == 0:
            z_img = archive['z_img']
            z_txt = z_txt
            print(z_img.shape, z_txt.shape)
            scores = (z_img * z_txt).sum(axis=-1)
            print(scores.mean().item(), scores.min().item(), scores.max().item())



if __name__ == '__main__':
    main(parse_args())



