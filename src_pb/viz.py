import jax
import jax.numpy as jnp
from einops import rearrange
import matplotlib.pyplot as plt

def viz_feature_maps(features):
    max_features_per_layer = max(jax.tree.map(lambda x: x.shape[-1], features))
    n_layers = len(features)
    n_layers, max_features_per_layer

    plt.figure(figsize=(1*max_features_per_layer, 1*n_layers))
    for i, layer_features in enumerate(features):
        for j, fmap in enumerate(rearrange(layer_features, 'h w c -> c h w')):
            plt.subplot(n_layers, max_features_per_layer, i*max_features_per_layer + j + 1)
            plt.imshow(fmap, cmap='bwr_r', vmin=-1.0, vmax=1.0); plt.xticks([]); plt.yticks([])
            if j==0:
                plt.ylabel(f"{i}", fontsize=25)
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
    # plt.subplot(n_layers, max_features_per_layer, (n_layers-1)*max_features_per_layer + (max_features_per_layer-1) + 1)
    # plt.imshow(rgb); plt.axis('off')
    plt.gcf().supylabel("Layer", fontsize=35, x=-0.01)
    plt.gcf().supxlabel("Neuron", fontsize=35)
    plt.suptitle("Feature Maps of CPPN", fontsize=35)
    plt.tight_layout()
    return plt.gcf()

# def viz_feature_maps(features):
#     features = [f.copy() for f in features]
#     max_fmaps = max([f.shape[-1] for f in features])

#     for i in range(len(features)):
#         layer = features[i]
#         if layer.shape[-1] < max_fmaps:
#             extra_maps = max_fmaps - layer.shape[-1]
#             shape = (*layer.shape[:2], extra_maps)
#             # print(layer.shape, shape)
#             layer = np.concatenate([layer, np.zeros(shape, dtype=layer.dtype)], axis=-1)
#             features[i] = layer
    
#     features = np.stack(features, axis=-2) # H, W, n_layers, n_features
#     features = np.pad(features, ((5, 5), (5, 5), (0, 0), (0, 0)), constant_values=-1.)
#     features = rearrange(features, "H W N D -> (N H) (D W)")

#     plt.figure(figsize=(20, 20))
#     plt.imshow(features, cmap='bwr_r', vmin=-1.0, vmax=1.0)
#     plt.xticks([]); plt.yticks([])
#     plt.show()