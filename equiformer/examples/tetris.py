import jax.random as jr
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import equinox as eqx
import jraph

import equiformer.layers as layers

TETRIS_BLOCKS = [
    [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
    [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, -1, 0]],  # chiral_shape_2
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # T
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # zigzag
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # L
]
TETRIS_BLOCKS_JNP = [jnp.array(block, dtype=float) for block in TETRIS_BLOCKS]


def mean_pool(g: jraph.GraphsTuple):
    """Mean pool of graph."""

    # Slightly ugly way of treating position coordinates differently
    globals_ = tu.tree_map_with_path(
        lambda l, n: jnp.mean(n, axis=(-3 if l[0].key > -1 else -2), keepdims=False),
        g.nodes,
    )
    return g._replace(globals=globals_)


class ShapeClassifier(eqx.Module):
    nn_layers: list

    def __init__(self, key: jr.PRNGKey):
        self.nn_layers = [
            layers.SelfInteractionLayer({0: (1, 4)}),
            layers.TensorProductLayer({0: [0, 1]}, {0: 4}),
            layers.SelfInteractionLayer({0: (4, 4), 1: (4, 4)}),
            layers.GateLayer({0: 4, 1: 4}),
            layers.TensorProductLayer({0: [0, 1], 1: [0, 1]}, {0: 4, 1: 4}, l_max=1),
            layers.SelfInteractionLayer({0: (8, 4), 1: (12, 4)}),
            layers.GateLayer({0: 4, 1: 4}),
            layers.TensorProductLayer({0: [0], 1: [1]}, {0: 4, 1: 4}, l_max=0),
            layers.SelfInteractionLayer({0: (8, 4)}),
            layers.GateLayer({0: 4}),
            mean_pool,
            (lambda g: jnp.mean(g.globals[0], axis=-1)),
            eqx.nn.Linear(4, 8, key=key),
            jax.nn.softmax,
        ]

    def __call__(self, g: jraph.GraphsTuple):
        for nn_layer in self.nn_layers:
            g = nn_layer(g)
        return g
