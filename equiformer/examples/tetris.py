import jax.random as jr
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import equinox as eqx
import jraph
import optax
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import equiformer.graphs as graphs
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


class TetrisDataset(Dataset):
    """Dataset containing Tetris blocks and their labels."""
    def __init__(self, rotate_seed: int | None = None):
        self.rotations = [jnp.eye(3) for _ in TETRIS_BLOCKS_JNP]
        if rotate_seed is not None:
            keys = jr.split(jr.PRNGKey(rotate_seed), num=len(TETRIS_BLOCKS_JNP))
            self.rotations = [jr.orthogonal(key, 3) for key in keys]

        self.graphs = [graphs.create_connected_graph(block @ rot.T) for rot, block in zip(self.rotations, TETRIS_BLOCKS_JNP)]
        self.labels = [i for i, _ in enumerate(self.graphs)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def labelled_graph_batcher(data):
    """Function for batching multiple graphs and labels."""
    gs, ys = tuple(zip(*data))
    return jraph.batch(gs), jnp.array(ys)


def pool_batched(g: jraph.GraphsTuple, aggregation_fn: callable = jax.ops.segment_sum):
    """Pool graph in a way that works for batched graphs.

    Parts taken from jraph source code.
    """
    nodes, _, _, _, _, n_node, _ = g
    sum_n_node = tu.tree_leaves(nodes)[0].shape[0]

    # Construct tensor to map from node to corresponding graph
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    # We use the aggregation function to pool the nodes/edges per graph.
    globals_ = tu.tree_map(
        lambda n: aggregation_fn(n, node_gr_idx, n_graph),
        nodes)
    return g._replace(globals=globals_)


class ShapeClassifier(eqx.Module):
    nn_layers: list

    def __init__(self, key: jr.PRNGKey):
        keys = jr.split(key, num=11)

        self.nn_layers = [
            layers.SelfInteractionLayer({0: (1, 4)}, keys[0]),
            layers.TensorProductLayer({0: [0, 1]}, {0: 4}, keys[1]),
            layers.SelfInteractionLayer({0: (4, 4), 1: (4, 4)}, keys[2]),
            layers.GateLayer({0: 4, 1: 4}, keys[3]),
            layers.TensorProductLayer({0: [0, 1], 1: [0, 1]}, {0: 4, 1: 4}, keys[4], l_max=1),
            layers.SelfInteractionLayer({0: (8, 4), 1: (12, 4)}, keys[5]),
            layers.GateLayer({0: 4, 1: 4}, keys[6]),
            layers.TensorProductLayer({0: [0], 1: [1]}, {0: 4, 1: 4}, keys[7], l_max=0),
            layers.SelfInteractionLayer({0: (8, 4)}, keys[8]),
            layers.GateLayer({0: 4}, keys[9]),
            pool_batched,
            (lambda g: jnp.mean(g.globals[0], axis=-1).T), # Doing mean to drop unary axis
            eqx.nn.Linear(4, 8, key=keys[10], use_bias=False),
            (lambda vec: vec.T),
        ]

    def __call__(self, g: jraph.GraphsTuple):
        for nn_layer in self.nn_layers:
            g = nn_layer(g)
        return g


@eqx.filter_value_and_grad
def compute_loss(model, g, label):
    """Cross entropy loss for labelling Tetris shape."""
    pred_label = model(g)
    return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(pred_label, label))


@eqx.filter_jit
def make_step(model, g, label, optim, opt_state):
    """Calculate loss & grad and update model according to optimiser."""
    loss, grads = compute_loss(model, g, label)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def main():
    """Main entrypoint."""
    model = ShapeClassifier(jr.PRNGKey(1))
    dataset = TetrisDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=labelled_graph_batcher)

    optim = optax.adam(5e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Training loop
    EPOCHS = 1500
    progress_bar = tqdm(range(EPOCHS))
    for _ in progress_bar:
        epoch_loss = 0.0
        for (g, label) in loader:
            loss, model, opt_state = make_step(model, g, label, optim, opt_state)
            epoch_loss += loss
        progress_bar.set_description(f"Loss: {epoch_loss:.4f}")

    # Test model accuracy and equivariance by creating rotated version of dataset
    rotated_dataset = TetrisDataset(rotate_seed = 4)

    def single_label(model, g) -> int:
        probs = jax.nn.softmax(model(g))
        return int(jnp.argmax(probs)), probs

    for ((g_rot, y_rot), (g, y)) in zip(dataset, rotated_dataset):
        y_pred_rot, probs_rot = single_label(model, g_rot)
        y_pred, probs = single_label(model, g)

        print(y_pred_rot, "vs", y_pred, "vs truth which is", y_rot, "=", y)
        print("max p diff", jnp.max(jnp.abs(probs - probs_rot)))


if __name__ == "__main__":
    main()
