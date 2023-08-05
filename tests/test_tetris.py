import pytest
import jax.random as jr
from torch.utils.data import Dataset
import equinox as eqx
import jax.numpy as jnp

import equiformer.examples.tetris as tetris
import equiformer.graphs as graphs


@pytest.fixture(scope="module")
def tetris_dataset() -> Dataset:
    return tetris.TetrisDataset()


def test_output_shape(tetris_dataset: Dataset):
    classifier = tetris.ShapeClassifier(jr.PRNGKey(1))

    for (g, _) in tetris_dataset:
        assert classifier(g).shape == (1, 8)

    # Also test on bigger graph
    rand_g = graphs.create_rand_graph(100, 200, {0: 1}, jr.PRNGKey(0))
    assert classifier(rand_g).shape == (1, 8)
