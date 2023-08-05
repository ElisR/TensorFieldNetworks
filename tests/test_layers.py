"""Module for testing NN layers."""
import pytest

import jax.random as jr
import equiformer.layers as layers
import equiformer.graphs as graphs


@pytest.mark.parametrize("n_c,l_f", [(4, 1), (4, 2)])
def test_spherical_filter(n_c: int, l_f: int):
    key = jr.PRNGKey(42)
    sph_key, coords_key = jr.split(key)
    spherical_layer = layers.SphericalFilter(n_c, l_f, num_basis=5, max_center=3.5, key=sph_key)

    coords = jr.normal(coords_key, shape=(100, 3))
    encoded = spherical_layer(coords)

    assert encoded.shape == (coords.shape[0], n_c, 2 * l_f + 1)


@pytest.mark.parametrize(
    "n_channels,filters,n_nodes,n_edges,expected_shapes",
    [
        ({0: 1, 1: 1}, {0: [0, 1], 1: [0, 1]}, 10, 20, {0: (10, 2, 1), 1: (10, 3, 3)}),
    ],
)
def test_tensor_product_layer(
    n_channels: dict[int, int],
    filters: dict[int, list[int]],
    n_nodes: int,
    n_edges: int,
    expected_shapes: dict[int, tuple[int]],
):
    key = jr.PRNGKey(0)
    g_key, tp_key = jr.split(key)
    g = graphs.create_rand_graph(n_nodes, n_edges, n_channels, g_key)

    assert g.nodes[-1].shape[0] == n_nodes

    tp_layer = layers.TFNLayer(filters, n_channels, tp_key)
    g_out = tp_layer(g)

    for l_o, expected_shape in expected_shapes.items():
        assert g_out.nodes[l_o].shape == expected_shape
