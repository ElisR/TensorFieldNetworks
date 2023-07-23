"""Module for creating and manipulating graphs."""
import jax
import jax.numpy as jnp
import jax.random as jr
import jraph

import equiformer.utils as utils

COORDS = -1


def create_rand_graph(
    n_nodes: int, n_edges: int, n_channels: dict[int, int], key: jr.PRNGKey
):
    """Create random graph with set number of nodes, edges and angular momentum channels."""
    # create random array of node features
    key, subkey, pos_subkey = jr.split(key, num=3)

    node_features = utils.generate_rand_features(n_channels, subkey, nodes=n_nodes)
    coords = jr.normal(pos_subkey, shape=(n_nodes, 3))

    # create random senders and receivers
    key, subkey1, subkey2 = jr.split(key, num=3)
    senders = jr.randint(subkey1, shape=(n_edges,), minval=0, maxval=n_nodes)
    receivers = jr.randint(subkey2, shape=(n_edges,), minval=0, maxval=n_nodes)

    # create random edge features with cartesian coordinates
    #edge_features = jr.normal(key, shape=(n_edges, 3))
    edge_features = create_edge_features(coords, senders, receivers)

    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
        nodes={COORDS: jnp.expand_dims(coords, -2), **node_features},
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        globals=None,
    )
    return graph


def create_edge_features(
    coords: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray
) -> jnp.ndarray:
    """Take input coordinates and return matrix of edges."""
    # Easier than using gather function semantics
    # Allowing for possibility of extra dimensions at the start
    return coords[receivers, ...] - coords[senders, ...]


def create_connected_graph(coords: jnp.ndarray, add_self_edges: bool = False):
    """Create connected graph from coordinates"""
    n_node = coords.shape[-2]

    # Create all to all graph
    senders, receivers = jnp.meshgrid(jnp.arange(n_node), jnp.arange(n_node))
    if not add_self_edges:
        senders = jax.vmap(jnp.roll)(senders, -jnp.arange(len(senders)))[:, 1:]
        receivers = receivers[:, 1:]
    senders = senders.flatten()
    receivers = receivers.flatten()
    assert senders.shape == receivers.shape
    n_edge = senders.shape[0]

    edges = create_edge_features(coords, senders, receivers)

    return jraph.GraphsTuple(
        nodes={-1: jnp.expand_dims(coords, -2), 0: jnp.ones((n_node, 1, 1), dtype=float)},
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=jnp.array([n_node]),
        n_edge=jnp.array([n_edge]),
    )
