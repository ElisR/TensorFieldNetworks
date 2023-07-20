"""Module containing utility functions."""
import jax.numpy as jnp
from jax import random

import jax.numpy as jnp
import jax.random as jr


def generate_rand_features(n_channels: dict[int, int], key: jr.PRNGKey, nodes: int = 100) -> jnp.ndarray:
    """
    Generates a random feature matrix with the given number of channels and key.

    Args:
        n_channels: A tuple of integers representing the number of channels for each angular momentum.
        key: A JAX random key.

    Returns:
        A JAX array representing the random feature matrix.
    """
    return {l: jr.uniform(key, (nodes, n, 2 * l + 1)) for l, n in n_channels.items()}


# TODO Remove this function
def infer_size(n_channels: tuple[int]) -> int:
    """Returns the size of the feature vector."""
    size = 0
    for l, n in enumerate(n_channels):
        size += (2 * l + 1) * n
    return size
