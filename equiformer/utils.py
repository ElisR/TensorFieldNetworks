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


def dict_to_e3nn_str(n_channels: dict[int, int]) -> str:
    """Converts between n_channels dictionary to e3nn irreps notation."""
    multiplities = [f"{mul}x{irrep}" for irrep, mul in n_channels.items()]
    return " ".join([multiplities])


def e3nn_str_to_dict(e3nn_str: str) -> dict[int, int]:
    """Converts between e3nn irreps notation and n_channels dictionary."""
    return {int(irrep): int(mul) for mul, irrep in [item.split('x') for item in e3nn_str.split()]}
