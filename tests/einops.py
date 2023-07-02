import pytest
from einops import rearrange
import jax.numpy as jnp
from equiformer.tensor_product import split_into_orbitals

@pytest.mark.unit
def test_rearrange():
    """Testing that rearranging works as I expect it to."""
    test = jnp.array([[1, 1, 1], [2, 2, 2]])
    assert jnp.all(rearrange(test, "i j -> (i j)") == jnp.array([1, 1, 1, 2, 2, 2]))


@pytest.mark.parametrize("n_channels", [(1, 2, 3), (2, 3, 4)])
def test_split_into_orbitals(n_channels):
    """Testing that split_into_orbitals works as I expect it to."""
    arr = jnp.ones(sum((2 * l + 1) * n for l, n in enumerate(n_channels)))
    orbitals = split_into_orbitals(arr, n_channels)
    assert len(orbitals) == len(n_channels)
    for i, orbital in enumerate(orbitals):
        assert orbital.shape[-1] == (2 * i + 1) * n_channels[i]