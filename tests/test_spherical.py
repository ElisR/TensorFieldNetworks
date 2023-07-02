"""Module for testing spherical harmonics in JAX."""
import pytest
import jax.numpy as jnp
import numpy as np

from equiformer.tensor_product import generate_cg_matrices

@pytest.mark.parametrize("l1,l2,l3", [(1, 1, 2), (2, 2, 2), (3, 3, 3)])
def test_cg_matrices(l1, l2, l3):
    # Make sure the matrices have the right shape
    cgr = generate_cg_matrices(l1, l2, l3)
    assert cgr.shape == (2*l1 + 1, 2*l2 + 1, 2*l3 + 1)