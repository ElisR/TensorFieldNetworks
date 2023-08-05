"""Module for testing spherical harmonics in JAX."""
import pytest
import jax.random as jr
import jax.numpy as jnp
from jax import grad

from equiformer.tensor_product import generate_cg_matrix
from equiformer.utils import generate_rand_features


@pytest.mark.parametrize("l1,l2,l3", [(1, 1, 2), (2, 2, 2), (3, 3, 3)])
def test_cg_matrices(l1, l2, l3):
    # Make sure the matrices have the right shape
    cgr = generate_cg_matrix(l1, l2, l3)
    assert cgr.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

"""
@pytest.mark.parametrize(
    "nc_f,nc_g",
    [
        (
            (0, 1, 1),
            (1, 1),
        ),
        (
            (1, 1),
            (1,),
        )
    ],
)
def test_tensor_product_multiple(nc_f, nc_g):
    key_f, key_g = jr.split(jr.PRNGKey(0), num=2)

    f = generate_rand_features(nc_f, key_f)
    g = generate_rand_features(nc_g, key_g)

    # Random function making use of tensor products
    def func(f, g, nc_f, nc_g):
        return jnp.sum(tensor_product_multiple(f, g, nc_f, nc_g))

    # Testing the function is differentiable
    grad_func = grad(func, argnums=(0, 1))
    gradded = grad_func(f, g, nc_f, nc_g)

    assert gradded[0].shape == f.shape
    assert gradded[1].shape == g.shape
"""
