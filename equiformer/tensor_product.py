"""Module holding functions for tensor product."""
from functools import lru_cache
from itertools import product

import jax.numpy as jnp
import numpy as np
from einops import einsum, rearrange
from sympy.physics.quantum.cg import CG


@lru_cache(maxsize=128)
def generate_cg_matrices(li: int, lf: int, lo: int):
    """Generate Clebsch-Gordan matrices for given angular momenta.

    Args:
        li: Initial angular momentum.
        lf: Filter angular momentum.
        lo: Output angular momentum.

    Returns:
        Three-dimensional tensor of shape (2li + 1, 2lf + 1, 2lo + 1) containing
        the Clebsch-Gordan coefficients.
    """
    # Define basis rotation
    ai, af, ao = basis_rotation(li), basis_rotation(lf), basis_rotation(lo)

    cg_mat = np.zeros((2 * li + 1, 2 * lf + 1, 2 * lo + 1), dtype=np.complex64)
    for i_o, mo in enumerate(range(-lo, lo + 1)):
        for i_f, mf in enumerate(range(-lf, lf + 1)):
            for i_i, mi in enumerate(range(-li, li + 1)):
                cg_mat[i_i, i_f, i_o] = CG(li, mi, lf, mf, lo, mo).doit().evalf()

    # Perform basis change to real basis
    cg_real = einsum(
        cg_mat,
        np.conj(ai),
        np.conj(af),
        ao,
        "mi mf mo, Mi mi, Mf mf, Mo mo -> Mi Mf Mo",
    )

    # Assert imaginary part is zero
    assert np.allclose(cg_real.imag, np.zeros_like(cg_real.imag))

    return jnp.array(cg_real.real)


def basis_rotation(ell: int) -> jnp.ndarray:
    """Generate basis rotation matrix for given angular momentum."""
    A = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=np.complex64)

    def ind(m):
        return m + ell

    for m in range(-ell, ell + 1):
        cs = (-1) ** m  # Condon-Shortley phase
        if m < 0:
            A[ind(m), ind(m)] = 1j / np.sqrt(2)
            A[ind(m), ind(-m)] = -cs * 1j / np.sqrt(2)
        elif m > 0:
            A[ind(m), ind(m)] = cs * 1 / np.sqrt(2)
            A[ind(m), ind(-m)] = 1 / np.sqrt(2)
        else:
            A[ind(m), ind(m)] = 1

    # Adding phase that makes CG real
    return jnp.array((-1j) ** ell * A)


def tensor_product(
    f: jnp.ndarray,
    g: jnp.ndarray,
    ws: list[jnp.ndarray] | None = None,
    lmax: int | None = None,
) -> list[jnp.ndarray]:
    """Calculate the weighted tensor product between two vectors of type l1 and l2.

    Uses Clebsch-Gordan coefficients to calculate the tensor product of two
    vectors of angular momentum l1 and l2. The resulting vector has angular
    momenta ranging from |l1 - l2| to l1 + l2.
    Vectors assumed to have channel index as penultimate dimension.

    Args:
        f: First vector of shape (..., nc1, 2l1 + 1).
        g: Second vector of shape (..., nc2, 2l2 + 1).
        ws: List of weights of shape (nc1, nc2) for each l3 output.
        lmax: Maximum angular momentum to include in tensor product.

    Returns:
        List of vectors of shape (..., nc1 * nc2, 2l3 + 1) for each l3 in l3s.
    """
    # Impute order of features
    l1, l2 = (f.shape[-1] - 1) // 2, (g.shape[-1] - 1) // 2
    # Allow for multiple channels
    nc1, nc2 = f.shape[-2], g.shape[-2]

    # Possible output angular momenta
    if lmax is None:
        lmax = l1 + l2
    l3s = range(abs(l1 - l2), min(lmax, l1 + l2) + 1)

    # If no weights are given, use ones
    if ws is None:
        ws = [jnp.ones((nc1, nc2)) for _ in l3s]

    # All Clebsch-Gordan matrices
    cg_mats = [generate_cg_matrices(l1, l2, l3) for l3 in l3s]

    tps = [
        einsum(cg_mat, f, g, w, "Mi Mf Mo, ... ci Mi, ... cf Mf, ci cf -> ... ci cf Mo")
        for cg_mat, w in zip(cg_mats, ws)
    ]
    return [rearrange(tp, "... ci cf Mo -> ... (ci cf) Mo") for tp in tps]


# Need to decide on how to handle multiple channels
# I think the channel index should be an extra dimension and tensors should only contain one order of object
def split_into_orbitals(arr: jnp.ndarray, n_channels: tuple[int]) -> list[jnp.ndarray]:
    """Split array into orbitals of shape (..., nc, l).

    Will have to use static argnums on second argument if jitting.
    """
    # Multiply n_channels by 2 * l + 1
    n_channels_multiplicity = tuple((2 * l + 1) * n for l, n in enumerate(n_channels))

    splitted = jnp.split(arr, np.cumsum(n_channels_multiplicity)[:-1], axis=-1)
    return [rearrange(s, "... (nc ms) -> ... nc ms", nc=nc, ms=2 * l + 1) for s, nc, l in zip(splitted, n_channels, range(len(n_channels)))]


def multiple_tensor_product(
    fs: list[jnp.ndarray], gs: list[jnp.ndarray], l_max: int, wss: list[list[jnp.ndarray]] | None = None
) -> list[jnp.ndarray]:
    """Calculate the tensor product between lists of features of different orders.

    All tensors have shape (..., nc, 2l + 1) where nc is the number of channels.

    Args:
        fs: List of features of different orders.
        gs: List of filters of different orders.
        l_max: Maximum angular momentum to include in tensor product.

    Returns:
        List of tensors containing the tensor product of all features and filters.
    """

    ncs_f, ncs_g = [f.shape[-2] for f in fs], [g.shape[-2] for g in gs]

    # Perform tensor product between all combinations of features and filters
    tps = [
        tensor_product(f, g, lmax=l_max)
        for f, g in product(fs, gs)
    ]

    # TODO Have to be careful with which filters get applied to which features.


def orchestrate_tensor_product(f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Orchestrate tensor product between tensors containing all channels and orders."""



# Had a separate thought to turn the Clebsch-Gordan matrix into a repeated tensor so that the tensor product could be done in one swoop.
# This tensor would have quite a few zero elements but maybe would be faster.
