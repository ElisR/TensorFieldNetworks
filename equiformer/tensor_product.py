"""Module holding functions for tensor product."""
from functools import lru_cache
from itertools import product

import jax.numpy as jnp
import numpy as np
from einops import einsum, rearrange, repeat
from sympy.physics.quantum.cg import CG


@lru_cache(maxsize=128)
def generate_cg_matrix(li: int, lf: int, lo: int) -> np.ndarray:
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

    # NOTE Not returning a JAX array anymore
    return cg_real.real


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

# TODO Add a function to generate list[int] of angular momenta multiplicities from e3nn notation
# e.g. "3x1 + 1x2" -> [0, 3, 1]

def generate_large_cg_matrix(ncs_1: list[int], ncs_2: list[int], lmax: int) -> jnp.ndarray:
    """Defines larger Clebsch-Gordan matrix for given angular momenta."""
    # Find the actual maximum angular momentum to calculate
    true_lmax = min(lmax, len(ncs_1) + len(ncs_2) - 2)

    # Define nested list to hold all blocks
    # Format needed for np.block
    cg_mats_all = [[[None] * (true_lmax + 1)] * len(ncs_2)] * len(ncs_1)

    for (l1, nc1), (l2, nc2) in product(enumerate(ncs_1), enumerate(ncs_2)):
        # Constructing matrix that will match input channels with output
        match_io = rearrange(np.eye(nc1 * nc2), "(nci ncf) ncif -> nci ncf ncif", nci=nc1, ncf=nc2)

        for l3 in range(0, true_lmax + 1):
            # CG matrix will be trivially zero if l3 is not in the range
            # Choosing to calculate them anyway for simplicity
            cg_mat = generate_cg_matrix(l1, l2, l3)

            # Repeating a cg_mat a number of times for each input channel
            cg_mat_repeated = repeat(cg_mat, "mi mf mo -> nc1 mi nc2 mf mo", nc1=nc1, nc2=nc2)

            # Perform einsum to properly expand cg_mat_repeated
            cg_mat_matching = einsum(
                cg_mat_repeated,
                match_io,
                "nc1 mi nc2 mf mo, nc1 nc2 ncif -> nc1 mi nc2 mf ncif mo",
            )
            # Reshaping to 3 dimensional matrix
            cg_mat_slurp = rearrange(cg_mat_matching, "nc1 mi nc2 mf ncif mo -> (nc1 mi) (nc2 mf) (ncif mo)", nc1=nc1, nc2=nc2, ncif=(nc1 * nc2))

            cg_mats_all[l1][l2][l3] = cg_mat_slurp

    # TODO Given that we eventually want it all collapsed into one axis, need to decide on an ordering from (l1, l2, l3) -> index, with the proviso that index_a > index_b if l3_a > l3_b

    # TODO Fix this function - currently giving wrong sizes
    return jnp.block(cg_mats_all)
    #return cg_mats_all


def tensor_product_singular(
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


def tensor_product_multiple(
        f: jnp.ndarray,
        g: jnp.ndarray,
        cg: jnp.ndarray,
        ws: list[jnp.ndarray] | None = None,
        lmax: int | None = None,
):
    """Calculate the weighted tensor product between two vectors of type l1 and l2.
    
    f and g are now larger tensors containing multiple channels of different angular momenta.
    """

# Had a separate thought to turn the Clebsch-Gordan matrix into a repeated tensor so that the tensor product could be done in one swoop.
# This tensor would have quite a few zero elements but maybe would be faster.
