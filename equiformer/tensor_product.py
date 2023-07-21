"""Module holding functions for tensor product."""
from functools import lru_cache
from itertools import product

import jax.numpy as jnp
import numpy as np
from einops import einsum, rearrange, repeat
from sympy.physics.quantum.cg import CG


@lru_cache(maxsize=128)
def generate_cg_matrix(l_i: int, l_f: int, l_o: int) -> np.ndarray:
    """Generate SO(3) Clebsch-Gordan matrix for given triplet of angular momenta.

    NOTE This is a linear transformation of the SU(2) Clebsch-Gordan matrix.

    Args:
        li: Initial angular momentum.
        lf: Filter angular momentum.
        lo: Output angular momentum.

    Returns:
        Three-dimensional tensor of shape (2li + 1, 2lf + 1, 2lo + 1) containing
        the Clebsch-Gordan coefficients.
    """
    # Define basis rotation
    a_i, a_f, a_o = _basis_rotation(l_i), _basis_rotation(l_f), _basis_rotation(l_o)

    cg_mat = np.zeros((2 * l_i + 1, 2 * l_f + 1, 2 * l_o + 1), dtype=np.complex64)
    for i_o, m_o in enumerate(range(-l_o, l_o + 1)):
        for i_f, m_f in enumerate(range(-l_f, l_f + 1)):
            for i_i, m_i in enumerate(range(-l_i, l_i + 1)):
                # Get SU(2) CG coefficients from SymPy
                cg_mat[i_i, i_f, i_o] = CG(l_i, m_i, l_f, m_f, l_o, m_o).doit().evalf()

    # Perform basis change to real basis
    cg_real = einsum(
        cg_mat,
        np.conj(a_i),
        np.conj(a_f),
        a_o,
        "mi mf mo, Mi mi, Mf mf, Mo mo -> Mi Mf Mo",
    )

    # Assert imaginary part is zero
    assert np.allclose(cg_real.imag, np.zeros_like(cg_real.imag))

    # NOTE Not returning a JAX array anymore
    return cg_real.real


def _basis_rotation(ell: int) -> jnp.ndarray:
    """Generate basis rotation matrix for given angular momentum.

    Changes from SU(2) irreps to SO(3).
    """
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


def _possible_output(l_i: int, l_f: int, l_max: int | None = None) -> list[int]:
    """Returns a list of the possible angular momenta from tensor product of l_i, l_f."""
    if l_max is None:
        l_max = l_i + l_f
    return list(range(abs(l_i - l_f), min(l_max, l_i + l_f) + 1))


def tensor_product(
    f: jnp.ndarray,
    g: jnp.ndarray,
    ws: dict[int, jnp.ndarray] | None = None,
    l_max: int | None = None,
    depthwise: bool = False,
) -> dict[int, jnp.ndarray]:
    """Calculate the weighted tensor product between two vectors of type l1 and l2.

    Uses Clebsch-Gordan coefficients to calculate the tensor product of two
    vectors of angular momentum l1 and l2. The resulting vector has angular
    momenta ranging from |l1 - l2| to l1 + l2.
    Vectors assumed to have channel index as penultimate dimension.
    Returns empty arrays for low l < |l1 - l2|.

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

    if depthwise:
        assert nc1 == nc2

    # Possible output angular momenta
    l3s = _possible_output(l1, l2, l_max=l_max)

    # If no weights are given, use ones
    if ws is None:
        ws = {}
    if not depthwise:
        _ = [ws.setdefault(l3, jnp.ones((nc1, nc2))) for l3 in l3s]
    else:
        _ = [ws.setdefault(l3, jnp.ones((nc1,))) for l3 in l3s]

    # All Clebsch-Gordan matrices
    cg_mats = {l3: jnp.array(generate_cg_matrix(l1, l2, l3)) for l3 in l3s}

    # Calculate tensor product as linear combination including weights
    weighted_prod = (
        "Mi Mf Mo, ... ci Mi, ... cf Mf, ci cf -> ... ci cf Mo"
        if not depthwise
        else "Mi Mf Mo, ... c Mi, ... c Mf, c -> ... c Mo"
    )
    tps = {
        l: einsum(
            cg_mats[l],
            f,
            g,
            ws[l],
            weighted_prod,
        )
        for l in l3s
    }
    if not depthwise:
        tps = {l: rearrange(tp, "... ci cf Mo -> ... (ci cf) Mo") for l, tp in tps.items()}
    return tps
