"""Module holding functions for tensor product."""
from sympy.physics.quantum.cg import CG
import jax.numpy as jnp
import numpy as np
from einops import einsum

# TODO Define Clebsch-Gordan coefficients
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

    cg_mat = np.zeros((2*li + 1, 2*lf + 1, 2*lo + 1), dtype=np.complex64)
    for i_o, mo in enumerate(range(-lo, lo + 1)):
        for i_f, mf in enumerate(range(-lf, lf + 1)):
            for i_i, mi in enumerate(range(-li, li + 1)):
                cg_mat[i_i, i_f, i_o] = CG(li, mi, lf, mf, lo, mo).doit().evalf()

    cg_real = einsum(cg_mat, np.conj(ai), np.conj(af), ao, "mi mf mo, Mi mi, Mf mf, mo Mo -> Mi Mf Mo")

    # assert imaginary part is zero
    #assert np.allclose(cg_real.imag, np.zeros_like(cg_real.imag))

    return jnp.array(cg_real.real)


def basis_rotation(ell: int) -> jnp.ndarray:
    """Generate basis rotation matrix for given angular momentum."""
    A = np.zeros((2*ell + 1, 2*ell + 1), dtype=np.complex64)

    def ind(m):
        return m + ell

    for m in range(-ell, ell + 1):
        cs = (-1)**m # Condon-Shortley phase
        if m < 0:
            A[ind(m), ind(m)] = 1j / np.sqrt(2)
            A[ind(m), ind(-m)] = -cs * 1j / np.sqrt(2)
        elif m > 0:
            A[ind(m), ind(m)] = cs * 1 / np.sqrt(2)
            A[ind(m), ind(-m)] = 1 / np.sqrt(2)
        else:
            A[ind(m), ind(m)] = 1

    # Adding phase that makes CG real
    return jnp.array((-1j)**ell * A)


# Need to decide on how to handle multiple channels