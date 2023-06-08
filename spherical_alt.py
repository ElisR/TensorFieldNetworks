import sympy
from sympy import Znm, Symbol, simplify, lambdify

from functools import partial, lru_cache

from jax import jit
import jax.numpy as jnp
from jax.scipy.special import sph_harm

# This is expected to be the slowest version of all
def SH_real(l, m):
    """Return JAX version of real spherical harmonics Y(theta, phi)."""
    theta, phi = Symbol("theta", real=True), Symbol("phi", real=True)

    ylm = sympy.simplify(Znm(l, m, theta, phi).expand(func=True))

    return lambdify([theta, phi], ylm, modules="jax")


# NOTE This is obviously broken because of control flow
def SH_real_native(l, m, thetas, phis):
    """Return JAX version of real spherical harmonics Y(theta, phi)."""
    if m == 0:
        return sph_harm(m, l, phis, thetas, n_max=l)
    elif m < 0:
        return jnp.sqrt(2) * (-1)**m * sph_harm(jnp.array([-m]), jnp.array([l]), phis, thetas, n_max=l).imag
    else:
        return jnp.sqrt(2) * (-1)**m * sph_harm(jnp.array([m]), jnp.array([l]), phis, thetas, n_max=l).real


def convert_to_spherical_coords(coords):
    """Convert cartesian coordinates to spherical coordinates."""
    # Get views
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]

    # Convert to spherical coords
    radii = jnp.sqrt(xs**2 + ys**2 + zs**2)
    thetas = jnp.nan_to_num(jnp.arccos(zs / radii), nan=0.0, copy=False)
    phis = jnp.arctan2(ys, xs)

    return thetas, phis, radii


@partial(jit, static_argnums=(1, 2))
def SH_real_jit(coords, l, m):
    """Return SH applied to an array of shape [*, 3], where last three coords are x, y, z.

    Expect this function to be slower than using the more refined expressions using cartesian coordinates directly.
    """

    thetas, phis, _ = convert_to_spherical_coords(coords)

    #SH = SH_real(l, m)
    #return SH(thetas, phis)
    return SH_real_native(l, m, thetas, phis)


def SH_real_jit_all(coords, l):
    """Return all SH applied to an array of shape [*, 3], where last three coords are x, y, z."""

    return jnp.stack([SH_real_jit(coords, l, m) for m in range(-l, l+1)], axis=-1)


def SH_real_cart(l, m):
    """Return JAX version of real spherical harmonics in cartesian coordinates Y(x, y, z).
    Uses a lot of SymPy algebraic simplification in the backend to generate fast expression.
    """
    theta, phi = Symbol("theta", real=True), Symbol("phi", real=True)
    x, y, z = Symbol("x", real=True), Symbol("y", real=True), Symbol("z", real=True)

    ylm = sympy.simplify(Znm(l, m, theta, phi).expand(func=True))
    ylm = sympy.expand_trig(ylm)

    # Extra step to try and expand complex exponentials
    # NOTE Recent addition
    ylm = sympy.expand(ylm, complex=True)
    ylm = sympy.expand_trig(ylm)
    #ylm = sympy.simplify(ylm)

    # Replacing spherical coords
    # TODO Be careful that phi is being substituted correctly with y < 0
    ylm = ylm.subs(theta, sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2)))
    ylm = ylm.subs(phi, sympy.acos(x / sympy.sqrt(x**2 + y**2)))

    # Manipulating fractions
    ylm = simplify(ylm)
    ylm = sympy.cancel(ylm)
    ylm = sympy.simplify(ylm)
    ylm = sympy.expand_power_base(ylm, force=True)
    ylm = sympy.powdenest(ylm, force=True)
    ylm = sympy.simplify(ylm)
    ylm = sympy.factor(ylm)

    # Condon-Shortley Phase
    ylm = ylm if (m % 2 == 0) else -ylm

    # Display for debugging
    #display(ylm)

    return lambdify([x, y, z], ylm, modules="jax")

@partial(jit, static_argnums=(1, 2))
def SH_sympy_jit(coords, l, m):
    """Return SH applied to an array of shape [*, 3], where last three coords are x, y, z.
    Expect this function to be slower than using the algebraic methods above.
    """
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    SH = SH_real_cart(l, m)

    return SH(xs, ys, zs)