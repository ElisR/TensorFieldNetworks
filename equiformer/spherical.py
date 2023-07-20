"""Module with custom JAX spherical harmonics implementation, using cartesian coordinates."""
import math
from functools import lru_cache, partial

import jax.numpy as jnp
import sympy
from jax import jit
from sympy import Expr, S, Symbol, lambdify, simplify
from sympy.core.numbers import I, pi
from sympy.functions.combinatorial.factorials import binomial, factorial


def _scale(l: int) -> Expr:
    """Internal function for scaling factor of spherical harmonics."""
    r = Symbol("r", real=True, positive=True)

    return sympy.sqrt((sympy.Integer(2) * l + 1) / (2 * pi)) * r ** (-l)


def _ab(m: int) -> tuple[Expr, Expr]:
    """Internal function for A and B coefficients of spherical harmonics."""
    x, y = Symbol("x", real=True), Symbol("y", real=True)

    power = (x + I * y) ** m
    return sympy.re(power), sympy.im(power)


def _pi(l: int, m: int) -> Expr:
    """Internal function for Pi coefficient of spherical harmonics.

    Args:
        l: Order of spherical harmonics.
        m: Index of spherical harmonics.
    """
    z, r = Symbol("z", real=True), Symbol("r", real=True, positive=True)

    prefac = (
        sympy.sqrt(factorial(l - m) / factorial(l + m))
        * sympy.sqrt(2 - sympy.KroneckerDelta(m, 0))
        / sympy.sqrt(2)
    )

    summation = 0
    for k in range(0, math.floor((l - m) / 2) + 1):
        summation += (
            S.NegativeOne**k
            * sympy.Integer(2) ** (-l)
            * binomial(l, k)
            * binomial(2 * l - 2 * k, l)
            * (factorial(l - 2 * k) / factorial(l - 2 * k - m))
            * r ** (2 * k)
            * z ** (l - 2 * k - m)
        )

    return prefac * summation


def solid_harmonics(l: int) -> dict[int, Expr]:
    """Returns a dictionary of real spherical harmonics :math:`Y_{\ell m}(x, y, z)`.

    Calculated using SymPy in the backend.

    Args:
        l: Maximum degree of spherical harmonics.
    """
    SH_dict = {}

    for m in range(0, l + 1, 1):
        A, B = _ab(m)
        pre = _pi(l, m) * _scale(l)
        y_plus = simplify(pre * A)
        y_minus = simplify(pre * B)

        SH_dict[m] = y_plus

        if m > 0:
            SH_dict[-m] = y_minus

    return SH_dict


@lru_cache(maxsize=None)
def solid_harmonics_jax(l: int) -> dict[int, callable]:
    """Jaxified version of solid_harmonics. Derived from SymPy expression."""
    sh_dict = solid_harmonics(l)

    x, y, z = Symbol("x", real=True), Symbol("y", real=True), Symbol("z", real=True)
    r = Symbol("r", real=True, positive=True)

    return {m: lambdify([x, y, z, r], sh, modules="jax") for m, sh in sh_dict.items()}


@partial(jit, static_argnums=(1, 2))
def solid_harmonic_jit(coords: jnp.ndarray, l: int, m: int) -> jnp.ndarray:
    """Applying one spherical harmonic to input."""
    # TODO See if this can be moved outside
    sh_dict = solid_harmonics_jax(l)

    # Get views
    # TODO Investigate if exploiting row major order is faster
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = jnp.sqrt(xs**2 + ys**2 + zs**2)

    return sh_dict[m](xs, ys, zs, rs)


@partial(jit, static_argnums=(1,))
def solid_harmonics_jit(coords: jnp.ndarray, l: int) -> jnp.ndarray:
    r"""Applying all real spherical harmonics :math:`Y_{\ell m}(x, y, z)` to input.

    .. math::
        r^{\ell}\left(\begin{array}{c}Y_{\ell m} \\ Y_{\ell-m}\end{array}\right)
        =\sqrt{\frac{2 \ell+1}{2 \pi}} \bar{\Pi}_{\ell}^m(z)
        \left(\begin{array}{c} A_m \\ B_m \end{array}\right),
        \quad m>0

    .. math::
        \bar{\Pi}_{\ell}^m(z)=
        \left[\frac{(\ell-m) !}{(\ell+m) !}\right]^{1 / 2} \sum_{k=0}^{\lfloor(\ell-m) / 2\rfloor}(-1)^k 2^{-\ell}
        \left(\begin{array}{l}\ell \\k\end{array}\right)
        \left(\begin{array}{c}2 \ell-2 k \\ \ell\end{array}\right)
        \frac{(\ell-2 k) !}{(\ell-2 k-m) !} r^{2 k} z^{\ell-2 k-m}

    .. math::
        A_m(x, y) \equiv \Re{[(x+i y)^m]}, \qquad B_m(x, y) \equiv \Im{[(x+i y)^m]}
    """

    # TODO See if this can be moved outside
    sh_dict = solid_harmonics_jax(l)

    # Get views
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = jnp.sqrt(xs**2 + ys**2 + zs**2)

    return jnp.stack([sh_dict[m](xs, ys, zs, rs) for m in range(-l, l + 1)], axis=-1)
