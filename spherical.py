import math

import sympy
from sympy import S, lambdify, Symbol, simplify
from sympy.core.numbers import I, pi
from sympy.functions.combinatorial.factorials import binomial, factorial

from functools import partial, lru_cache

from jax import jit
import jax.numpy as jnp

from einops import rearrange

def Scale(l):
    r = Symbol("r", real=True, positive=True)

    return sympy.sqrt((sympy.Integer(2) * l + 1) / (2 * pi)) * r**(-l)

def AB(m):
    x, y = Symbol("x", real=True), Symbol("y", real=True)

    power = (x + I * y)**m
    return sympy.re(power), sympy.im(power)

def Pi(l, m):
    z, r = Symbol("z", real=True), Symbol("r", real=True, positive=True)

    prefac = sympy.sqrt(factorial(l - m) / factorial(l + m)) * sympy.sqrt(2 - sympy.KroneckerDelta(m, 0)) / sympy.sqrt(2)

    summation = 0
    for k in range(0, math.floor((l - m) / 2) + 1):
        summation += S.NegativeOne**k * sympy.Integer(2)**(-l) * binomial(l, k) * binomial(2 * l - 2 * k, l) * (factorial(l - 2 * k) / factorial(l - 2 * k - m)) * r**(2 * k) * z**(l - 2 * k - m)

    return prefac * summation

def SolidHarmonics(l):
    SH_dict = {}

    for m in range(0, l + 1, 1):
        A, B = AB(m)
        pre =  Pi(l, m) * Scale(l)
        y_plus = simplify(pre * A)
        y_minus = simplify(pre * B)

        SH_dict[m] = y_plus

        if m > 0:
            SH_dict[-m] = y_minus

    return SH_dict

@lru_cache(maxsize=None)
def SolidHarmonicsJax(l):
    sh_dict = SolidHarmonics(l)

    x, y, z = Symbol("x", real=True), Symbol("y", real=True), Symbol("z", real=True)
    r = Symbol("r", real=True, positive=True)

    return {m: lambdify([x, y, z, r], sh, modules="jax") for m, sh in sh_dict.items()}

@partial(jit, static_argnums=(1, 2))
def SolidHarmonic_jit(coords, l, m):
    """Applying one spherical harmonic to input."""
    
    # TODO See if this can be moved outside
    sh_dict = SolidHarmonicsJax(l)

    # Get views
    # TODO Investigate if exploiting row major order is faster
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = jnp.sqrt(xs**2 + ys**2 + zs**2)

    return sh_dict[m](xs, ys, zs, rs)

# TODO Make this stacked function a lot faster
@partial(jit, static_argnums=(1,))
def SolidHarmonics_jit(coords, l):
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
    sh_dict = SolidHarmonicsJax(l)

    # Get views
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = jnp.sqrt(xs**2 + ys**2 + zs**2)

    # Stacking operation is quite a bit slower than calling  individual results
    return jnp.stack([sh_dict[m](xs, ys, zs, rs) for m in range(-l, l+1)], axis=-1)
    #return rearrange([sh_dict[m](xs, ys, zs, rs) for m in range(-l, l+1)], "m i -> m i") # Equivalent but maybe slower