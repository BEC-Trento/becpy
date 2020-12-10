#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 10-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
hfsolver: a numerical solver for the Hartree-Fock equation of state of a homogeneous Bose gas
https://github.com/carmelom/hfsolver
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from .functions import _g32

from ..constants import pi, kB, hbar, mass, z32
from ..constants import interaction_constant_g as g_int


_rtol = 1e-8
_xtol = np.sqrt(3 * _rtol)

_z0 = np.sqrt(_rtol) / 2


def _fun(z, eta, nu, alpha):
    ret = _g32(z) - nu / 2 + eta * np.log(z) / (2 * alpha)
    return ret


def _solver(nu, alpha):
    eta = np.sign(z32 - nu / 2)
    if eta == 0:  # avoid pushing the solver to the domain boundary
        return (z32, 0.)
    else:
        # cut where g23(z) - z <= _rtol
        if alpha * nu <= 0.5 * np.log(3 * _rtol):
            return (np.exp(alpha * nu), 0)
        else:
            z = brentq(_fun, _z0, 1, args=(eta, nu, alpha))
            xt = nu / 2 - eta * np.log(z) / (2 * alpha)
            xc = 0 if eta >= 0 else (nu - 2 * xt)
            return (xt, xc)


def _solver_mu(x, alpha):
    """
    return: (nu, xc)
    """
    # if alpha == 0:
    #     return (x, x)
    if x <= _xtol:
        return (np.log(x) / alpha, 0)
    elif x == z32:
        return (x, 0)
    else:
        if x < z32:
            def F(z, x): return _g32(z) - x
            z = brentq(F, 0, 1, args=(x,))
            xc = 0
        else:
            def F(z, x, alpha): return _g32(z) - x - np.log(z) / alpha
            z = brentq(F, 0, 1 - _z0, args=(x, alpha))
            xc = np.log(z) / alpha
        nu = 2 * x + np.log(z) / alpha
        return nu, xc


# ------------- physics

def lambda_therm(T):
    return ((2 * pi * hbar**2) / (mass * kB * T))**(1. / 2)


def T_crit(n):
    return 2 * pi * hbar**2 * (n / z32)**(2. / 3) / (mass * kB)


def _alpha(T):
    return g_int / (lambda_therm(T)**3 * kB * T)


def _nu(mu, T):
    return mu * lambda_therm(T)**3 / g_int


def radial_harmonic_trap(r, omega_ho=2 * pi):
    return 0.5 * mass * omega_ho**2 * r**2


def integrate_N_harmonic_trap(mu0, omega_ho, n_radial, mu_radial):
    return -4 * pi / np.sqrt(mass * omega_ho**2)**3 * trapz(n_radial * np.sqrt(2 * (mu0 - mu_radial)), mu_radial)


def solver_harmonic_trap(mu0, T, omega_ho, r=None, dr=1e-6, Rmax=None):
    """Returns interpolating functions to calculate the density as a function of ``mu(r)``.
    Those can be evaluated on an arbitrary grid of values for ``mu``.

    All physical values are expected to be given in SI units.

    Parameters
    ----------
    mu0 : float
        Chemical potential at trap center.
    T : float
        Temperature.
    omega_ho : float
        Averaged trap frequency.
    dr : float, optional
        Spatial resolution on which to compute the interpolating grid for ``mu``.
        Defaults to 1 um.

    Returns
    -------
    fun_n : ``scipy.interpolate.interp1d`` object
        Interpolating function that returns n (the total density) as a function of mu
    fun_n0 : ``scipy.interpolate.interp1d`` object
        Interpolating function that returns n0 (BEC density) as a function of mu
    mu : ndarray
        The grid of interpolating points for mu
    r : ndarray
        The (radial, evenly spaced) spatial grid of points corresponding to ``mu``
    alpha : float
        Temperature parameter
    """
    lt = lambda_therm(T)
    alpha = _alpha(T)
    if r is None:
        if Rmax is None:
            if mu0 > 0:
                Rmax = np.sqrt(30 * mu0 / mass) / omega_ho  # almost 4\sigma
            else:
                Rmax = 3 * np.sqrt(kB * T / mass) / omega_ho  # 4\sigma thermal
        else:
            Rmax = Rmax
        r = np.arange(0, Rmax, dr)
    else:
        r = r
    mu = mu0 - radial_harmonic_trap(r, omega_ho)
    nu = _nu(mu, T)
    x_t = np.empty(nu.shape)
    x_c = np.empty(nu.shape)
    for j, n in enumerate(nu):
        x_t[j], x_c[j] = _solver(n, alpha)
    x_tot = x_t + x_c
    _n = x_tot.astype(float) / lt**3
    _n0 = x_c.astype(float) / lt**3
    fun_n = interp1d(mu, _n, kind='cubic')
    fun_n0 = interp1d(mu, _n0, kind='cubic')
    return fun_n, fun_n0, mu, r, alpha


def _solver_LDA(mu0, T, V):
    lt = lambda_therm(T)
    alpha = _alpha(T)
    nu = _nu(mu0 - V, T)
    x_t, x_c = _solver(nu, alpha)
    x_tot = x_t + x_c
    _n = float(x_tot) / lt**3
    _n0 = float(x_c) / lt**3
    return _n, _n0


solver_LDA = np.vectorize(_solver_LDA, excluded={'mu0', 'T'})


def _physics_solver(mu0, T):
    lt = lambda_therm(T)
    alpha = _alpha(T)
    nu = _nu(mu0, T)
    x_t, x_c = _solver(nu, alpha)
    x_tot = x_t + x_c
    _n = float(x_tot) / lt**3
    _n0 = float(x_c) / lt**3
    return _n, _n0


physics_solver = np.vectorize(_physics_solver)


def _physics_solver_mu(n, T):
    if T == 0:
        return (g_int * n, n)
    else:
        lt = lambda_therm(T)
        alpha = _alpha(T)
        x = n * lt**3
        nu, x_c = _solver_mu(x, alpha)
        mu = g_int * nu / lt**3
        n0 = float(x_c) / lt**3
        return mu, n0


physics_solver_mu = np.vectorize(_physics_solver_mu)
