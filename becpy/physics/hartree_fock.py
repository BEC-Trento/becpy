#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 12-2018 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""
import matplotlib.pyplot as plt
import sys
import numpy as np
from uncertainties import unumpy as unp
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from .functions import g12, g32, g52
from ..constants import pi, kB, mass, z32, z52
from ..constants import scattering_length as a_scatt, interaction_constant_g as g_int

from .hfsolver import solver_harmonic_trap, solver_LDA, physics_solver_mu, physics_solver, integrate_N_harmonic_trap
from .hfsolver import T_crit as _hf_T_crit, lambda_therm

thismodule = sys.modules[__name__]


def theta(x):
    return 0.5 + 0.5 * np.sign(x)


def n_crit(T):
    return z32 / lambda_therm(T)**3


def T_crit(n, mf_correction=False):
    Tc = _hf_T_crit(n)
    if mf_correction:
        Tc *= 1 + 1.3 * n * a_scatt**3
    return Tc


def mu_HF(T, n):
    """
    Returns mu vs T solving the true HF at given density n
    """
    T, n = np.broadcast_arrays(np.atleast_1d(T), np.atleast_1d(n))
    mu, _ = physics_solver_mu(n, T)
    return mu


def eta_HF(n):
    return g_int * n / kB / T_crit(n)


def mu_HF_approx(t, n):
    """
    Returns mu / gn vs t = T/Tc at lowest order in eta
    """
    eta = eta_HF(n)
    return 1 + t**(3. / 2) - 2 * np.sqrt(pi) / z32 * t * np.sqrt(eta * (1 - t**(3. / 2)))


def p_HF(T, n):
    """
    Returns pressure vs T solving the true HF at given density n
    """
    mu, n0 = physics_solver_mu(n, T)
    zeta = np.exp((mu - 2 * g_int * n) / kB / T)
    p = g_int * (n**2 - 0.5 * n0**2) + kB * T * g52(zeta) / lambda_therm(T)**3
    return p


def k_HF(T, n, h=1e-3):
    """
    Returns the isothermal compressibility k
    solve HF at density n(1 +/- h) and estimate the derivative dn / dmu
    """
    mu_p, _ = physics_solver_mu(n * (1 + h), T)
    mu_m, _ = physics_solver_mu(n * (1 - h), T)
    dn_dmu = 2 * h / (mu_p - mu_m)  # second order diff
    return dn_dmu / n


def get_pressure(V, n, initial=0):
    # pressure = cumtrapz(n, x=V, initial=np.nan)
    # pressure = pressure[-1] - pressure
    # pressure = - cumtrapz(n[::-1], x=V[::-1], initial=-initial)[::-1]
    pressure = cumtrapz(n[::-1], x=-V[::-1], initial=0)[::-1]
    return pressure + initial


def MF_density(zita, T):
    return z32 / lambda_therm(T)**3 + kB * T / g_int * np.log(1 / zita)


def MF_pressure(zita, T):
    return z52 + lambda_therm(T) / 4 / a_scatt * np.log(zita)**2


def TH_density(zita, T):
    return g32(1 / zita) / lambda_therm(T)**3


def TH_pressure(zita, T):
    return g52(1 / zita)


def n_ideal(mu0, T, V):
    mu = mu0 - V
    zita = np.exp(-mu / kB / T)
    n = np.piecewise(zita, [zita <= 1, zita > 1],
                     [MF_density, TH_density], T)
    return n


def p_ideal(mu0, T, V):
    mu = mu0 - V
    zita = np.exp(-mu / kB / T)
    p = np.piecewise(zita, [zita <= 1, zita > 1],
                     [MF_pressure, TH_pressure], T)
    return p * kB * T / lambda_therm(T)**3


def n_semi_ideal(mu0, T, V, split=False):
    mu = mu0 - V
    zita = np.exp(-np.abs(mu) / kB / T)
    nt = g32(zita) / lambda_therm(T)**3
    n0 = np.maximum(0, mu / g_int)
    if split:
        return n0, nt
    else:
        return n0 + nt


def k_semi_ideal(mu0, T, V):
    mu = mu0 - V
    zita = np.exp(-np.abs(mu) / kB / T)
    k0 = 1 / g_int * theta(mu)
    kt = -np.sign(mu) / kB / T / lambda_therm(T)**3 * g12(zita)
    return (k0 + kt) / n_semi_ideal(mu0, T, V)**2


def p_semi_ideal(mu0, T, V):
    n = n_semi_ideal(mu0, T, V)
    # p = get_pressure(V, n)
    p1 = p_HF(T, n[-1])
    p = get_pressure(V, n, initial=p1)
    return p


def n_hartree_fock(mu0, T, V, split=False, solver_kwargs={}):
    n, n0 = solver_LDA(mu0, T, V)
    if split:
        nt = n - n0
        return n0, nt
    else:
        return n


def n_hartree_fock_interp(mu0, T, V, split=False, solver_kwargs={}):
    solver_kw = dict(omega_ho=2 * pi * 60, dr=3e-7, Rmax=3e-3)
    solver_kw.update(solver_kwargs)
    fun_n_sim, fun_n0_sim, mu_array, r, alpha = solver_harmonic_trap(
        mu0, T, **solver_kw)
    mu_local = np.atleast_1d(mu0 - V)
    n = fun_n_sim(mu_local)
    if split:
        n0 = fun_n0_sim(mu_local)
        nt = n - n0
        return n0, nt
    else:
        return n


def k_hartree_fock(mu0, T, V, solver_kwargs={}):
    n = n_hartree_fock(mu0, T, V, solver_kwargs=solver_kwargs)
    k = - np.gradient(n, V) / n**2
    return k


def p_hartree_fock(mu0, T, V, solver_kwargs={}):
    n0, nt = n_hartree_fock(mu0, T, V, solver_kwargs=solver_kwargs, split=True)
    n = n0 + nt
    mu = mu0 - V - 2 * g_int * n
    zeta = np.exp(mu / kB / T)
    zeta = np.clip(zeta, a_min=None, a_max=1)
    p = g_int * (n**2 - 0.5 * n0**2) + kB * T * g52(zeta) / lambda_therm(T)**3
    # n = n_hartree_fock(mu0, T, V, solver_kwargs=solver_kwargs, split=False)
    # mu = mu0 - V[-1] - 2*g_int*n[-1]
    # zeta = np.exp(mu / kB / T)
    # p1 = g_int*n[-1]**2 + kB * T * g52(zeta) / lambda_therm(T)**3
    # p = get_pressure(V, n, initial=p1)
    return p


def get_eos(mu0, T, V, n='hartree_fock', n_std=None):
    if n in ['hartree_fock', 'semi_ideal', 'ideal']:
        n = eval(f"n_{n}(mu0, T, V,)")
        n_std = None
    valid = n > 0
    n = n.copy()
    n[~valid] = np.nan
    if n_std is not None:
        n = unp.uarray(n, n_std)
        print('-----------------\n', type(n))
    mu_local = mu0 - V
    u0 = mu_local / g_int / n
    t0 = T / T_crit(n)
    return t0, u0


def integrate_density(r, n, AR=1):
    N = np.trapz(4 * np.pi * AR * n * r**2, x=r)
    return N


# def get_N(mu0, T, omega_rho, AR, solver_kwargs={}):
#     """ This requires mu0 and T in SI units"""
#     solver_kw = dict(omega_ho=omega_rho, dr=3e-7, Rmax=0.5e-3)
#     solver_kw.update(solver_kwargs)
#     fun_n_sim, fun_n0_sim, mu_array, r, alpha = solver_harmonic_trap(mu0, T, **solver_kw)
#     n = fun_n_sim(mu_array)
#     return integrate_density(r, n, AR)
#
# def get_mu0(N, T, omega_rho, AR, mu0_lims=(-10, 100), solver_kwargs={}):
#     """
#     In: N, T in SI units
#     returns mu0 in nK
#     """
#     def fun(mu0, N, T, omega_rho, AR, solver_kwargs):
#         mu0 = mu0*1e-9*kB
#         return get_N(mu0, T, omega_rho, AR, solver_kwargs) - N
#     return brentq(fun, *mu0_lims, args=(N, T, omega_rho, AR, solver_kwargs))


def get_N(mu0, T, omega_rho, AR, density_model='hartree_fock', split=False, *args, **kwargs):
    """ This requires mu0 and T in SI units"""
    Rmax = kwargs.get('Rmax', 3e-3)
    dr = kwargs.get('dr', 3e-7)
    r = np.arange(0, Rmax, dr)
    V = 0.5 * mass * omega_rho**2 * r**2
    fun = getattr(thismodule, f"n_{density_model}")
    kw = {'omega_ho': omega_rho, 'dr': dr, 'Rmax': Rmax}
    kw.update(kwargs)
    if split:
        n0, nt = fun(mu0, T, V, split=True, solver_kwargs=kw)
        N0 = integrate_density(r, n0, AR)
        Nt = integrate_density(r, nt, AR)
        return N0, Nt
    else:
        n = fun(mu0, T, V, solver_kwargs=kw)
        return integrate_density(r, n, AR)


def get_mu0(N, T, omega_rho, AR, mu0_lims=(-30, 300), density_model='hartree_fock', *args, **kwargs):
    """
    In: N, T in SI units
    returns mu0 in nK
    kwargs (to get_N):
        dr: spatial res to compute density [default: 2e-6]
        Rmax: max range to compute density [default: 3e-3]
    """
    if density_model == 'hartree_fock':
        # kw = {'omega_ho': omega_rho, 'dr': 2e-6, 'Rmax': 3e-3}
        # kw.update(kwargs)
        # args += (False, kw,)
        return get_mu0_HF(N, T, omega_rho, AR, mu0_lims=mu0_lims)

    def fun(mu0, N, T, omega_rho, AR, density_model, *args, **kwargs):
        mu0 = mu0 * 1e-9 * kB
        return get_N(mu0, T, omega_rho, AR, density_model, *args, **kwargs) - N
    return brentq(fun, *mu0_lims, args=(N, T, omega_rho, AR, density_model,) + args)


def get_mu0_HF_quad(N, T, omega_rho, AR, mu0_lims=(-30, 300), **kwargs):
    Rmax = kwargs.get('Rmax', 100e-6)
    T1 = T * 1e9
    mu_max = mu0_lims[1] * 1e-9 * kB
    mu_min = mu_max - 0.5 * mass * omega_rho**2 * Rmax**2
    dmu = 0.3e-9 * kB
    _mu = np.arange(mu_min, mu_max, dmu)
    n, n0 = physics_solver(_mu, T)

    # wrap it to extrapolate: https://stackoverflow.com/a/2745496
    def extrap1d(interpolator):
        xmin = interpolator.x.min()

        def pointwise(x):
            # print(x)
            if x < xmin:
                return np.exp(x / T1) / lambda_therm(T)**3
            else:
                return interpolator(x)

        def ufunclike(xs):
            # print(type(xs))
            xs = xs if isinstance(xs, np.ndarray) else np.array([xs])
            return np.array(list(map(pointwise, xs)))

        return ufunclike

    density = extrap1d(
        interp1d(_mu * 1e9 / kB, n, kind='linear', fill_value='extrapolate'))

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # # uu = np.arange(-1000, mu0_lims[1], 1)
    # uu = _mu*1e9/kB * 2
    # ax.semilogy(uu, density(uu), '-o')
    # ax.plot(uu, np.exp(uu / T1) / lambda_therm(T)**3, '-o')
    # plt.show()

    def fun(mu0):
        def integrand(mu):
            return 4 * np.pi * AR * 1e-9 * kB * density(mu) * np.sqrt(2 * (mu0 - mu) * 1e-9 * kB) / (mass * omega_rho**2)**(3 / 2)
        ret = quad(integrand, a=-np.inf, b=mu0)
        return ret[0] - N

    return brentq(fun, *mu0_lims)


def get_mu0_HF(N, T, omega_rho, AR, mu0_lims=(-30, 300), **kwargs):
    """
    In: N, T in SI units
    returns mu0 in nK
    kwargs (to hf_solver):
        dr: spatial res to compute density [default: 0.5 um]
        Rmax: max range to compute density [default: 200 um]
    """
    solver_kw = dict(omega_ho=omega_rho, dr=0.5e-7, Rmax=2e-4)
    solver_kw.update(kwargs)
    mu_upper = mu0_lims[1] * 1e-9 * kB
    fun_n_sim, _, mus, r, _ = solver_harmonic_trap(mu_upper, T, **solver_kw)
    ns = fun_n_sim(mus)

    # import matplotlib.pyplot as plt
    # fig, (ax, ax1) = plt.subplots(2, sharex=True)
    # ax.plot(r, ns)
    # ax1.plot(r, mus * 1e9 / kB)
    # ax1.axhline(mu0_lims[0])
    # plt.show()

    def fun(mu0):
        mu0 = mu0 * 1e-9 * kB
        where = mus <= mu0
        return AR * integrate_N_harmonic_trap(mu0, omega_rho, ns[where], mus[where]) - N
    return brentq(fun, *mu0_lims)


def plot_pressure_data(ax, mu0, T, p, V, *args, **kwargs):
    # Let's default to fit agains V, and plot against (inverse) fugacity
    zita = np.exp(-(mu0 - V) / kB / T)
    p = p * lambda_therm(T)**3 / kB / T
    return ax.semilogx(zita, p, *args, **kwargs)


def compare_models(mu0, T, r, omega):
    V = 0.5 * mass * omega**2 * r**2
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ax_n, ax_p, ax_eos = axes
    for model in 'ideal', 'semi_ideal', 'hartree_fock':
        n = eval(f"n_{model}(mu0, T, V)")
        ax_n.plot(r * 1e6, n, label=model)
        p = eval(f"p_{model}(mu0, T, V)")
        plot_pressure_data(ax_p, mu0, T, p, V, label=model)
        t, u = get_eos(mu0, T, V, n)
        ax_eos.plot(t, u, label=model)
        ax_eos.set(xlim=(0, 1.5), ylim=(-2.5, 3),
                   xlabel='T/Tc', ylabel='mu/gn')
    ax_n.legend()
    for ax in axes:
        ax.grid(True)
    return fig, axes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['toolbar'] = 'toolmanager'
    # plt.rcParams['figure.dpi'] = 72

    # import os, sys
    # sys.path.insert(0, os.path.abspath('..'))
    # from visualization import fig2clipboard

    from hfsolver.physics import mass, kB
    tf = 90
    omega = 2 * np.pi * tf
    r = np.arange(0, 50, 0.1) * 1e-6
    Vtrap = 0.5 * mass * omega**2 * r**2

    mu0 = 75 * 1e-9 * kB
    T = 200e-9

    fig, axes = compare_models(mu0, T, r, omega)
    fig.suptitle(f"Compare models\nmu0 = {mu0*1e9/kB:g} nK, T = {T*1e9:g} nK")

    n0, nt = n_hartree_fock(mu0, T, Vtrap, split=True)
    n = n0 + nt
    fig, ax = plt.subplots()
    ax.plot(r * 1e6, n)
    ax.plot(r * 1e6, nt, '--')
    ax.plot(r * 1e6, n0)
    ax.set(xlabel='r [um]', ylabel='n [at/m^3]',)
    ax.grid(True)
    fig.suptitle(f'HF density profile\nmu0 {mu0:g} nK, T {T*1e9:g} nK')

    # mu = np.linspace(-10, 100, 20)
    # AR = 10
    # N = np.empty_like(mu)
    # N0 = np.empty_like(mu)
    # for j, mu0 in enumerate(mu):
    #     mu0 = mu0*1e-9*kB
    #     n0, nt = n_hartree_fock(mu0, T, Vtrap, split=True)
    #     n = n0 + nt
    #     N[j] = integrate_density(r, n, AR)
    #     N0[j] = integrate_density(r, n0, AR)
    #
    # NN = 5e6
    # mu0_N = get_mu0(NN, T, omega, AR)
    # print(f"found mu0 = {mu0_N:.2f}")
    # fig, ax = plt.subplots()
    # ax.plot(mu, N*1e-6, label='N')
    # ax.plot(mu, (N-N0)*1e-6, label='Nth')
    # ax.plot(mu, N0*1e-6, label='Nbec')
    # ax.plot(mu0_N, NN*1e-6, 'or', mew=2, mfc='none')
    # ax.set(xlabel='mu0 [nK]', ylabel='N [M]')
    # ax.legend()
    # ax.grid()
    # fig.suptitle(f"Natoms vs mu0 at {T*1e9:.0f} nK\ntrap freqs: {tf/AR} x {tf} Hz")

    fig, ax = plt.subplots()
    t0, u0 = get_eos(mu0, T, V=Vtrap, n=n)
    Rtf = np.sqrt(2 * mu0 / mass) / omega
    dr = 0.5e-6
    r_bins = np.arange(0.5 * Rtf, 1.5 * Rtf, dr)
    V_bins = 0.5 * mass * omega**2 * r_bins**2
    t, u, t_err, u_err = get_eos(mu0, T, V=Vtrap, n=n, V_bins=V_bins)

    ax.plot(t0, u0, 'k')
    ax.errorbar(t, u, xerr=t_err, yerr=u_err, marker='o', ls='')
    ax.set(xlim=(0, 1.5), ylim=(-2.5, 3), xlabel='T/Tc', ylabel='mu/gn')
    ax.grid(True)
    plt.show()
