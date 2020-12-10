#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 11-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Module docstring

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from becpy.data import hf_scan
from becpy.physics import trapped_hartree_fock as thf

mu0, T, N0, Nt, freq_x, freq_y = [hf_scan[k] for k in ['mu0', 'T', 'N0', 'Nt', 'freq_x', 'freq_y']]

omega_x = 2 * np.pi * freq_x
omega_rho = 2 * np.pi * freq_y
omega_ho = (omega_x * omega_rho**2)**(1 / 3)

N = N0 + Nt
bec_fraction = N0 / N

mu0_1 = mu0[:, 0]
T_1 = T[0]


mu0_lims = mu0_1.min(), mu0_1.max()

print('interpolating...')
interp_kwargs = dict(kx=1, ky=1, s=0)
_N = RectBivariateSpline(mu0_1, T_1, N, **interp_kwargs)
_N0 = RectBivariateSpline(mu0_1, T_1, N0, **interp_kwargs)


def __mu0(N, T):
    def fun(mu0, N, T):
        return _N(mu0, T*1e9, grid=False) - N
    return brentq(fun, *mu0_lims, args=(N, T))


print('done')
_mu0 = np.vectorize(__mu0)


def _lda_bec_fraction(N, T):
    return _N0(__mu0(N, T), T*1e9, grid=False) / N


lda_bec_fraction = np.vectorize(_lda_bec_fraction)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
c = ax.contourf(T, mu0, bec_fraction, 200)
ax.contour(T, mu0, bec_fraction, [0.05], colors='r')
plt.colorbar(c, ax=ax)
ax.set(xlabel='T [nK]', ylabel='mu0 [nK]')

c = ax2.contourf(T, mu0, N, levels=np.linspace(1e5, N.max(), 200))
plt.colorbar(c, ax=ax2)


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
t = np.linspace(0.5, 1.3, 200)

f2, a2 = plt.subplots()
a2.hist(N.ravel(), 200)

for N in 1e6, 5e6, 1e7, 3e7:
    Tc = thf.critical_temperature(N, omega_ho)
    T = t * Tc

    eta = thf.eta(N, omega_ho)
    u0 = eta * Tc * 1e-9
    f0 = thf.bec_fraction(N, T, omega_ho, clip=False)

    u1 = _mu0(N, T)
    f1 = lda_bec_fraction(N, T)

    l, = ax.plot(t, f1, label=f"{N*1e-6:.1f} M")
    print(l)
    ax.plot(t, f0, '--', color=l.get_color())

    ax1.plot(T, u1)

ax.legend()
plt.show()
