#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 08-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Functions and formulas for HF theory of a trapped Bose gas
Refs:
[1] Pitaevskii, Stringari, Bose-Einstein condensation and superfluidity, 2nd ed. Chap 13
[2] S. Giorgini, L. P. Pitaevskii, and S. Stringari, Phys. Rev. A 54, R4633 (1996).

"""

import numpy as np
from ..constants import hbar, mass, scattering_length, kB, z2, z3
from .functions import g3_inv


def eta(N, omega_ho):
    a_ho = np.sqrt(hbar / mass / omega_ho)
    eta = 0.5 * z3**(1 / 3) * (15 * N**(1 / 6) * scattering_length / a_ho)**(2 / 5)
    return eta


def critical_temperature(N, omega_ho):
    # ideal gas in HT
    Tc = hbar * omega_ho * (N / z3)**(1 / 3) / kB
    return Tc


def tc_shift_interactions(N, omega_ho):
    # [2] Eq 18
    a_ho = np.sqrt(hbar / mass / omega_ho)
    shift = - 1.33 * scattering_length / a_ho * N**(1 / 6)
    return shift


def tc_shift_finite_size(N, AR):
    # AR = omega_rad / omega_x, assume symmetric trap around x
    # [2] Eq. 3
    shift = - 0.73 * (AR**(-2 / 3) + 2 * AR**(1 / 3)) / 3 / N**(1 / 3)
    return shift


def _bec_fraction(t, eta):
    # [1] Eq. 13.40
    f0 = np.maximum(1 - t**3, 0)
    bec_fraction = f0 - z2 / z3 * eta * t**2 * f0**(2. / 5)
    return bec_fraction.clip(0)


def bec_fraction(N, T, omega_ho):
    _eta = eta(N, omega_ho)
    t = T / critical_temperature(N, omega_ho)
    return _bec_fraction(t, _eta)


def __mu_t(t, eta):
    if t <= 1:
        return eta * (1 - t**3)**(2/5)
    else:
        # z = brentq(lambda z: g3(z) - z3 / t**3, 0, 1)
        z = g3_inv(z3 / t**3)
        return t * np.log(z)


_mu_t = np.vectorize(__mu_t, doc="mu / k Tc0 vs. T/Tc0")
