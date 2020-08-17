#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 08-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Functions and formulas for HF theory of a trapped Bose gas
Ref: Pitaevskii, Stringari, 2nd ed. Chap 13

"""

import numpy as np
from ..constants import hbar, mass, scattering_length, kB, z2, z3


def eta(N, omega_ho):
    a_ho = np.sqrt(hbar / mass / omega_ho)
    eta = 0.5 * z3**(1 / 3) * (15 * N**(1 / 6) * scattering_length / a_ho)**(2 / 5)
    return eta


def critical_temperature(N, omega_ho):
    Tc = hbar * omega_ho * (N / z3)**(1 / 3) / kB
    return Tc


def _bec_fraction(t, eta):
    # Eq. 13.40
    f0 = np.maximum(1 - t**3, 0)
    bec_fraction = f0 - z2 / z3 * eta * t**2 * f0**(2./5)
    return bec_fraction.clip(0)


def bec_fraction(N, T, omega_ho):
    _eta = eta(N, omega_ho)
    t = T / critical_temperature(N, omega_ho)
    return _bec_fraction(t, _eta)
