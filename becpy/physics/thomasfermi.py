#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 09-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np
from ..constants import hbar, mass, scattering_length, interaction_constant_g as g_int


def mu0(N, omega_ho):
    a_ho = np.sqrt(hbar / mass / omega_ho)
    return 0.5 * hbar * omega_ho * (15 * N * scattering_length / a_ho)**(2/5)


def castin_dum_expansion_ax(t, omega_x, omega_rho, R0):
    tau = omega_rho * t
    eps = omega_x / omega_rho
    return R0 * (1 + eps**2 * (tau * np.arctan(tau) - 0.5*np.log(1 + tau**2)))


def castin_dum_expansion_rad(t, omega_rho, R0):
    return R0 * np.sqrt(1 + (omega_rho * t)**2)
