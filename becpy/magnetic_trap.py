#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np

from scipy.constants import pi, hbar, atomic_mass
mass = 23 * atomic_mass
gravity = 9.806


def trap_detuning(x, z, trap_x, trap_z):
    AR = trap_z / trap_x
    omega_rho = 2 * pi * trap_z
    z_sag = gravity / omega_rho**2
    return 3 * mass * omega_rho**2 / (2 * hbar) * (x**2 / AR**2 + z**2 - 2 * z * z_sag)


def trap_detuning_xy(x, y, trap_x, trap_y):
    AR = trap_y / trap_x
    omega_rho = 2 * pi * trap_y
    return 3 * mass * omega_rho**2 / (2 * hbar) * (x**2 / AR**2 + y**2)


def rabi(t, omega, delta):
    dd = delta / omega
    return 1 / (1 + dd**2) * np.sin(0.5 * omega * t * np.sqrt(1 + dd**2))**2


def avg_rabi(t, omega, delta0):
    tau = 2 * pi * omega * t
    D = delta0 / omega
    b = tau * D**2 / (1 + 2 * D**2)
    return 0.5 * (1 - np.cos(tau + 0.5 * np.arctan(b)) / (1 + b**2)**(1. / 4)) / np.sqrt(1 + 2 * D**2)


def marconi_time_correction(t):
    """Corrects the marconi pulsetime [us] by 0.059 us"""
    return t - 0.059
