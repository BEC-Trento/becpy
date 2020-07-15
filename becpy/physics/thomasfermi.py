#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 09-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np


def expansion_axial(t, omega_x, omega_rho, R0):
    tau = omega_rho * t
    eps = omega_x / omega_rho
    return R0 * (1 + eps**2 * (tau * np.arctan(tau) - 0.5*np.log(1 + tau**2)))


def expansion_radial(t, omega_rho, R0):
    return R0 * np.sqrt(1 + (omega_rho * t)**2)


if __name__ == '__main__':
    from becpy.physics import BECTF_3d

    omega_rho = 2*np.pi*150
    omega_ax = 2*np.pi*12

    Ry0 = 1
    Rx0 = omega_rho / omega_ax
    AR = Rx0 / Ry0
    print(AR)

    t = 4e-3

    rx = expansion_axial(t, omega_ax, omega_rho, Rx0)
    ry = expansion_radial(t, omega_rho, Ry0)

    print(rx/ry)
