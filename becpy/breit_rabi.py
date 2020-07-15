#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 03-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Breit--Rabi energy levels for ground state od 23Na

"""

import numpy as np
from scipy.constants import h, physical_constants

from scipy.optimize import brentq

I_nuclear = 3 / 2
gJ = 2.0023
muB = physical_constants['Bohr magneton'][0]
Ehf_freq = 1771.626 * 1e6
Ehf = h * Ehf_freq


def breit_rabi(B, F, mF):
    """F = 1, F = 2 Na B-field shift from Breit rabi formula.
    freq = breit_rabi(B, F, mF)

    Args:
        B : B field (Gauss)
        F : Hyperfine level F. Must be 1 or 2 (raises AssertionError instead)
        mF: Zeeman sublevel. Must be abs(mF) <= F (raises AssertionError)

    Returns:
        freq: frequency shift (MHz) wrt 3S_{1/2} fine structure energy.
              Note that mF = 0 levels have nonzero energy in this way.
    """
    # B (Gauss) -> freq (MHz)
    assert F in (1, 2)
    assert np.abs(mF) <= F

    x = gJ * muB * B * 1e-4 / Ehf
    if F == 2:
        sign = 1
    elif F == 1:
        sign = -1
    if mF == -2:
        sign *= np.sign(1 - x)
    return (-0.5 / (2 * I_nuclear + 1) + sign * 0.5 * np.sqrt(1 + 4 * mF * x / (2 * I_nuclear + 1) + x**2)) * Ehf_freq * 1e-6


def freq_shift(B, F, mF):
    return breit_rabi(B, F, mF) - breit_rabi(0, F, mF)


def transition_frequency(F, mF, F_prime, mF_prime, B=0):
    return breit_rabi(B, F_prime, mF_prime) - breit_rabi(B, F, mF)


def B_field(freq, F, mF, F_prime, mF_prime, *args, **kwargs):
    kw = dict(a=0, b=100)
    kw.update(kwargs)
    return brentq(lambda b: transition_frequency(F, mF, F_prime, mF_prime, b) - freq, *args, **kw)


if __name__ == '__main__':
    # cfr Steck, Figure 4.
    import matplotlib.pyplot as plt

    B = np.linspace(0, 3000, )

    for mF in range(-2, 3):
        plt.plot(B, breit_rabi(B, 2, mF) * 1e-3, 'C2', label=mF)
    for mF in range(-1, 2):
        plt.plot(B, breit_rabi(B, 1, mF) * 1e-3, 'C0', label=mF)
    plt.ylim(-5, 5)
    plt.show()
