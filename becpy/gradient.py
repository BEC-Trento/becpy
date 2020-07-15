#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np
from scipy.special import binom


def smooth_gradient_n(y, N, dx=1, pad_mode='edge', axis=-1):
    # http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    assert N % 2 == 1  # N must be odd
    m = (N - 3) // 2
    M = (N - 1) // 2
    cks = (2**(-(2 * m + 1)) * (binom(2 * m, m - k + 1)
                                - binom(2 * m, m - k - 1))
           for k in range(1, M + 1))

    pw = [(0, 0)] * y.ndim
    pw[axis] = (N, N)
    y = np.pad(y, pad_width=pw, mode=pad_mode)
    if np.ndim(dx) != 0:
        # FIXME this actually assumes that x has the same dimensions as y
        x = np.pad(dx, pad_width=pw, mode=pad_mode)
        even = False
    else:
        even = True
    d = np.zeros_like(y)
    for j, ck in enumerate(cks):
        k = j + 1
        dx = dx if even else (np.roll(x, -j - 1, axis) -
                              np.roll(x, j + 1, axis)) / 2 / k
        d += ck * (np.roll(y, -j - 1, axis) - np.roll(y, j + 1, axis)) / dx
    S = [slice(None)] * y.ndim
    S[axis] = slice(N, -N)
    y_grad = d[tuple(S)]
    return y_grad
