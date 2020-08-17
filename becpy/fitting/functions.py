#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np
from scipy.special import spence


def g2(x):
    """ polylog(2, x) implemented via scipy.special.spence.
        This translation is due to scipy's definition of the dilogarithm
    """
    return spence(1.0 - x)


z2 = g2(1)


def guess_1d(x, data):
    # print(data)
    data = np.nan_to_num(data).clip(0)
    # print(data)
    if np.all(data == 0):
        data = np.ones_like(x)
    mx = np.average(x, weights=data)
    sx = np.sqrt(np.average((x - mx)**2, weights=data))
    return mx, sx


def guess_from_peak_2d(image, x=None, y=None):
    # print('image ---------------')
    # print(image)
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    mx, sx = guess_1d(x, image.sum(0))
    my, sy = guess_1d(y, image.sum(1))
    # amp = image[int(my), int(mx)]
    amp = np.nanmax(image)
    return [amp, mx, my, sx, sy]


def rotate(x, y, mx, my, alpha):
    alpha = (np.pi / 180.) * float(alpha)
    xr = + (x - mx) * np.cos(alpha) - (y - my) * np.sin(alpha)
    yr = + (x - mx) * np.sin(alpha) + (y - my) * np.cos(alpha)
    return xr + mx, yr + my


def gaussian2d(x, y, amp, mx, my, sx, sy, offset=0, alpha=0):
    """gaussian2d(x, y, amplitude, mx, my, sx, sy)
       A 2d gaussian function.
    """
    if alpha:
        x, y = rotate(x, y, mx, my, alpha)
    return amp * np.exp(-(x - mx)**2 / (2 * sx**2) - (y - my)**2 / (2 * sy**2)) + offset


def bose2d(x, y, amp, mx, my, sx, sy, offset=0, alpha=0):
    """
    Bose function for the integrated density profile of a (not so hot) thermal cloud.
    """
    return amp * g2(gaussian2d(x, y, 1., mx, my, sx, sy, 0, alpha)) / z2 + offset


def thomasfermi2d(x, y, amp, mx, my, rx, ry, offset=0, alpha=0):
    """thomasfermi2d(x, y, amplitude, mx, my, sx, sy)
       A 2d Thomas-Fermi (integrated inverted parabola) function.
    """
    if alpha:
        x, y = rotate(x, y, mx, my, alpha)
    b = np.maximum(0, 1 - ((x - mx) / rx)**2 - ((y - my) / ry)**2)**(3. / 2)
    return amp * b + offset


def bimodalbose2d(x, y, amp_g, amp_tf, mx, my, sx, sy, rx, ry, offset=0, alpha=0, return_split=False):
    if alpha:
        x, y = rotate(x, y, mx, my, alpha)
    g = bose2d(x, y, amp_g, mx, my, sx, sy)
    b = thomasfermi2d(x, y, amp_tf, mx, my, rx, ry,)
    if return_split:
        return g, b, offset
    else:
        return g + b + offset
