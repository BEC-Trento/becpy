#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Created on Sun Mar  5 18:59:12 2017
# Copyright (C) 2016  Carmelo Mordini
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
A library of commonly used functions. This is shared between the modules

    * fitting
    * imageprocess
"""

import numpy as np
from scipy.special import spence

try:
    import mpmath as mp #fast, low precision implementation
    from functools import partial
    
    _g32 = partial(mp.fp.polylog, 3/2)
    _g52 = partial(mp.fp.polylog, 5/2)
    
    g32 = np.vectorize(_g32, otypes=[np.float64,], doc="polylog(3/2, x). Vectorized by numpy.")
    g52 = np.vectorize(_g52, otypes=[np.float64,], doc="polylog(5/2, x). Vectorized by numpy.")
except ImportError:
    #FIXME: find an alternative for this
    print('mpmath module not found. The g32 and g52 functions will not be availlabe')
    def g32():
        raise NotImplementedError
    def g52():
        raise NotImplementedError

def g2(x):
    """ polylog(2, x) implemented via scipy.special.spence.
        This translation is due to scipy's definition of the dilogarithm
    """
    return spence(1.0 - x)

def guess_from_peak_2d(image, X=None, Y=None):
    """
        Guess and returns the initial 2d peak-model parameters
    """    
    height, width = image.shape
    ym, xm = np.unravel_index(np.nanargmax(image), image.shape)
    amp = image[ym, xm]
    if X is None:
        mx = xm
        x = np.arange(width)-mx
    else:
        mx = X[ym, xm]
        x = X[ym,:]
    if Y is None:
        my = ym
        y = np.arange(height)-my
    else:
        my = Y[ym, xm]
        y = Y[:,xm]
    sx = np.nanstd((x-mx)*image[ym,:]/amp)
    sy = np.nanstd((y-my)*image[:,xm]/amp)
    return amp, mx, my, sx, sy

def gauss1d(X, A, mx, rx, offs=0,):
    return A*np.exp(-(X-mx)**2/(2*sx**2)) + offs

def bimodal1d(X, A, B, mx, sx, rx, offs=0, return_both=False):
    g = A*np.exp(-(X-mx)**2/(2*sx**2))
    b = B*np.maximum(0, 1 - ((X-mx)/rx)**2)**2
    if return_both:
        return g + offs, b + offs
    else:
        return g + b + offs


def gaussian2d(X, Y, amp, mx, my, sx, sy, offset=0):
    """gaussian2d(X, Y, amplitude, mx, my, sx, sy)
       A 2d gaussian function.
    """
    return amp * np.exp(-(X-mx)**2/(2*sx**2) -(Y-my)**2/(2*sy**2)) + offset

def bose2d(X, Y, amp, mx, my, sx, sy, offset=0):
    """
    Bose function for the integrated density profile of a (not so hot) thermal cloud.
    """
    return amp * g2(gaussian2d(X, Y, 1., mx, my, sx, sy, 0)) + offset

def thomasfermi2d(X, Y, amp, mx, my, rx, ry, offset=0):
    """thomasfermi2d(X, Y, amplitude, mx, my, sx, sy)
       A 2d Thomas-Fermi (integrated inverted parabola) function.
    """
    b = 1 -((X-mx)/rx)**2 -((Y-my)/ry)**2
    b = np.maximum(0, b**3)
    return amp * np.sqrt(b) + offset

def bimodal2d(XY, A, B, mx, my,sx, sy, rx, ry, offs=0, return_both=False):
    X, Y = XY
    g = gaussian2d(X, Y, A, mx, my, sx, sy)
    b = thomasfermi2d(X, Y, B, mx, my, rx, ry,)
    if return_both:
        return g + offs, b + offs
    else:
        return g + b + offs

def bimodal2d_fit(XY, A, B, mx, my, sx, sy, rx, ry, offs=0):
    return bimodal2d(XY, A, B, mx, my, sx, sy, rx, ry, offs=0).ravel()

def rotate(X, Y, mx, my, alpha):
    alpha = (np.pi / 180.) * float(alpha)
    xr = + (X-mx) * np.cos(alpha) - (Y-my) * np.sin(alpha)
    yr = + (X-mx) * np.sin(alpha) + (Y-my) * np.cos(alpha)
    return xr + mx, yr + my
    
def thomasfermi2d_rotate(X, Y, amp, mx, my, rx, ry, alpha, offset=0):
    """thomasfermi2d(X, Y, amplitude, mx, my, sx, sy)
       A 2d Thomas-Fermi (integrated inverted parabola) function.
       alpha must be in radians
    """
    xr, yr = rotate(X, Y, mx, my, alpha)
    b = 1 -((X-mx)/rx)**2 -((Y-my)/ry)**2
    b = np.maximum(0, b**3)
    return amp * np.sqrt(b) + offset





