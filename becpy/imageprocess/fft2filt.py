#!/usr/bin/python
# -*- coding: utf-8 -*-
#
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
import numpy as np

try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2
    
def gauss2(x, y, mx, my, ax, ay):
    return np.exp(-(x-mx)**2 / (2*ax**2)) * np.exp(-(y-my)**2 / (2*ay**2))#/ (np.sqrt(2*np.pi)*a)
def invgauss2(x, y, mx, my, ax, ay):
    return 1 - gauss2(x, y, mx, my, ax, ay)
def filter2(FX, FY, mx, my, ss):
    return invgauss2(FX, FY, mx, my, ss, ss) * invgauss2(FX, FY, -mx, -my, ss, ss)
