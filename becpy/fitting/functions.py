#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np


def gaussian2d(x, y, amp, mx, my, sx, sy, offset=0):
    """gaussian2d(x, y, amplitude, mx, my, sx, sy)
       A 2d gaussian function.
    """
    return amp * np.exp(-(x - mx)**2 / (2 * sx**2) - (y - my)**2 / (2 * sy**2)) + offset


def guess_1d(x, data):
    # print(data)
    data = np.nan_to_num(data).clip(0)
    # print(data)
    if np.all(data == 0):
        data = np.ones_like(x)
    mx = np.average(x, weights=data)
    sx = np.sqrt(np.average((x - mx)**2, weights=data))
    return mx, sx


def guess_from_peak_2d(image, X=None, Y=None):
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
