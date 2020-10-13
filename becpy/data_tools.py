#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 01-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>
"""
Module docstring
"""

import numpy as np


def average_repeat(x, y):
    xu = np.unique(x)
    yu = np.empty(xu.shape, dtype=y.dtype)
    yerr = np.empty_like(xu)
    for j, v in enumerate(xu):
        ix = (x == v)
        yu[j] = y[ix].mean()
        yerr[j] = y[ix].std()
    return xu, yu, yerr
