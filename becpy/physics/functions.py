#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 08-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np

import mpmath as mp  # fast, low precision implementation
from functools import partial

_g12 = partial(mp.fp.polylog, 1 / 2)
_g32 = partial(mp.fp.polylog, 3 / 2)
_g52 = partial(mp.fp.polylog, 5 / 2)
_g3 = partial(mp.fp.polylog, 3)

g12 = np.vectorize(_g12, otypes=[np.float64], doc="polylog(1/2, x). Vectorized by numpy.")
g32 = np.vectorize(_g32, otypes=[np.float64], doc="polylog(3/2, x). Vectorized by numpy.")
g52 = np.vectorize(_g52, otypes=[np.float64], doc="polylog(5/2, x). Vectorized by numpy.")
g3 = np.vectorize(_g3, otypes=[np.float64], doc="Vectorized mp.polylog(3, x).")
