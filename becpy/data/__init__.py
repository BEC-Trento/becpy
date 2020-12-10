#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 11-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Module docstring

"""
import sys
import numpy as np
from importlib.resources import path

thismodule = sys.modules[__name__]

with path(thismodule, 'hf_scan.npz') as f:
    hf_scan = np.load(f)
