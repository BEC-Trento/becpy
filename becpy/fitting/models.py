#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 03-2017 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
lmfit predefined models

"""

import lmfit
from . import functions as funcs


class Model2D(lmfit.Model):
    def __init__(self, func, *args, **kwargs):
        super().__init__(func, independent_vars=['x', 'y'],
                         nan_policy='omit', *args, **kwargs)


class Gaussian2DModel(Model2D):
    __doc__ = funcs.gaussian2d.__doc__ + \
        lmfit.models.COMMON_DOC if funcs.gaussian2d.__doc__ else ""

    def __init__(self, *args, **kwargs):
        super().__init__(
            funcs.gaussian2d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('sy', min=0)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = funcs.guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp=amp, mx=mx, my=my, sx=sx, sy=sy, offset=0, alpha=0)


class ThomasFermi2DModel(Model2D):
    __doc__ = funcs.gaussian2d.__doc__ + \
        lmfit.models.COMMON_DOC if funcs.thomasfermi2d.__doc__ else ""

    def __init__(self, *args, **kwargs):
        super().__init__(
            funcs.thomasfermi2d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('ry', min=0)
        self.set_param_hint('ry', min=0)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = funcs.guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp=amp, mx=mx, my=my, rx=sx, ry=sy, offset=0, alpha=0)


class BimodalBose2DModel(Model2D):
    __doc__ = funcs.gaussian2d.__doc__ + \
        lmfit.models.COMMON_DOC if funcs.bimodalbose2d.__doc__ else ""

    def __init__(self, *args, **kwargs):
        super().__init__(
            funcs.bimodalbose2d, *args, **kwargs)
        self.set_param_hint('amp_tf', min=0)
        self.set_param_hint('amp_g', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = funcs.guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp_g=amp / 2, amp_tf=amp, mx=mx, my=my,
                                sx=sx, sy=sy, rx=sx, ry=sy / 2, offset=0, alpha=0)
