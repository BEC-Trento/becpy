#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Created on Fri Feb 10 21:23:33 2017
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

from lmfit import Parameters
from lmfit.model import Model, CompositeModel, operator
from lmfit.models import update_param_vals, COMMON_DOC

from ..functions import gaussian2d, thomasfermi2d, bose2d, guess_from_peak_2d, thomasfermi2d_rotate


class Model2D(Model):
    def __init__(self, func, X=None, Y=None, *args, **kwargs):
        super(Model2D, self).__init__(func, independent_vars=['X', 'Y'], missing='drop', *args, **kwargs)
        
#    def fit(self, data, *args, **kwargs):
#        if 'X' not in kwargs or 'Y' not in kwargs:
#            h, w = data.shape
#            Y, X = np.mgrid[:h, :w]
#            kwargs.update({'X': X, 'Y': Y})
#        output = super(Model2D, self).fit(data, *args, **kwargs)
#        return
        
    

class Gaussian2DModel(Model2D):
    __doc__ = gaussian2d.__doc__ + COMMON_DOC if gaussian2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Gaussian2DModel, self).__init__(gaussian2d, *args, **kwargs)
        self.set_param_hint('sx', min=0)
        self.set_param_hint('sy', min=0)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        pars = self.make_params(amp=amp, mx=mx, my=my, sx=sx, sy=sy)
        return update_param_vals(pars, self.prefix, **kwargs)

class ThomasFermi2DModel(Model2D):
    __doc__ = thomasfermi2d.__doc__ + COMMON_DOC if thomasfermi2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(ThomasFermi2DModel, self).__init__(thomasfermi2d, *args, **kwargs)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)
#        self.physical_params = Parameters()
#        self.physical_params.add_many(
#                ('sigma0', cross_section, False, None, None, None),
#                ('N', 0, False, None, None, '1e-3*1.25*amp *rx*ry /sigma0'))

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        pars = self.make_params(amp=amp, mx=mx, my=my, rx=sx, ry=sy)
        return update_param_vals(pars, self.prefix, **kwargs)
    
class ThomasFermi2DModel_rotate(Model2D):
    __doc__ = thomasfermi2d_rotate.__doc__ + COMMON_DOC if thomasfermi2d_rotate.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(ThomasFermi2DModel_rotate, self).__init__(thomasfermi2d_rotate, *args, **kwargs)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)
#        self.physical_params = Parameters()
#        self.physical_params.add_many(
#                ('sigma0', cross_section, False, None, None, None),
#                ('N', 0, False, None, None, '1e-3*1.25*amp *rx*ry /sigma0'))

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        pars = self.make_params(amp=amp, mx=mx, my=my, rx=sx, ry=sy, alpha=0)
        return update_param_vals(pars, self.prefix, **kwargs)

class Bimodal2DModel(CompositeModel):
    __doc__ = "Composite model for a Gaussian + TF function\n" + COMMON_DOC
    def __init__(self, link_centers=True, *args, **kwargs):
        self.gaussmod = Gaussian2DModel(prefix='g_',*args, **kwargs)
        self.tfmod = ThomasFermi2DModel(*args, **kwargs)
        super(Bimodal2DModel, self).__init__(self.tfmod, self.gaussmod, operator.add,
             independent_vars=['X', 'Y'], missing='drop', *args, **kwargs)
        if link_centers:
            self.link()
        self.set_param_hint('g_offset', value=0, vary=None)
        self.set_param_hint('sx', expr='g_sx')
        self.set_param_hint('sy', expr='g_sy')
        

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        amp2 = amp*3
        pars = self.make_params(g_amp=amp, g_mx=mx, g_my=my, g_sx=5*sx, g_sy=5*sy,
                                amp=amp2, mx=mx, my=my, rx=sx, ry=sy)
        return update_param_vals(pars, self.prefix, **kwargs)
    
    def link(self):
        self.set_param_hint('g_mx', vary=False, expr='mx')
        self.set_param_hint('g_my', vary=False, expr='my')
        
    def unlink(self):
        self.set_param_hint('g_mx', vary=True, expr=None)
        self.set_param_hint('g_my', vary=True, expr=None)