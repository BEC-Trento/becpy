# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:32:11 2014 by simone
Edited on Apr 2015 

@author: carmelo

Module containing the fitting routines.
"""
# pylint: disable=E1101

from scipy.optimize import curve_fit
import numpy as np
from mpmath import polylog

from .fitting1d import gaussian1d
#from libraries.core_eos import *

def rotate(X, Y, alpha):
    alpha = (np.pi / 180.) * float(alpha)
    xr = + X * np.cos(alpha) - Y * np.sin(alpha)
    yr = + X * np.sin(alpha) + Y * np.cos(alpha)
    X, Y = xr, yr
    return X, Y

def gaussian(X, Y, mx, my, sx, sy, alpha=0):
    """
    Returns the result of a Gaussian.

    Args:
        (X, Y) (tuple of np.array): matrices of the coordinates to be combined,
        usually results of np.meshgrid
        mx (float): horizontal center
        my (float): vertical center
        sx (float): horizontal sigma
        sy (float): vertical sigma
        alpha (float): rotation angle in degrees 
        """
    X = X-mx
    Y = Y-my
    if alpha != 0:
        X, Y = rotate(X, Y, alpha)
    
    return np.exp(- X**2 / (2*sx**2) - Y**2 / (2*sy**2))


def thomasfermi(X, Y, mx, my, rx, ry, alpha=0):
    """
    Returns the result of a Thomas-Fermi function (inverted parabola).

    Args:
        (X, Y) (tuple of np.array): matrices of the coordinates to be combined,
        usually results of np.meshgrid
        mx (float): horizontal center
        my (float): vertical center
        rx (float): horizontal TF radius
        ry (float): vertical TF radius
        alpha (float): rotation angle in degrees 
    """
    X = X-mx
    Y = Y-my
    if alpha != 0:
        X, Y = rotate(X, Y, alpha)
    
    b = (1 - (X/rx)**2 - (Y/ry)**2)
    b = np.maximum(b, 0)
    b = np.sqrt(b)
    return b**3

_bose_g = np.vectorize(polylog)
def bose_g(z, nu, order):
    if order == np.infty:
        s = _bose_g(nu, z)
        return s.astype(float)
    else:
        summed = np.zeros(z.shape)
        for k in range(order):
            k+=1
            summed += z**k / k**nu
    #    print('z: ',z)
    #    print('sum: ',summed)
        return summed

            
''' ora i fit 2d per gaussiana, thomasfermi e bimodale '''

class Fitting2d(object):
    """
    Base class for fitting routines. It has some common methods, other must be
    overridden for the specific fitting types with child classes.
    """

    def __init__(self, img0, par0=None):
        """
        Initialize the fitting routine with a given image.

        Args:
            img0 (np.array): image for the fit
            par0 (list): initial guess for the fit s(optional, not yet used)
        """
        self.img0 = img0
        self.par0 = par0 #TODO: not used

        #calculates the matrices with the x and y coordinates
        ny, nx = self.img0.shape
        x = np.arange(nx)
        y = np.arange(ny)
#        self.X, self.Y = np.meshgrid(x, y)
        X, Y = np.meshgrid(x, y)
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        #the fitted image result is initialized to None
        self.fitted = None

        #list of the parameter string names, must be implemented
        self.par_names = tuple([])

    def guess_gauss_par0(self, slc_main, slc_max, slc_bkg):
        """
        Guess and returns the initial gaussian parameters from the slices

        Args:
            slc_main (tuple): tuple of 2 slices for the coordinates of the main
            area for the gaussian center guess
            slc_max (tuple): tuple of 2 slices for the coordinates of the
            maximal area for the gaussian amplitude guess
            slc_bkg (tuple): tuple of 2 slices for the coordinates of the
            background area for the gaussian offset guess

        Returns:
            Tuple with the Gaussian guessed parameters
        """
        if slc_main is None:
            height, width = self.img0.shape
            slc_main = slice(None)
            ym, xm = np.unravel_index(self.img0.argmax(), self.img0.shape)
        else:
            height, width = self.img0[slc_main].shape
            xm = np.mean(slc_main[1].indices(self.img0.shape[1])[0:-1])
            ym = np.mean(slc_main[0].indices(self.img0.shape[0])[0:-1])

        offs = np.mean(self.img0[slc_bkg])
        if slc_max is None:
            slc_max = slice(None)
            
        amp1 = np.mean(self.img0[slc_max])
        if offs is np.masked:
#            print('bkg mascherato')
            offs = 0
#        print(type(offs))
#        print(offs, amp1, xm, ym, width/4.0, height/4.0)
        return (offs, amp1, xm, ym, width/4.0, height/4.0)

    def guess_par0(self, *args, **kwargs):
        """
        Parameters guess for the specific function (must be overridden).
        """
        pass

    def function(self, *args, **kwargs):
        """
        Specific function (must be overridden).
        """
        pass

    def fit(self, sigma=None, output_cov=False):
        """
        Performs the fitting operations and returns a dictionary with the
        fitted parameters.
        outputs: results_dict, errors_dict [, covmat (optional)]
        """
        frame = self.img0
        #TODO handle when there is a wrong fit and set fit options
        #TODO consider parameters uncertanties
        try:
            results = curve_fit(self.function, (self.X, self.Y), frame.ravel(),
                                p0=self.par0, sigma=sigma)
            results_dict = dict(list(zip(self.par_names, results[0])))
            errors_dict = dict(list(zip(self.par_names,
                                    np.sqrt(np.diag(results[1]))))) 
        except RuntimeError:
            print("Error while fitting")
            results = [self.par0, [None for r in self.par0]]
            results_dict = dict(list(zip(self.par_names, results[0])))
            errors_dict = dict(list(zip(self.par_names,
                                    results[1])))
                                    
        self.par_fit = results_dict
        if output_cov:
            covmat = results[1]
            return results_dict, errors_dict, covmat
        else:
            return results_dict, errors_dict
        

class Gauss2d(Fitting2d):
    """
    Gaussian 2D fit.
    """
    def __init__(self, img0, par0=None):
        super(Gauss2d, self).__init__(img0, par0)
        self.par_names = ["offs", "amp1", "mx", "my", "sx", "sy", "alpha"]

    def function(self, XY, offs, amp1, mx, my, sx, sy, alpha):
        """
        Implements the gaussian fitting function.
        (see gaussian() and thomasfermi())
        """
        self.fitted = amp1*gaussian(XY, mx, my, sx, sy, alpha) + offs
        return self.fitted.ravel()

    def guess_par0(self, slc_main, slc_max, slc_bkg):
        """
        Implements the gaussian parameter guess from slices.
        (see Fitting.guess_gauss_par0())
        """
        offs, amp1, mx, my, sx, sy = self.guess_gauss_par0(slc_main,
                                                           slc_max,
                                                           slc_bkg)
        alpha0 = 10.
        par0 = (offs, amp1, mx, my, sx, sy, alpha0)

        self.par0 = par0
        return par0


class ThomasFermi2d(Fitting2d):
    """
    Thomas-Fermi 2D fit (inverted parabola).
    """
    def __init__(self, img0, par0=None):
        super(ThomasFermi2d, self).__init__(img0, par0)
        self.par_names = ["offs", "amp1", "mx", "my", "rx", "ry", "alpha"]

    def function(self, XY, offs, amp2, mx, my, rx, ry, alpha):
        """
        Implements the Thomas-Fermi fitting function.
        (see gaussian() and thomasfermi())
        """
        self.fitted = amp2*thomasfermi(XY, mx, my, rx, ry, alpha) + offs
        return self.fitted.ravel()

    def guess_par0(self, slc_main, slc_max, slc_bkg):
        """
        Implements the Thomas-Fermi parameter guess.
        (see Fitting.guess_gauss_par0())
        """
        offs, amp1, mx, my, sx, sy = self.guess_gauss_par0(slc_main,
                                                           slc_max,
                                                           slc_bkg)
        alpha0 = 10.
        par0 = (offs, amp1, mx, my, sx*2.0, sy*2.0, alpha0)

        self.par0 = par0
        return par0


class Bimodal2d(Gauss2d, ThomasFermi2d):
    """
    Gaussian+Thomas Fermi bimodal 2D fit.
    """
    def __init__(self, img0, par0=None):
        super(Bimodal2d, self).__init__(img0, par0)
        self.par_names = ["offs", "amp1", "mx", "my", "sx", "sy",
                          "amp2", "rx", "ry", "alpha"]

    def function(self, XY, offs, amp1, mx, my, sx, sy, amp2, rx, ry, alpha):
        """
        Implements the bimodal fitting function.
        (see gaussian() and thomasfermi())
        """
        self.fitted = amp1*gaussian(XY, mx, my, sx, sy, alpha) +\
                      amp2*thomasfermi(XY, mx, my, rx, ry, alpha) + offs
        return self.fitted.ravel()

    def guess_par0(self, slc_main, slc_max, slc_bkg):
        """
        Implements the bimodal parameter guess.
        (see Fitting.guess_gauss_par0())
        """
        offs, amp1, mx, my, sx, sy = self.guess_gauss_par0(slc_main,
                                                           slc_max,
                                                           slc_bkg)
        alpha0 = 10.
        par0 = (offs, amp1/2.0, mx, my, sx, sy, amp1/2.0, sx*2.0, sy*2.0, alpha0)

        self.par0 = par0
        return par0




