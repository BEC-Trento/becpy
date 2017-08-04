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
#from libraries.core_eos import *


def gaussian1d(x, mx, sx):
    """
    Returns the result of a  1D Gaussian.

    Args:
        (X, Y) (tuple of np.array): matrices of the coordinates to be combined,
        usually results of np.meshgrid
        mx (float): horizontal center
        my (float): vertical center
        sx (float): horizontal sigma
        sy (float): vertical sigma
        alpha (float): rotation angle in degrees 
        """
    return np.exp(- (x-mx)**2 / (2*sx**2))


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

class Fitting1d(object):
    """
    Base class for fitting routines. It has some common methods, other must be
    overridden for the specific fitting types with child classes.
    """

    def __init__(self, data, xvalues=None, par0=None):
        """
        Initialize the fitting routine with a given image.

        Args:
            data (np.array): image for the fit
            xvalues (np.array): data for x coordinate
            par0 (list): initial guess for the fit s(optional, not yet used)
        """
        self.data = data
        self.par0 = par0 #TODO: not used
        
        if xvalues is None:
            self.X = np.arange(self.data.size)
        else:
            self.X = xvalues
        
        #the fitted image result is initialized to None
        self.fitted = None
        self.par_fit = None
        self.err_fit = None
        #list of the parameter string names, must be implemented
        self.par_names = tuple([])

    def init_par0(self, *args, **kwargs):
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
        """
        #TODO handle when there is a wrong fit and set fit options
        #TODO consider parameters uncertanties
        try:
            results = curve_fit(self.function, self.X, self.data,
                                p0=self.par0, sigma=sigma, full_output=True)
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
        self.err_fit = errors_dict
        if output_cov:
            covmat = results[1]
            return results_dict, errors_dict, covmat
        else:
            return results_dict, errors_dict
        
class BoseDistribution(Fitting1d):
    """
    Fit per la densità sull'asse e per la pressione
    """
    
    def __init__(self, data, xvalues=None, par0=None, B_pot=1, nu=3/2, order=6):
        super(BoseDistribution, self).__init__(data, xvalues, par0)
        self.par_names = ["amp1", "mu", "temp"]
        self.nu = nu
        self.order = order
        self.B = B_pot
        if par0 is None:
            self.init_par0()

    def function(self, X, amp1, mu, temp):
        """
        Implements the density/pressure fitting function, in form
        
        """
        zeta = np.exp(mu - (self.B / temp) * X**2)    
#        summed = np.zeros(X.shape)
#        for k in range(self.order):
#            k+=1
#            summed += zeta**k / k**self.nu
            
        self.fitted = amp1*bose_g(zeta, self.nu, self.order)
        return self.fitted

    def init_par0(self):
        ''' initializes set of parameters: amp, fug, pot
        '''
        amp0 = self.data.max()
        mu0 = 0.1 #Salomon misura mu_0 = 0.1 kT
        s = self.data - self.data.mean()      
        temp0 = np.sqrt(np.sum(s**2)/s.size) / self.B
        
        par0 = (amp0, mu0, temp0)
        self.par0 = par0
        return par0
        
class BoseSimple(Fitting1d):
    """
    Fit per la densità sull'asse e per la pressione
    """
    
    def __init__(self, data, xvalues=None, par0=None, B_potential=1, nu=3/2, order=6):
        super(BoseSimple, self).__init__(data, xvalues, par0)
        self.par_names = ["amp1", "mu", "temp"]
        self.nu = None
        self.order = None
        self.B = B_potential
        if par0 is None:
            self.init_par0()

    def function(self, X, amp1, mu, temp):
        """
        Implements the density/pressure fitting function, in form
        
        """
        self.fitted = amp1*np.exp(mu/temp - (self.B / temp) * X**2)   
        return self.fitted

    def init_par0(self):
        ''' initializes set of parameters: amp, fug, pot
        '''
        amp0 = self.data.max()
#        mu0 = 0.1 #Salomon misura mu_0 = 0.1 kT
        s = self.data - self.data.mean()      
        temp0 = np.sqrt(np.sum(s**2)/s.size) / self.B
        mu0 = 0.1 * temp0
        
        par0 = (amp0, mu0, temp0)
        self.par0 = par0
        return par0
        
class Unosux(Fitting1d):
    """
    Classe di fit stupida, per una prova
    """
    
    def __init__(self, data, xvalues=None, par0=None):
        super(Unosux, self).__init__(data, xvalues, par0)
        self.par_names = ["a", "c"]
        if par0 is None:
            self.init_par0()

    def function(self, X, a, c):
        """
        Implements the density/pressure fitting function, in form
        f(x) = amp * g_nu(fug * exp(-pot * x**2)) where
        fug = exp(mu_0 / kT)        fugacity
        pot = m*omega_ax**2 / 2kT   potential coefficient
        """
        
        self.fitted = a / X**2 + c
        return self.fitted

    def init_par0(self):
        ''' initializes set of parameters: amp, fug, pot
        '''
        a = 1
        c = 0
        
        par0 = (a, c)
        self.par0 = par0
        return par0        
