#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 11-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Module docstring

"""
import numpy as np
import matplotlib.pyplot as plt

import concurrent.futures
from tqdm import tqdm

from becpy.physics import hartree_fock as hf
from scipy.constants import Boltzmann as kB

npoints = 100

freq_x = 12.28
freq_y = 137.65
omega_x = 2 * np.pi * freq_x
omega_rho = 2 * np.pi * freq_y


T = np.linspace(50, 1000, npoints)  # nK
mu0 = np.linspace(-1000, 400, npoints)  # nK

omega_ho = (omega_x * omega_rho**2)**(1 / 3)
AR = omega_rho / omega_x

mu0, T = np.meshgrid(mu0, T, indexing='ij')

shape = mu0.shape

N0 = np.empty_like(T) * np.nan
Nt = np.empty_like(T) * np.nan

density_model = 'hartree_fock'

values = list(zip(mu0.ravel(), T.ravel()))
K = list(range(len(values)))


def main(k):
    _mu0, _T = values[k]
    # print(ix)
    return hf.get_N(_mu0 * 1e-9 * kB, _T * 1e-9, omega_rho, AR,
                    split=True, density_model=density_model,
                    Rmax=5e-3, dr=3e-7)


with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    for k, (_N0, _Nt) in zip(K, tqdm(executor.map(main, K), total=len(K))):
        ix = np.unravel_index(k, shape)
        N0[ix] = _N0
        Nt[ix] = _Nt


# np.savez_compressed('../becpy/data/hf_scan.npz', mu0=mu0, T=T, N0=N0, Nt=Nt)
np.savez_compressed('../becpy/data/hf_scan.npz', mu0=mu0, T=T, N0=N0, Nt=Nt, freq_x=freq_x, freq_y=freq_y)


plt.imshow(N0)
plt.show()
