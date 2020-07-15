#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Constants for 23Na
Let's reorganize this module later in case I need to add new atomic species

"""

from scipy.constants import pi, physical_constants

hbar = physical_constants['reduced Planck constant'][0]
atomic_mass = physical_constants['atomic mass constant'][0]
aB = physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

mass = 23 * atomic_mass
scattering_length = 54.54 * aB  # 1, -1

wavelength = 589.162e-9  # D2 line
linewidth = 2 * pi * 9.80e6
cross_section = 3 * wavelength**2 / 2 / pi
