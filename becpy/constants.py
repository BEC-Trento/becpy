#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Constants for 23Na
Let's reorganize this module later in case I need to add new atomic species

"""

from scipy.constants import pi, hbar, physical_constants

atomic_mass = physical_constants['atomic mass constant'][0]
aB = physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

mass = 23 * atomic_mass
scattering_length = 54.54 * aB  # 1, -1
interaction_constant_g = 4 * pi * hbar**2 * scattering_length / mass

wavelength = 589.162e-9  # D2 line
linewidth = 2 * pi * 9.80e6
cross_section = 3 * wavelength**2 / 2 / pi

# Riemann zeta function values
z2 = 1.644934066
z3 = 1.202056903
z32 = 2.612375348685
z52 = 1.341487257250
