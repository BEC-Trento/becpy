#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np
import h5py

from .correction import probe_correction_beta


def get_od(atoms, probe, dark, alpha, chi_sat, pulsetime, mask=None):
    atoms = (atoms - dark).clip(1)
    probe = (probe - dark).clip(1)

    if mask is not None:
        probe = probe_correction_beta(atoms, probe, mask)
    return alpha * np.log(probe / atoms) + (probe - atoms) / chi_sat / pulsetime


def get_raw_data(filename, camera='cam_y2', pulsetime='probe_pulsetime'):

    with h5py.File(filename, 'r') as f:
        _images = f[f'data/{camera}/images']
        images = _images[:].astype(float)
        img_names = _images.attrs['img_names']
        images = dict(zip(img_names, images))

        calibrations = dict(f[f'calibrations/{camera}'].attrs)
        calibrations['pixel_size'] /= calibrations['magnification']
        calibrations['pulsetime'] = f['globals'].attrs['probe_pulsetime']

    return images, calibrations
