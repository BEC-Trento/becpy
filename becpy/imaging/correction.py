#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""Module docstring

"""

import numpy as np

from scipy.ndimage.interpolation import shift
from skimage.transform import rotate


def lsq_match(a, b, mask):
    """
    beta is best lstsq coefficient for
        a - beta * b
    so that b matches a when multiplied by beta
    """
    mask = np.where(mask)
    a = a.copy()[mask]
    b = b.copy()[mask]
    beta = (a @ b) / (b @ b)
    return beta


def probe_correction_beta(atoms, probe, mask, verbose=True):
    mask = mask.astype(bool)
    beta = lsq_match(atoms, probe, mask)
    if verbose:
        print(f"probe matched to atoms with beta = {beta:.6f}")
    probe_correct = probe * beta
    return probe_correct


def center_and_trim(od, roi, center, angle=None, verbose=True):
    """
    center and rotate an image wrt to a given roi, then crop

    Params:
    -------
        od: the image
        roi: tuple of slices or np.index_exp. If None, the image will not be cropped
        center: (mx, my) wrt the non-cropped image
        angle: clock(?)wise angle. Rotation is skipped by default (angle = None)

    Returns:
    --------
        od1: the image after shifting, rotation and crop
    """

    if roi is None:
        h, w = od.shape
        roi = (slice(0, h), slice(0, w))
    # mx, my = center
    # C0 = mx - roi[1].start, my - roi[0].start
    C0 = center
    C1 = 0.5 * (roi[1].start + roi[1].stop), 0.5 * (roi[0].start + roi[0].stop)
    center = C0

    # shift in y, x
    dxy = C1[1] - C0[1], C1[0] - C0[0]
    if verbose:
        print(f"Shift by {dxy}")
    od = shift(od, dxy, mode='wrap')
    center = C1

    if angle:
        if verbose:
            print(f"Rotate image by an angle {angle}")
        od = rotate(od, -angle, resize=False, center=center, mode='wrap',
                     order=3, preserve_range=True,)

    od = od[roi]
    # shape = od.shape
    # assert shape[0] % 2 == 1
    # assert shape[1] % 2 == 1

    return od
