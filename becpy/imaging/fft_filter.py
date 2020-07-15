#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 07-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Cut-and-paste FFT image filtering

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

try:
    from pyfftw.interfaces import numpy_fft as fft
except ModuleNotFoundError:
    from numpy import fft


def fft_filter(im, rois, shift, plot=True, imshow_kwargs={}):
    f0 = fft.fftshift(fft.fft2(im))
    f1 = f0.copy()

    if plot:
        kw = dict(vmin=-0.05, vmax=1)
        kw.update(imshow_kwargs)
        fig, axes = plt.subplots(2, 2, figsize=(
            9, 4), sharex='col', sharey='col')
        (ax, ax_f), (ax1, ax1_f) = axes
        ax.imshow(im, **kw)
        ax_f.imshow(abs(f0), norm=LogNorm())

    dx, dy = shift
    for j, roi in enumerate(rois):
        print(j, roi)
        roi_t = translate_roi(roi, dx, dy)
        roi_symm = symmetric_roi(roi, im.shape)
        roi_symm_t = symmetric_roi(roi_t, im.shape)

        f1[roi] = f1[roi_t]
        f1[roi_symm] = f1[roi_symm_t]
        if plot:
            roi_to_rect(roi, ax=ax_f, index=j)
            roi_to_rect(roi, ax=ax1_f)
            roi_to_rect(roi_t, ax=ax_f, ec='r')
            roi_to_rect(roi_symm, ax=ax_f)
            roi_to_rect(roi_symm, ax=ax1_f)
            roi_to_rect(roi_symm_t, ax=ax_f, ec='r')

    im1 = fft.ifft2(fft.ifftshift(f1)).real
    if plot:
        ax1.imshow(im1, **kw)
        ax1_f.imshow(abs(f1), norm=LogNorm())

    return im1


def symmetric_roi(roi, shape):
    r0, r1 = roi
    cy, cx = shape[0] // 2, shape[1] // 2
    return slice(2 * cy - r0.stop, 2 * cy - r0.start), slice(2 * cx - r1.stop, 2 * cx - r1.start)


def translate_roi(roi, dx, dy):
    r0, r1 = roi
    return slice(r0.start + dy, r0.stop + dy), slice(r1.start + dx, r1.stop + dx)


def roi_to_rect(roi, ax=None, index=None, **kwargs):
    slice_h, slice_w = roi
    xy = (slice_w.start, slice_h.start)
    width = slice_w.stop - slice_w.start
    height = slice_h.stop - slice_h.start
    r_kw = dict(fill=False,
                edgecolor='white',
                linewidth=1.)
    r_kw.update(kwargs)
    rect = Rectangle(xy, width, height, **r_kw)
    if ax is not None:
        ax.add_patch(rect)
        if index is not None:
            ax.text(*xy, index, color=r_kw['edgecolor'])
    return rect


if __name__ == '__main__':

    im = np.load('od_fringes.npz', allow_pickle=True)['od']
    rois = [
        np.index_exp[78:85, 210:220],
        np.index_exp[75:80, 220:225],
    ]

    fft_filter(im, rois, shift=(30, 15), imshow_kwargs=dict(vmin=-0.05, vmax=0.3), plot=True)
    plt.show()
