#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-1970 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
Vortex finding functions for 2D GPE
code by C. J. Foster (2008) <https://arxiv.org/abs/1302.0470>
source: https://github.com/c42f/vortutils
translated from MATLAB by CM
"""


import numpy as np
from scipy.signal import convolve2d

from tqdm.autonotebook import tqdm

L1 = np.asarray([
    [-0.5, -0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5, 0.5],
    [0.75, 0.25, 0.25, -0.25]
])

filter1 = np.asarray([[-1, 1], [1, -1]])
filter2 = - filter1


def pseudovorticity(x, y, psiR, psiI, axis=None):
    dx_psiR, dy_psiR = np.gradient(psiR, x, y, axis=axis)
    dx_psiI, dy_psiI = np.gradient(psiI, x, y, axis=axis)
    omega = dx_psiR * dy_psiI - dy_psiR * dx_psiI
    return omega


def phase_winding2d(phase):
    vorticity = (convolve2d(np.unwrap(phase, axis=0), filter1, mode='valid') + \
                 convolve2d(np.unwrap(phase, axis=1), filter2, mode='valid')) / 2 / np.pi
    return vorticity


def vortex_detect2d(x, y, psiR, psiI, cutoff_radii=None):
    # this is explained (not so well) also here
    # https://loriordan.netlify.app/post/vortex_2d/

    phase = np.arctan2(psiI, psiR)
    vorticity = phase_winding2d(phase)

    # phase_winding has mode='valid', so these are the upper-left corner of the plaquette
    indices = np.asarray(np.where(np.round(vorticity) != 0)).T  # shape = (nvortices, 2)
    coords = np.stack([x[indices[:, 0]], y[indices[:, 1]]], axis=1)

    if cutoff_radii is not None:
        rmin, rmax = cutoff_radii
        r = np.hypot(coords[:, 0], coords[:, 1])
        select = (r >= rmin) & (r < rmax)
        indices = indices[select]
        coords = coords[select]

    dx = [np.diff(x).mean(), np.diff(y).mean()]

    c0 = coords.copy()

    for j, (ix, iy) in enumerate(indices):
        # for sure there is a way to vectorise ALL of this
        sl = np.index_exp[ix:ix + 2, iy:iy + 2]
        a1, b1, c1 = L1 @ psiR[sl].ravel()
        a2, b2, c2 = L1 @ psiI[sl].ravel()
        A = np.asarray([[a1, b1], [a2, b2]])
        c = np.asarray([c1, c2])
        shift = np.linalg.solve(A, -c) * dx

#         psi = np.stack([psiR[sl].ravel(), psiI[sl].ravel()], axis=0)
#         coeff = psi @ L1.T
#         shift = np.linalg.solve(coeff[:, :2], -coeff[:, 2]) * dx
#         coords[j] += shift

        coords[j] += shift

    return indices, coords, c0


def vortex_tracker(x, y, t, psiR, psiI, cutoff_radii=None, threshold_distance=0.4):
    frames = []

    for j in tqdm(range(len(psiR))):
        indices, coords, c0 = vortex_detect2d(x, y, psiR[j], psiI[j], cutoff_radii)
        frames.append(coords)

    # the most idiotic, non vectorized thing we can do

    # make sure you have enough containers
    nvort = max(len(v) for v in frames)
    print(f"Found {nvort} vortices total")

    vortices = {}
    # maybe I do need a class for this
    for j in range(nvort):
        vortices[j] = {'time': [], 'coords': []}

    for i in range(len(frames)):
        time = t[i]
        vs = frames[i]
        # print(f"time {i}: {len(vs)} vortices")
        for j, v in enumerate(vs):
            # print(f"try vortex {j}")
            for uid, vx in vortices.items():
                if len(vx['coords']) > 0:
                    # print(f"  box {uid} non-empty")
                    v1 = vx['coords'][-1]  # take the last position where you've seen the vortex with id uid
                    d = np.linalg.norm(v1 - v)
                    if d < threshold_distance:
                        vx['time'].append(time)
                        vx['coords'].append(v)
                        # print(f"  vortex {j} goes in box {uid}")
                        break
                    # else:
                        # print(f"  vortex {j} too far to stay in box {uid}")
                else:
                    # print(f"  box {uid} empty")
                    vx['time'].append(time)
                    vx['coords'].append(v)
                    # print(f"  vortex {j} goes in box {uid}")
                    break

    for uid, vx in vortices.items():
        print(uid, len(vx['time']), len(vx['coords']))
        vx['time'] = np.asarray(vx['time'])
        vx['coords'] = np.stack(vx['coords'], axis=0)

    return vortices
