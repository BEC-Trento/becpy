#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 06-2020 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
!!!! Module under test !!!
Still developing.
The Eigenfaces class should work, the incremental version I'm not sure.
"""

import numpy as np

from sklearn.decomposition import PCA, IncrementalPCA


class Eigenfaces:

    def fit(self, probes, mask, **pca_kwargs):
        self.shape = mask.shape
        self.probes = probes.reshape(probes.shape[0], -1)
        self.mask = mask
        self.mean = self.probes.mean(axis=0)

        X = np.stack([p[mask] for p in probes], axis=0)
        self.pca = PCA(n_components=None, **pca_kwargs)

        print('Loading PCA...')
        print('Fit...')
        self.pca.fit(X)

        print('Preparing matrices...')
        # TODO: implement n_components.
        # pU will not be square, so I need to invert it in the right subspace
        pU = self.pca.transform(X)
        U = np.linalg.inv(pU)
        self.extended_components_ = U @ (self.probes - self.mean)
        print('done.')

    def transform(self, image, n_components=None):
        _im = image[self.mask].reshape(1, -1)
        coeffs = self.pca.transform(_im).ravel()
        best = coeffs[0:n_components] @ self.extended_components_[0:n_components, :] + self.mean
        return best.reshape(self.shape)


class IncrementalEigenfaces:
    """This implememtation is wrong.
    the mask should be fixed formo the beginning like in the above."""

    def __init__(self, n_components, **pca_kwargs):
        self.n_samples = 0
        self.pca = IncrementalPCA(
            n_components=n_components, batch_size=None, whiten=False, **pca_kwargs)

    def add_probes(self, X):
        """NB: batch_size >= n_components"""
        self.pca.partial_fit(X)
        self.n_samples += len(X)

    def _transform(self, X, mask):
        """Override pca.transform to project over the unmasked region"""
        X = X - self.pca.mean_
        X_transformed = X[mask] @ self.pca.components_.T[mask, :]
        return X_transformed

    def best_probe(self, atoms, mask):
        shape = atoms.shape

        atoms = atoms.ravel()
        mask = mask.ravel()
        coeffs = self._transform(atoms, mask)
        best = self.pca.inverse_transform(coeffs)
        return best.reshape(shape)


if __name__ == '__main__':
    ef = Eigenfaces(n_components=90)

    def make_batch(batch_filenames):
        batch = []

        for filename in batch_filenames:

            with h5py.File(filename, 'r') as file:
                camera = p.in_situ_camera
                images = file[f'data/{camera}/images'][:].astype(float)

            images = {k: v for k, v in zip(p.images_names, images)}

            probe = images['probe']
            bg = images['bg']
            probe -= bg
            batch.append(probe.ravel())

        return np.stack(batch, axis=0)

    X = make_batch(filenames[100:])

    ef.add_probes(X)
