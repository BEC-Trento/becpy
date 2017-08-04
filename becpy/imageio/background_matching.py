#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Created on Mon July 10
# Copyright (C) 2017  Carmelo Mordini
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import os
from datetime import datetime
import json
from . import read_sis, write_sis

class BackgroundManager():
    def __init__(self, name, savedir):
        self._name = name
        self.savedir = savedir
        self.B_dataset = None
        self.dataset_name = None
        self.imshape = None
        self.mask = None
        self.beta = None
        self.BB_inv = None
        

    def get_timestamped_name(self,):
        fmt='%Y-%m-%d-%H-%M-%S-{fname}'
        return datetime.now().strftime(fmt).format(fname=self._name)
    
        
    def build_acquired_bg(self, frames_list, match_bg_fun=None):
        dataset = np.concatenate([f[np.newaxis, ...] for f in frames_list], axis=0)
        return self.save_bg(dataset)
        
    def save_bg(self, dataset):
        self.name = self.get_timestamped_name()
        path = os.path.join(self.savedir, self.name + '.npy')
        np.save(path, dataset)
        print('saved bg dataset %s'%path)
        return None
        
    def default_mask(self, shape):
        h, w = shape
        mask = np.zeros(shape)
        mask[:, :w//6] = 1
        mask[:, w-w//6:] = 1
        mask[:h//6, :] = 1
        mask[h-h//6:, :] = 1
        return mask.astype(bool)
        
    def load_dataset(self, file):
        self.B_dataset = np.load(file)
        print('Loaded dataset with shape', self.B_dataset.shape)
        self.imshape = self.B_dataset[0].shape
        # TODO: implement custom mask and saving via np.savez_compressed
        mask = self.default_mask(self.imshape)
        print('gen default mask')
        self.load_mask(mask)
        self.compute_bg_matrix()
        
    def load_mask(self, mask):
        self.mask = mask
        self.wmask = np.where(mask)
        
    def compute_bg_matrix(self,):
        if self.B_dataset is None:
            print('You must load a dataset first')
            return
        if self.mask is None:    
            print('You must set a mask first')
            return
        beta = np.concatenate([b[self.wmask][np.newaxis, :] for b in self.B_dataset], axis=0)
        self.BB = np.dot(beta, beta.T)
        print('Match area shape:', beta.shape)        
        self.beta = beta
        # return beta, BB_inv

    def compute_bg(self, image, output_coeff=False):
        alpha = np.dot(self.beta, image[self.wmask])
        try:        
            c = np.linalg.solve(self.BB, alpha)
        except np.linalg.LinAlgError as err:
            print(err)
            print('BB matrix is singular: will pseudo-solve')
            c = np.linalg.lstsq(self.BB, alpha)[0]
        B_opt = np.sum(self.B_dataset*c[:, np.newaxis, np.newaxis], axis=0)
        if output_coeff:
            return c, B_opt
        else:
            return B_opt
        

