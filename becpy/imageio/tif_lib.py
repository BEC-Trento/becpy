#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Created on Fri Feb 10 12:44:42 2017
# Copyright (C) 2016  Carmelo Mordini
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
from PIL import Image
        
def read_tif(filename, full_output=False):
    image = Image.open(str(filename))
    header = ''
    image = np.asarray(image)
    if full_output:
        return header, image
    else:
        return image

def write_tif():
    raise NotImplementedError