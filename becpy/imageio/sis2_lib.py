#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2016  Simone Serafini
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

import numpy as np

import time
import datetime


def readsis(filename, verbose=False):
    ''' Read sis files, both old version and SisV2
    Parameters
    ----------
    filename : stringa con il nome o il path relativo del file.
    Returns im1, im2, image, rawdata, stringa, block
    -------
    im1 : ndarray 2D
        first half of the image [righe 0 : height/2-1].
    im2 : ndarray 2D
        second half of the image [righe height/2 : height-1].
    image : ndarray 2D
        whole image
    rawdata : ndarray 1D
        raw data read from file
    stringa : string
        comment and datestamp of the image
    block : tuple (Bheight,Bwidth)
        Bheight : int
            y dimension of the sub-block
        Bwidth : int
            x dimension of the sub-block
    Notes
    -----
    NB all the outputs are slices of the raw data: any modifications of them will be reflected also in the linked rawdata elements
    '''
    f = open(filename, 'rb')        # open in reading and binary mode
    header = f.read(512)            # read the fixed size header: 512 bytes
    if header[0] == 48:             # 48 is ASCII code for '0': all older sis start with 0
        sisVersion = '  sis1'
        datestamp = ' we do not know'
        commitProg = '  nothing'
        comment = ' nothing'
    elif header[0:5] == b'SisV2':   # SisV2 print header
        sisVersion = str(header[0:5])
        datestamp = str(header[18:37])
        commitProg = str(header[40:48])
        comment = str(header[48:])
    else:
#        print('No valid header found')
        return readsis_quiet(filename, verbose)

    if verbose:
        print("You are opening the " + sisVersion[2:-1] + " file: " + filename)
    f.close()
    # Sometimes it gives error if the file is opened a sigle time

    f = open(filename, 'rb')                        # open in reading and binary mode
    rawdata = np.fromfile(f,np.uint16).astype(np.int)
    # 'H' = uint16
    # types are listed in np.typeDict
    # put in an array the data formatted uint16 and casted to int
    ''' NB fundamental to cast to int:
        unsigned short gives overflow '''
    f.close()

    # Dimension of the whole image
    width=rawdata[6]  # N cols
    height=rawdata[5] # N rows

    # Dimension of the sub-blocks
    Bwidth=rawdata[8]  # N subcols
    Bheight=rawdata[7] # N subrows

    # Length of comments
    ls = rawdata[19]

    # Reading the images
    image = rawdata[-width*height:]
    image.resize(height,width)

    # defines the two images in the sis
    im0 = image[:height//2, :]
    im1 = image[height//2:, :]

    if header[0] == 48:
        stringa = sisVersion[1:-1] + ' ' + datestamp[2:-1] + ' ' + comment[2:]
    elif header[0:5] == b'SisV2':
        stringa = sisVersion[1:-1] + ' ' + datestamp[2:-1] + ' ' + comment[2:ls+2]
    block = (Bheight,Bwidth)
    
    if verbose:
        print("It says: " + comment[1:ls+2] + "...")
        print("...and was created on " + datestamp[1:])
        print("The commit of the program is: " + commitProg[2:])

    return im0, im1, image, rawdata, commitProg, stringa , block

def readsis_quiet(filename, verbose=False):
    ''' Read sis files, both old version and SisV2
    Parameters
    ----------
    filename : stringa con il nome o il path relativo del file.
    Returns im1, im2, image, rawdata, stringa, block
    -------
    im1 : ndarray 2D
        first half of the image [righe 0 : height/2-1].
    im2 : ndarray 2D
        second half of the image [righe height/2 : height-1].
    image : ndarray 2D
        whole image
    rawdata : ndarray 1D
        raw data read from file
    stringa : string
        comment and datestamp of the image
    block : tuple (Bheight,Bwidth)
        Bheight : int
            y dimension of the sub-block
        Bwidth : int
            x dimension of the sub-block
    Notes
    -----
    NB all the outputs are slices of the raw data: any modifications of them will be reflected also in the linked rawdata elements
    '''
    f = open(filename, 'rb')        # open in reading and binary mode
    header = f.read(512)            # read the fixed size header: 512 bytes
    

    if verbose:
        print("Quietly opening the file: " + filename)
    f.close()
    # Sometimes it gives error if the file is opened a sigle time

    f = open(filename, 'rb')                        # open in reading and binary mode
    rawdata = np.fromfile(f,np.uint16).astype(np.int)
    # 'H' = uint16
    # types are listed in np.typeDict
    # put in an array the data formatted uint16 and casted to int
    ''' NB fundamental to cast to int:
        unsigned short gives overflow '''
    f.close()

    # Dimension of the whole image
    width=rawdata[6]  # N cols
    height=rawdata[5] # N rows

    # Dimension of the sub-blocks
    Bwidth=rawdata[8]  # N subcols
    Bheight=rawdata[7] # N subrows

    # Length of comments
    ls = rawdata[19]

    # Reading the images
    image = rawdata[-width*height:]
    image.resize(height,width)

    # defines the two images in the sis
    im0 = image[:height//2, :]
    im1 = image[height//2:, :]


    return im0, im1


def sis_write(image, filename, Bheight, Bwidth, commitProg, stamp, sisposition=None):
        """
        Low-level interaction with the sis file for writing it.
        Writes the whole image, with the unused part filled with zeros.

        Args:
            image (np.array): the 2d-array that must be writed after conversion
            to 16-bit unsigned-integers (must be already normalized)
            filename (string): sis filename
            Bheight (int): the y dimension of the eventual block
            Bwidth (int): the x dimension of the eventual block
            stamp (string): a string to describe who, why and what you want
        """
        #keep the double-image convention for sis files, filling the unused
        #with zeros
        if sisposition == 0:
            image = np.concatenate((image, np.zeros_like(image)))
        elif sisposition == 1:
            image = np.concatenate((np.zeros_like(image), image))
        elif sisposition is None:
            image = np.concatenate((image, image))

        with open(str(filename), 'w+b') as fid:
            # Write here SisV2 + other 4 free bytes
            head = 'SisV2' + '.' + '0'*4
            fid.write(head.encode())

            # This is OK
            height, width = image.shape
            size = np.array([height, width], dtype=np.uint16)
            size.tofile(fid)

            # Here we put 2*2 more bytes with the sub-block dimension
            Bsize = np.array([Bheight, Bwidth], dtype=np.uint16)
            Bsize.tofile(fid)

            # Also a timestamp
            ts = time.time()
            phead = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.')
            fid.write(phead.encode())

            # More: commitProg + descriptive stamp
            ls = np.array([len(stamp)], dtype=np.uint16) # length of the stamp coded at the 38+39 byte
            ls.tofile(fid)
            fid.write(commitProg[:8].encode())
            fid.write(stamp.encode())
            freeHead = '0'*(472-len(commitProg[:8]+stamp))
            fid.write(freeHead.encode())

            image.astype(np.uint16).tofile(fid)
        print('sis written to ' + filename)

def sis_write_off(self, OD, filename, Bheight, Bwidth, stamp):
        """
        Low-level interaction with the sis file for writing it.
        Writes the whole image, with the unused part filled with zeros.

        Args:
            image (np.array): the 2d-array that must be writed after conversion
            to 16-bit unsigned-integers (must be already normalized)
            filename (string): sis filename
            Bheight (int): the y dimension of the eventual block
            Bwidth (int): the x dimension of the eventual block
            stamp (string): a string to describe who, why and what you want
        """
        #norm = np.array((2,1))
        norm = np.append(OD.min(), OD.max())
        par = OD + abs(norm[0])
        image = par * 6553.6
        #image = par.astype(np.uint16)
        print(norm.astype(np.uint16))

        #keep the double-image convention for sis files, filling the unused
        #with zeros
        if self == 0:
            image = np.concatenate((image, np.zeros_like(image)))
        elif self == 1:
            image = np.concatenate((np.zeros_like(image), image))

        with open(str(filename), 'w+b') as fid:
            # Write here SisV2 + other 4 free bytes
            head = 'SisV2' + '.' + '0'*4
            fid.write(head.encode())

            # This is OK
            height, width = image.shape
            size = np.array([height, width], dtype=np.uint16)
            size.tofile(fid)

            # Here we put 2*2 more bytes with the sub-block dimension
            Bsize = np.array([Bheight, Bwidth], dtype=np.uint16)
            Bsize.tofile(fid)

            # Also a timestamp
            ts = time.time()
            phead = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.')
            fid.write(phead.encode())

            # More: descriptive stamp
            ls = np.array([len(stamp)], dtype=np.uint16) # length of the stamp coded at the 38+39 byte
            ls.tofile(fid)
            fid.write(stamp.encode())
            freeHead = '0'*(472-len(stamp))
            fid.write(freeHead.encode())

            image.astype(np.uint16).tofile(fid)

# compatibility aliasing
read_sis = readsis
write_sis = sis_write
