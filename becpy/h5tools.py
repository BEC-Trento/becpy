#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 11-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
save / load data to some h5file
freely inspired by databandit & friends

"""

import h5py
import pandas as pd
from collections import OrderedDict
import logging

log = logging.getLogger(__name__)


def savedataframe(filename, dataframe):
    dataframe.to_hdf(filename, 'dataframe', append=True)


def loaddataframe(filename):
    try:
        return pd.read_hdf(filename, 'dataframe')
        log.info('Loading dataframe from file.')
    except KeyError:
        return pd.DataFrame()


# def _loadgroup(filename, group, idx, load_datasets=True):
#     group = f"{group}/{idx:03d}" if idx is not None else group

def load_data(filename, group, load_datasets=True):

    def to_dict(group, data):
        data.update(dict(group.attrs))
        for name, obj in group.items():
            if isinstance(obj, h5py.Group):
                sub = {}
                to_dict(obj, sub)
                data[name] = sub
            else:
                if load_datasets:
                    data[name] = obj[()]

    with h5py.File(filename, 'r') as f:
        group = f[group]
        data = {}
        to_dict(group, data)
    return data


def load_keys(filename, group):
    with h5py.File(filename, 'r') as f:
        try:
            group = f[group]
        except ValueError:
            group = f
        except KeyError:
            return []
        return list(group.keys()) + list(group.attrs.keys())


# def loaddata(filename, group, idx=None, load_datasets=True):
#
#     if idx is not None:
#         return _loadgroup(filename, group, idx, load_datasets)
#
#     with h5py.File(filename, 'r') as f:
#         keys = list(f[group].keys())
#         data = {}
#         # for key in keys:
#         #     data[int(idx)] = dict(f[f"{group}/{idx}"].attrs)
#     for idx in keys:
#         data[int(idx)] = _loadgroup(filename, group, int(idx), load_datasets)
#     return OrderedDict(data)


def _save_dataset(filename, name, value, dtype=None):
    """Delete the dataset before rewriting. h5 is a pain with deleting
    stuff but this approach keeps file size from growing too much.
    No it doesnt
    """

    try:
        filename.create_dataset(name, data=value, maxshape=(None,), dtype=dtype)
    except RuntimeError:
        del filename[name]
        filename.create_dataset(name, data=value, maxshape=(None,), dtype=dtype)


def save_data(filename, data, group, idx=None):
    group = f"{group}/{idx:03d}" if idx is not None else group
    with h5py.File(filename, 'a') as h5file:
        g = h5file.require_group(group)
        for key, value in data.items():
            # print(key)
            if isinstance(value, dict):
                _group = f"{group}/{key}"
                save_data(filename, data=value, group=_group)
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                _save_dataset(g, key, value)
            else:
                g.attrs[key] = value
