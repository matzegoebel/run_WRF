#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:22:36 2019

Miscellaneous functions for submit_jobs.py

@author: c7071088
"""

import copy
import itertools
import numpy as np
import pandas as pd
import math


def find_nproc(n, min_n_per_proc=25, even_split=False):
    """
    Find number of processors needed for a given number grid points in WRF.

    Parameters
    ----------
    n : int
        number of grid points
    min_n_per_proc : int, optional
        Minimum number of grid points per processor. The default is 25.
    even_split : bool, optional
        Force even split of grid points between processors.

    Returns
    -------
    int
        number of processors.

    """

    if n <= min_n_per_proc:
        return 1
    elif even_split:
        for d in np.arange(min_n_per_proc, n+1):
            if n%d==0:
                return int(n/d)
    else:
        return math.floor(n/min_n_per_proc)


def bool_to_fort(b):
    """Convert python boolean to fortran boolean (as str)."""
    if b:
        return ".true."
    else:
        return ".false."

def transpose_list(l):
    """
    Transpose list of lists.

    Parameters
    ----------
    l : list
        list of lists with equal lenghts.

    Returns
    -------
    list :
        transposed list.

    """
    return list(map(list, zip(*l)))

def flatten_list(l):
    """Flattens list-like variables. Works for lists, tuples and numpy arrays."""
    flat_l = []
    for item in l:
        if type(item) in [list, tuple, np.ndarray]:
            flat_item = flatten_list(item)
            flat_l.extend(flat_item)
        else:
            flat_l.append(item)

    return flat_l


def grid_combinations(param_grid):
    """
    Create list of all combinations of parameter values defined in dictionary param_grid.
    Two or more parameters can be varied simultaneously by defining a composite
    parameter with a dictionary as value.
    E.g.,
    d = dict(
            input_sounding=["stable","unstable"],
            topo=["cos", "flat"],
            c1={"res" : [200,1000], "nz" : [120, 80]})


    Parameters
    ----------
    param_grid : dictionary of lists or dictionaries
        input dictionary containing the parameter values

    Returns
    -------
    combs : list of dictionaries
            parameter combinations

    """
    d = copy.deepcopy(param_grid)
    param_grid_flat = copy.deepcopy(param_grid)
    params = []
    composite_params = []
    for param,val in d.items():
        if type(val) == dict:
            val_list = list(val.values())
            params.extend(val.keys())
            composite_params.extend(val.keys())
            d[param] = transpose_list(val_list)
            for k in val.keys():
                param_grid_flat[k] = val[k]
            del param_grid_flat[param]
        else:
            params.append(param)

    combs = list(itertools.product(*d.values()))
    for i,comb in enumerate(combs):
        c = flatten_list(comb)
        combs[i] = dict(zip(params,c))

    return pd.DataFrame(combs), param_grid_flat, composite_params