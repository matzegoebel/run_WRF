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
import os
from collections import OrderedDict as odict
import glob
from datetime import datetime


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

def flatten_list(l):
    '''Flattens list. Works for list elements that are lists or tuples'''
    flat_l = []
    for item in l:
        if type(item) in [list, tuple, np.ndarray]:
            flat_item = flatten_list(item)
            flat_l.extend(flat_item)
        else:
            flat_l.append(item)

    return flat_l

def overprint(message):
    print("\r", message, end="")

def elapsed_time(start):
    time_diff = datetime.now() - start
    return time_diff.total_seconds()

def print_progress(start, counter=None, length=None, message="Elapsed time", prog=True):
    '''
    Print progress and elapsed time since start date.
    '''
    msg = ""
    if prog != False:
        if prog == "%":
            msg = "Progress: {} %    ".format(round(float(counter)/length*100,2))
        else:
            msg = "Progress: {}/{}   ".format(counter, length)

    etime = elapsed_time(start)
    hours, remainder = divmod(etime, 3600)
    minutes, seconds = divmod(remainder, 60)
    overprint(msg + "%s: %s hours %s minutes %s seconds" % (message, int(hours),int(minutes),round(seconds)))


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
    composite_params = {}
    for param,val in d.items():
        if type(val) == dict:
            val_list = list(val.values())
            lens = np.array([len(l) for l in val_list])
            if (lens[0] != lens).any():
                raise ValueError("All parameter ranges that belong to the same composite must have equal lengths!")
            val_list.append(np.arange(lens[0]))
            params.extend([*val.keys(), param + "_idx"])
            composite_params[param] = list(val.keys())

            d[param] = transpose_list(val_list)
            for k in val.keys():
                param_grid_flat[k] = val[k]
            del param_grid_flat[param]
        else:
            params.append(param)

    combs = list(itertools.product(*d.values()))
    for i,comb in enumerate(combs):
        c = flatten_list(comb)
        combs[i] = odict(zip(params,c))

    return pd.DataFrame(combs), param_grid_flat, composite_params

def get_runtime_all(id_filter, dirs, subdir="", all_times=True, levels=None, remove=None, length=1e7, verbose=True):

    dirs = [os.path.expanduser(d) for d in dirs]
    runs = [glob.glob(d + "/wrf/runs/" + subdir + "/WRF_*{}*".format(id_filter)) for d in dirs]
    runs = flatten_list(runs)
    if levels is not None:
        nlevels = len(levels)
        columns = levels.copy()
    else:
        columns = []
    columns.extend(["nx", "ny", "ide", "jde", "timing"])
    index = None
    if all_times:
        index = np.arange(int(length))
        columns.append("time")
    timing = pd.DataFrame(columns=columns, index=index)
    counter = 0
    for j,r in enumerate(runs):
        if verbose:
            print_progress(counter=j, length=len(runs))
        ID = r.split("/")[-1][4:]
        IDl = [i for i in ID.split("_") if i not in remove]
        if len(IDl) != nlevels:
            print("{} does not have not the correct ID length".format(ID))
            continue
        for i,a in enumerate(IDl):
            try:
                a = float(a)
                if a == int(a):
                    a = int(a)
                IDl[i] = a
            except:
                pass
            _, counter = get_runtime(r, timing=timing, counter=counter, all_times=all_times)
    return timing

def get_runtime(run_dir, timing=None, counter=None, all_times=True):

    if os.path.isfile(run_dir + "/rsl.error.0000"):
        f = open(run_dir + "/rsl.error.0000")
    elif os.path.isfile(run_dir + "/run.log"):
        f = open(run_dir + "/run.log")
    else:
        raise FileNotFoundError("No log file found!")
    if timing is None:
        timing = pd.DataFrame(columns=["nx", "ny", "ide", "jde", "timing"])
        counter = 0
    log = f.readlines()
    settings = []
    if "WRF V" in "".join(log[4:15]): #check that it is no init run
        timing_ID = []
        times = []
        for line in log:
            if "Timing for main" in line:
                linesp = line.split(":")
                timing_l = linesp[-1].strip()
                try:
                    timing_l = float(timing_l[:timing_l.index("elapsed")])
                except ValueError:
                    continue
                timing_ID.append(timing_l)
                if all_times:
                    time = line[line.index("time"):line.index("on domain")][5:-1]
                    times.append(time)
            elif "Ntasks" in line:
                nx = int(line[line.index("X")+1: line.index(",")].replace(" ", ""))
                ny = int(line[line.index("Y")+1: line.index("\n")].replace(" ", ""))
                settings.extend([nx, ny])
            elif "ids,ide" in line:
                _, _, ide, _, jde  = [l for l in line[:-1].split(" ") if l != ""]
                settings.extend([int(ide), int(jde)])
        timing_ID = np.array(timing_ID)
        if len(timing_ID) > 0:
            if all_times:
                timing.iloc[counter:counter+len(timing_ID), -1] = times
                timing.iloc[counter:counter+len(timing_ID), -2] = timing_ID
                timing.iloc[counter:counter+len(timing_ID), -7:-2] = settings
                counter += len(timing_ID)
            else:
                timing.loc[counter] = [*settings, np.nanmean(timing_ID)]
                counter += 1
        else:
            counter += 1
    f.close()
    return timing, counter
