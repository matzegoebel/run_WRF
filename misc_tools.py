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
import subprocess as sp


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
    '''Flattens list. Works for list elements that are lists or tuples'''
    flat_l = []
    for item in l:
        if type(item) in [list, tuple, np.ndarray]:
            flat_item = flatten_list(item)
            flat_l.extend(flat_item)
        else:
            flat_l.append(item)

    return flat_l

def make_list(o):
    if type(o) not in [tuple, list, dict, np.ndarray]:
        o = [o]
    return o

def overprint(message):
    print("\r", message, end="")

def elapsed_time(start):
    time_diff = datetime.now() - start
    return time_diff.total_seconds()

def print_progress(start=None, counter=None, length=None, message="Elapsed time", prog=True):
    '''
    Print progress and elapsed time since start date.
    '''
    msg = ""
    if prog != False:
        if prog == "%":
            msg = "Progress: {} %    ".format(round(float(counter)/length*100,2))
        else:
            msg = "Progress: {}/{}   ".format(counter, length)
    if start is not None:
        etime = elapsed_time(start)
        hours, remainder = divmod(etime, 3600)
        minutes, seconds = divmod(remainder, 60)
        msg += "%s: %s hours %s minutes %s seconds" % (message, int(hours),int(minutes),round(seconds))
    overprint(msg)

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

def output_id_from_config(param_comb, param_grid, param_names=None, runID=None):
    """
    Creates ID for output files. Param_names can be used to replace parameter values.
    
    Parameters
    ----------
    param_comb : pandas Dataframe
        current parameter configuration
    param_grid : dictionary of lists or dictionaries
        input dictionary containing the parameter values
    param_names : dictionary of lists or dictionaries
        names of parameter values for output filenames
    runID : str or None
        General ID for the requested simulation series used as prefix in filenames
    
    Returns
    -------
    ID : str
        output ID

    """  
    ID = param_comb.copy()
    for p, v in param_grid.items():
        if type(v) == dict:
            if param_names is None:
                raise ValueError("param_names cannot be None if composite parameters are used!")
            for pc in v.keys():
                del ID[pc]
        if (param_names is not None) and (p in param_names):
            namep = param_names[p]
            if type(v) == dict:
                ID[p] = namep[ID[p + "_idx"]]
                del ID[p + "_idx"]
            else:
                if type(namep) == dict:
                    ID[p] = namep[ID[p]]
                else:
                    ID[p] = namep[param_grid[p].index(ID[p])]

    ID_str = "_".join(ID.values.astype(str))
    if runID is not None:
        ID_str =  runID + "_" + ID_str

    return ID_str, ID


def get_runtime_all(runs=None, id_filter=None, dirs=None, all_times=False, levels=None, remove=None, verbose=False):
    """
    Get runtime per timestep from all given run directories or for all directories in dirs that pass the filter id_filter.

    Parameters
    ----------
    runs : list of str, optional
        List of directory paths that contain log outputs. The default is None.
    id_filter : TYPE, optional
        Filter to apply to directories in dirs when runs is None. The default is None.
    dirs : list of str, optional
        Directories to search in, when runs is None. The default is None.
    all_times : bool, optional
        Output information for all timestamps instead of averaging. The default is False.
    levels : list of str, optional
        Names of the parameters used to construct the filenames. The default is None.
    remove : list of str, optional
        Parameter values to remove from filenames. The default is None.
    verbose : bool, optional
        Verbose output. The default is False.

    Returns
    -------
    timing : pandas DataFrame
        Information about timing, number of MPI slots and domain size for all given runs.
    """
    if runs is None:
        dirs =  make_list(dirs)
        dirs = [os.path.expanduser(d) for d in dirs]
        runs = [glob.glob(d + "/WRF_*{}*".format(id_filter)) for d in dirs]
        runs = flatten_list(runs)
    if remove is None:
        remove = []
    if levels is not None:
        nlevels = len(levels)
        columns = list(levels.copy())
    else:
        IDs = [r.split("/")[-1][4:] for r in runs]
        IDls = [[i for i in ID.split("_") if i not in remove] for ID in IDs]
        nlevels = max([len(ID) for ID in IDls])
        columns = list(np.arange(nlevels))

    columns.extend(["path", "nx", "ny", "ide", "jde", "timing", "timing_sd"])
    index = None
    if all_times:
        #estimate number of lines in all files
        runs_mpi = [r for r in runs if os.path.isfile(r + "/rsl.error.0000")]
        runs_serial = [r for r in runs if os.path.isfile(r + "/run.log") and not os.path.isfile(r + "/rsl.error.0000")]
        num_lines = [sum(1 for line in open(r + "/rsl.error.0000")) for r in runs_mpi]
        num_lines.extend([sum(1 for line in open(r + "/run.log")) for r in runs_serial])
        index = np.arange(sum(num_lines))
        columns.append("time")
    timing = pd.DataFrame(columns=columns, index=index)
    counter = 0

    for j,(r, ID) in enumerate(zip(runs, IDs)):
        IDl = [i for i in ID.split("_") if i not in remove]
        if len(IDl) > nlevels:
            print("{} does not have not the correct id length".format(ID))
            continue
        for i,a in enumerate(IDl):
            try:
                a = float(a)
                if a == int(a):
                    a = int(a)
                IDl[i] = a
            except:
                pass

        _, new_counter = get_runtime(r, timing=timing, counter=counter, all_times=all_times)
        timing.iloc[counter:new_counter, :len(IDl)] = IDl
        timing.loc[counter:new_counter-1, "path"] = r
        counter = new_counter
        if verbose:
            print_progress(counter=j+1, length=len(runs))

    timing = timing.dropna(axis=0,how="all")
    timing = timing.dropna(axis=1,how="all")
    return timing

def get_runtime(run_dir, timing=None, counter=None, all_times=False):
    """
    Get runtime, MPI slot and domain size information from log file in run_dir.
    This information is written to timing starting from counter or, if not given, a new Dataframe is created.

    Parameters
    ----------
    run_dir : str
        Path of directory.
    timing : pandas DataFrame, optional
        DataFrame to write information to. The default is None.
    counter : int, optional
        Index at which to start writing information in timing. The default is None.
    all_times : bool, optional
        Output all timestamps instead of averaging. The default is False.

    Raises
    ------
    FileNotFoundError
        No log file found.

    Returns
    -------
    timing : pandas DataFrame
        Updated timing data.
    counter : int
        Current counter for writing.

    """

    if os.path.isfile(run_dir + "/rsl.error.0000"):
        f = open(run_dir + "/rsl.error.0000")
    elif os.path.isfile(run_dir + "/run.log"):
        f = open(run_dir + "/run.log")
    else:
        raise FileNotFoundError("No log file found!")

    log = f.readlines()
    settings = {}
    if "WRF V" in "".join(log[:15]): #check that it is no init run
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
                settings["nx"] = line[line.index("X")+1: line.index(",")].replace(" ", "")
                settings["ny"] = line[line.index("Y")+1: line.index("\n")].replace(" ", "")
            elif "ids,ide" in line:
                _, _, settings["ide"], _, settings["jde"]  = [l for l in line[:-1].split(" ") if l != ""]
            elif ("WRF TILE" in line) and ("ide" not in settings):
                settings["ide"] = line[line.index("IE")+2: line.index("JS")]
                settings["jde"] = line[line.index("JE")+2:]
        if "nx" not in settings:
            settings["nx"] = 1
            settings["ny"] = 1
        for k, val in settings.items():
            settings[k] = int(val)
        timing_ID = np.array(timing_ID)
        if timing is None:
            columns = ["nx", "ny", "ide", "jde", "timing"]
            if all_times:
                columns.append("time")
                ind = np.arange(len(timing_ID))
            else:
                ind = [0]

            timing = pd.DataFrame(columns=columns, index=ind)
            counter = 0
        if len(timing_ID) > 0:
            if all_times:
                timing.loc[counter:counter+len(timing_ID)-1, "time"] = times
                timing.loc[counter:counter+len(timing_ID)-1, "timing"] = timing_ID
                timing.loc[counter:counter+len(timing_ID)-1, settings.keys()] = list(settings.values())
               # timing.iloc[counter:counter+len(timing_ID), -7:-2] = settings
                counter += len(timing_ID)
            else:
                timing.loc[counter, settings.keys()] = list(settings.values())
                timing.loc[counter, "timing"] = np.nanmean(timing_ID)
                timing.loc[counter, "timing_sd"] = np.nanstd(timing_ID)
                counter += 1
        else:
            counter += 1
    f.close()
    return timing, counter

def get_runtime_id(run_dir, rt_search_paths):
    """
    Look for simulations in rt_search_paths with identical namelist file as the one in
    run_dir (except for irrelevant parameters defined in ignore_params in the code).
    Then collect the runtime per time step for these simulations and return it.

    Parameters
    ----------
    run_dir : str
        Path to the reference directory.
    rt_search_paths : str of list of str
        Paths, in which to search for identical simulations.

    Returns
    -------
    tuple of floats
        runtime per timestep (s), mean and standard deviation

    """
    rt_search_paths = make_list(rt_search_paths)
    print("Search for runtime values in previous runs.")
    ignore_params = ["start_year", "start_month","start_day", "start_hour",
                     "start_minute","run_hours", "_outname"]
    identical_runs = []
    for search_path in rt_search_paths:
        search_path += "/"
        runs_all = next(os.walk(search_path))[1]
        runs = []
        for r in runs_all:
            r_files = os.listdir(search_path + r)
            if ("namelist.input" in r_files) and (("run.log" in r_files) or ("rsl.error.0000" in r_files)):
                runs.append(search_path + r)
        namelist_ref = namelist_to_dict("{}/namelist.input".format(run_dir))
        for r in runs:
            namelist_r = namelist_to_dict("{}/namelist.input".format(r))
            identical = True
            params = list(set([*namelist_r.keys(), *namelist_ref.keys()]))
            for param in params:
                if param not in ignore_params:
                    if (param not in namelist_r) or (param not in namelist_ref) or (namelist_ref[param] != namelist_r[param]):
                        identical = False
                        break
            if identical:
                identical_runs.append(r)
                print("{} has same namelist parameters".format(r))

    if len(identical_runs) > 0:
        timing = get_runtime_all(runs=identical_runs, dirs=rt_search_paths, all_times=False)
        return timing["timing"].mean(), timing["timing_sd"].mean()


#%%

def namelist_to_dict(path, verbose=False):
    """Convert namelist file to dictionary."""
    with open(path) as f:
        namelist_str = f.read().replace(" ", "").replace("\t", "").split("\n")
    namelist_dict = {}
    for line in namelist_str:
        if line != "":
            if verbose:
                print("\n" + line)
            param_val = get_namelist_param_val(line, verbose=verbose)
            if param_val is not None:
                namelist_dict[param_val[0]] = param_val[1]
    return namelist_dict

def get_namelist_param_val(line, verbose=False):
    """Get parameter name and value from line in namelist file."""
    line = line.replace(" ", "")
    line = line.replace("\t", "")
    if "=" not in line:
        if verbose:
            print("Line contains not parameters")
        return
    elif line[0] == "!":
        if verbose:
            print("Line is commented out")
        return
    else:
        if "!" in line:
            line = line[:line.index("!")]

        param, val = line.split("=")
        val = mod_namelist_val(val)
        if verbose:
            print(param, val)

    return param, val

def mod_namelist_val(val):
    """Remove unnecessary dots and commas from namelist value and use only one type of quotation marks."""
    val = val.replace('"', "'")
    if val[-1] == ",":
        val = val[:-1]
    if val[-1] == ".":
        val = val[:-1]
    return val


#%%

def prepare_restart(wdir, outpath, output_streams, end_time):

    """
    Prepare restart runs.

    Search for restart files, select the most recent one and adapt the namelist file of the simulation.
    The output of the previous simulation is backed up in a folder called rst.

    Parameters
    ----------
    wdir : str
        Path of the simulation folder that contains the restart files.
    outpath : str
        Directory where simulation output is placed.
    output_streams : list
        List of output stream names.
    end_time : datetime.datetime
        End time of simulation.

    Returns
    -------
    run_hours : datetime.datetime
        Start time of restart run.
    """

    rstfiles = os.popen("ls -t {}/wrfrst*".format(wdir)).read()
    if rstfiles == "":
        print("WARNING: no restart files found")

    ID = "_".join(wdir.split("/")[-1].split("_")[1:])
    restart_time = rstfiles.split("\n")[0].split("/")[-1].split("_")[-2:]
    print("Restart run from {}".format(" ".join(restart_time)))
    start_time_rst = datetime.strptime("_".join(restart_time), '%Y-%m-%d_%H:%M:%S')
    rst_date, rst_time = restart_time
    rst_date = rst_date.split("-")
    rst_time = rst_time.split(":")
    run_hours = (end_time - start_time_rst).total_seconds()/3600

    rst_opt = "restart .true. start_year {} start_month {} start_day {}\
    start_hour {} start_minute {} start_second {} run_hours {}".format(*rst_date, *rst_time, run_hours)
    os.makedirs("{}/rst/".format(outpath), exist_ok=True) #move previous output in backup directory
    outfiles = [glob.glob("{}/{}_{}".format(outpath, stream, ID)) for stream in output_streams]
    for f in flatten_list(outfiles):
        fname = f.split("/")[-1]
        rst_ind = 0
        while os.path.isfile("{}/rst/{}_rst_{}".format(outpath, fname, rst_ind)):
            rst_ind += 1
        os.rename(f, "{}/rst/{}_rst_{}".format(outpath, fname, rst_ind))
    os.environ["code_dir"] = os.path.curdir
    err = os.system("bash search_replace.sh {0}/namelist.input {0}/namelist.input {1}".format(wdir, rst_opt))
    if err != 0:
        raise RuntimeError("Error in preparing restart run! Failed to modify namelist values!")

    return run_hours
