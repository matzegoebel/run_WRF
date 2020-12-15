#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:22:36 2019

Miscellaneous functions for submit_jobs.py

@author: c7071088
"""

from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import math
import os
from collections import OrderedDict as odict
import glob
from datetime import datetime
from io import StringIO
import sys
from run_wrf import get_namelist
from run_wrf import vertical_grid
from pathlib import Path as fopen
import xarray as xr
import importlib
#%%nproc


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


#%%general helpful functions

def list_equal(*elements):
    """Check if all given elements are equal."""
    l0 = elements[0]
    for l in elements[1:]:
        if l != l0:
            return False

    return True

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
    """Flattens list. Works for list elements that are lists or tuples."""
    flat_l = []
    for item in l:
        if type(item) in [list, tuple, np.ndarray]:
            flat_item = flatten_list(item)
            flat_l.extend(flat_item)
        else:
            flat_l.append(item)

    return flat_l

def make_list(o):
    """Put object in list if it is not already an iterable."""
    if type(o) not in [tuple, list, dict, np.ndarray]:
        o = [o]
    return o

def overprint(message):
    """Print over previous print by rewinding the record."""
    print("\r", message, end="")

def elapsed_time(start):
    """Elapsed time in seconds since start."""
    time_diff = datetime.now() - start
    return time_diff.total_seconds()

def print_progress(start=None, prog=True, counter=None, length=None, message="Elapsed time"):
    """Print progress and elapsed time since start date."""
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

def format_timedelta(td):
    """Format td in seconds to HHH:MM:SS"""
    td = int(td)
    hours, remainder = divmod(td, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:03}:{:02}:{:02}'.format(hours, minutes, seconds)

def ls_t(pattern):
    """Return files matching pattern sorted by modification date."""
    files = os.popen("ls -t {}".format(pattern)).read().split("\n")
    files = remove_empty_str(files)

    return files

def remove_empty_str(l):
    """Remove empty strings from list."""
    return [i for i in l if i != ""]
def read_file(file):
    return fopen(file).read_text()

def extract_times(ds):
    time = [datetime.fromisoformat(str(t.values.astype(str))) for t in ds["Times"]]
    time = pd.DatetimeIndex(time)
    return time.values
#%%config grid related

def grid_combinations(param_grid, add_params=None, param_names=None, runID=None):
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
    add_params : pandas DataFrame
        additional parameters to add to result
    param_names : dictionary of lists or dictionaries
        names of parameter values for output filenames
    runID : str or None
        General ID for the requested simulation series used as prefix in filenames

    Returns
    -------
    combs : list of dicts or pandas DataFrame
        parameter combinations


    """
    if (param_grid is None) or param_grid == {}:
        combs = [odict({})]
        params = None
        IDi, IDi_d = output_id_from_config(param_names=param_names, runID=runID)
        index = [IDi]
        fnames = index
    else:
        #prepare grid
        d = deepcopy(param_grid)
        params = []
        composite_params = []
        for param,val in d.items():
            if type(val) == dict:
                val_list = list(val.values())
                lens = np.array([len(l) for l in val_list])
                if (lens[0] != lens).any():
                    raise ValueError("All parameter ranges that belong to the same composite must have equal lengths!")
                val_list.append(np.arange(lens[0]))
                params.extend([*val.keys(), param + "_idx"])
                composite_params.append( param + "_idx")
                d[param] = transpose_list(val_list)
            else:
                params.append(param)

        combs = list(itertools.product(*d.values()))
        index = []
        fnames = []
        #create parameter grid
        for i,comb in enumerate(combs):
            c = flatten_list(comb)
            combs[i] = odict(zip(params,c))
            IDi, IDi_d = output_id_from_config(combs[i], param_grid, param_names=param_names, runID=runID)
            index.append(str(IDi_d))
            #IDs used for filenames
            fnames.append(IDi)
    if add_params is not None:
        #combine param grid and additional settings
        for param, val in add_params.items():
            for i,comb in enumerate(combs):
                if param not in comb:
                    comb[param] = val

    combs = pd.DataFrame(combs).astype("object")
    combs.index = index
    combs["fname"] = fnames
    if params is not None:
        #add flags for core parameters that are defined in param_grid
        core = combs.iloc[0].copy()
        core[:] = False
        core[params] = True
        core = core.rename("core_param")
        combs = combs.append(core)
        #mark _idx colums
        composite = core.rename("composite_idx")
        composite[:] = False
        composite[composite_params] = True
        combs = combs.append(composite)
    return combs

def output_id_from_config(param_comb=None, param_grid=None, param_names=None, runID=None):
    """
    Create ID for output files. Param_names can be used to replace parameter values.

    Parameters
    ----------
    param_comb : pandas Dataframe, optional
        current parameter configuration
    param_grid : dictionary of lists or dictionaries, optional
        input dictionary containing the parameter values
    param_names : dictionary of lists or dictionaries, optional
        names of parameter values for output filenames
    runID : str or None, optional
        General ID for the requested simulation series used as prefix in filenames

    Returns
    -------
    ID : str
        output ID

    """
    ID_str = ""
    ID = None
    if param_grid is not None:
        ID = {}
        for p, v in param_grid.items():
            if (param_names is not None) and (p in param_names):
                namep = param_names[p]
                if type(v) == dict:
                    ID[p] = namep[param_comb[p + "_idx"]]
                else:
                    if type(namep) == dict:
                        ID[p] = namep[param_comb[p]]
                    else:
                        ID[p] = namep[param_grid[p].index(param_comb[p])]
            elif type(v) == dict:
                raise ValueError("param_names need to be defined for composite parameters!")
            else:
                ID[p] = param_comb[p]

        for p, val in ID.items():
            ID_str += "_{}={}".format(p, val)

    if runID is not None:
        ID_str = runID + ID_str

    return ID_str, ID

#%%runtime
def get_runtime_all(runs=None, id_filter=None, dirs=None, all_times=False, levels=None, remove=None, use_median=False, verbose=False):
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
    use_median : bool, optional
        Use median instead of mean to get average timing. The default is False.
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
    else:
        runs = make_list(runs)
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

    columns.extend(["path", "nx", "ny", "ide", "jde", "timing", "timing_sd", "nsteps"])
    index = None
    runlogs = {}
    runlogs_list = []
    for run, ID in zip(runs, IDs):
        runlogs[ID] = glob.glob(run  + "/run*.log") +  glob.glob(run  + "/rsl.error.0000")
        runlogs_list.extend(runlogs[ID])
    if all_times:
        #estimate number of lines in all files
        num_lines = [sum(1 for line in open(rl)) for rl in runlogs_list]
        index = np.arange(sum(num_lines))
        columns.append("time")
    timing = pd.DataFrame(columns=columns, index=index)
    counter = 0

    for j,(runpath, ID) in enumerate(zip(runs, IDs)):
        IDl = [i for i in ID.split("_") if i not in remove]
        if len(IDl) > nlevels:
            print("{} does not have not the correct id length".format(ID))
            continue
        for runlog in runlogs[ID]:
            _, new_counter = get_runtime(runlog, timing=timing, counter=counter, all_times=all_times, use_median=use_median)
            timing.iloc[counter:new_counter, :len(IDl)] = IDl
            timing["path"].iloc[counter:new_counter] = runpath
            counter = new_counter
        if verbose:
            print_progress(counter=j+1, length=len(runs))



    timing = timing.dropna(axis=0,how="all")
    timing = timing.dropna(axis=1,how="all")
    return timing

def get_runtime(runlog, timing=None, counter=None, all_times=False, use_median=False):
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
    use_median : bool, optional
        Use median instead of mean to get average timing. The default is False.

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
    if os.path.isfile(runlog):
        with open(runlog) as f:
            log = f.readlines()
    else:
        raise FileNotFoundError("No log file found!")

    settings = {}
    if "Timing for main" in "".join(log): #check that it is no init run
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
            timing.loc[:, "nsteps"] = 1
            counter = 0
        if len(timing_ID) > 0:
            if all_times:
                timing.loc[counter:counter+len(timing_ID)-1, "time"] = times
                timing.loc[counter:counter+len(timing_ID)-1, "timing"] = timing_ID
                timing.loc[counter:counter+len(timing_ID)-1, settings.keys()] = list(settings.values())
                counter += len(timing_ID)
            else:
                timing.loc[counter, settings.keys()] = list(settings.values())
                if use_median:
                    timing.loc[counter, "timing"] = np.nanmedian(timing_ID)
                else:
                    timing.loc[counter, "timing"] = np.nanmean(timing_ID)
                timing.loc[counter, "timing_sd"] = np.nanstd(timing_ID)
                timing.loc[counter, "nsteps"] = len(timing_ID)
                counter += 1
        else:
            counter += 1
    return timing, counter

def get_identical_runs(run_dir, search_paths, ignore_nxy=False):
    """
    Look for simulations in search_paths with identical namelist file as the one in
    run_dir (except for irrelevant parameters defined in ignore_params in the code).

    Parameters
    ----------
    run_dir : str
        Path to the reference directory.
    search_paths : str of list of str
        Paths, in which to search for identical simulations.
    ignore_nxy : bool
        Ignore number of processes in x and y-direction (for vmem retrieval only)

    Returns
    -------
    identical_runs : list
        list of runs with identical namelist file

    """
    search_paths = make_list(search_paths)
    ignore_params = ["start_year", "start_month","start_day", "start_hour",
                     "start_minute","run_hours", "_outname"]
    if ignore_nxy:
        ignore_params.extend(["nproc_x", "nproc_y"])
    identical_runs = []
    for search_path in search_paths:
        search_path += "/"
        if len(list(os.walk(search_path))) == 0:
            continue
        runs_all = next(os.walk(search_path))[1]
        runs = []
        for r in runs_all:
            r_files = os.listdir(search_path + r)
            if "namelist.input" in r_files:
                runs.append(search_path + r)

        namelist_ref = get_namelist.namelist_to_dict("{}/namelist.input".format(run_dir))
        for r in runs:
            namelist_r = get_namelist.namelist_to_dict("{}/namelist.input".format(r))
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


    return identical_runs


def get_vmem(runs):
    """
    Get maximum used virtual memory of SGE jobs from log files in the given directories.

    Parameters
    ----------
    runs : list of str
        Directories to search in.
    logfile : str
        Name of logfile.

    Returns
    -------
    float
        Maximum virtual memory for all jobs.

    """
    if len(runs) == 0:
        return
    vmem = []
    for i, r in enumerate(runs):
        rfiles = glob.glob(r + "/resources*")
        for resource_file in rfiles:
            vmem_r_str = get_job_usage(resource_file)["maxvmem"]
            vmem_r = None
            for mag, factor in zip(("M", "G"), (1, 1000)):
                if mag in vmem_r_str:
                    vmem_r = float(vmem_r_str[:vmem_r_str.index(mag)])*factor
                    break
            vmem.append(vmem_r)
    if len(vmem) > 0:
        return vmem


def get_job_usage(resource_file):
    """
    Get usage statistics from job scheduler.

    Parameters
    ----------
    resource_file : str
        File where command output was saved.

    Returns
    -------
    usage : dict
        Usage statistics.

    """
    usage = fopen(resource_file).read_text()
    if "usage" in usage:
        usage = usage[usage.index("\nusage"):].split("\n")[1]
        usage = usage[usage.index(":")+1:].strip().split(",")
        usage = dict([l.strip().split("=") for l in usage])
    elif "MaxVMSize" in usage:
        kv = usage.split("\n")
        keys = kv[0].split("|")[:-1]
        vals = kv[-2].split("|")[:-1]
        usage = {k : v for k,v in zip(keys, vals) if v != ""}
        if "MaxVMSize" in usage:
            usage["maxvmem"] = usage["MaxVMSize"]
    else:
        raise RuntimeError("File format for job usage not known!")

    return usage


def set_vmem_rt(args, run_dir, conf, run_hours, nslots=1, pool_jobs=False, test_run=False, request_vmem=True):
    """Set vmem and runtime per time step  based on settings in config file."""
    skip = False

    resource_search_paths = [conf.run_path, *conf.resource_search_paths]

    #get runtime per timestep
    n_steps = 3600*run_hours/args["dt_f"]
    identical_runs = None
    runtime_per_step = None
    print_rt_step = False
    if test_run:
        runtime_per_step = conf.rt_test*60/n_steps/conf.rt_buffer
        args["n_rep"] = 1
    elif conf.rt is not None:
        runtime_per_step = conf.rt*60/n_steps/conf.rt_buffer
    elif conf.runtime_per_step is not None:
        runtime_per_step = conf.runtime_per_step
    else:
        print("Get runtime from previous runs")
        run_dir_0 = run_dir + "_0" #use rep 0 as reference
        identical_runs = get_identical_runs(run_dir_0, resource_search_paths)
        if len(identical_runs) > 0:
            timing = get_runtime_all(runs=identical_runs, all_times=False, use_median=conf.rt_use_median)
            if len(timing) > 0:
                runtime_per_step = np.average(timing.timing, weights=timing.nsteps)
                rt_sd = np.average(timing.timing_sd, weights=timing.nsteps)
                print("Runtime per time step standard deviation: {0:.5f} s".format(rt_sd))
            else:
                skip = True
                print("Could not retrieve runtime from previous runs. Skipping...")

        else:
            print("No previous runs found. Skipping...")
            skip = True

        print_rt_step = True

    if not skip:
        args["rt_per_timestep"] = runtime_per_step*conf.rt_buffer
        if print_rt_step:
            print("Runtime per time step: {0:.5f} s".format(runtime_per_step))

    #virtual memory
    if request_vmem:
        vmemi = None
        if test_run:
            vmemi = conf.vmem_test*nslots
        elif conf.vmem is not None:
            vmemi = conf.vmem*nslots
        elif conf.vmem_per_grid_point is not None:
            print("Use vmem per grid point")
            vmemi = int(conf.vmem_per_grid_point*args["e_we"]*args["e_sn"])*conf.vmem_buffer
            if conf.vmem_min is not None:
                vmemi = max(vmemi, conf.vmem_min*nslots)
        else:
            print("Get vmem from previous runs")
            if identical_runs is None:
                run_dir_0 = run_dir + "_0" #use rep 0 as reference
                identical_runs = get_identical_runs(run_dir_0, resource_search_paths, ignore_nxy=False)

            vmemi = get_vmem(identical_runs)
            if vmemi is None:
                skip = True
                if len(identical_runs) == 0:
                    print("No previous runs found. Skipping...")
                else:
                    print("Could not retrieve vmem from previous runs. Skipping...")
            else:
                vmemi = max(vmemi)*conf.vmem_buffer

        if vmemi is not None:
            print("Use vmem per slot: {0:.1f}M".format(vmemi/nslots))
            args["vmem"] = vmemi

    return args, skip



#%%init
def read_input_sounding(path, scm=False):
    """Read potential temperature, height and surface pressure from WRF input sounding file"""
    input_sounding_df = pd.read_csv(path, sep=" ", header=None, index_col=0, names=np.arange(5), skipinitialspace=True)
    if scm:
        cols = ["u", "v", "theta", "qv"]
        t0, q0, p0 = input_sounding_df.iloc[0,-3:]
        input_sounding_df = input_sounding_df.iloc[:, :-1]
    else:
        cols = ["theta", "u", "v", "qv"]
        input_sounding_df[4] /= 1000
        header = input_sounding_df.iloc[0]
        p0, t0, q0 = header.name, *header.iloc[:2]
        q0 /= 1000
        input_sounding_df = input_sounding_df.iloc[1:]

    input_sounding_df.columns = cols
    input_sounding_df.index.name = "level"
    input_sounding = xr.Dataset(input_sounding_df)

    return input_sounding, p0

def prepare_init(args, conf, wrf_dir, namelist_check=True):
    """Sets some namelist parameters based on the config files settings."""
    print("Setting namelist parameters\n")
    wrf_build = "{}/{}".format(conf.build_path, wrf_dir)
    namelist_path = "{}/test/{}/namelist.input".format(wrf_build, conf.ideal_case)
    namelist = get_namelist.namelist_to_dict(namelist_path)
    namelist_all = get_namelist.namelist_to_dict(namelist_path, build_path=wrf_build, registries=conf.registries)

    namelist_upd_all = deepcopy(namelist_all)
    namelist_upd_all.update(args)
    namelist_upd = deepcopy(namelist)
    namelist_upd.update(args)

    check_p_diff = lambda p,val=0: (p in namelist_upd_all) and (namelist_upd_all[p] != val)
    check_p_diff_2 = lambda p,val=0: (p not in args) and check_p_diff(p, val)

    #timestep
    dt_int = math.floor(args["dt_f"])
    args["time_step"] = dt_int
    args["time_step_fract_num"] = round((args["dt_f"] - dt_int)*10)
    args["time_step_fract_den"] = 10
    if "radt" not in args:
        args["radt"] = max(args["radt_min"], 10*dt_int/60) #rule of thumb

    #vert. domain
    if "eta_levels" not in args:
        # input_sounding_path = "{}/test/{}/input_sounding_{}".format(wrf_build, conf.ideal_case, args["input_sounding"])
        # input_sounding, p0 = read_input_sounding(input_sounding_path, scm="scm" in conf.ideal_case)
        # theta = input_sounding["theta"]
        if ("dzmax" in args) and (args["dzmax"] == "dx"):
            args["dzmax"] = args["dx"]
        vert_keys = ["nz", "dzmax", "etaz1", "etaz2", "n2", "z1", "z2", "alpha"]
        vert_args = {}
        for key in vert_keys:
            if key in args:
                vert_args[key] = args[key]
                del args[key]

        args["eta_levels"], dz = vertical_grid.create_levels(args["ztop"], args["dz0"], method=args["dz_method"],
                                                             plot=False, table=False, **vert_args)
        args["e_vert"] = len(args["eta_levels"])
        print("Created vertical grid:\n{0} levels\nlowest level at {1:.1f} m\nthickness of uppermost layer: {2:.1f} m\n".format(args["e_vert"], dz[0], dz[-2]))

    args["eta_levels"] = "'" + ",".join(["{0:.6f}".format(e) for e in  args["eta_levels"]])  + "'"
    if "scm" in conf.ideal_case:
        print("WARNING: Eta levels are neglected in the standard initialization of the single column model case!")

    #output streams
    for stream, (_, out_int) in args["output_streams"].items():
        out_int_m = math.floor(out_int)
        out_int_s = round((out_int - out_int_m)*60)
        if (out_int_m == 0) and (out_int_s == 0):
            raise ValueError("Found output interval of {0:.2f} s for output stream {1}. Output intervals below 1 s are not allowed!".format(out_int*60, stream))
        if stream > 0:
            stream_full = "auxhist{}".format(stream)
        else:
            stream_full = "history"
        args["{}_interval_m".format(stream_full)] = out_int_m
        args["{}_interval_s".format(stream_full)] = out_int_s

    #specified heat fluxes or land surface model
    if "spec_hfx" in args:
        print("Specified heatflux used:")
        phys = ["ra_sw_physics", "ra_lw_physics", "sf_surface_physics"]
        for p in phys:
            if check_p_diff(p):
                print("Setting {}=0".format(p))
                args[p] = 0
        args["tke_heat_flux"] = args["spec_hfx"]
        if "isfflx" in args:
            if args["isfflx"] == 1:
                print("Isfflx={} not compatible with specified heat flux. Setting isfflx=2".format(args["isfflx"]))
                args["isfflx"] = 2
        elif check_p_diff("isfflx", 2):
            args["isfflx"] = 2
            print("Setting isfflx=2.")
        print("\n")
    else:
        for p in ["tke_heat_flux", "tke_drag_coefficient"]:
            if check_p_diff(p):
                print("WARNING: Setting {}=0 as spec_hfx is not set".format(p))
                args[p] = 0.
        if check_p_diff("isfflx", 1):
            args["isfflx"] = 1
            print("Setting isfflx=1")


    if "bl_pbl_physics" in args:
        pbl_scheme = args["bl_pbl_physics"]

        #choose surface layer scheme that is compatible with PBL scheme
        if pbl_scheme in [1,2,3,4,5,7,10]:
            sfclay_scheme = pbl_scheme
        elif pbl_scheme == 6:
            sfclay_scheme = 5
        else:
            sfclay_scheme = 1
        p = "sf_sfclay_physics"
        if check_p_diff_2(p, sfclay_scheme):
            print("Setting sf_sfclay_physics={} for compatibility with PBL scheme.".format(sfclay_scheme))
            args[p] = sfclay_scheme

    if namelist_upd_all["sf_sfclay_physics"] in [-1,0]:
        print("WARNING: no surface layer scheme selected")

    if "iofields_filename" in args:
        if args["iofields_filename"] == "":
            args["iofields_filename"] = "NONE_SPECIFIED"
        args["iofields_filename"] = '''"'{}'"'''.format(args["iofields_filename"])
        # args_str = args_str + """ iofields_filename "'{}'" """.format(args["iofields_filename"])

    # delete non-namelist parameters
    del_args = conf.del_args
    args_clean = deepcopy(args)
    for del_arg in del_args:
        if del_arg in namelist_all:
            raise RuntimeError("Parameter {} used in submit_jobs.py already defined in namelist.input! Rename this parameter!".format(del_arg))
        if del_arg in args_clean:
            del args_clean[del_arg]
    for key, val in args_clean.items():
        if type(val) == bool:
            if val:
                args_clean[key] = ".true."
            else:
                args_clean[key] = ".false."

    namelist_all.update(args_clean)

    if "dz" in locals():
        namelist_all["dz"] = dz
    if namelist_check:
        check_namelist_best_practice(namelist_all)
    args_df = pd.DataFrame(args_clean, index=[0])

    #use integer datatype for integer parameters
    new_dtypes = {}
    for arg in args_df.keys():
        typ = args_df.dtypes[arg]
        if (typ == float) and (args_df[arg].dropna() == args_df[arg].dropna().astype(int)).all():
            typ = int
        new_dtypes[arg] = typ
    args_dict = args_df.astype(new_dtypes).iloc[0].to_dict()
    args_str = " ".join(["{} {}".format(param, val) for param, val in args_dict.items()])

    return args, args_str

def check_namelist_best_practice(namelist):
    """Check consistency of namelist parameters and print warnings if strange parameter combinations are used"""

    raise_err = False
    dx = namelist["dx"]
    dy = namelist["dy"]
    print("Checking consistency of namelist settings for horizontal grid spacing of dx={0:.1f} m, dy={0:.1f} m".format(dx, dy))

    #dt
    dt =  namelist["time_step"] + namelist["time_step_fract_num"]/namelist["time_step_fract_den"]
    if dt > 6*min(dx, dy)/1000:
        print("WARNING: time step is larger then recommended. This may cause numerical instabilities. Recommendation: dt(s) = 6 * min(dx, dy)(km)")

    #vertical grid
    if "dz" in namelist:
        dz_max = np.nanmax(namelist["dz"])
        if round(dz_max, 1) > min(dx, dy):
            print("ERROR: There are levels with dz > min(dx, dy) (dz_max={0:.1f} m). Use more vertical levels, a lower model top or a higher dx!".format(dz_max))
            raise_err = True
        if dz_max > 1000:
            print("ERROR: There are levels with dz > 1000 m (dz_max={0:.1f} m). Use more vertical levels or a lower model top!".format(dz_max))
            raise_err = True
        if (np.nanmin(namelist["dz"]) < 0.5 * max(dx, dy)) and (namelist["mix_isotropic"] == 1):
            print("WARNING: At some levels the vertical grid spacing is less than half the horizontal grid spacing."
                  " Consider using anisotropic mixing (mix_isotropic=0).")

    #MP_physics
    if  namelist["mp_physics"] != 0:
        graupel = namelist["mp_physics"] not in [1,3,4,14]
        if (not graupel) and (max(dx, dy) <= 4000):
            print("WARNING: Microphysics scheme with graupel necessary at convection-permitting resolution. Avoid the following settings for mp_physics: 1,3,4 or 14")
        elif graupel and (min(dx, dy) >= 10000):
            print("HINT: Microphysics scheme with graupel not necessary for grid spacings above 10 km. You can instead use one of the following settings for mp_physics: 1,3,4 or 14")


    #pbl scheme, LES and turbulence
    if (namelist["bl_pbl_physics"] == 0) and (max(dx, dy) >= 500) and (namelist["km_opt"] != 5):
        print("WARNING: PBL scheme recommended for dx >= 500 m")
    elif namelist["bl_pbl_physics"] != 0:
        if max(dx, dy) <= 100:
            print("WARNING: No PBL scheme necessary for dx <= 100 m, use LES!")
        else:
            if ("dz" in namelist) and (namelist["dz"][0] > 100):
                print("WARNING: First vertical level should be within surface layer (max. 100 m). Current lowest level at {0:.2f} m".format(namelist["dz"][0]))
            if namelist["bl_pbl_physics"] in [3, 7, 99]:
                eta_levels = eval("np.array([{}])".format(namelist["eta_levels"][1:-1]))
                if eta_levels[1] > 0.995:
                    print("WARNING: First eta level should not be larger than 0.995 for ACM2, GFS and MRF PBL schemes. Found: {0:.4f}".format(eta_levels[1]))


    if (namelist["bl_pbl_physics"] == 0) and (max(dx, dy) <= 500): #LES
        if namelist["km_opt"]==4:
            print("ERROR: For LES, eddy diffusivity based on horizontal deformation (km_opt=4) is not appropriate.")
            raise_err = True
        if namelist["diff_opt"] != 2:
            print("ERROR: LES requires horizontal diffusion in physical space (diff_opt=2).")
            raise_err = True
        if namelist["sf_sfclay_physics"] not in [0,1,2]:
            print("WARNING: Surface layer scheme {} not recommended for LES. Rather use setting 1 or 2.".format(namelist["sf_sfclay_physics"]))


    if (namelist["km_opt"] in [2,3]) and (namelist["diff_opt"] != 2):
        print("ERROR: Horizontal diffusion in physical space (diff_opt=2) needed for 3D SGS turbulence scheme (km_opt=2 or 3).")
        raise_err = True
    elif namelist["km_opt"] == 1:
        if namelist["khdif"] == 0:
            print("WARNING: using constant eddy diffusivity (km_opt=1), but horizontal eddy diffusivity khdif=0")
        if namelist["kvdif"] == 0:
            print("WARNING: using constant eddy diffusivity (km_opt=1), but vertical eddy diffusivity kvdif=0")

    if (namelist["km_opt"] != 4) and (namelist["bl_pbl_physics"] != 0):
        print("WARNING: If boundary layer scheme is used, SGS turbulent mixing should be based on 2D deformation (km_opt=4).")

    #Cumulus
    if (max(dx, dy) >= 10000) and (namelist["cu_physics"] == 0):
        print("WARNING: For dx >= 10 km, the use of a cumulus scheme is strongly recommended.")
    elif (max(dx, dy) <= 4000) and (namelist["cu_physics"] != 0):
        print("WARNING: For dx <= 4 km, a cumulus scheme is probably not needed.")
    elif (max(dx, dy) > 4000) and (min(dx, dy) < 10000) and (namelist["cu_physics"] not in [3, 5, 11, 14]):
        print("WARNING: The grid spacing lies in the gray zone for cumulus convection. Consider using a scale-aware cumulus parametrization (cu_physics={3, 5, 11, 14})")

    if (max(dx, dy) >= 500) and (namelist["shcu_physics"] == 0) and (namelist["bl_pbl_physics"] not in [4,5,6,10]):
        print("WARNING: For dx >= 500 m, a shallow cumulus scheme or PBL scheme with mass flux component (bl_pbl_physics=4,5,6 or 10) is recommended")
    elif (max(dx, dy) < 500) and (namelist["shcu_physics"] != 0):
        print("WARNING: For dx < 500 m, usually no shallow convection scheme is needed")



    #advection scheme
    basic_adv = np.array([namelist[adv + "_adv_opt"] for adv in ["moist", "scalar", "momentum"]])
    any_basic_adv = (basic_adv < 2).any()
    if (max(dx, dy) > 100) and (min(dx, dy) < 1000) and any_basic_adv:
        print("WARNING: Monotonic or non-oscillatory advection options are recommended for 100m < dx < 1km (moist/scalar/momentum_adv_opt >= 2)")

    #surface fluxes
    for flux in ["tke_drag_coefficient", "tke_heat_flux"]:
        if flux == "tke_heat_flux":
            use_fluxes = [0,2]
        else:
            use_fluxes = [0]

        flux_val = 0
        if flux in namelist:
            flux_val = namelist[flux]
        if (namelist["isfflx"] in use_fluxes) and (flux_val == 0):
            print("WARNING: {}=0, although it is used as surface flux for isfflx={}".format(flux,namelist["isfflx"]))


    #dynamics
    damp_f = namelist["zdamp"]/namelist["ztop"]
    if damp_f < 0.2:
        print("WARNING: the damping depth zdamp={0:.0f} m seems rather small given a domain height of ztop={1:.0f} m".format(namelist["zdamp"],namelist["ztop"]))
    elif damp_f > 0.4:
        print("WARNING: the damping depth zdamp={0:.0f} m seems rather large given a domain height of ztop={1:.0f} m".format(namelist["zdamp"],namelist["ztop"]))


    print("\n")
    if raise_err:
        raise ValueError("Critical inconsistencies found in namelist settings. Fix the issues or use -n option to ignore them.")

#%%restart

def get_restart_times(wdir, end_time):
    """
    Search for restart files, select the most recent one, and update the model start and end times.

    Parameters
    ----------
    wdir : str
        Path of the simulation folder that contains the restart files.
    end_time : datetime.datetime
        End time of simulation.

    Returns
    -------
    run_hours : datetime.datetime
        Start time of restart run.
    rst_opt : dict
        Updated namelist options for restart run.
    """
    #check if already finished
    runlogs = glob.glob(wdir + "/run_*.log")
    if len(runlogs) > 0:
        if len(runlogs) > 1:
            timestamp = sorted([r.split("/")[-1].split("_")[1].split(".")[0] for r in runlogs])[-1]
            runlog = wdir + "/run_{}.log".format(timestamp)
        else:
            runlog = runlogs[0]
        with open(runlog) as f:
            runlog = f.readlines()

        if "d01 {} wrf: SUCCESS COMPLETE WRF\n".format(end_time) in runlog:
            print("Run already complete")
            return -1, None

    #search rst files and determine start time
    rstfiles = os.popen("ls -t {}/wrfrst*".format(wdir)).read()
    if rstfiles == "":
        print("no restart files found. Run from start...")
        return None, None

    restart_time = rstfiles.split("\n")[0].split("/")[-1].split("_")[-2:]
    print("Restart run from {}".format(" ".join(restart_time)))

    start_time_rst = datetime.strptime("_".join(restart_time), '%Y-%m-%d_%H:%M:%S')
    times = {}
    rst_date, rst_time = restart_time
    times["start"] = rst_date.split("-")
    times["start"].extend(rst_time.split(":"))

    end_time_dt = datetime.strptime(end_time, '%Y-%m-%d_%H:%M:%S')
    end_d, end_t = end_time.split("_")
    times["end"] = end_d.split("-")
    times["end"].extend(end_t.split(":"))

    run_hours = (end_time_dt - start_time_rst).total_seconds()/3600
    if run_hours  <= 0:
        print("Run already complete")
        return -1, None

    rst_opt = "restart .true."
    for se in ["start", "end"]:
        for unit, t in zip(["year", "month", "day", "hour", "minute", "second"], times[se]):
            rst_opt += " {}_{} {}".format(se, unit, t)


    return run_hours, rst_opt

def prepare_restart(wdir, rst_opt):
    """
    Prepare restart runs.

    Adapt the namelist file of the restart simulation.

    Parameters
    ----------
    wdir : str
        Path of the simulation folder that contains the restart files.
    rst_opt : dict
        Updated namelist options for restart run.

    """

    os.environ["code_dir"] = os.path.curdir
    err = os.system("bash search_replace.sh {0}/namelist.input {0}/namelist.input 0 {1}".format(wdir, rst_opt))
    if err != 0:
        raise RuntimeError("Error in preparing restart run! Failed to modify namelist values!")

def concat_output(config_file=None):
    """Concatenate all output files for each run defined in config_file and delete them."""

    if config_file is None:
        args = sys.argv[1:]
        if (len(args) != 1) or (args[0] in ["-h", "--help", "-help"]):
            print("Concatenate all output files for each run defined in config_file and delete them.\nUsage:\n concat_output config_file")
            sys.exit()
        config_file = args[0]

    #get runs
    if config_file[-3:] == ".py":
        config_file = config_file[:-3]
    try:
        conf = importlib.import_module("run_wrf.configs.{}".format(config_file))
    except ModuleNotFoundError:
        conf = importlib.import_module(config_file)

    if "param_combs" in dir(conf):
        param_combs = conf.param_combs
    else:
        param_combs = grid_combinations(conf.param_grid, conf.params, param_names=conf.param_names, runID=conf.runID)
    ind = [i for i in param_combs.index if i not in ["core_param", "composite_idx"]]
    param_combs = param_combs.loc[ind]

    for cname, param_comb in param_combs.iterrows():
        for rep in range(param_comb["n_rep"]): #repetion loop
            rundir = param_comb["fname"] + "_" + str(rep)
            outpath = os.path.join(conf.outpath, rundir, "") #WRF output path
            print("Concatenate files in " + outpath)
            for stream, (outfile, _) in param_comb["output_streams"].items():
                #get all files
                outfiles = sorted(glob.glob(outpath + outfile + "*"))
                if len(outfiles) == 0:
                    print("No files to concatenate for stream {}!".format(outfile))
                    continue
                elif len(outfiles) == 1:
                    print("Only 1 file available for stream {}!".format(outfile))
                    continue
                outfiles_names = [f.split("/")[-1] for f in outfiles]
                print("Files: {}".format(" ".join(outfiles_names)))
                outfiles = outfiles[::-1]

                #cut out overlapping time stamps
                time_prev = None
                all_times = []
                for outfile in outfiles:
                    ds = xr.open_dataset(outfile, decode_times=False)
                    time = extract_times(ds)
                    all_times.append(time)
                    ds.close()
                    if time_prev is not None:
                        if time_prev[0] <= time[-1]:
                            stop_idx = np.where(time_prev[0]==time)[0][0]-1
                            if stop_idx > -1:
                                err = os.system("ncks -O -d Time,0,{0} {1} {1}".format(stop_idx, outfile))
                                if err != 0:
                                    raise Exception("Error in ncks when reducing {}".format(outfile))
                            else:
                                print("File {} is redundant!".format(outfile))
                                os.remove(outfile)
                                continue

                    time_prev = time
                all_times = np.array(sorted(set(np.concatenate(all_times))))
                #concatenate files
                err = os.system("ncrcat --rec_apn {} {}".format(" ".join(sorted(outfiles[:-1])), outfiles[-1]))
                if err != 0:
                    raise Exception("Error in ncrcat when concatenating output of original and restarted runs" )

                #check final file
                ds = xr.open_dataset(outfiles[-1], decode_times=False)
                concat_times = extract_times(ds)
                if (len(all_times) != len(concat_times)) or (all_times != concat_times).any():
                    raise RuntimeError("Error in concatenated time dimension")
                for f in outfiles[:-1]:
                    os.remove(f)

#%%job scheduler
def get_node_size_slurm(queue):
    queue = queue.split(",")
    ncpus = []
    for q in queue:
        n = os.popen("sinfo -o %c -h -p {}".format(q)).read()
        if n == "":
            raise ValueError("Job queue {} not available!".format(q))
        ncpus.append(n)
    node_size = np.array([int(int(n)/2) for n in ncpus])
    if any(node_size[0] != node_size):
        print("WARNING: Different node sizes for the given queues: {}\n Choosing smaller one...".format(dict(zip(queue, node_size))))
    return node_size.min()
#%%



class Capturing(list):
    """Class to use for capturing print output of python function."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
