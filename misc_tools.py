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
from netCDF4 import Dataset
import wrf
from datetime import datetime
from io import StringIO
import sys
import get_namelist
import vertical_grid

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

def print_progress(start=None, counter=None, length=None, message="Elapsed time", prog=True):
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
    hours, remainder = divmod(td, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:03}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


#%%config grid related

def grid_combinations(param_grid, add_params=None):
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

    Returns
    -------
    param_combs : pandas DataFrame
        parameter combinations
    combs_full : pandas DataFrame
        param_combs combined with add_params
    param_grid_flat : dictionary
        input param_grid with composite parameters flattened
    composite_params : dictionary
        included parameters for each composite parameter

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

    param_combs = pd.DataFrame(combs)

    combs_full = param_combs.copy()
    if add_params is not None:
        #combine param grid and additional settings
        for param, val in add_params.items():
            if param not in combs_full:
                combs_full[param] = val

    return param_combs, combs_full, param_grid_flat, composite_params

def output_id_from_config(param_comb, param_grid, param_names=None, runID=None):
    """
    Create ID for output files. Param_names can be used to replace parameter values.

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

#%%runtime
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
        run_logs= [r  + "/run.log" for r in runs if os.path.isfile(r + "/run.log")]
        num_lines = [sum(1 for line in open(rl)) for rl in run_logs]
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
        try:
            _, new_counter = get_runtime(r, timing=timing, counter=counter, all_times=all_times)
            timing.iloc[counter:new_counter, :len(IDl)] = IDl
            timing.loc[counter:new_counter-1, "path"] = r
            counter = new_counter
            if verbose:
                print_progress(counter=j+1, length=len(runs))
        except FileNotFoundError:
            if verbose:
                print("No log file found")


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
    runlog = run_dir + "/run.log"
    if os.path.isfile(runlog):
        f = open(runlog)
    else:
        raise FileNotFoundError("No log file found!")

    log = f.readlines()
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

def get_identical_runs(run_dir, search_paths):
    """
    Look for simulations in search_paths with identical namelist file as the one in
    run_dir (except for irrelevant parameters defined in ignore_params in the code).

    Parameters
    ----------
    run_dir : str
        Path to the reference directory.
    search_paths : str of list of str
        Paths, in which to search for identical simulations.

    Returns
    -------
    identical_runs : list
        list of runs with identical namelist file

    """
    search_paths = make_list(search_paths)
    print("Search for runtime values in previous runs.")
    ignore_params = ["start_year", "start_month","start_day", "start_hour",
                     "start_minute","run_hours", "_outname"]
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


def get_vmem(runs, logfile="qstat.info"):
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
        qstat_file = r + "/" + logfile
        if os.path.isfile(qstat_file):
            vmem_r = get_job_usage(qstat_file)["maxvmem"]
            for mag, factor in zip(("M", "G"), (1, 1000)):
                if mag in vmem_r:
                    vmem_r_num = float(vmem_r[:vmem_r.index(mag)])*factor
            vmem.append(vmem_r_num)
    if len(vmem) > 0:
        return vmem


def get_job_usage(qstat_file):
    """
    Get usage statistics from qstat -j $JOB_ID command.

    Parameters
    ----------
    qstat_file : str
        File where command output was saved.

    Returns
    -------
    usage : dict
        Usage statistics.

    """
    qstat = open(qstat_file).read()
    usage = qstat[qstat.index("\nusage"):].split("\n")[1]
    usage = usage[usage.index(":")+1:].strip().split(",")
    usage = dict([l.strip().split("=") for l in usage])

    return usage


def set_vmem_rt(args, run_dir, conf, run_hours, nslots=1, pool_jobs=False):
    """Set vmem and runtime per time step  based on settings in config file."""
    skip = False

    #get runtime per timestep
    r = args["dx"]
    identical_runs = None
    runtime_per_step = None
    if conf.rt is not None:
        runtime_per_step = conf.rt*args["dt"]/(3600*run_hours)/conf.rt_buffer #runtime per time step
    elif (conf.runtime_per_step_dict is not None) and (r in conf.runtime_per_step_dict):
        runtime_per_step = conf.runtime_per_step_dict[r]
        print("Use runtime dict")
    else:
        print("Get runtime from previous runs")
        run_dir_0 = run_dir + "_0" #use rep 0 as reference
        identical_runs = get_identical_runs(run_dir_0, conf.resource_search_paths)
        if len(identical_runs) > 0:
            timing = get_runtime_all(runs=identical_runs, all_times=False)
            if len(timing) > 0:
                runtime_per_step, rt_sd = timing["timing"].mean(), timing["timing_sd"].mean()
                print("Runtime per time step standard deviation: {0:.5f} s".format(rt_sd))

        if runtime_per_step is None:
            print("No runtime specified and no previous runs found. Skipping...")
            skip = True
    args["rt_per_timestep"] = runtime_per_step

    #virtual memory
    vmemi = None
    if pool_jobs:
        vmemi = conf.vmem_pool
    elif conf.vmem is not None:
        vmemi = conf.vmem
    elif conf.vmem_per_grid_point is not None:
        print("Use vmem per grid point")
        vmemi = int(conf.vmem_per_grid_point*args["e_we"]*args["e_sn"])/nslots
        if conf.vmem_min is not None:
            vmemi = max(vmemi, conf.vmem_min)
    else:
        print("Get vmem from previous runs")
        if identical_runs is None:
            run_dir_0 = run_dir + "_0" #use rep 0 as reference
            identical_runs = get_identical_runs(run_dir_0, conf.resource_search_paths)

        vmemi = get_vmem(identical_runs)
        if vmemi is None:
            print("No vmem specified and no previous runs found. Skipping...")
            skip = True
        else:
            vmemi = max(vmemi)
    if vmemi is not None:
        vmemi *= nslots*conf.vmem_buffer

    args["vmem"] = vmemi

    return args, skip

#%%init
def prepare_init(args, conf, wrf_dir_i):
    """Sets some namelist parameters based on the config files settings."""

    r = args["dx"]
    if "dy" not in args:
        args["dy"] = r

    if "isotropic_res" in args:
        if r <= args["isotropic_res"]:
            args["mix_isotropic"] = 1
        else:
            args["mix_isotropic"] = 0

    #timestep

    dt_int = math.floor(args["dt"])
    args["time_step"] = dt_int
    args["time_step_fract_num"] = round((args["dt"] - dt_int)*10)
    args["time_step_fract_den"] = 10
    if "radt" not in args:
        args["radt"] = max(args["radt_min"], 10*dt_int/60) #rule of thumb

    #vert. domain
    args["e_vert"] = args["nz"]
    eta, dz = vertical_grid.create_levels(nz=args["nz"], ztop=args["ztop"], method=args["dz_method"],
                                                      dz0=args["dz0"], plot=False, table=False)
    eta_levels = "'" + ",".join(eta.astype(str)) + "'"

    #split output in one timestep per file
    one_frame = False
    if r <= conf.split_output_res:
        args["frames_per_outfile"] = 1
        for out_ind in conf.output_streams.keys():
            if out_ind != 0:
                args["frames_per_auxhist{}".format(out_ind)] = 1
        one_frame = True

    #output streams
    for stream, (_, out_int) in conf.output_streams.items():
        out_int_m = math.floor(out_int)
        out_int_s = round((out_int - out_int_m)*60)
        if stream > 0:
            stream_full = "auxhist{}".format(stream)
        else:
            stream_full = "history"
        args["{}_interval_m".format(stream_full)] = out_int_m
        args["{}_interval_s".format(stream_full)] = out_int_s

    #specified heat fluxes or land surface model
    if "spec_hfx" in args:
        args["ra_lw_physics"] = 0
        args["ra_sw_physics"] = 0
        args["tke_heat_flux"] = args["spec_hfx"]
        args["sf_surface_physics"] = 0
        if "isfflx" not in args:
            args["isfflx"] = 2
    else:
        if "isfflx" not in args:
            args["isfflx"] = 1
        args["tke_heat_flux"] = 0.
        args["tke_drag_coefficient"] = 0.

    #pbl scheme
    if r >= args["pbl_res"]:
        pbl_scheme = args["bl_pbl_physics"]
        if "km_opt" not in args:
            args["km_opt"] = 4
    else:
        pbl_scheme = 0
        if "km_opt" not in args:
            args["km_opt"] = 2

    args["bl_pbl_physics"] = pbl_scheme

    #choose surface layer scheme that is compatible with PBL scheme
    if "sf_sfclay_physics" not in args:
        if pbl_scheme in [1,2,3,4,5,7,10]:
            sfclay_scheme = pbl_scheme
        elif pbl_scheme == 6:
            sfclay_scheme = 5
        else:
            sfclay_scheme = 1
        args["sf_sfclay_physics"] = sfclay_scheme

    # delete non-namelist parameters
    del_args = [*conf.del_args, *[p + "_idx" for p in conf.composite_params.keys()]]
    with open("{}/{}/test/{}/namelist.input".format(conf.build_path, wrf_dir_i, conf.ideal_case)) as namelist_file:
        namelist = namelist_file.read().replace(" ", "").replace("\t","")
    args_df = args.copy()
    for del_arg in del_args:
        if "\n"+del_arg+"=" in namelist:
            raise RuntimeError("Parameter {} used in submit_jobs.py already defined in namelist.input! Rename this parameter!".format(del_arg))
        if del_arg in args_df:
            del args_df[del_arg]

    args_df = pd.DataFrame(args_df, index=[0])

    #use integer datatype for integer parameters
    new_dtypes = {}
    for arg in args_df.keys():
        typ = args_df.dtypes[arg]
        if (typ == float) and (args_df[arg].dropna() == args_df[arg].dropna().astype(int)).all():
            typ = int
        new_dtypes[arg] = typ
    args_dict = args_df.astype(new_dtypes).iloc[0].to_dict()

    args_str = " ".join(["{} {}".format(param, val) for param, val in args_dict.items()])

    if "iofields_filename" in args:
        args_str = args_str + """ iofields_filename "'{}'" """.format(args["iofields_filename"])
    args_str += " eta_levels " + eta_levels

    return args, args_str, one_frame

#%%restart

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
        print("WARNING: no restart files found. Skipping...")
        return

    ID = "_".join(wdir.split("/")[-1].split("_")[1:])
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
    rst_opt = "restart .true"
    for se in ["start", "end"]:
        for unit, t in zip(["year", "month", "day", "hour", "minute", "second"], times[se]):
            rst_opt += " {}_{} {}".format(se, unit, t)

    os.makedirs("{}/rst/".format(outpath), exist_ok=True) #move previous output in backup directory
    outfiles = [glob.glob("{}/{}_{}".format(outpath, stream, ID)) for stream in output_streams]
    for f in flatten_list(outfiles):
        fname = f.split("/")[-1]
        rst_ind = 0
        while os.path.isfile("{}/rst/{}_rst_{}".format(outpath, fname, rst_ind)):
            rst_ind += 1
        os.rename(f, "{}/rst/{}_rst_{}".format(outpath, fname, rst_ind))
    os.environ["code_dir"] = os.path.curdir
    err = os.system("bash search_replace.sh {0}/namelist.input {0}/namelist.input 0 {1}".format(wdir, rst_opt))
    if err != 0:
        raise RuntimeError("Error in preparing restart run! Failed to modify namelist values!")

    return run_hours

def concat_restart(path, id_filter=""):
    """Concatenate output from restarted run and original run."""
    path = path + "/"
    old_files_0 = glob.glob("{}/rst/*{}_rst_0".format(path, id_filter))
    if len(old_files_0) == 0:
        print("No files to concatenate!")
    for file in old_files_0:
        ID = file.split("/")[-1]
        ID = ID[:ID.index("rst")-1]
        print("Process file {}".format(ID))
        new_file = path + ID

        old_files = glob.glob("{}/rst/{}_rst_*".format(path, ID))
        old_files = [o for o in old_files if o[-4:] != "_cut"]
    #    old_files = os.popen("ls -t {}/rst/{}_rst_*".format(path, ID)).read().split("\n")[::-1]
        rst_inds = [int(f.split("_")[-1]) for f in old_files]
        rst_inds = list(np.argsort(rst_inds))
        if "" in old_files:
            old_files.remove("")
        time = None
        if os.path.isfile(new_file):
            all_files = [*old_files, new_file]
            rst_inds.append(-1)
        else:
            all_files = old_files

        all_files_cut = []
        for rst_ind in rst_inds:
            cfile = all_files[rst_ind]
            ds = Dataset(cfile, "r+")
            time_next = wrf.extract_times(ds, timeidx=None)
            if time is not None:
                if time[-1] >= time_next[0]:
                    start_idx = np.where(time[-1]==time_next)[0][0] + 1
                    ds.close()
                    if start_idx < len(time_next):
                        err = os.system("ncks -O -d Time,{},{} {} {}".format(start_idx, len(time_next)-1, cfile, cfile + "_cut"))
                        if err != 0:
                            raise Exception("Error in ncks when reducing {}".format(cfile))
                    else:
                        print("File {} is redundant!".format(cfile))
                        continue
                else:
                    os.system("cp {} {}".format(cfile, cfile + "_cut"))
            else:
                os.system("cp {} {}".format(cfile, cfile + "_cut"))

            all_files_cut.append(cfile + "_cut")
            time = time_next

        if len(all_files_cut) > 1:
            new_concat = new_file + "_concat"
            if os.path.isfile(new_concat):
                os.remove(new_concat)
            err = os.system("ncrcat {} {}".format(" ".join(all_files_cut), new_concat))
            if err != 0:
                raise Exception("Error in ncrcat when concatenating output of original and restarted runs" )
            file_final = Dataset(new_file + "_concat")
            time_final = wrf.extract_times(file_final, timeidx=None)
            file_final.close()

            res = np.median(time_final[1:] - time_final[:-1])
            time_corr = np.arange(time_final[0], time_final[-1] + res, res)
            if (len(time_corr) != len(time_final)) or (time_corr != time_final).any():
                raise RuntimeError("Error in concatenated time dimension for final file {}".format(file))
            for f in all_files_cut + all_files:
                os.remove(f)
            os.rename(new_file + "_concat", new_file)
        else:
            print("\nNo files to concatenate")

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
