#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

@author: matze

"""
import os
from collections import OrderedDict as odict
import misc_tools

#%%
'''Simulations settings'''
params = {} #parameter dict for params not used in param_grid

wrf_dir_pre = "WRF" #prefix for WRF build directory (_debug or _mpi are appended automatically)
ideal_case = "em_les" #idealized WRF case
runID = "test" #name for this simulation series

outpath = os.environ["wrf_res"] #WRF output path root
outdir = "test/" #subdirectory for WRF output if not set in command line
run_path = os.environ["wrf_runs"] #path where run directories of simulations will be created
build_path = os.environ["wrf_builds"] #path where different versions of the compiled WRF model code reside

#Define parameter grid for simulations (any namelist parameters and some additional ones can be used)
param_grid = odict(mp_physics=[1,2],
                   res={"dx" : [200,4000], "dz0" : [10,50], "nz" : [100,60]})

# names of parameter values for output filenames; either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics" : {1: "kessler", 2: "lin"},
               "sf_sfclay_physics" : {1 : "mm5", 2: "eta", 5 : "mynn"},
               "res"         : ["LES", "MYJ"]}

#Set additional namelist parameters (only active if they are not present in param_grid)
#any namelist parameters and some additional ones can be used


params["start_time"] = "2018-06-20_00:00:00" #format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_02:00:00" #format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1 #number of repetitions for each configuration

#horizontal grid
params["dx"] = 500 #horizontal grid spacing (m)
params["lx"] = 1 #horizontal extent in east west (m)
params["ly"] = 1 #minimum horizontal extent in north south (m)
#use minimum number of grid points set below:
use_min_gridpoints = True #"x", "y", True (for both) or False
params["min_gridpoints_x"] = 10 #minimum number of grid points in x direction
params["min_gridpoints_y"] = 10 #minimum number of grid points in y direction
#if use_min_gridpoints: force x and y extents to be multiples of lx and ly, respectively
force_domain_multiple = False #"x", "y", True (for both) or False

#vertical grid
params["ztop"] = 5000 #top of domain (m)
params["zdamp"] = int(params["ztop"]/3) #depth of damping layer (m)
params["nz"] = 60 #number of vertical levels
params["dz0"] = 20 #height of first model level (m)
params["dz_method"] = 0 #method for creating vertical grid as defined in vertical_grid.py
params["dt"] = None  #time step (s), if None calculated as dt = 6 s/m *dx/1000
#minimum time between radiation calls (min); if radt is not specified: radt=max(radt_min, 10*dt)
params["radt_min"] = 1

params["input_sounding"] = "meanwind" #name of input sounding to use (final name is then created: input_sounding_$name)

params["isotropic_res"] = 100 #dx (m) from and below which mixing is isotropic
params["pbl_res"] = 500 #dx (m) from and above which to use PBL scheme; this also changes km_opt
params["spec_hfx"] = None #None specified surface heat flux instead of radiation

#other standard namelist parameters
params["mp_physics"] = 2
params["bl_pbl_physics"] = 2

#indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
output_streams = {0: ["wrfout", 30] }

# filename where output variables for standard and auxiliary streams are modified:
params["iofields_filename"] = None

params["restart_interval"] = 240 #restart interval (min)

split_output_res = 0 #dx (m) below which to split output in one timestep per file

# non-namelist parameters that will not be included in namelist file
del_args =   ["start_time", "end_time", "nz", "dz0","dz_method", "min_gridpoints_x", "min_gridpoints_y", "lx", "ly", "spec_hfx", "input_sounding",
              "n_rep", "isotropic_res", "pbl_res", "dt", "radt_min"]
#%%
'''Settings for resource requirements of SGE jobs'''
cluster_name = "leo" #this name should appear in the variable $HOSTNAME to detect if cluster settings should be used
queue = "std.q" #queue for SGE

#virtual memory: numbers need adjustment
vmem_init_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for WRF initialization (ideal.exe)
vmem_init_min = 2000 #minimum virtual memory (MB) for WRF initialization

vmem = None #virtual memory per job (MB)  to request for running WRF (wrf.exe)

#if vmem is None:
vmem_per_grid_point = None #vmem (MB) per horizontal grid point; will be divided by number of slots
vmem_min = None #minimum virtual memory (MB) for running WRF

vmem_pool = 2000 #virtual memory to request per slot if pooling is used

vmem_buffer = 1.3 #buffer factor for virtual memory

# runtime: specify either rt or runtime_per_step or None
# if None: runtime is estimated from previous identical runs if present
rt = None #None or job runtime in seconds
rt_buffer = 1.5 #buffer factor to multiply rt with
# if rt is None: runtime per time step in seconds for different dx
runtime_per_step_dict = None #{ 100: 3., 500: 0.5, 1000: 0.3}

#paths to search for log files to determine runtime and/or vmem if not specified
resource_search_paths = [run_path]

# slots
nslots_dict = {} #set number of slots for each dx
min_n_per_proc = 16 #25, minimum number of grid points per processor
even_split = False #force equal split between processors

#%%
'''Slot configurations for personal computer and cluster'''

reduce_pool = True #reduce pool size to the actual uses number of slots; do not use if you do not want to share the node with others

if (("HOSTNAME" in os.environ) and (cluster_name in os.environ["HOSTNAME"])):
    cluster = True
    #maximum number of slots that will be requested for the x and y directions
    max_nslotsy = None
    max_nslotsx = None
    pool_size = 28 #number of cores per pool if job pooling is used
else:
    cluster = False
    max_nslotsy = None
    max_nslotsx = None
    pool_size = 16

#%%

param_combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid)

#combine param grid and additional settings
combs = param_combs.copy()
for param, val in params.items():
    if param not in combs:
        combs[param] = val

#Below you can manually add parameters to the DataFrame combs
