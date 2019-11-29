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

wrf_dir_pre = "WRF_test" #prefix for WRF build directory (_debug and _mpi will be added later)
ideal_case = "em_les" #idealized WRF case
runID = "pytest" #name for this simulation series

outpath = os.environ["wrf_res"]#WRF output path root
outdir = "test/" + runID #subdirectory for WRF output if not set in command line
run_path = os.environ["wrf_runs"] + "/" + runID #path where run directories of simulations will be created
build_path = os.environ["wrf_builds"] + "tests" #path where different versions of the compiled WRF model code reside

#Define parameter grid for simulations (any namelist parameters and some additional ones can be used)
param_grid = odict(sf_sfclay_physics=[1, 2])
 #          res={"dx" : [100,4000], "bl_pbl_physics": [0,1], "dz0" : [10,50], "nz" : [350,60]})

# names of parameter values for output filenames; either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics" : {1: "kessler", 2: "lin"},
               "sf_sfclay_physics" : {1 : "mm5", 2: "eta", 5 : "mynn"},
               "res"         : ["LES", "MYJ"]}

#Set additional namelist parameters (only applies if they are not present in param_grid)
#any namelist parameters and some additional ones can be used
start_time = "2018-06-20_00:00:00" #"2018-06-20_20:00:00"; format %Y-%m-%d_%H:%M:%S
end_time = "2018-06-20_02:00:00" #"2018-06-23_00:00:00"; format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1 #number of repetitions for each configuration

#horizontal grid
params["dx"] = 500 #horizontal grid spacing (m)
params["lx"] = 50 #16000, horizontal extent in east west (m)
params["ly"] = 50 #4000, minimum horizontal extent in north south (m)
use_gridpoints = True #use minimum number of grid points set below
params["gridpoints"] = 2 #16, minimum number of grid points in each direction -1
force_domain_multiple = True #if use_gridpoints: force domain with x and y extents that are multiples of lx and ly, respectively

#vertical grid
params["ztop"] = 5000 #15000, top of domain (m)
params["zdamp"] = int(params["ztop"]/3) #depth of damping layer (m)
params["nz"] = 60 #176, number of vertical levels
params["dz0"] = 20 #10, height of first model level (m)
params["dz_method"] = 0 #method for creating vertical grid as defined in vertical_grid.py
params["dt"] = None #1 #time step (s), if None calculated as dt = 6 s/m *dx/1000
#minimum time between radiation calls (min); if radt is not specified: radt=max(radt_min, 10*dt)
params["radt_min"] = 1

params["input_sounding"] = "" #name of input sounding to use (should be named input_sounding_name)

params["isotropic_res"] = 100 #resolution below which mixing is isotropic
params["pbl_res"] = 500 #resolution above which to use PBL scheme (m); this also changes km_opt
params["spec_hfx"] = None #None specified surface heat flux instead of radiation

#standard namelist parameters
params["mp_physics"] = 2
params["bl_pbl_physics"] = 1

#indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
output_streams = {0: ["wrfout", 30], 7: ["fastout", 5.5] }

# filename where output variables for standard and auxiliary streams are modified:
params["iofields_filename"] = "NONE_SPECIFIED"

params["restart_interval"] = 240 #restart interval (min)

split_output_res = 0 #resolution below which to split output in one timestep per file

# non-namelist parameters that will not be included in namelist file
del_args =   ["dx", "nz", "dz0","dz_method", "gridpoints", "lx", "ly", "spec_hfx", "spec_sw",
            "pert_res", "input_sounding", "repi", "n_rep", "isotropic_res", "pbl_res", "dt"]
#%%
'''Settings for resource requirements of SGE jobs'''
cluster_name = "leo" #this name should appear in the variable $HOSTNAME to detect if cluster settings should be used
queue = "std.q" #queue for SGE

#virtual memory: numbers need adjustment
#TODO: approach is not optimized yet!
vmem_init_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for WRF initialization (ideal.exe)
vmem_init_min = 2000 #minimum virtual memory (MB) for WRF initialization

vmem_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for running WRF (wrf.exe); will be divided by number of slots
vmem_min = 600 #minimum virtual memory (MB) for running WRF
vmem_pool = 2000 #virtual memory to request per slot if pooling is used

vmem_buffer = 1.2 #buffer factor for virtual memory

# runtime: specify either rt or runtime_per_step or None
# if None: runtime is estimated from short test run
# if qsub: run for a few minutes; check runtime and vmem and resubmit
rt = None #None or job runtime in seconds
rt_buffer = 1.5 #buffer factor to multiply rt with

# if rt is None: runtime per time step in seconds for different dx
runtime_per_step_dict = None#{ 62.5 : 3., 100: 3., 125. : 3. , 250. : 1., 500: 0.5, 1000: 0.3, 2000.: 0.3, 4000.: 0.3}
#paths to search for log files to determine runtime if not specified
rt_search_paths = [run_path, run_path + "/old"]
rt_check = 180 #runtime (s) of test jobs  for --check_rt option

# slots
nslots_dict = {} #set number of slots for each resolution
min_n_per_proc = 16 #25, minimum number of grid points per processor
even_split = False #1, force equal split between processors

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
