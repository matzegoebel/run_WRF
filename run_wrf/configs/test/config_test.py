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
from run_wrf import misc_tools

#%%
'''Simulations settings'''
params = {} #parameter dict for params not used in param_grid

wrf_dir_pre = "WRF_test" #prefix for WRF build directory (_debug or _mpi are appended automatically)
ideal_case = "em_les" #idealized WRF case
runID = "pytest" #name for this simulation series

outpath = os.environ["wrf_res"] #WRF output path root
outdir = "test/" + runID #subdirectory for WRF output if not set in command line
run_path = os.environ["wrf_runs"] + "/" + runID #path where run directories of simulations will be created
build_path = os.environ["wrf_builds"] + "tests" #path where different versions of the compiled WRF model code reside

#Define parameter grid for simulations (any namelist parameters and some additional ones can be used)
param_grid = odict(mp_physics=[1,2])

# names of parameter values for output filenames; either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics" : {1: "kessler", 2: "lin"},
               "sf_sfclay_physics" : {1 : "mm5", 2: "eta", 5 : "mynn"},
               "res"         : ["LES", "MYJ"]}

#Set additional namelist parameters (only active if they are not present in param_grid)
#any namelist parameters and some additional ones can be used


params["start_time"] = "2018-06-20_07:00:00" #format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_07:30:00" #format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1 #number of repetitions for each configuration

#horizontal grid
params["dx"] = 500 #horizontal grid spacing x-direction(m)
params["dy"] = None #horizontal grid spacing y-direction (m), if None: dy = dx
params["lx"] = 1000 #minimum horizontal extent in east west (m)
params["ly"] = 1000 #minimum horizontal extent in north south (m)
#use minimum number of grid points set below:
use_min_gridpoints = False #"x", "y", True (for both) or False
params["min_gridpoints_x"] = 2 #minimum number of grid points in x direction (including boundary)
params["min_gridpoints_y"] = 2 #minimum number of grid points in y direction (including boundary)
#if use_min_gridpoints: force x and y extents to be multiples of lx and ly, respectively
force_domain_multiple = False #"x", "y", True (for both) or False

#control vertical grid creation (see vertical_grid.py for details on the different methods)
params["ztop"] = 2000 #top of domain (m)
params["zdamp"] = int(params["ztop"]/3) #depth of damping layer (m)
params["nz"] = None #number of vertical levels
params["dz0"] = 20 #height of first model level (m)
params["dzmax"] = 300 #if nz is None and for dz_method=0 only: specify maximum vertical grid spacing instead of nz; either float or "dx" to make it equal to dx
params["dz_method"] = 3 #method for creating vertical grid as defined in vertical_grid.py

params["dt_f"] = None  #time step (s), if None calculated as dt = 6 s/m *min(dx,dy)/1000; can be float
#minimum time between radiation calls (min); if radt is not specified: radt=max(radt_min, 10*dt)
params["radt_min"] = 1

params["input_sounding"] = "meanwind" #name of input sounding to use (final name is then created: input_sounding_$name)

params["spec_hfx"] = None #None specified surface heat flux instead of radiation (K m s-1)

#other standard namelist parameters
params["mp_physics"] = 0

params["bl_pbl_physics"] = 2

#indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {24: ["wrfout", 10.], 0: ["fastout", 5.] }

# filename where output variables for standard and auxiliary streams are modified:
# if None: use specified value in namelist.input: if "" no file is used
params["iofields_filename"] = "IO_test.txt"

params["restart_interval_m"] = 30 #restart interval (min)


registries = ["Registry.EM_COMMON", "registry.hyb_coord", "registry.les", "registry.io_boilerplate"] #registries to look for default namelist parameters

# non-namelist parameters that will not be included in namelist file
del_args =   ["output_streams", "start_time", "end_time", "dz0", "dz_method", "min_gridpoints_x", "min_gridpoints_y", "lx", "ly", "spec_hfx", "input_sounding",
              "n_rep", "dt_f", "radt_min"]
#%%
'''Settings for resource requirements of batch jobs'''

#virtual memory (only relevant for SGE)
#numbers need adjustment
vmem_init_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for WRF initialization (ideal.exe)
vmem_init_min = 2000 #minimum virtual memory (MB) for WRF initialization

vmem = None #virtual memory per slot (MB) to request for running WRF (wrf.exe)

#if vmem is None:
vmem_per_grid_point = None #vmem (MB) per horizontal grid point per job (not per slot!)
vmem_min = None #minimum virtual memory (MB) per slot for running WRF

vmem_buffer = 1.5 #buffer factor for virtual memory (not used for test runs or if vmem is given)
vmem_test = 1000  #virtual memory per slot (MB) for test runs


#stack size (MB) for ideal.exe (SGE only)
h_stack_init = 128
#stack size (MB) for wrf.exe
h_stack = None

#runtime (min) for ideal.exe
rt_init = 10
# runtime for wrf.exe: specify either rt or runtime_per_step or None
# if None: runtime is estimated from previous identical runs if present
rt_buffer = 2 #buffer factor to multiply rt with (not used for test runs or if rt is given)
rt = None #None or job runtime in minutes; buffer not used
rt_test = 5 #runtime (min) for test runs


send_rt_signal = 10 #seconds before requested runtime is exhausted and signal is sent to job
send_rt_signal_restart = 120 #send rt signal earlier for concatenation of output files in restart runs

#paths to search for log files to determine runtime and/or vmem if not specified
resource_search_paths = [run_path]

# slots
min_nx_per_proc = 16 #25, minimum number of grid points per processor in x-direction
min_ny_per_proc = 16 #25, minimum number of grid points per processor in y-direction
even_split = False #force equal split between processors


#%%
'''Slot configurations and cluster settings'''
mail_address = "matthias.goebel@uibk.ac.at"
clusters = ["leo", "vsc"]
reduce_pool = True #reduce pool size to the actual uses number of slots; do not use if you do not want to share the node with others

host = os.popen("hostname -d").read()
module_load = ""
request_vmem = False #request specific values for virtual memory
force_pool = False

if any([c in host for c in clusters]):
    cluster = True
    #maximum number of slots that will be requested for the x and y directions
    max_nslotsy = None
    max_nslotsx = None
    if  "leo" in host:
        job_scheduler = "sge"
        #modules to load
        module_load = "module load intel/18.0u1 netcdf-4"
        queue = "std.q" #batch queue for SGE
        bigmem_queue = "bigmem.q"
        bigmem_limit = 25e3 #limit (MB) where bigmem_queue is used
        pool_size = 28 #number of cores per pool if job pooling is used
        request_vmem = True


    elif "vsc" in host:
        job_scheduler = "slurm"
        module_load = "module load intel/16.0.3 intel-mpi/5.1.3 hdf5/1.8.16 pnetcdf/1.5.0 netcdf/4.3.2;\
                       export NETCDF=/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2;\
                       export PNETCDF=/opt/sw/x86_64/glibc-2.12/ivybridge-ep/parallel/netcdf/1.5.0/intel-14.0.2"
        queue = "mem_0064" #partition on vsc3
        qos = "normal_0064"
         #minimum pool size; should be equal to the number of available CPUs per node
        pool_size = misc_tools.get_node_size_slurm(queue)
        force_pool = True #always use pooling as vsc only offers exclusive nodes
else:
    job_scheduler = "slurm"
    queue = "std"
    qos = None
    pool_size = 4
    cluster = False
    max_nslotsy = None
    max_nslotsx = None
    force_pool = True

#%%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

#Below you can manually add parameters to the DataFrame combs
