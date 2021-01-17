#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

@author: Matthias Göbel

"""
import os
from collections import OrderedDict as odict
from run_wrf import misc_tools

# %%
'''Simulations settings'''
params = {}  # parameter dict for params not used in param_grid

ideal_case = "em_les"  # idealized WRF case
runID = "test"  # name for this simulation series

outpath = os.environ["wrf_res"] + "test"  # WRF output path root
run_path = os.environ["wrf_runs"]  # path where run directories of simulations will be created
build_path = os.environ["wrf_builds"]  # path where different versions of the compiled WRF model code reside
serial_build = "WRF"  # used if nslots=1
parallel_build = "WRF_mpi"  # used if nslots > 1
debug_build = "WRF_debug"  # used for -d option

# Define parameter grid for simulations
# any namelist parameters and some additional ones can be used
# if None: only 1 configuration is run
param_grid = odict(mp_physics=[1, 2],
                   res={"dx": [200, 4000], "dz0": [10, 50], "nz": [100, 60]})

# names of parameter values for output filenames;
# either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics": {1: "kessler", 2: "lin"},
               "sf_sfclay_physics": {1: "mm5", 2: "eta", 5: "mynn"},
               "res": ["LES", "MYJ"]}

# Set additional namelist parameters (only active if they are not present in param_grid)
# any namelist parameters and some additional ones can be used


params["start_time"] = "2018-06-20_00:00:00"  # format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_02:00:00"  # format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1  # number of repetitions for each configuration

# horizontal grid
params["dx"] = 1000  # horizontal grid spacing x-direction(m)
params["dy"] = None  # horizontal grid spacing y-direction (m), if None: dy = dx
params["lx"] = 1000  # horizontal extent in east west (m)
params["ly"] = 1000  # minimum horizontal extent in north south (m)
# use minimum number of grid points set below:
use_min_gridpoints = False  # "x", "y", True (for both) or False
params["min_gridpoints_x"] = 10  # minimum number of grid points in x direction
params["min_gridpoints_y"] = 10  # minimum number of grid points in y direction
# if use_min_gridpoints: force x and y extents to be multiples of lx and ly, respectively
force_domain_multiple = False  # "x", "y", True (for both) or False

# vertical grid
params["ztop"] = 5000  # top of domain (m)
params["zdamp"] = int(params["ztop"] / 3)  # depth of damping layer (m)
params["nz"] = 60  # number of vertical levels
params["dz0"] = 20  # height of first model level (m)
# if nz is None and for dz_method=0 only: specify maximum vertical grid spacing instead of nz;
# either float or "dx" to make it equal to dx
params["dzmax"] = None
params["dz_method"] = 0  # method for creating vertical grid as defined in vertical_grid.py #TODO: change all 3 to 1

params["dt_f"] = None  # time step (s), if None calculated as dt = 6 s/m *min(dx,dy)/1000; can be float
# minimum time between radiation calls (min); if radt is not specified: radt=max(radt_min, 10*dt)
params["radt_min"] = 1

params["input_sounding"] = "shalconv"  # name of input sounding to use (final name is then created: input_sounding_$name)

params["spec_hfx"] = None  # None specified surface heat flux instead of radiation (K m s-1)

# other standard namelist parameters
params["mp_physics"] = 2
params["bl_pbl_physics"] = 2

# indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {0: ["wrfout", 30]}

# filename where output variables for standard and auxiliary streams are modified:
# if None: use specified value in namelist.input: if "" no file is used
params["iofields_filename"] = None

params["restart_interval_m"] = 240  # restart interval (min)

registries = ["Registry.EM_COMMON", "registry.hyb_coord", "registry.les", "registry.io_boilerplate"]  # registries to look for default namelist parameters

# non-namelist parameters that will not be included in namelist file
del_args = ["output_streams", "start_time", "end_time", "dz0", "dz_method", "min_gridpoints_x", "min_gridpoints_y", "lx", "ly", "spec_hfx", "input_sounding",
            "n_rep", "dt_f", "radt_min"]
# %%
'''Settings for resource requirements of batch jobs'''

# virtual memory (only relevant for SGE)
# numbers need adjustment
vmem_init_per_grid_point = 0.3  # virtual memory (MB) per horizontal grid point to request for WRF initialization (ideal.exe)
vmem_init_min = 2000  # minimum virtual memory (MB) for WRF initialization

vmem = None  # virtual memory per slot (MB) to request for running WRF (wrf.exe)

# if vmem is None:
vmem_per_grid_point = None  # vmem (MB) per horizontal grid point per job (not per slot!)
vmem_min = None  # minimum virtual memory (MB) per slot for running WRF

vmem_buffer = 1.5  # buffer factor for virtual memory (not used for test runs or if vmem is given)
vmem_test = 2000  # virtual memory per slot (MB) for test runs


# stack size (MB) for ideal.exe (SGE only)
h_stack_init = 128
# stack size (MB) for wrf.exe
h_stack = None

# runtime (min) for ideal.exe
rt_init = 10
# runtime for wrf.exe: specify either rt or runtime_per_step or None
# if None: runtime is estimated from previous identical runs if present
runtime_per_step = None
rt_buffer = 2  # buffer factor to multiply rt with (not used for test runs or if rt is given)
rt = None  # None or job runtime in minutes; buffer not used
rt_test = 5  # runtime (min) for test runs
rt_use_median = False  # use median when averaging runtime of previous runs, may be necessary if previous runs are short


send_rt_signal = 20  # seconds before requested runtime is exhausted and signal is sent to job

# additional paths (other than run_path) to search for log files to determine runtime and/or vmem if not specified
resource_search_paths = []

# slots
min_nx_per_proc = 50  # minimum number of grid points per processor in x-direction
min_ny_per_proc = 25  # minimum number of grid points per processor in y-direction
even_split = False  # force equal split between processors


# %%
'''Slot configurations and cluster settings'''
mail_address = ""
clusters = ["leo", "vsc3", "vsc4"]
# reduce pool size to the actual uses number of slots
# do not use if you do not want to share the node with others
reduce_pool = True
host = os.popen("hostname -d").read()
module_load = ""
request_vmem = False  # request specific values for virtual memory
force_pool = False

if any([c in host for c in clusters]):
    cluster = True
    # maximum number of slots that will be requested for the x and y directions
    max_nslotsy = None
    max_nslotsx = None
    if "leo" in host:
        job_scheduler = "sge"
        # modules to load
        module_load = "module load intel/18.0u1 netcdf-4"
        queue = "std.q"  # batch queue for SGE
        bigmem_queue = "bigmem.q"
        bigmem_limit = 25e3  # limit (MB) where bigmem_queue is used
        pool_size = 28  # number of cores per pool if job pooling is used
        request_vmem = True
    elif "vsc3" in host:
        job_scheduler = "slurm"
        module_load = "module load intel/19 intel-mpi/2019 hdf5/1.8.12-MPI pnetcdf/1.10.0 netcdf_C/4.4.1.1 netcdf_Fortran/4.4.4;\
                       export NETCDF=/opt/sw/x86_64/glibc-2.17/ivybridge-ep/netcdf_Fortran/4.4.4/intel/19/intel-mpi/2019/hdf5/1.8.12-MPI/pnetcdf/1.10.0/netcdf_C/4.4.1.1/"
        queue = "vsc3plus_0064"  # partition on vsc3
        qos = "vsc3plus_0064"
        # queue = "mem_0064"
        # qos = "normal_0064"
        # minimum pool size; should be equal to the number of available CPUs per node
        pool_size = misc_tools.get_node_size_slurm(queue)
        force_pool = True  # always use pooling as vsc only offers exclusive nodes
    elif "vsc4" in host:
        job_scheduler = "slurm"
        module_load = "module load intel intel-mpi;\
                       export NETCDF=$HOME/wrf/lib/LIBRARIES/netcdf"
        queue = "mem_0096"
        qos = "mem_0096"
        # minimum pool size; should be equal to the number of available CPUs per node
        pool_size = misc_tools.get_node_size_slurm(queue)
        force_pool = True  # always use pooling as vsc only offers exclusive nodes
else:
    pool_size = 4
    cluster = False
    max_nslotsy = None
    max_nslotsx = None
    force_pool = True

# %%

param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

# Below you can manually add parameters to the DataFrame combs
