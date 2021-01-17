#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Base settings for submit_jobs.py

@author: Matthias Göbel

"""
import os
from run_wrf import misc_tools

# %%
'''Simulations settings'''
params = {}  # parameter dict for params not used in param_grid

ideal_case = "em_les"  # idealized WRF case

outpath = os.environ["wrf_res"]  # WRF output path root
run_path = os.environ["wrf_runs"]  # path where run directories of simulations will be created
build_path = os.environ["wrf_builds"]  # path where different versions of the compiled WRF model code reside
serial_build = "WRF"  # used if nslots=1
parallel_build = "WRF_mpi"  # used if nslots > 1
debug_build = "WRF_debug"  # used for -d option

# registries to look for default namelist parameters
registries = ["Registry.EM_COMMON", "registry.hyb_coord", "registry.les", "registry.io_boilerplate"]

# %%
'''Settings for resource requirements of batch jobs'''

# virtual memory (only relevant for SGE)
# numbers need adjustment
vmem_init = 2000  # virtual memory (MB) for WRF initialization

# virtual memory per slot (MB) to request for running WRF (wrf.exe)
# if None: try to estimate from previous runs and multiply with vmem_buffer
vmem = None
vmem_buffer = 1.5  # buffer factor for virtual memory (not used for test runs or if vmem is given)
vmem_test = 1000  # virtual memory per slot (MB) for test runs

# stack size (MB) for ideal.exe (SGE only)
h_stack_init = 128
# stack size (MB) for wrf.exe
h_stack = None

# runtime (min) for ideal.exe
rt_init = 10
# runtime for wrf.exe: specify either rt or runtime_per_step or None
# if None: runtime is estimated from previous identical runs if present
runtime_per_step = None  # runtime per timestep
rt_buffer = 2  # buffer factor to multiply rt with (not used for test runs or if rt is given)
rt = None  # None or job runtime in minutes; buffer not used
rt_test = 5  # runtime (min) for test runs
# use median when averaging runtime of previous runs, may be necessary if previous runs are short
rt_use_median = False
send_rt_signal = 10  # seconds before requested runtime is exhausted and signal is sent to job

# additional paths (other than run_path) to search for log files
# to determine runtime and/or vmem if not specified
resource_search_paths = []

# slots
min_nx_per_proc = 16  # 25, minimum number of grid points per processor in x-direction
min_ny_per_proc = 16  # 25, minimum number of grid points per processor in y-direction
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
