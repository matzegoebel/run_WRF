#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Base settings for launch_jobs.py

@author: Matthias Göbel

"""
import os
from run_wrf import tools


# %%
'''Simulations settings'''
# parameter dict for params not used in param_grid
# parameters defined here but not as elements of params cannot be used in param_grid
params = {}

params["ideal_case_name"] = "em_les"  # idealized WRF case

if "wrf_res" in os.environ:
    params["outpath"] = os.environ["wrf_res"]  # WRF output path root
if "wrf_runs" in os.environ:
    params["run_path"] = os.environ["wrf_runs"]  # path where run directories of simulations will be created
if "wrf_builds" in os.environ:
    params["build_path"] = os.environ["wrf_builds"]  # path where different versions of the compiled WRF model code reside
params["serial_build"] = "WRF"  # used if nslots=1
params["parallel_build"] = "WRF_mpi"  # used if nslots > 1
params["debug_build"] = "WRF_debug"  # used for -d option

# registries to look for default namelist parameters
registries = ["Registry.EM_COMMON", "registry.hyb_coord", "registry.les", "registry.io_boilerplate"]

# %%
'''Settings for resource requirements of batch jobs'''

# virtual memory (only relevant for SGE)
# numbers need adjustment
params["vmem_init"] = 2000  # virtual memory (MB) for WRF initialization

# virtual memory per slot (MB) to request for running WRF (wrf.exe)
# if None: try to estimate from previous runs and multiply with vmem_buffer
params["vmem"] = None
params["vmem_buffer"] = 1.5  # buffer factor for virtual memory (not used for test runs or if vmem is given)
params["vmem_test"] = 1000  # virtual memory per slot (MB) for test runs

# stack size (MB) for ideal.exe (SGE only)
params["h_stack_init"] = 128
# stack size (MB) for wrf.exe
params["h_stack"] = None

# runtime (min) for ideal.exe
params["runtime_init"] = 10
# runtime for wrf.exe: specify either rt or runtime_per_step or None
# if None: runtime is estimated from previous identical runs if present
params["runtime_per_step"] = None  # runtime per timestep
params["rt_buffer"] = 2  # buffer factor to multiply rt with (not used for test runs or if rt is given)
params["runtime"] = None  # None or job runtime in minutes; buffer not used
params["runtime_test"] = 5  # runtime (min) for test runs
# use median when averaging runtime of previous runs, may be necessary if previous runs are short
params["rt_use_median"] = False
params["send_rt_signal"] = 90  # seconds before requested runtime is exhausted and signal is sent to job

# additional paths (other than run path) to search for log files
# to determine runtime and/or vmem if not specified
resource_search_paths = []

# slots
params["min_nx_per_proc"] = 16  # 25, minimum number of grid points per processor in x-direction
params["min_ny_per_proc"] = 16  # 25, minimum number of grid points per processor in y-direction


# %%
'''Slot configurations and cluster settings'''
# path to mpiexec or mpirun executable to use; if None: use the system default
mpiexec = None

mail_address = ""  # mail address for job scheduler
clusters = ["leo", "vsc3", "vsc4"]
force_pool = False  # always use pooling
# reduce pool size to the actual used number of slots
# do not use if you do not want to share the node with others
reduce_pool = True
host = os.popen("hostname -d").read()
params["module_load"] = ""  # module load and other architecture specific shell command
request_vmem = False  # request specific values for virtual memory
# maximum number of slots that will be requested for the x and y directions
params["max_nslotsx"] = None
params["max_nslotsy"] = None
if any([c in host for c in clusters]):
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
        module_load = "module load intel/19 intel-mpi/2019 hdf5/1.8.12-MPI pnetcdf/1.10.0 netcdf_C/4.4.1.1 netcdf_Fortran/4.4.4;"\
                      "export NETCDF=/opt/sw/x86_64/glibc-2.17/ivybridge-ep/netcdf_Fortran/4.4.4/intel/19/intel-mpi/2019/hdf5/1.8.12-MPI/pnetcdf/1.10.0/netcdf_C/4.4.1.1/"
        queue = "vsc3plus_0064"  # partition on vsc3
        qos = "vsc3plus_0064"
        # queue = "mem_0064"
        # qos = "normal_0064"
        # minimum pool size; should be equal to the number of available CPUs per node
        pool_size = tools.get_node_size_slurm(queue)
        force_pool = True  # vsc only offers exclusive nodes
    elif "vsc4" in host:
        job_scheduler = "slurm"
        module_load = "module load intel intel-mpi;"\
                      "export NETCDF=$HOME/wrf/lib/LIBRARIES/netcdf"
        queue = "mem_0096"
        qos = "mem_0096"
        # minimum pool size; should be equal to the number of available CPUs per node
        pool_size = tools.get_node_size_slurm(queue)
        force_pool = True  # vsc only offers exclusive nodes
    params["module_load"] = "module purge;" + module_load

else:
    pool_size = 4
