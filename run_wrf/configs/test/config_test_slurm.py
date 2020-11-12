#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

For MPI testing with SLURM.

@author: matze

"""

from run_wrf.configs.test.config_test_mpi import *

#%%

job_scheduler = "slurm"
module_load = "module load intel/19 intel-mpi/2019 hdf5/1.8.12-MPI pnetcdf/1.10.0 netcdf_C/4.4.1.1 netcdf_Fortran/4.4.4;\
               export NETCDF=/opt/sw/x86_64/glibc-2.17/ivybridge-ep/netcdf_Fortran/4.4.4/intel/19/intel-mpi/2019/hdf5/1.8.12-MPI/pnetcdf/1.10.0/netcdf_C/4.4.1.1/"

queue = "mem_0064" #partition on vsc3
qos = "normal_0064"
 #minimum pool size; should be equal to the number of available CPUs per node
pool_size = 8
request_vmem = False
force_pool = True #always use pooling