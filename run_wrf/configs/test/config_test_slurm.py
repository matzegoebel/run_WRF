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
module_load = "module load intel/16.0.3 intel-mpi/5.1.3 hdf5/1.8.16 pnetcdf/1.5.0 netcdf/4.3.2;\
               export NETCDF=/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2;\
               export PNETCDF=/opt/sw/x86_64/glibc-2.12/ivybridge-ep/parallel/netcdf/1.5.0/intel-14.0.2"
queue = "mem_0064" #partition on vsc3
qos = "normal_0064"
 #minimum pool size; should be equal to the number of available CPUs per node
pool_size = 8
request_vmem = False
force_pool = True #always use pooling