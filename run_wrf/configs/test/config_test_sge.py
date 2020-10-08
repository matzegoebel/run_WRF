#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

For MPI testing with SGE.

@author: matze

"""

from run_wrf.configs.test.config_test_mpi import *

#%%

job_scheduler = "sge"
#modules to load
module_load = "module load intel/18.0u1 netcdf-4"
queue = "std.q" #batch queue for SGE
bigmem_queue = "bigmem.q"
bigmem_limit = 25e3 #limit (MB) where bigmem_queue is used
pool_size = 28 #number of cores per pool if job pooling is used
request_vmem = True