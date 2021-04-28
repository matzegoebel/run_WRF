#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Test settings for automated tests.

For MPI testing with SLURM.

@author: Matthias Göbel

"""

from run_wrf.configs.test.config_test_mpi import *
from copy import deepcopy
params = deepcopy(params)

# %%
params["end_time"] = "2018-06-20_10:00:00"  # format %Y-%m-%d_%H:%M:%S

job_scheduler = "slurm"
params["module_load"] = "module load intel"
queue = "mem_0064"  # partition on vsc3
qos = "normal_0064"
# minimum pool size; should be equal to the number of available CPUs per node
pool_size = 8
request_vmem = False
force_pool = True  # always use pooling
