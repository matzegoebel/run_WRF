#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

For MPI testing with SLURM.

@author: Matthias Göbel

"""
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.test.config_test import *
if "param_combs" in dir():
    del param_combs
from copy import deepcopy

# %%

param_grid = odict(mp_physics=[1, 2])
params = deepcopy(params)

params["lx"] = 16000
params["ly"] = 5000

mail_address = "test@test.com"
# %%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)
