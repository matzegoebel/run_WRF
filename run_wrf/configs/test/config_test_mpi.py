#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

For MPI testing with SLURM.

@author: Matthias Göbel

"""
import os
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.test.config_test import *
if "param_combs" in dir():
    del param_combs
from copy import deepcopy

#%%

param_grid = odict(mp_physics=[1, 2])
params = deepcopy(params)

use_min_gridpoints = True #"x", "y", True (for both) or False
params["min_gridpoints_x"] = 33 #minimum number of grid points in x direction
params["min_gridpoints_y"] = 11 #minimum number of grid points in y direction

#%%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)
