#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

For MPI testing.

@author: matze

"""
import os
from collections import OrderedDict as odict
import misc_tools
from configs.test.config_test import *

#%%

param_grid = odict(mp_physics=[1, 2])
params = params.copy()

params["end_time"] = "2018-06-20_00:11:00" #format %Y-%m-%d_%H:%M:%S

use_min_gridpoints = True #"x", "y", True (for both) or False
params["min_gridpoints_x"] = 33 #minimum number of grid points in x direction
params["min_gridpoints_y"] = 11 #minimum number of grid points in y direction
pool_size = 4

#%%
param_combs, combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid, params)
