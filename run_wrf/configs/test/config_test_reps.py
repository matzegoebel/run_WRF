#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

To test repetitions functionality
@author: matze

"""
import os
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.test.config_test import *
from copy import deepcopy

#%%
param_grid = odict(mp_physics=[1])
params = deepcopy(params)
params["n_rep"] = 2 #number of repetitions for each configuration

#%%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

