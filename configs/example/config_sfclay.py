#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Example settings

@author: matze

"""
import os
from collections import OrderedDict as odict
import misc_tools
from configs.config import *
from copy import deepcopy


#%%
runID = "sfclay" #name for this simulation series
outdir = "test/" + runID #subdirectory for WRF output if not set in command line

param_grid = odict(sf_sfclay_physics=[1, 2, 5])

params = deepcopy(params)

params["dx"] = 500 #horizontal grid spacing (m)

#%%
param_combs, combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid, params)


