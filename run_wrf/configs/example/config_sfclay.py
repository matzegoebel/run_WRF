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
from run_wrf import misc_tools
from run_wrf.configs.config import *
if "param_combs" in dir():
    del param_combs
from copy import deepcopy


#%%
runID = "sfclay" #name for this simulation series
outdir = "test/" + runID #subdirectory for WRF output if not set in command line

param_grid = odict(sf_sfclay_physics=[1, 2, 5])

params = deepcopy(params)

params["dx"] = 500 #horizontal grid spacing x-direction(m)
params["dy"] = None #horizontal grid spacing y-direction (m), if None: dy = dx

#%%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)


