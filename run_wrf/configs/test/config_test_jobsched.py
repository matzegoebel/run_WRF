#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

To test run with job scheduler

@author: Matthias Göbel

"""
import os
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.test.config_test import *
if "param_combs" in dir():
    del param_combs

#%%

vmem = 500

#%%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

