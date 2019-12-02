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

vmem = 200

#%%
param_combs, combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid, params)

