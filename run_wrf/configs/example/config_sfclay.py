#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Example settings

@author: Matthias Göbel

"""
from collections import OrderedDict as odict
from run_wrf.configs.config import *
from copy import deepcopy
params = deepcopy(params)
param_combs = None


# %%

runID = "sfclay"  # name for this simulation series

param_grid = odict(sf_sfclay_physics=[1, 2, 5])

params["dx"] = 500  # horizontal grid spacing x-direction(m)
params["dy"] = None  # horizontal grid spacing y-direction (m), if None: dy = dx
