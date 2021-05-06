#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for launch_jobs.py
Test settings for automated tests.

For MPI testing with SLURM.

@author: Matthias Göbel

"""
from collections import OrderedDict as odict
from run_wrf import tools
from run_wrf.configs.test.config_test import *
from copy import deepcopy
params = deepcopy(params)

# %%
params["ztop"] = 15000  # top of domain (m)

# use sounding to convert height to pressure levels
params["sounding_path"] = f"{params['build_path']}/WRF/test/em_les/input_sounding_cops"
params["input_sounding"] = "cops"
