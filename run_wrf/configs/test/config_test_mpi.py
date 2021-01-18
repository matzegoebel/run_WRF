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

param_grid = odict(mp_physics=[1, 2])

params["lx"] = 16000
params["ly"] = 5000

mail_address = "test@test.com"
