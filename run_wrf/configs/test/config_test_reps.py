#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Settings for launch_jobs.py
Test settings for automated tests.

To test repetitions functionality
@author: Matthias Göbel

"""
from collections import OrderedDict as odict
from run_wrf.configs.test.config_test import *
from copy import deepcopy
params = deepcopy(params)

# %%
param_grid = odict(mp_physics=[5])
params["n_rep"] = 2  # number of repetitions for each configuration
