#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

Assert RuntimeError due to dx being in del_args

@author: Matthias Göbel

"""

from run_wrf.configs.test.config_test import *

# non-namelist parameters that will not be included in namelist file
del_args = ["dx", "start_time", "end_time", "dz0", "dz_method", "min_gridpoints_x",
            "min_gridpoints_y", "lx", "ly", "spec_hfx", "input_sounding",
            "n_rep", "dt_f", "radt_min"]
