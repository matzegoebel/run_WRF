#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

To test restart functionality

@author: Matthias Göbel

"""
from run_wrf.configs.test.config_test import *
from copy import deepcopy
params = deepcopy(params)

# %%

params["end_time"] = "2018-06-20_07:08:00"  # format %Y-%m-%d_%H:%M:%S
