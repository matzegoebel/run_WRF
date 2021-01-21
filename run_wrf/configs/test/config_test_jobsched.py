#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Settings for submit_jobs.py
Test settings for automated tests.

To test run with job scheduler

@author: Matthias Göbel

"""
from run_wrf.configs.test.config_test import *
from copy import deepcopy
params = deepcopy(params)

vmem = 500
