#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

@author: matze

"""
import os
from collections import OrderedDict as odict
import misc_tools
from config_test import *

# non-namelist parameters that will not be included in namelist file
del_args = ["dx", "nz", "dz0","dz_method", "gridpoints", "lx", "ly", "spec_hfx", "spec_sw",
            "pert_res", "input_sounding", "repi", "n_rep", "isotropic_res", "pbl_res", "dt"]

