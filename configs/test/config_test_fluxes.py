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
wrf_dir_pre = "WRF" #prefix for WRF build directory (_debug or _mpi are appended automatically)
build_path = os.environ["wrf_builds"] #path where different versions of the compiled WRF model code reside

param_grid = odict(fluxcalc=dict(do_avgflx_em=[0,1], iofields_filename=["", "LES_IO.txt"]))
param_names = {"fluxcalc": ["offline", "online"]}
params = params.copy()
params["dx"] = 150 #horizontal grid spacing (m)
params["dt"] = 1  #time step (s), if None calculated as dt = 6 s/m *dx/1000
params["lx"] = 400 #minimum horizontal extent in east west (m)
params["ly"] = 400 #minimum horizontal extent in north south (m)
params["nz"] = 100 #number of vertical levels
params["dz0"] = 10 #height of first model level (m)

#%%
param_combs, combs, param_grid_flat, composite_params = misc_tools.grid_combinations(param_grid, params)

combs[0][ "output_streams"] = {0: ["fastout", 1./60 ]}
combs[1][ "output_streams"] = {0: ["fastout", 10.], 7 : ["slowout", 30.]}
