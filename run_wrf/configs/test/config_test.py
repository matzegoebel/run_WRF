#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:36:46 2019

Settings for submit_jobs.py
Test settings for automated tests.

@author: Matthias Göbel
"""
import os
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.base_config import *
from copy import deepcopy
params = deepcopy(params)#TODOm: why?

# %%
'''Simulations settings'''

runID = "pytest"  # name for this simulation series

outpath = os.environ["wrf_res"] + "/test/" + runID  # WRF output path root
run_path = os.environ["wrf_runs"] + "/" + runID  # path where run directories of simulations will be created
build_path = os.environ["wrf_builds"] + "/tests"  # path where different versions of the compiled WRF model code reside

# Define parameter grid for simulations (any namelist parameters and some additional ones can be used)
param_grid = odict(mp_physics=[1, 2])

# names of parameter values for output filenames; either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics": {1: "kessler", 2: "lin"},
               "sf_sfclay_physics": {1: "mm5", 2: "eta", 5: "mynn"},
               "res": ["LES", "MYJ"]}

# Set additional namelist parameters (only active if they are not present in param_grid)
# any namelist parameters and some additional ones can be used


params["start_time"] = "2018-06-20_07:00:00"  # format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_07:06:00"  # format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1  # number of repetitions for each configuration

# horizontal grid
params["dx"] = 500  # horizontal grid spacing x-direction(m)
params["dy"] = None  # horizontal grid spacing y-direction (m), if None: dy = dx
params["lx"] = 1000  # minimum horizontal extent in east west (m)
params["ly"] = 1000  # minimum horizontal extent in north south (m)
# use minimum number of grid points set below:
use_min_gridpoints = False  # "x", "y", True (for both) or False
params["min_gridpoints_x"] = 2  # minimum number of grid points in x direction (including boundary)
params["min_gridpoints_y"] = 2  # minimum number of grid points in y direction (including boundary)

# control vertical grid creation (see vertical_grid.py for details on the different methods)
params["ztop"] = 2000  # top of domain (m)
params["zdamp"] = int(params["ztop"] / 3)  # depth of damping layer (m)
params["nz"] = None  # number of vertical levels
params["dz0"] = 20  # height of first model level (m)
# if nz is None and for vgrid_method=0 only: specify maximum vertical grid spacing instead of nz
# either float or "dx" to make it equal to dx
params["dzmax"] = 300
params["vgrid_method"] = 1  # method for creating vertical grid as defined in vertical_grid.py

params["dt_f"] = 3  # time step (s), can be float

params["input_sounding"] = "meanwind"  # name of input sounding to use (final name is then created: input_sounding_$name)

params["spec_hfx"] = None  # None specified surface heat flux instead of radiation (K m s-1)

# other standard namelist parameters
params["mp_physics"] = 0

params["bl_pbl_physics"] = 2

# indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {24: ["wrfout", 2.], 0: ["fastout", 1.]}

# filename where output variables for standard and auxiliary streams are modified:
# if None: use specified value in namelist.input: if "" no file is used
params["iofields_filename"] = "IO_test.txt"

params["restart_interval_m"] = 4  # restart interval (min)


# %%
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

# Below you can manually add parameters to the DataFrame combs
