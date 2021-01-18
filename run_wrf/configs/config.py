#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Settings for submit_jobs.py

@author: Matthias Göbel

"""
from collections import OrderedDict as odict
from run_wrf import misc_tools
from run_wrf.configs.base_config import *
from copy import deepcopy
params = deepcopy(params)

# %%
'''Simulations settings'''

runID = "test"  # name for this simulation series

# Define parameter grid for simulations
# any namelist parameters can be used and parameters with default values set in dict params below
# if None: only 1 configuration is run
param_grid = odict(mp_physics=[1, 2],
                   res={"dx": [200, 4000], "dz0": [10, 50], "nz": [100, 60]})

# names of parameter values for output filenames;
# either dictionaries or lists (not for composite parameters)
param_names = {"mp_physics": {1: "kessler", 2: "lin"},
               "sf_sfclay_physics": {1: "mm5", 2: "eta", 5: "mynn"},
               "res": ["LES", "MYJ"]}

# Fill dictionary params with default values to be used for parameters not present in param_grid

params["start_time"] = "2018-06-20_00:00:00"  # format %Y-%m-%d_%H:%M:%S
params["end_time"] = "2018-06-20_02:00:00"  # format %Y-%m-%d_%H:%M:%S

params["n_rep"] = 1  # number of repetitions for each configuration

# horizontal grid
params["dx"] = 1000  # horizontal grid spacing x-direction(m)
params["dy"] = None  # horizontal grid spacing y-direction (m), if None: dy = dx
params["lx"] = 1000  # horizontal extent in east west (m)
params["ly"] = 1000  # horizontal extent in north south (m)

# vertical grid
params["ztop"] = 5000  # top of domain (m)
params["zdamp"] = int(params["ztop"] / 3)  # depth of damping layer (m)
params["nz"] = 60  # number of vertical levels
params["dz0"] = 20  # height of first model level (m)
# if nz is None and for vgrid_method=0 only: specify maximum vertical grid spacing instead of nz;
# either float or "dx" to make it equal to dx
params["dzmax"] = None
# method for creating vertical grid as defined in vertical_grid.py
# if None: do not change eta_levels
params["vgrid_method"] = 0

params["dt_f"] = 1  # time step (s), can be float

params["input_sounding"] = "shalconv"  # name of input sounding to use (final name is then created: input_sounding_$name)

params["spec_hfx"] = None  # None specified surface heat flux instead of radiation (K m s-1)

# other standard namelist parameters
params["mp_physics"] = 2
params["bl_pbl_physics"] = 2

# indices for output streams and their respective name and output interval (minutes, floats allowed)
# 0 is the standard output stream
params["output_streams"] = {0: ["wrfout", 30]}

# filename where output variables for standard and auxiliary streams are modified:
# if None: use specified value in namelist.input: if "" no file is used
params["iofields_filename"] = None

params["restart_interval_m"] = 240  # restart interval (min)

# override values of base_config
params["ideal_case_name"] = "em_b_wave"  # idealized WRF case


# %%

# create parameter grid, if not set or None, grid is created in submit_jobs.py
param_combs = misc_tools.grid_combinations(param_grid, params, param_names=param_names, runID=runID)

# Below you can manually add or change parameters in param_combs
