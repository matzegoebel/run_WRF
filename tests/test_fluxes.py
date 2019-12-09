#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:06:38 2019

@author: c7071088
"""



import os
from submit_jobs import submit_jobs
import pytest
from misc_tools import Capturing
from collections import Counter
import configs.test.config_test_fluxes as conf
import shutil
import time
from netCDF4 import Dataset
import misc_tools
import wrf
import pandas as pd
import tools
import itertools
import matplotlib.pyplot as plt
import numpy as np
#%%

def test_submit_jobs():
    outpath = conf.outpath + "/test/test_fluxes"
    for d in [outpath, os.environ["wrf_runs"] + "/test_fluxes"]:
        if os.path.isdir(d):
            shutil.rmtree(d)

    os.chdir("..")
    submit_jobs(init=True, exist="o", config_file="test.config_test_fluxes")

    with Capturing() as output:
        submit_jobs(init=False, wait=True, pool_jobs=True, exist="o", config_file="test.config_test_fluxes")
    count = Counter(output)
    for m in ['d01 2018-06-20_08:00:00 wrf: SUCCESS COMPLETE WRF']:
        assert count[m] == 2

    file_on = Dataset(outpath + "/slowout_test_fluxes_online_0")
    file_off = Dataset(outpath + "/fastout_test_fluxes_offline_0")

    #%%
    var = ["U", "V", "W", "TH", "Q"]
    var_wrf = ["ua", "va", "wa", "th", "QVAPOR"]
    #covs = list(itertools.combinations_with_replacement(var, 2))
    dat_off = {v : wrf.getvar(file_off, v_wrf, timeidx=None) for v, v_wrf in zip(var, var_wrf)}

    i = 0
    for v1 in var:
        for v2 in [None] + var:
            v = v1
            if v2 is not None:
                v += v2
            v += "MEAN"
            if v not in file_on.variables:
                print("Variable {} not available.".format(v))
                continue
            i += 1
            cov_on = wrf.getvar(file_on, v, timeidx=None)[1:].values.flatten()
            cov_off_hres = dat_off[v1]
            if v2 is not None:
                cov_off_hres = cov_off_hres * dat_off[v2]
            cov_off_hres = cov_off_hres.where(cov_off_hres!=np.inf, np.nan)
            cov_off = cov_off_hres.resample(Time="30min", closed="right", label="right").mean()[1:].values.flatten()
            joint = np.append(cov_off, cov_on)
            ax = plt.subplot(3,6,i)
            ax.scatter(cov_on, cov_off, s=1)
            plt.ylabel("offline")
            plt.xlabel("online")
            #vmin, vmax = np.quantile(joint, q=[0.01,0.99])
            # plt.ylim(vmin, vmax)
            # plt.xlim(vmin, vmax)
            ax.set_title(v)





