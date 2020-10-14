#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:14:23 2019

@author: c7071088
"""
from run_wrf import misc_tools
from run_wrf.misc_tools import Capturing
import pytest
import numpy as np
import subprocess as sp
import os
from collections import OrderedDict as odict
from run_wrf import get_namelist
import pandas as pd
from collections import Counter

runs_dir = "./test_data/runs/"
#%%
def test_get_runtime():
    print(os.getcwd())
    timing, counter = misc_tools.get_runtime(runs_dir + "WRF_pytest_eta_0/run_2018-04-10T06:13:14.log", all_times=True)
    assert counter == 86400
    assert timing.notna().all().all()
    timing_m, _ = misc_tools.get_runtime(runs_dir + "WRF_pytest_eta_0/run_2018-04-10T06:13:14.log", all_times=False, counter=None, timing=None)
    np.testing.assert_allclose( timing_m["timing"].values.mean(), timing["timing"].mean())
    assert (timing_m.values[0,:-2] == np.array([1,1,6,6])).all()
    with pytest.raises(FileNotFoundError):
        timing, counter = misc_tools.get_runtime(runs_dir + "/run.log", all_times=True)

    dirs = runs_dir
    timing_m = misc_tools.get_runtime_all(id_filter="", dirs=dirs, all_times=False, levels=None, remove=None, verbose=True)
    timing = misc_tools.get_runtime_all(id_filter="", dirs=dirs, all_times=True, remove=None, verbose=True)
    assert len(timing) == 345600
    assert timing.iloc[:,-7:].notna().all().all()

    with Capturing() as output:
        identical_runs = misc_tools.get_identical_runs(runs_dir + "WRF_pytest_eta_0", "test_data/runs/")
    print(output)
    assert len(output) == 2
    output = [o.split("/")[-1] for o in output]
    assert 'WRF_pytest_2_eta_0 has same namelist parameters' in output
    assert 'WRF_pytest_eta_0 has same namelist parameters'  in output
    vmem = misc_tools.get_vmem(identical_runs)
    assert sorted(vmem) == [114.105, 160.0]

def test_job_usage():
    resource_file = "test_data/resources.info"
    usage = misc_tools.get_job_usage(resource_file)
    control = {'cpu': '00:02:00',
     'mem': '13.13958 GB s',
     'io': '0.02315 GB',
     'vmem': '114.105M',
     'maxvmem': '114.105M'}
    assert control == usage

def test_namelist_to_dict():
    path = 'test_data/namelists/namelist.test'
    namelist_dict = get_namelist.namelist_to_dict(path, verbose=False)
    correct = {'run_days': 0,
     'iofields_filename': "'LES_IO.txt'",
     'input_2d': '.false',
     'auxhist7_outname': "'test'",
     'shadlen': 25000}
    for k, v in correct.items():
        assert v == namelist_dict[k]

def test_grid_combinations():
    param_grid = odict(input_sounding=["stable", "unstable"], res={"dx" : [200,4000], "dz0" : [10,50]})
    param_names = dict(res = [200, 4000])
    param_combs = misc_tools.grid_combinations(param_grid, param_names=param_names, runID="")
    param_combs = param_combs.drop(columns="fname")
    param_combs_corr = pd.DataFrame(columns=["input_sounding", "dx", "dz0"], index=np.arange(5))
    param_combs_corr.loc[:,:] = np.array([['stable', 200, 10],
                                          ['stable', 4000, 50],
                                          ['unstable', 200, 10],
                                          ['unstable', 4000, 50],
                                          [True, True, True]], dtype=object)
    param_combs_corr.index = param_combs.index
    pd.testing.assert_frame_equal(param_combs.astype(object), param_combs_corr)
