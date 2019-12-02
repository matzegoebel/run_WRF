#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:26:51 2019

Test submit_jobs function

@author: c7071088
"""

import os
from submit_jobs import submit_jobs
import pytest
from misc_tools import Capturing
from collections import Counter
import configs.test.config_test as conf
import shutil
import time
from netCDF4 import Dataset
import misc_tools
import wrf
import pandas as pd
#%%

def test_submit_jobs():
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)

    for add in ["_mpi", ""]:
        target_dir = "{}/{}{}/test/{}/".format(conf.build_path, conf.wrf_dir_pre, add, conf.ideal_case)
        shutil.copy("test_data/IO_test.txt", target_dir)
        shutil.copy("test_data/namelists/namelist.input", target_dir + "namelist.input")

    os.chdir("..")
    with pytest.raises(RuntimeError):
        submit_jobs(config_file="test.config_test_del_args", init=True)

    #check skipping non-initialized runs
    with Capturing() as output:
        submit_jobs(init=False, config_file="test.config_test")
    assert Counter(output)["Run not initialized yet! Skipping..."] == 2


    #initialize and run wrf
    combs = submit_jobs(init=True, config_file="test.config_test")
    nruns = combs["n_rep"].sum()
    submit_jobs(init=False, wait=True, config_file="test.config_test")

    #check output data
    outd = os.path.join(conf.outpath, conf.outdir)
    outfiles = ['fastout_pytest_lin_0','wrfout_pytest_lin_0', 'fastout_pytest_kessler_0', 'wrfout_pytest_kessler_0']
    assert sorted(os.listdir(outd)) == sorted(outfiles)
    file = Dataset(outd + "/fastout_pytest_lin_0")
    t = wrf.extract_times(file, timeidx=None)
    t_corr = pd.date_range(start="2018-06-20T00:00:00", end='2018-06-20T02:00:00', freq="10min")
    assert (t == t_corr).all()

    #test behaviour if output exists
    for run in os.listdir(conf.run_path):
        file = "{}/{}/wrfinput_d01".format(conf.run_path, run)
        if os.path.isfile(file):
            os.remove(file)
    exist_message = (("s", "Redoing initialization..."), ("s", "Skipping..."), ("o", "Overwriting..."), ("b", "Creating backup..."))
    success = {True : 'wrf: SUCCESS COMPLETE IDEAL INIT', False : 'd01 2018-06-20_02:00:00 wrf: SUCCESS COMPLETE WRF'}
    for init in [True, False]:
        for i, (exist, message) in enumerate(exist_message):
            if init or i > 0:
                print(exist, message)
                with Capturing() as output:
                    submit_jobs(init=init, exist=exist, wait=True, config_file="test.config_test")
                count = Counter(output)
                assert count[message] == nruns
                if "Skipping..." not in message:
                    assert count[success[init]] == nruns


    #check backup creation
    submit_jobs(init=False, exist="b", wait=True, config_file="test.config_test")
    bak = ['fastout_pytest_lin_0_bak_0',
           'wrfout_pytest_lin_0_bak_0',
           'fastout_pytest_lin_0_bak_1',
           'wrfout_pytest_lin_0_bak_1',
           'fastout_pytest_kessler_0_bak_0',
           'wrfout_pytest_kessler_0_bak_0',
           'fastout_pytest_kessler_0_bak_1',
           'wrfout_pytest_kessler_0_bak_1']
    assert sorted(os.listdir(outd + "/bak")) == sorted(bak)

    with pytest.raises(ValueError, match="Value 'a' for -e option not defined!"):
        submit_jobs(init=True, exist="a", config_file="test.config_test")


    #check restart
    with Capturing() as output:
        submit_jobs(init=False, restart=True, wait=True, config_file="test.config_test_rst")
    count = Counter(output)
    for m in ["Restart run from 2018-06-20 02:00:00", 'd01 2018-06-20_04:00:00 wrf: SUCCESS COMPLETE WRF']:
        assert count[m] == nruns
    #check output data
    outd = os.path.join(conf.outpath, conf.outdir)
    outfiles = ['rst', 'bak', 'fastout_pytest_lin_0','wrfout_pytest_lin_0', 'fastout_pytest_kessler_0', 'wrfout_pytest_kessler_0']
    assert sorted(os.listdir(outd)) == sorted(outfiles)
    file = Dataset(outd + "/fastout_pytest_lin_0")
    t = wrf.extract_times(file, timeidx=None)
    t_corr = pd.date_range(start="2018-06-20T00:00:00", end='2018-06-20T04:00:00', freq="10min")
    assert (t == t_corr).all()

    #check repeats
    combs = submit_jobs(init=True, exist="o", config_file="test.config_test_reps")
    with Capturing() as output:
        submit_jobs(init=False, wait=True, exist="o", config_file="test.config_test_reps")
    count = Counter(output)
    assert count[success[False]] == combs["n_rep"].sum()


    #check mpi and pool
    combs = submit_jobs(init=True, wait=True, exist="o", config_file="test.config_test_mpi")

    with Capturing() as output:
        submit_jobs(init=False, pool_jobs=True, wait=True, exist="o", config_file="test.config_test_mpi")
    count = Counter(output)
    m = "Submit IDs: ['pytest_kessler_0', 'pytest_lin_0']"
    assert count[m] == 1
    m = "d01 2018-06-20_00:11:00 wrf: SUCCESS COMPLETE WRF"
    assert count[m] == combs["n_rep"].sum()

    #test get_rt and vmem, qsub
    for run in os.listdir(conf.run_path):
        rundir ="{}/{}/".format(conf.run_path, run)
        shutil.copy("tests/test_data/qstat.info", rundir)

    with Capturing() as output:
        combs = submit_jobs(init=False, check_args=True, use_qsub=True, exist="o", config_file="test.config_test_mpi")
    count = Counter(output)
    messages = ['Get runtime from previous runs', 'Get vmem from previous runs', 'Use vmem per slot: 148.3365M']
    for m in messages:
        assert count[m] == combs["n_rep"].sum()
    rundirs = [rd + "_0" for rd in combs["run_dir"].values]
    timing = misc_tools.get_runtime_all(rundirs, all_times=False)["timing"].values
    messages = ["Runtime per time step: {0:.5f} s".format(t) for t in timing]
    for m in messages:
        assert count[m] == 1

    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/pytest"]:
        shutil.rmtree(d)
# os.chdir("..")


#TODO
#Domain size must be multiple of lx
#check name list changes
# test history streams
