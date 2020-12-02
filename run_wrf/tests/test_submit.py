#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:26:51 2019

Test submit_jobs function

@author: c7071088
"""

import os
from run_wrf.submit_jobs import submit_jobs
import pytest
from run_wrf.misc_tools import Capturing
from collections import Counter
import run_wrf.configs.test.config_test as conf
import shutil
import time
import xarray as xr
from run_wrf import misc_tools
from run_wrf import get_namelist
import glob
import pandas as pd

success = {True : 'wrf: SUCCESS COMPLETE IDEAL INIT', False : 'd01 2018-06-20_07:30:00 wrf: SUCCESS COMPLETE WRF'}
outd = os.path.join(conf.outpath, conf.outdir)

test_dir = os.getcwd()
code_dir = "/".join(test_dir.split("/")[:-1])

batch_dict = {"slurm" : "sbatch", "sge" : "qsub"}

#%%tests

def test_basic():
    """
    Test basic submit functionality.

    Initialize and run WRF; Check behaviour when run already exists; Restart run; Check that errors are raised
    """
    with pytest.raises(RuntimeError, match="Parameter dx used in submit_jobs.py already defined in namelist.input! Rename this parameter!"):
        submit_jobs(config_file="test.config_test_del_args", init=True)

    #run wrf
    submit_jobs(init=True, exist="o", config_file="test.config_test")
    combs = submit_jobs(init=False, verbose=True, wait=True, exist="o", config_file="test.config_test")

    #test namelist
    rpath = combs["run_dir"][0] + "_0"
    ID = "_".join(rpath.split("/")[-1].split("_")[1:])
    namelists = []
    namelists.append(get_namelist.namelist_to_dict(rpath + "/namelist.input"))
    namelists.append(get_namelist.namelist_to_dict("./tests/test_data/namelists/namelist.{}".format(ID)))
    for key in set([*namelists[0].keys(), *namelists[1].keys()]):
        if "_outname" not in key:
            equal = namelists[0][key] == namelists[1][key]
            if not equal:
                print(key, namelists[0][key], namelists[1][key])
            assert equal


    input_sounding = misc_tools.read_file(rpath + "/input_sounding")
    input_sounding_corr = misc_tools.read_file(rpath + "/input_sounding_meanwind")
    assert input_sounding == input_sounding_corr

    #check output data
    outfiles = ['fastout_pytest_mp_physics=kessler_0',
                'fastout_pytest_mp_physics=lin_0',
                'wrfout_pytest_mp_physics=kessler_0',
                'wrfout_pytest_mp_physics=lin_0']
    assert sorted(os.listdir(outd)) == sorted(outfiles)
    file = xr.open_dataset(outd + "/fastout_pytest_mp_physics=lin_0")
    t = file["XTIME"]
    t_corr = pd.date_range(start="2018-06-20T07:00:00", end='2018-06-20T07:30:00', freq="5min")
    assert (len(t) == len(t_corr)) and (t == t_corr).all()

    #test behaviour if run already exists
    for run in os.listdir(conf.run_path):
        file = "{}/{}/wrfinput_d01".format(conf.run_path, run)
        if os.path.isfile(file):
            os.remove(file)
    exist_message = (("s", "Redoing initialization..."), ("s", "Skipping..."), ("o", "Overwriting..."), ("b", "Creating backup..."))
    for init in [True, False]:
        for i, (exist, message) in enumerate(exist_message):
            if init or i > 0:
                print("\n\n")
                print(exist, message)
                combs, output = capture_submit(init=init, exist=exist, wait=True, config_file="test.config_test")
                print("\n".join(output))
                count = Counter(output)
                assert count[message] == combs["n_rep"].sum()
                if "Skipping..." not in message:
                    assert count[success[init]] == combs["n_rep"].sum()

    #backup created?
    bak = ['fastout_pytest_mp_physics=kessler_0_bak_0',
           'fastout_pytest_mp_physics=lin_0_bak_0',
           'wrfout_pytest_mp_physics=kessler_0_bak_0',
           'wrfout_pytest_mp_physics=lin_0_bak_0']
    assert sorted(os.listdir(outd + "/bak")) == sorted(bak)

    with pytest.raises(ValueError, match="Value 'a' for -e option not defined!"):
        combs = submit_jobs(init=True, exist="a", config_file="test.config_test")
    _, output = capture_submit(init=False, exist="r", wait=True, config_file="test.config_test_rst")
    count = Counter(output)
    print("\n".join(output))
    for m in ["Restart run from 2018-06-20 07:30:00", 'd01 2018-06-20_08:00:00 wrf: SUCCESS COMPLETE WRF']:
        assert count[m] == combs["n_rep"].sum()

    #check output data
    outfiles = ['rst', 'bak',
                'fastout_pytest_mp_physics=kessler_0',
                'fastout_pytest_mp_physics=lin_0',
                'wrfout_pytest_mp_physics=kessler_0',
                'wrfout_pytest_mp_physics=lin_0']
    assert sorted(os.listdir(outd)) == sorted(outfiles)
    file = xr.open_dataset(outd + "/fastout_pytest_mp_physics=lin_0")
    t = file["XTIME"]
    t_corr = pd.date_range(start="2018-06-20T07:00:00", end='2018-06-20T08:00:00', freq="5min")
    assert (len(t) == len(t_corr)) and (t == t_corr).all()

def test_repeats():
    """Test config repetitions functionality."""
    combs = submit_jobs(init=True, exist="o", config_file="test.config_test_reps")
    _, output = capture_submit(init=False, wait=True, exist="o", config_file="test.config_test_reps")
    print("\n".join(output))
    count = Counter(output)
    assert count[success[False]] == combs["n_rep"].sum()


def test_mpi_and_batch():
    """Test MPI runs and check commands generated for job schedulers (without running them)"""
    combs = submit_jobs(init=True, wait=True, exist="o", config_file="test.config_test_mpi")
    _, output = capture_submit(init=False, pool_jobs=True, wait=True, exist="o", config_file="test.config_test_mpi")
    print("\n".join(output))
    count = Counter(output)
    m = "Submit IDs: ['pytest_mp_physics=kessler_0', 'pytest_mp_physics=lin_0']"
    assert count[m] == 1
    m = "d01 2018-06-20_07:30:00 wrf: SUCCESS COMPLETE WRF"
    assert count[m] == combs["n_rep"].sum()

    rundirs = []
    for rundir in combs["run_dir"]:
        rundir += "_0"
        rundirs.append(rundir)
        runlogs = glob.glob(rundir + "/run*.*") + glob.glob(rundir + "/rsl.error.0000")
        for runlog in runlogs:
            os.remove(runlog)
        shutil.copy("tests/test_data/resources.info", rundir)
        shutil.copy("tests/test_data/runs/WRF_pytest_eta_0/run_2018-04-10T06:13:14.log", rundir)
    #test SGE
    _, output = capture_submit(init=False, check_args=True, verbose=True, use_job_scheduler=True, exist="o", config_file="test.config_test_sge")
    print("\n".join(output))
    count = Counter(output)
    c = output[-1]
    date = c[c.index("lin_0"):c.index(".out")][6:]
    batch_comm = "qsub -cwd -q std.q -o {0}/logs/run_pytest_mp_physics=lin_0_{1}.out -e {0}/logs/run_pytest_mp_physics=lin_0_{1}.err -l h_rt=000:00:15 "\
                 " -pe openmpi-fillup 2 -M matthias.goebel@uibk.ac.at -m ea -N pytest_mp_physics=lin_0 -V  -l h_vmem=85M "\
                 " run_wrf.job".format(conf.run_path, date)
    assert batch_comm == c

    messages = ['Get runtime from previous runs', 'Get vmem from previous runs', 'Use vmem per slot: 85.6M']
    for m in messages:
        assert count[m] == combs["n_rep"].sum()

    rundirs_0 = combs["run_dir"].values + "_0"
    timing = misc_tools.get_runtime_all(rundirs_0, all_times=False)["timing"].values
    message_rt = "Runtime per time step: {0:.5f} s".format(timing[0])
    assert count[message_rt] == 2

    #test SLURM
    _, output = capture_submit(init=False, check_args=True, verbose=True, use_job_scheduler=True, exist="o", config_file="test.config_test_slurm")
    print("\n".join(output))
    count = Counter(output)
    c = output[-1]
    date = c[c.index("lin_0"):c.index(".out")][6:]
    batch_comm = "sbatch -p mem_0064 -o {0}/logs/run_pool_pytest_mp_physics=kessler_0_pytest_mp_physics=lin_0_{1}.out -e {0}/logs/run_pool_pytest_mp_physics=kessler_0_pytest_mp_physics=lin_0_{1}.err --time=000:00:15 "\
                 "--ntasks-per-node=4 -N 1 --mail-user=matthias.goebel@uibk.ac.at --mail-type=END,FAIL -J pool_pytest_mp_physics=kessler_0_pytest_mp_physics=lin_0 "\
                 "--export=ALL  --qos=normal_0064  run_wrf.job".format(conf.run_path, date)
    assert batch_comm == c
    assert count['Get runtime from previous runs'] == combs["n_rep"].sum()
    assert count[message_rt] == 2


def test_scheduler_full():
    """Test runs using a job scheduler if available"""
    #Check if job scheduler is available
    if os.popen("command -v {}".format(batch_dict[conf.job_scheduler])).read() != "":
        combs = submit_jobs(init=True, exist="o", config_file="test.config_test_mpi")
        _, output = capture_submit(init=False, use_job_scheduler=True, test_run=True, exist="o", verbose=True, config_file="test.config_test_mpi")
        print("\n".join(output))
        job_sched = conf.job_scheduler.lower()

        if job_sched  == "slurm":
            jobs = ["pool_pytest_mp_physics=kessler_0_pytest_mp_physics=lin_0"]
            comm = "squeue -n "
        elif job_sched  == "sge":
            jobs = ["pytest_mp_physics=kessler_0", "pytest_mp_physics=lin_0"]
            comm = "qstat -j "
        else:
            raise ValueError("Job scheduler {} not known!".format(job_sched))

        finished = False
        while not finished:
            finished = True
            time.sleep(5)
            for j in jobs:
                status = os.popen(comm + j).read().split("\n")
                status = misc_tools.remove_empty_str(status)

                if "Following jobs do not exist:" in status:
                    pass
                elif len(status) > 1:
                    finished = False

        rundirs = combs["run_dir"].values + "_0"
        print(rundirs)
        for rundir in rundirs:
            runlog = glob.glob(rundir + "/run_*.log")[0]
            runlog = misc_tools.read_file(runlog)
            m = "d01 2018-06-20_07:30:00 wrf: SUCCESS COMPLETE WRF"
            assert m in runlog


#%%helper functions

@pytest.fixture(autouse=True)
def run_around_tests():
    """Delete test data before and after every test and transfer test namelist and iofiles before the tests"""

    # Code that will run before each test

    os.chdir(code_dir)
    #remove test data
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    #copy namelist and io file for tests
    for add in ["_mpi", ""]:
        target_dir = "{}/{}{}/test/{}/".format(conf.build_path, conf.wrf_dir_pre, add, conf.ideal_case)
        shutil.copy("{}/test_data/IO_test.txt".format(test_dir), target_dir)
        shutil.copy("{}/test_data/namelists/namelist.input".format(test_dir), target_dir)

    #check skipping non-initialized runs
    _, output = capture_submit(init=False, config_file="test.config_test")
    assert Counter(output)["Run not initialized yet! Skipping..."] == 2


    # A test function will be run at this point
    yield

    # Code that will run after each test
    for d in [os.environ["wrf_res"] + "/test/pytest", os.environ["wrf_runs"] + "/pytest"]:
        if os.path.isdir(d):
            shutil.rmtree(d)

def capture_submit(*args, **kwargs):
    try:
        with Capturing() as output:
            combs = submit_jobs(*args, **kwargs)
    except Exception as e:
        print(output)
        raise(e)

    return combs, output

