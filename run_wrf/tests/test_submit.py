#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:26:51 2019

Test launch_jobs function

@author: Matthias Göbel
"""

import os
from run_wrf.launch_jobs import launch_jobs
import pytest
from run_wrf.tools import Capturing
from collections import Counter
import run_wrf.configs.test.config_test as conf
import shutil
import time
import xarray as xr
from run_wrf import tools, get_namelist, vertical_grid
import glob
import pandas as pd
from pathlib import Path

end_time = "2018-06-20_07:06:00"
success = {True: 'wrf: SUCCESS COMPLETE IDEAL INIT',
           False: 'd01 {} wrf: SUCCESS COMPLETE WRF'.format(end_time)}
params = conf.params
outd = params["outpath"]
rund = params["run_path"]

test_dir = os.getcwd()
code_dir = "/".join(test_dir.split("/")[:-1])

batch_dict = {"slurm": "sbatch", "sge": "qsub"}

# %%tests


def test_basic():
    """
    Test basic submit functionality.

    Initialize and run WRF; Check behaviour when run already exists
    Restart run; Check that errors are raised
    """
    # run wrf
    launch_jobs(init=True, exist="o", config_file="test.config_test")
    combs = launch_jobs(init=False, verbose=True, wait=True, exist="o",
                        config_file="test.config_test")

    # test namelist
    rpath = combs["run_dir"][0] + "_0"
    ID = "_".join(rpath.split("/")[-1].split("_")[1:])
    namelists = []
    namelists.append(get_namelist.namelist_to_dict(rpath + "/namelist.input"))
    namelists.append(get_namelist.namelist_to_dict("./tests/test_data/namelists/namelist.{}".format(ID)))
    for key in set([*namelists[0].keys(), *namelists[1].keys()]):
        if "_outname" not in key:
            equal = namelists[0][key] == namelists[1][key]
            if not equal:
                print("unequal namelist settings:", key, namelists[0][key], namelists[1][key])
            assert equal

    input_sounding = tools.read_file(rpath + "/input_sounding")
    build = os.path.join(combs["build_path"][0], combs["parallel_build"][0], "test", "em_les")
    input_sounding_corr = tools.read_file(build + "/input_sounding_meanwind")
    assert input_sounding == input_sounding_corr

    # check output data
    for run in os.listdir(outd):
        outfiles = sorted(os.listdir(os.path.join(outd, run)))
        outfiles_corr = ['fastout_d01_2018-06-20_07:00:00', 'wrfout_d01_2018-06-20_07:00:00']
        assert outfiles_corr == outfiles
        for f, freq in zip(outfiles, ["1", "2"]):
            ds = xr.open_dataset(os.path.join(outd, run, f), decode_times=False, engine="scipy")
            t = tools.extract_times(ds)
            t_corr = pd.date_range(start="2018-06-20T07:00:00", end=end_time.replace("_", "T"),
                                   freq=freq + "min")
            assert (len(t) == len(t_corr)) and (t == t_corr).all()

    # test behaviour if run already exists
    for run in os.listdir(rund):
        file = "{}/{}/wrfinput_d01".format(rund, run)
        if os.path.isfile(file):
            os.remove(file)
    exist_message = (("s", "Redoing initialization..."), ("s", "Skipping..."),
                     ("o", "Overwriting..."), ("b", "Creating backup..."))
    for init in [True, False]:
        for i, (exist, message) in enumerate(exist_message):
            if init or i > 0:
                print("\n\n")
                print(exist, message)
                combs, output = capture_submit(init=init, exist=exist, wait=True,
                                               config_file="test.config_test")
                print("\n".join(output))
                count = Counter(output)
                assert count[message] == combs["n_rep"].sum()
                if "Skipping..." not in message:
                    assert count[success[init]] == combs["n_rep"].sum()

    # backup created?
    bak = ['fastout_d01_2018-06-20_07:00:00_bak_0',
           'wrfout_d01_2018-06-20_07:00:00_bak_0']
    for run in os.listdir(outd):
        outfiles = sorted(os.listdir(os.path.join(outd, run, "bak")))
        assert outfiles == bak

    with pytest.raises(ValueError, match="Value 'a' for -e option not defined!"):
        combs = launch_jobs(init=True, exist="a", config_file="test.config_test")
    _, output = capture_submit(init=False, exist="r", wait=True,
                               config_file="test.config_test_rst")
    count = Counter(output)
    print("\n".join(output))
    for m in ["Restart run from 2018-06-20 07:04:00",
              'd01 2018-06-20_07:08:00 wrf: SUCCESS COMPLETE WRF']:
        assert count[m] == combs["n_rep"].sum()

    outfiles_corr = ['bak',
                     'fastout_d01_2018-06-20_07:00:00',
                     'fastout_d01_2018-06-20_07:05:00',
                     'wrfout_d01_2018-06-20_07:00:00',
                     'wrfout_d01_2018-06-20_07:06:00']
    # check output
    for run in os.listdir(outd):
        outfiles = sorted(os.listdir(os.path.join(outd, run)))
        assert outfiles == outfiles_corr

    # concat output and check
    tools.concat_output("test.config_test")
    for run in os.listdir(outd):
        outfiles = sorted(os.listdir(os.path.join(outd, run)))
        outfiles_corr = ['fastout_d01_2018-06-20_07:00:00', 'wrfout_d01_2018-06-20_07:00:00']
        assert outfiles_corr == outfiles[1:]
        for f, freq in zip(outfiles_corr, ["1", "2"]):
            ds = xr.open_dataset(os.path.join(outd, run, f), decode_times=False, engine="scipy")
            t = tools.extract_times(ds)
            t_corr = pd.date_range(start="2018-06-20T07:00:00", end='2018-06-20T07:08:00',
                                   freq=freq + "min")
            assert (len(t) == len(t_corr)) and (t == t_corr).all()


def test_repeats():
    """Test config repetitions functionality."""
    combs = launch_jobs(init=True, exist="o", config_file="test.config_test_reps")
    _, output = capture_submit(init=False, wait=True, exist="o",
                               config_file="test.config_test_reps")
    print("\n".join(output))
    count = Counter(output)
    assert count[success[False]] == combs["n_rep"].sum()


def test_mpi_and_batch():
    """Test MPI runs and check commands generated for job schedulers (without running them)"""
    combs = launch_jobs(init=True, wait=True, exist="o", config_file="test.config_test_mpi")
    _, output = capture_submit(init=False, pool_jobs=True, wait=True, exist="o",
                               config_file="test.config_test_mpi")
    print("\n".join(output))
    count = Counter(output)
    m = "Submit IDs: ['pytest_mp_physics=eta_0', 'pytest_mp_physics=lin_0']"
    assert count[m] == 1
    m = success[False]
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
    # test SGE
    _, output = capture_submit(init=False, check_args=True, verbose=True, use_job_scheduler=True,
                               exist="o", config_file="test.config_test_sge")
    print("\n".join(output))
    count = Counter(output)
    c = output[-1]
    batch_comm = "qsub -cwd -q std.q -o {0}/logs/ -e {0}/logs/ -l h_rt=000:01:31 "\
                 " -pe openmpi-fillup 2  -M test@test.com -m ea -N run_pytest -V  "\
                 "-l h_vmem=85M  run_wrf.job".format(rund)
    assert batch_comm == c

    messages = ['Get runtime from previous runs', 'Get vmem from previous runs',
                'Use vmem per slot: 85.6M']
    for m in messages:
        assert count[m] == combs["n_rep"].sum()

    rundirs_0 = combs["run_dir"].values + "_0"
    timing = tools.get_runtime_all(rundirs_0, all_times=False)["timing"].values
    message_rt = "Runtime per time step: {0:.5f} s".format(timing[0])
    assert count[message_rt] == 2

    # test SLURM
    _, output = capture_submit(init=False, check_args=True, verbose=True, use_job_scheduler=True,
                               exist="o", config_file="test.config_test_slurm")
    print("\n".join(output))
    count = Counter(output)
    c = output[-1]
    batch_comm = "sbatch -p mem_0064 -o {0}/logs/run_pytest.o_%j -e {0}/logs/run_pytest.e_%j "\
                 "--time=000:01:31 --ntasks-per-node=4 -N 1 "\
                 "--mail-user=test@test.com --mail-type=END,FAIL -J run_pytest "\
                 "--export=ALL  --qos=normal_0064  run_wrf.job".format(rund)
    assert batch_comm == c
    assert count['Get runtime from previous runs'] == combs["n_rep"].sum()
    assert count[message_rt] == 2
    # TODO: also check environment variables


def test_scheduler_full():
    """Test runs using a job scheduler if available"""
    # Check if job scheduler is available
    if ("job_scheduler" in dir(conf)) and \
            os.popen("command -v {}".format(batch_dict[conf.job_scheduler])).read() != "":
        combs = launch_jobs(init=True, exist="o", config_file="test.config_test_mpi")
        _, output = capture_submit(init=False, use_job_scheduler=True, test_run=True, exist="o",
                                   verbose=True, config_file="test.config_test_mpi")
        print("\n".join(output))
        job_sched = conf.job_scheduler.lower()
        if job_sched == "slurm":
            comm = "squeue -n "
        elif job_sched == "sge":
            comm = "qstat -j "
        else:
            raise ValueError("Job scheduler {} not known!".format(job_sched))

        finished = False
        first_loop = True
        while not finished:
            finished = True
            status = os.popen(comm + "run_pytest").read().split("\n")
            status = tools.remove_empty_str(status)
            if "Following jobs do not exist:" in status:
                pass
            if len(status) > 1:
                finished = False
            elif first_loop:
                raise RuntimeError("Batch job was not submitted!")
            time.sleep(5)
            first_loop = False
        rundirs = combs["run_dir"].values + "_0"
        print(rundirs)
        for rundir in rundirs:
            runlog = glob.glob(rundir + "/run_*.log")[0]
            runlog = tools.read_file(runlog)
            m = success[False]
            assert m in runlog


def test_vgrid():
    """Test vertical grid creation.

       Actual vertical grid spacings should be close to desired ones (within +- 1 m).
    """
    combs = launch_jobs(init=True, exist="o", config_file="test.config_test_vgrid")
    rpath = combs["run_dir"][0] + "_0"

    wrfinput = xr.open_dataset(rpath + "/wrfinput_d01", engine="scipy").isel(Time=0)
    z = (wrfinput["PHB"] + wrfinput["PH"]) / 9.81
    z = z.mean(["west_east", "south_north"])
    dz = z.diff("bottom_top_stag")
    c = combs.iloc[0]
    grid = vertical_grid.create_levels(ztop=c["ztop"], dz0=c["dz0"], method=c["vgrid_method"],
                                       dzmax=c["dzmax"], nz=c["nz"])
    err = dz - grid.dz[:-1].values
    assert all(abs(err) < 1)

# %%helper functions

@pytest.fixture(autouse=True)
def run_around_tests():
    """Delete test data before and after every test and transfer test namelist and iofiles before the tests"""

    # Code that will run before each test
    os.chdir(code_dir)
    # remove test data
    for d in [outd, rund]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    # copy namelist and io file for tests
    for build in [params["parallel_build"], params["serial_build"]]:
        target_dir = "{}/{}/test/{}/".format(params["build_path"], build, params["ideal_case_name"])
        shutil.copy("{}/test_data/IO_test.txt".format(test_dir), target_dir)
        shutil.copy("{}/test_data/input_sounding_cops".format(test_dir), target_dir)
        shutil.copy("{}/test_data/namelists/namelist.input".format(test_dir), target_dir)

    # check skipping non-initialized runs
    _, output = capture_submit(init=False, config_file="test.config_test")
    assert Counter(output)["Run not initialized yet! Skipping..."] == 2

    # A test function will be run at this point
    yield

    # Code that will run after each test
    for d in [outd, rund]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.chdir(Path(__file__).parent)


def capture_submit(*args, **kwargs):
    try:
        with Capturing() as output:
            combs = launch_jobs(*args, **kwargs)
    except Exception as e:
        print(output)
        raise(e)

    return combs, output
