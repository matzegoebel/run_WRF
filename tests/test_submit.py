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


def test_submit_jobs():
    os.chdir("..")

    submit_jobs(check_args=True, init=True, config_file="test.config_test")
    combs = submit_jobs(check_args=True, config_file="test.config_test")
    with pytest.raises(RuntimeError):
        submit_jobs(config_file="test.config_test_del_args", init=True)

    nruns = len(combs)
    submit_jobs(init=True, config_file="test.config_test")

    #test behaviour if output exists
    for run in os.listdir(conf.run_path):
        file = "{}/{}/wrfinput_d01".format(conf.run_path, run)
        if os.path.isfile(file):
            os.remove(file)
    exist_message = (("s", "Redoing initialization..."), ("s", "Skipping..."), ("o", "Overwriting..."), ("b", "Creating backup..."))
    success = 'wrf: SUCCESS COMPLETE IDEAL INIT'
    for init in [True, False]:
        for i, (exist, message) in enumerate(exist_message):
            print(exist, message)
            with Capturing() as output:
                submit_jobs(init=init, exist=exist, wait=True, config_file="test.config_test")
            count = Counter(output)
            if init or i > 0:
                assert count[message] == nruns
            if "Skipping" not in message:
                assert count[success[init]] == nruns

    with pytest.raises(ValueError, match="Value 'a' for -e option not defined!"):
        submit_jobs(init=True, exist="a", config_file="test.config_test")


    time.sleep(5)


    shutil.rmtree(conf.run_path)
    shutil.rmtree(conf.outpath)



#test run and restart run, check name list changes
#test check_rt
# check if bak is created

#check if wrf successful in log or as parameter form submit_jobs?