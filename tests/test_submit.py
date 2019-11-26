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

def test_submit_jobs():
    os.chdir("..")
    submit_jobs(check_args=True, init=True, config_file="test.config_test")
    submit_jobs(check_args=True, config_file="test.config_test")
    submit_jobs(init=True, config_file="test.config_test")
    with pytest.raises(RuntimeError):
        submit_jobs(config_file="test.config_test_del_args", init=True)

