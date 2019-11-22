#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:20:58 2019

@author: matze
"""

import os

os.system("python submit_jobs.py -t -i -c config_test")
os.system("python submit_jobs.py -t -c config_test")
os.system("python submit_jobs.py -i -c config_test")
