#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:04:02 2019

Concatenate output from restarted run and original run

@author: c7071088
"""

import os
from netCDF4 import Dataset
import wrf
import numpy as np

path = os.environ["wrf_res"] + "/test/"

#%%
path = path + "/"
for file in os.listdir(path + "bk"):
    if not os.path.isfile(path + file):
        print("Original and restarted runs could not be concatenated. Restarted run not found!")
        continue
    print("Processing file {}".format(file))
    file1_p = path + "bk/" + file
    file2_p = path + file

    file1 = Dataset(file1_p, "r+")
    file2 = Dataset(file2_p, "r+")

    time1 = wrf.extract_times(file1, timeidx=None)
    time2 = wrf.extract_times(file2, timeidx=None)

    stop_idx = np.where(time1==time2[0])[0][0] - 1

    time_all = np.concatenate((time1[:stop_idx+1], time2))
    res = time_all[1] - time_all[0]
    time_corr = np.arange(time1[0], time2[-1] + res, res)
    if (len(time_corr) != len(time_all)) or (time_corr != time_all).any():
        raise RuntimeError("Error in concatenated time dimension for file {}".format(file))
    file1.close()
    file2.close()
    err = os.system("ncks -d Time,0,{} {} {}".format(stop_idx, file1_p, file1_p + "_cut"))
    if err != 0:
        raise Exception("Error in ncks when reducing original run")

    err = os.system("ncrcat {} {} {}".format(file1_p + "_cut", file2_p, path + file + "_concat"))
    if err != 0:
        raise Exception("Error in ncrcat when concatenating output of original and restarted runs" )
    file_final = Dataset(path + file + "_concat")
    time_final = wrf.extract_times(file_final, timeidx=None)
    file_final.close()

    if (len(time_final) != len(time_all)) or (time_final != time_all).any():
        raise RuntimeError("Error in concatenated time dimension for final file {}".format(file))
    for f in [file1_p, file1_p + "_cut", file2_p]:
        os.remove(f)
    os.rename(path + file + "_concat", path + file )
