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
import glob

path = os.environ["wrf_res"] + "/test/fluxdelay/"

#%%
path = path + "/"
old_files_0 = glob.glob("{}/rst/*_rst_0".format(path))
for file in old_files_0:
    ID = file.split("/")[-1]
    ID = ID[:ID.index("rst")-1]
    print("Processing file {}".format(ID))
    new_file = path + ID
    if not os.path.isfile(new_file):
        print("Original and restarted runs could not be concatenated. Restarted run not found!")
        continue
    old_files = os.popen("ls -t {}/rst/{}_rst_*".format(path, ID)).read().split("\n")[::-1]
    if "" in old_files:
        old_files.remove("")
    time = None
    all_files =  [*old_files, new_file]
    old_files_cut = []
    for cfile in all_files:
        ds = Dataset(cfile, "r+")
        time_next = wrf.extract_times(ds, timeidx=None)
        if time is not None:
            start_idx = np.where(time[-1]==time_next)[0][0] + 1 
            ds.close()
            if start_idx < len(time_next):
                err = os.system("ncks -O -d Time,{} {} {}".format(start_idx, cfile, cfile + "_cut"))
                if err != 0:
                    raise Exception("Error in ncks when reducing {}".format(cfile))
                old_files_cut.append(cfile + "_cut")
            else:
                print("File is redundant!")
        time = time_next
    
    all_files_cut = [*old_files_cut, new_file]
    if len(all_files_cut) > 1:
        err = os.system("ncrcat {} {} {}".format(*all_files_cut, new_file + "_concat"))
        if err != 0:
            raise Exception("Error in ncrcat when concatenating output of original and restarted runs" )
        file_final = Dataset(new_file + "_concat")
        time_final = wrf.extract_times(file_final, timeidx=None)
        file_final.close()
    
        res = np.median(time_final[1:] - time_final[:-1])
        time_corr = np.arange(time_final[0], time_final[-1] + res, res)
        if (len(time_corr) != len(time_final)) or (time_corr != time_final).any():
            raise RuntimeError("Error in concatenated time dimension for final file {}".format(file))
        for f in all_files_cut:
            os.remove(f)
        os.rename(path + file + "_concat", path + file )
    else:
        print("\nNo files to concatenate")
