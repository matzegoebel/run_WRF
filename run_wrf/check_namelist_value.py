#!/usr/bin/env python3
#TODO rather put somewhere else and make an entry point

import sys
from run_wrf.get_namelist import get_namelist_param_val, mod_namelist_val
line = sys.argv[1]
new_val = sys.argv[2]

param_val = get_namelist_param_val(line)
if param_val is not None:
    old_val = param_val[1]
    new_val = mod_namelist_val(new_val)
    if old_val == new_val:
        print("Old and new value are equal")
