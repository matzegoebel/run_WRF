#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:17:50 2019

@author: c7071088
"""


def namelist_to_dict(path, verbose=False):
    """Convert namelist file to dictionary."""
    with open(path) as f:
        namelist_str = f.read().replace(" ", "").replace("\t", "").split("\n")
    namelist_dict = {}
    for line in namelist_str:
        if line != "":
            if verbose:
                print("\n" + line)
            param_val = get_namelist_param_val(line, verbose=verbose)
            if param_val is not None:
                namelist_dict[param_val[0]] = param_val[1]
    return namelist_dict

def get_namelist_param_val(line, verbose=False):
    """Get parameter name and value from line in namelist file."""
    line = line.replace(" ", "")
    line = line.replace("\t", "")
    if "=" not in line:
        if verbose:
            print("Line contains no parameters")
        return
    elif line[0] == "!":
        if verbose:
            print("Line is commented out")
        return
    else:
        if "!" in line:
            line = line[:line.index("!")]

        param, val = line.split("=")
        val = mod_namelist_val(val)
        if verbose:
            print(param, val)

    return param, val

def mod_namelist_val(val):
    """Remove unnecessary dots and commas from namelist value and use only one type of quotation marks."""
    val = val.replace('"', "'")
    if val[-1] == ",":
        val = val[:-1]
    if val[-1] == ".":
        val = val[:-1]
    return val

