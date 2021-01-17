#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to read in and modify WRF namelists

@author: Matthias Göbel
"""


def namelist_to_dict(path, verbose=False, first_domain_only=True,
                     registries=None, build_path=None):
    """Convert namelist file to dictionary."""
    with open(path) as f:
        namelist_str = f.read().replace(" ", "").replace("\t", "").split("\n")
    namelist_dict = {}
    for line in namelist_str:
        if line != "":
            if verbose:
                print("\n" + line)
            param_val = get_namelist_param_val(line, verbose=verbose,
                                               first_domain_only=first_domain_only)
            if param_val is not None:
                namelist_dict[param_val[0]] = param_val[1]
    if (registries is not None) and (build_path is not None):
        for reg in registries:
            with open(build_path + "/Registry/" + reg) as f:
                reglines = f.readlines()
            for line in reglines:
                line = line.replace("\t", " ")
                if line[:7] == "rconfig":
                    rconfig = [i for i in line.split(" ") if i != ""]
                    if rconfig[2] not in namelist_dict:
                        namelist_dict[rconfig[2]] = mod_namelist_val(rconfig[5])
                    if ("namelist," not in rconfig[3]) and (rconfig[3] != "derived"):
                        raise Exception("Strange format of registry file {}!".format(reg))
    return namelist_dict


def get_namelist_param_val(line, verbose=False, first_domain_only=True):
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
        if "\n" in line:
            raise ValueError("Duplicate entry in namelist file:\n {}".format(line))
        try:
            param, val = line[:line.index("=")], line[line.index("=") + 1:]
        except ValueError as e:
            e.args = ("Error in line: {}\n".format(line) + e.args[0], )
            raise(e)
        if (param != "eta_levels") and first_domain_only:
            # use only first domain value
            val = val.split(",")[0]

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

    try:
        val = float(val)
        if val == int(val):
            val = int(val)
    except ValueError:
        pass

    return val
