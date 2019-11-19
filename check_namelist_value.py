#!/usr/bin/env python3
import sys


line = sys.argv[1]
new_val = sys.argv[2]


line = line.replace(" ", "")
line = line.replace("\t", "")

if line[0] == "!":
    print("Line is commented out")
else:
    if "!" in line:
        line = line[:line.index("!")]

    param, old_val = line.split("=")

    old_val = old_val.replace('"', "'")
    new_val = new_val.replace('"', "'")

    #remove dot and comma if unnecessary
    if old_val[-1] == ",":
        old_val = old_val[:-1]
    if old_val[-1] == ".":
        old_val = old_val[:-1]
    if new_val[-1] == ",":
        new_val = new_val[:-1]
    if new_val[-1] == ".":
        new_val = new_val[:-1]


    if old_val == new_val:
        print("Old and new value are equal")
