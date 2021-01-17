#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:52:16 2019

Get list of hosts available for certain job

@author: Matthias Göbel
"""

import sys
import os
# TODO: doesn't need to be separate file

job = str(sys.argv[1])
qstat = str(sys.argv[2])

qstat = qstat.split("\n")

hosts = []
read_in = False
for line in qstat:
    i = [i for i in line.split(" ") if i != ""]
    if os.environ["USER"] in i:
        ljob = i[0]
        if ljob == job:
            read_in = True
        else:
            read_in = False
    if read_in and (i[-1] == "SLAVE"):
        host = [h for h in i if "intern" in h][0]
        start = host.index("@")
        end = host.index(".intern")
        hosts.append(host[start+1:end])

print(" ".join(hosts))
