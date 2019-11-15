#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:52:16 2019

Get list of hosts available for certain job

@author: c7071088
"""

import sys
import os

job = str(sys.argv[1])
qstat = str(sys.argv[2])

qstat = qstat.split("\n")

hosts = []
read_in = False
for line in qstat:
    l = [l for l in line.split(" ") if l !=""]
    if os.environ["USER"] in l:
        ljob = l[0]
        if ljob == job:
            read_in = True
        else:
            read_in = False
    if read_in and (l[-1] == "SLAVE"):
        host = [h for h in l if "intern" in h][0]
        start = host.index("@")
        end = host.index(".intern")
        hosts.append(host[start+1:end])

print(" ".join(hosts))