#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:52:16 2019

@author: c7071088
"""

import sys

job = str(sys.argv[1])
qstat = str(sys.argv[2])

qstat = qstat.split("\n")
#print(qstat)
#f=open("qstat")
#job = "101779"
#qstat = f.readlines()

hosts = []
read_in = False
for line in qstat:
    l = [l for l in line.split(" ") if l !=""]
    if "c7071088" in l:
        ljob = l[0]
        if ljob == job:
            read_in = True
        else:
            read_in = False
   # print(l)
    if read_in and (l[-1] == "SLAVE"):
        host = [h for h in l if "intern" in h][0]
        start = host.index("@")
        end = host.index(".intern")
        hosts.append(host[start+1:end])  
 
print(" ".join(hosts))