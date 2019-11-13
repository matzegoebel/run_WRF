#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:25:41 2019

@author: c7071088
"""

import sys
from collections import Counter

#           rank 0=aa slot=1
#           rank 1=bb slot=8
#           rank 2=cc slot=6

#print(sys.argv)
hosts = sys.argv[1]
si = int(sys.argv[2])
ns = int(sys.argv[3])

#hosts = "n015 n015 n015 n015 n015 n015 n015 n045 n045 n045 n045 n019 n019 n032 n016 n016"
#si = 6
#ns = 4

hosts = hosts.split(" ")
hosts = Counter(hosts)

rankfile=""
it = 0
i = 0
for n,slots in hosts.items():
    for s in range(slots):
        if (it >= si) and (it < si + ns):
            rankfile += "rank {}={} slot={}\n".format(i,n,s)
            i += 1
        it += 1
print(rankfile)