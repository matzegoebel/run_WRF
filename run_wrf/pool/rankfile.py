#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:25:41 2019

Produce rankfile to pin jobs to certain processors.

@author: Matthias Göbel
"""

import sys
from collections import Counter


hosts = sys.argv[1]
si = int(sys.argv[2])
ns = int(sys.argv[3])

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