#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:42:10 2021

Functions related to batch processing with qsub

@author: Matthias Göbel
"""
import sys
from collections import Counter
import os


def rankfile():
    """Produce rankfile to pin jobs to certain processors."""
    hosts = sys.argv[1]
    start_slot = int(sys.argv[2])
    nslots = int(sys.argv[3])

    hosts = hosts.split(" ")
    hosts = Counter(hosts)

    rankfile = ""
    it = 0
    i = 0
    for n, slots in hosts.items():
        for s in range(slots):
            if (it >= start_slot) and (it < start_slot + nslots):
                rankfile += "rank {}={} slot={}\n".format(i, n, s)
                i += 1
            it += 1
    print(rankfile)


def get_hosts():
    """Get list of hosts available for certain job."""
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
            hosts.append(host[start + 1:end])

    print(" ".join(hosts))


def get_hosts_set():
    """Drop duplicates from list of hosts."""
    hosts = sys.argv[1]

    hosts = hosts.split(" ")
    hosts = list(set(hosts))
    hosts = ",".join(hosts)

    print(hosts)
