#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 00:10:19 2019

Drop duplicates from list of hosts.

@author: c7071088
"""

import sys

hosts = sys.argv[1]

hosts = hosts.split(" ")
hosts = list(set(hosts))
hosts = ",".join(hosts)

print(hosts)
