#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 00:10:19 2019

@author: c7071088
"""

import sys

hosts = sys.argv[1]

#hosts = "n015 n015 n015 n015 n015 n015 n015 n045 n045 n045 n045 n019 n019 n032 n016 n016"
hosts = hosts.split(" ")
hosts = list(set(hosts))
hosts = ",".join(hosts)

print(hosts)
