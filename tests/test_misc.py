#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:14:23 2019

@author: c7071088
"""
import misc_tools
import pytest
import numpy as np
import subprocess as sp
import os
from io import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def test_get_runtime():
    timing, counter = misc_tools.get_runtime("WRF_fluxdelay_eta_0", all_times=True)
    assert counter == 86400
    assert timing.notna().all().all()
    timing_m, _ = misc_tools.get_runtime("WRF_fluxdelay_eta_0", all_times=False, counter=None, timing=None)
    np.testing.assert_allclose( timing_m["timing"].values.mean(), timing["timing"].mean())
    assert (timing_m.values[0,:-2] == np.array([1,1,6,6])).all()
    with pytest.raises(FileNotFoundError):
        timing, counter = misc_tools.get_runtime("./", all_times=True)

    dirs = "./"
    timing_m = misc_tools.get_runtime_all(id_filter="", dirs=dirs, all_times=False, levels=None, remove=None, verbose=True)
    timing = misc_tools.get_runtime_all(id_filter="", dirs=dirs, all_times=True, remove=None, verbose=True)
    assert len(timing) == 313200
    assert timing.iloc[:,-7:].notna().all().all()

    with Capturing() as output:
        rt, sd = misc_tools.get_runtime_id("./WRF_fluxdelay_eta_0", "./")
    assert len(output) == 3
    assert output[1].split("/")[-1] == 'WRF_fluxdelay_2_eta_0 has same namelist parameters'
    assert output[2].split("/")[-1] == 'WRF_fluxdelay_eta_0 has same namelist parameters'
    assert round(rt,5) == 0.01268


def test_namelist_to_dict():
    path = './namelist.test'
    namelist_dict = misc_tools.namelist_to_dict(path, verbose=False)
    correct = {'run_days': '0',
     'iofields_filename': "'LES_IO.txt'",
     'input_2d': '.false',
     'auxhist7_outname': "'test'",
     'shadlen': '25000'}
    for k, v in correct.items():
        assert v == namelist_dict[k]

