#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:14:23 2019

@author: c7071088
"""
import misc_tools
import pytest
import numpy as np

def test_get_runtime():
    timing, counter = misc_tools.get_runtime("../", all_times=True)
    assert counter == 100799
    assert timing.notna().all().all()
    timing_m, _ = misc_tools.get_runtime("../", all_times=False)
    np.testing.assert_allclose( timing_m["timing"].values[0], timing["timing"].mean())
    assert (timing_m.values[0,:-1] == np.array([4,1,65,17])).all()
    with pytest.raises(FileNotFoundError):
        timing, counter = misc_tools.get_runtime("./", all_times=True)

    dirs = "./"
    timing_m = misc_tools.get_runtime_all("", dirs, subdir="", all_times=False, remove=None, verbose=True)
    timing = misc_tools.get_runtime_all("", dirs, subdir="", all_times=True, remove=None, length=1e6, verbose=True)
    assert len(timing) == 172800
    assert timing.notna().all().all()

#misc_tools.get_runtime_all("", dirs, subdir="", all_times=False, remove=None, verbose=True)
#timing = misc_tools.get_runtime_all("", dirs, subdir="", all_times=True, remove=None, length=1e6, verbose=True)


