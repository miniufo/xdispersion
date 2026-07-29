# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import pytest
from xdispersion.utils import get_overlap_indices

def test_get_overlap_indices():
    dset = xr.open_dataset('./data/glad32.nc')
    
    cases = [[slice(10, 20), slice( 5, 10)],
             [slice(10, 20), slice(20, 25)],
             [slice(10, 20), slice(12, 18)],
             [slice(10, 20), slice( 8, 20)],
             [slice(10, 20), slice(11, 25)]]

    for case in cases:
        times1 = dset.time.values[case[0]]
        times2 = dset.time.values[case[1]]
        
        i1, i2, i3, i4 = get_overlap_indices(times1, times2)
        if i1 != None:
            assert len(times1[i1:i2])==len(times2[i3:i4])
            assert (times1[i1:i2]==times2[i3:i4]).all()
        else:
            assert (i1 == i2 and i1 == i3 and i1 == i4)
        
        times1 = dset.time.values[case[1]]
        times2 = dset.time.values[case[0]]
        
        i1, i2, i3, i4 = get_overlap_indices(times1, times2)
        if i1 != None:
            assert len(times1[i1:i2])==len(times2[i3:i4])
            assert (times1[i1:i2]==times2[i3:i4]).all()
        else:
            assert (i1 == i2 and i1 == i3 and i1 == i4)

