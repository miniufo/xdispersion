# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import pytest
from xdispersion.analytics import ana_r2, ana_Ku, ana_K2, ana_CIST


@pytest.mark.parametrize("kappa, r0", [
    (0.1, 1e-1), ( 0.2, 1e-2), ( 1, 1e-3),
    (2.0, 1e-4), (10.0, 1e-5), (20, 1e-6),
])
def test_relative_dispersion(kappa, r0):
    rtime = xr.DataArray(np.logspace(-3, 2, 101), dims='rtime', coords={'rtime':np.logspace(-3, 2, 101)})
    
    r2_rich = ana_r2(rtime, params={'beta':kappa, 'r0':r0}, regime='richardson')
    r2_rich2 = ana_r2(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-5/3}, regime='generallocal')
    
    np.testing.assert_allclose(r2_rich, r2_rich2)
    
    r2_GM = ana_r2(rtime, params={'lambda':kappa, 'r0':r0}, regime='-2')
    r2_GM2 = ana_r2(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-2}, regime='generallocal')
    
    np.testing.assert_allclose(r2_GM, r2_GM2)
    


@pytest.mark.parametrize("kappa, r0", [
    (0.1, 1e-1), ( 0.2, 1e-2), ( 1, 1e-3),
    (2.0, 1e-4), (10.0, 1e-5), (20, 1e-6),
])
def test_kurtosis(kappa, r0):
    rtime = xr.DataArray(np.logspace(-3, 2, 101), dims='rtime', coords={'rtime':np.logspace(-3, 2, 101)})
    
    ku1 = ana_Ku(rtime, params={'beta':kappa, 'r0':r0}, regime='richardson')
    ku2 = ana_Ku(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-5/3}, regime='generallocal')
    
    np.testing.assert_allclose(ku1, ku2)
    
    ku1 = ana_Ku(rtime, params={'lambda':kappa, 'r0':r0}, regime='-2')
    ku2 = ana_Ku(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-2}, regime='generallocal')
    
    np.testing.assert_allclose(ku1, ku2)
    
    ku1 = ana_Ku(rtime, params={'beta':kappa, 'r0':r0}, regime='richardson-a')
    ku2 = ana_Ku(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-5/3}, regime='generallocal-a')
    
    np.testing.assert_allclose(ku1, ku2)
    
    ku1 = ana_Ku(rtime, params={'lambda':kappa, 'r0':r0}, regime='-2-a')
    ku2 = ana_Ku(rtime, params={'kappa':kappa, 'r0':r0, 'slope':-2}, regime='generallocal-a')
    
    np.testing.assert_allclose(ku1, ku2)


@pytest.mark.parametrize("kappa, r0", [
    (0.1, 1e-1), ( 0.2, 1e-2), ( 1, 1e-3),
    (2.0, 1e-4), (10.0, 1e-5), (20, 1e-6),
])
def test_relative_diffusivity(kappa, r0):
    r = xr.DataArray(np.logspace(-3, 2, 101), dims='r', coords={'r':np.logspace(-3, 2, 101)})

    K2_rich = ana_K2(r, params={'beta':kappa, 'r0':r0}, regime='richardson')
    K2_rich2 = ana_K2(r, params={'kappa':kappa, 'r0':r0, 'slope':-5/3}, regime='generallocal')
    np.testing.assert_allclose(K2_rich, K2_rich2)

    K2_GM = ana_K2(r, params={'lambda':kappa, 'r0':r0}, regime='-2')
    K2_GM2 = ana_K2(r, params={'kappa':kappa, 'r0':r0, 'slope':-2}, regime='generallocal')
    np.testing.assert_allclose(K2_GM, K2_GM2)


@pytest.mark.parametrize("kappa, r0", [
    (0.1, 1e-1), ( 0.2, 1e-2), ( 1, 1e-3),
    (2.0, 1e-4), (10.0, 1e-5), (20, 1e-6),
])
def test_CIST(kappa, r0):
    alpha = 1.2
    r = xr.DataArray(np.logspace(-3, 2, 101), dims='r', coords={'r':np.logspace(-3, 2, 101)})

    cist1 = ana_CIST(r, alpha, params={'beta':kappa, 'r0':r0}, regime='richardson')
    cist2 = ana_CIST(r, alpha, params={'kappa':kappa, 'r0':r0, 'slope':-5/3}, regime='generallocal')

    np.testing.assert_allclose(cist1, cist2)

    cist1 = ana_CIST(r, alpha, params={'lambda':kappa, 'r0':r0}, regime='-2')
    cist2 = ana_CIST(r, alpha, params={'kappa':kappa, 'r0':r0, 'slope':-2}, regime='generallocal')

    np.testing.assert_allclose(cist1, cist2)
    
