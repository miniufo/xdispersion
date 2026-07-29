#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate reference measure NC files for testing.

This script computes all measures via ``cal_measures`` for three datasets
and saves the results as compressed NetCDF files (zlib) suitable for
uploading to GitHub.

Usage::

    cd xdispersion
    python tests/generate_ref.py

The generated files are::

    data/ref_measures_ragged.nc    — GLAD drifters   (ragged,  latlon)
    data/ref_measures_local.nc     — Particles_local (non-ragged, cartesian)
    data/ref_measures_nonlocal.nc  — Particles_nonlocal (non-ragged, cartesian)

Each file contains ~42 variables (const-t, const-r, and derived measures)
computed with ``ensemble=0`` (no bootstrapping).
"""
import os
import xarray as xr
from xdispersion.core import RelativeDispersion
from xdispersion.template import cal_measures
from xdispersion.utils import gen_rbins


rbins = gen_rbins(0.01, 1000, alpha=1.2)


def save_nc(ds, path):
    """Save *ds* as a zlib-compressed NetCDF file."""
    encoding = {v: {'zlib': True, 'complevel': 5} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    size = os.path.getsize(path)
    print(f'  saved {path}  ({size / 1024:.1f} KB)')


# --------------------------------------------------------------------------- #
def generate_ragged():
    """GLAD drifters — ragged, latlon coordinates."""
    print('Generating reference for ragged (GLAD) ...')
    dset = xr.open_dataset('./data/glad32.nc')
    rd = RelativeDispersion(dset, xpos='longitude', ypos='latitude',
                            uvel='ve', vvel='vn', maxtlen=4 * 24 * 80,
                            time='time', Rearth=6371.2, ID='traj',
                            coord='latlon', ragged=True, chunk=50)
    p_all = rd.get_all_pairs()
    pairs = rd.get_original_pairs(p_all, r0=[0.08, 0.18])
    print(f'  {len(pairs.pair)} original pairs')
    ref = cal_measures(rd, pairs, rbins, one_by_one=True)
    save_nc(ref, './data/ref_measures_ragged.nc')


def generate_local():
    """Particles_local — non-ragged, cartesian coordinates."""
    print('Generating reference for local (Particles_local) ...')
    dset = xr.open_dataset('./data/Particles_local.nc')
    rd = RelativeDispersion(dset, xpos='xpos', ypos='ypos',
                            uvel='uvel', vvel='vvel',
                            time='time', ID='ID',
                            coord='cartesian', ragged=False, chunk=200)
    p_all = rd.get_all_pairs()
    pairs = rd.get_original_pairs(p_all, r0=[0.0099, 0.0101])
    print(f'  {len(pairs.pair)} original pairs')
    ref = cal_measures(rd, pairs, rbins, one_by_one=True)
    save_nc(ref, './data/ref_measures_local.nc')


def generate_nonlocal():
    """Particles_nonlocal — non-ragged, cartesian coordinates."""
    print('Generating reference for nonlocal (Particles_nonlocal) ...')
    dset = xr.open_dataset('./data/Particles_nonlocal.nc').isel(ID=slice(0, 40))
    rd = RelativeDispersion(dset, xpos='xpos', ypos='ypos',
                            uvel='uvel', vvel='vvel',
                            time='time', ID='ID',
                            coord='cartesian', ragged=False, chunk=200)
    p_all = rd.get_all_pairs()
    pairs = rd.get_original_pairs(p_all, r0=[0.0099, 0.0101])
    print(f'  {len(pairs.pair)} original pairs')
    ref = cal_measures(rd, pairs, rbins, one_by_one=True)
    save_nc(ref, './data/ref_measures_nonlocal.nc')


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    generate_ragged()
    generate_local()
    generate_nonlocal()
    print('Done!')
