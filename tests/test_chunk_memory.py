#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test how the ``chunk`` parameter affects peak memory usage.

For both ragged and non-ragged datasets, this script tries several chunk
sizes and reports the peak RSS during:

  1. get_all_pairs + get_original_pairs   (pair generation)
  2. separation_measures                   (building block)
  3. cal_measures(one_by_one=True)         (all measures)

Usage::

    cd xdispersion
    python tests/test_chunk_memory.py
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import psutil

from xdispersion.core import RelativeDispersion
from xdispersion.template import cal_measures
from xdispersion.utils import gen_rbins

# Reuse MemMonitor from profile_memory
from profile_memory import MemMonitor, _mb

rbins = gen_rbins(0.01, 1000, alpha=1.2)


# Chunk sizes to test
RAGGED_CHUNKS   = [10, 25, 50, 100, None]
NONRAGGED_CHUNKS = [50, 100, 200, 500, None]


def test_ragged_chunk(chunk):
    """Test a single chunk size for the ragged dataset."""
    gc.collect()
    label = str(chunk) if chunk is not None else 'None'
    proc = psutil.Process()

    # Phase 1: setup + pairs
    with MemMonitor() as m1:
        dset = xr.open_dataset('./data/glad32.nc')
        rd = RelativeDispersion(dset, xpos='longitude', ypos='latitude',
                                uvel='ve', vvel='vn', maxtlen=4 * 24 * 80,
                                time='time', Rearth=6371.2, ID='traj',
                                coord='latlon', ragged=True, chunk=chunk)
        p_all = rd.get_all_pairs()
        pairs = rd.get_original_pairs(p_all, r0=[0.08, 0.18])

    # Phase 2: separation_measures (representative building block)
    with MemMonitor() as m2:
        rx, ry, rxy, r, rpb = rd.separation_measures(pairs)
        for v in [rx, ry, rxy, r, rpb]:
            if hasattr(v, 'load'):
                v.load()

    # Phase 3: cal_measures
    with MemMonitor() as m3:
        ds = cal_measures(rd, pairs, rbins, one_by_one=True)

    # Save results before cleanup
    npairs_all = len(p_all.pair)
    npairs_ori = len(pairs.pair)

    # cleanup
    del dset, rd, p_all, pairs, rx, ry, rxy, r, rpb, ds
    gc.collect()

    return {
        'chunk': label,
        'npairs': npairs_all,
        'npairs_ori': npairs_ori,
        'setup_peak': m1.peak,
        'setup_delta': m1.delta,
        'sep_peak': m2.peak,
        'sep_delta': m2.delta,
        'cal_peak': m3.peak,
        'cal_delta': m3.delta,
    }


def test_nonragged_chunk(chunk, dataset='nonlocal', nparticles=40):
    """Test a single chunk size for a non-ragged dataset."""
    gc.collect()
    label = str(chunk) if chunk is not None else 'None'

    with MemMonitor() as m1:
        dset = xr.open_dataset(f'./data/Particles_{dataset}.nc').isel(ID=slice(0, nparticles))
        rd = RelativeDispersion(dset, xpos='xpos', ypos='ypos',
                                uvel='uvel', vvel='vvel',
                                time='time', ID='ID',
                                coord='cartesian', ragged=False, chunk=chunk)
        p_all = rd.get_all_pairs()
        pairs = rd.get_original_pairs(p_all, r0=[0, 0.01])

    with MemMonitor() as m2:
        rx, ry, rxy, r, rpb = rd.separation_measures(pairs)
        for v in [rx, ry, rxy, r, rpb]:
            if hasattr(v, 'load'):
                v.load()

    with MemMonitor() as m3:
        ds = cal_measures(rd, pairs, rbins, one_by_one=True)

    npairs_all = len(p_all.pair)
    npairs_ori = len(pairs.pair)

    del dset, rd, p_all, pairs, rx, ry, rxy, r, rpb, ds
    gc.collect()

    return {
        'chunk': label,
        'npairs': npairs_all,
        'npairs_ori': npairs_ori,
        'setup_peak': m1.peak,
        'setup_delta': m1.delta,
        'sep_peak': m2.peak,
        'sep_delta': m2.delta,
        'cal_peak': m3.peak,
        'cal_delta': m3.delta,
    }


def print_comparison(title, results):
    """Print a comparison table."""
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')
    print(f'  {"chunk":>6s}  {"npairs":>7s}  '
          f'{"setup Δ":>10s}  {"setup peak":>10s}  '
          f'{"sep Δ":>10s}  {"cal Δ":>10s}  {"cal peak":>10s}')
    print(f'  {"-"*6}  {"-"*7}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')

    for r in results:
        print(f'  {r["chunk"]:>6s}  {r["npairs"]:>7d}  '
              f'{_mb(r["setup_delta"]):>10s}  {_mb(r["setup_peak"]):>10s}  '
              f'{_mb(r["sep_delta"]):>10s}  '
              f'{_mb(r["cal_delta"]):>10s}  {_mb(r["cal_peak"]):>10s}')

    # Analysis
    deltas = [r['setup_peak'] for r in results]
    min_idx = deltas.index(min(deltas))
    max_idx = deltas.index(max(deltas))
    print(f'\n  Setup peak — lowest: chunk={results[min_idx]["chunk"]} '
          f'({_mb(deltas[min_idx])} MB),  '
          f'highest: chunk={results[max_idx]["chunk"]} '
          f'({_mb(deltas[max_idx])} MB),  '
          f'ratio: {deltas[max_idx]/deltas[min_idx]:.2f}x')


def main():
    # --- Ragged ---
    print('Testing ragged dataset (GLAD, 297 trajectories)...')
    ragged_results = []
    for chunk in RAGGED_CHUNKS:
        label = str(chunk) if chunk is not None else 'None'
        print(f'  chunk={label} ...', end=' ', flush=True)
        r = test_ragged_chunk(chunk)
        ragged_results.append(r)
        print(f'setup_peak={_mb(r["setup_peak"])} MB', flush=True)

    print_comparison('Ragged: chunk size vs memory', ragged_results)

    # --- Non-ragged (40 particles for speed) ---
    print('\nTesting non-ragged dataset (nonlocal, 40 particles)...')
    nonragged_results = []
    for chunk in NONRAGGED_CHUNKS:
        label = str(chunk) if chunk is not None else 'None'
        print(f'  chunk={label} ...', end=' ', flush=True)
        r = test_nonragged_chunk(chunk, dataset='nonlocal', nparticles=40)
        nonragged_results.append(r)
        print(f'setup_peak={_mb(r["setup_peak"])} MB', flush=True)

    print_comparison('Non-ragged: chunk size vs memory', nonragged_results)

    print(f'\n{"=" * 80}')
    print('  Done.')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
