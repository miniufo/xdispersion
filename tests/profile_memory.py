#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile peak memory usage for each diagnostic measure in ``measures.py``.

For each dataset (ragged / local / nonlocal) this script:

  1. Sets up ``RelativeDispersion`` and pairs (same params as ``test_measures.py``)
  2. Computes separation / velocity / acceleration building blocks one by one,
     recording peak RSS for each
  3. Calls **each measure function individually** while a background thread
     samples RSS at high frequency to capture the true peak
  4. Calls ``cal_measures`` with *all* measures at once for comparison
  5. Prints a formatted summary table sorted by peak memory

Usage::

    cd xdispersion
    python tests/profile_memory.py                     # all datasets
    python tests/profile_memory.py --dataset ragged    # one dataset
    python tests/profile_memory.py --interval 0.001    # finer polling

Requires: ``psutil``  (``pip install psutil``)
"""
import gc
import os
import sys
import time
import threading
import argparse

# Ensure the project root (parent of tests/) is on sys.path so that the
# ``xdispersion`` package can be imported when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

try:
    import psutil
except ImportError:
    sys.exit('psutil is required.  Install with:  pip install psutil')

from xdispersion.core import RelativeDispersion
from xdispersion.measures import (
    relative_dispersion,
    velocity_structure_function,
    relative_diffusivity,
    finite_amplitude_growth_rate,
    initial_memory,
    anisotropy,
    lagrangian_velocity_correlation,
    kurtosis,
    cencini_vulpiani_exponent,
    finite_size_lyapunov_exponent,
    cumulative_inverse_separation_time,
    probability_density_function,
    cumulative_density_function,
    principle_axis_components,
    rotational_divergent_components,
)
from xdispersion.template import cal_measures
from xdispersion.utils import gen_rbins


rbins = gen_rbins(0.01, 1000, alpha=1.2)

# Chunk sizes — must match test_measures.py / generate_ref.py
RAGGED_CHUNK    = 50
NONRAGGED_CHUNK = 200


# =========================================================================== #
#  Memory monitor                                                              #
# =========================================================================== #
class MemMonitor:
    """Monitor peak RSS of the current process via a background thread.

    Usage::

        with MemMonitor() as m:
            do_something()
        print(m.delta)   # peak RSS - baseline RSS
    """

    def __init__(self, interval=0.002):
        self.interval = interval
        self._proc    = psutil.Process()
        self._baseline = 0
        self._peak     = 0
        self._running  = False
        self._thread   = None

    def _poll(self):
        while self._running:
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        gc.collect()
        self._baseline = self._proc.memory_info().rss
        self._peak     = self._baseline
        self._running  = True
        self._thread   = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        if self._thread:
            self._thread.join()
        rss = self._proc.memory_info().rss
        if rss > self._peak:
            self._peak = rss
        return False

    @property
    def baseline(self):
        return self._baseline

    @property
    def peak(self):
        return self._peak

    @property
    def delta(self):
        return self._peak - self._baseline


def _mb(b):
    """Format bytes as MB."""
    return f'{b / 1048576:.1f}'


def _force_load(result):
    """Trigger computation of lazy (dask) results."""
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'load'):
                r.load()
    elif hasattr(result, 'load'):
        result.load()


# =========================================================================== #
#  Dataset setup (mirrors test_measures.py fixture)                            #
# =========================================================================== #
def setup_dataset(name):
    """Return (rd, pairs, building_blocks_dict).

    Building blocks are .load()-ed so that per-measure memory profiling
    reflects only the measure's own memory, not the building blocks'.
    """
    if name == 'ragged':
        dset = xr.open_dataset('./data/glad32.nc')
        rd = RelativeDispersion(dset, xpos='longitude', ypos='latitude',
                                uvel='ve', vvel='vn', maxtlen=4 * 24 * 80,
                                time='time', Rearth=6371.2, ID='traj',
                                coord='latlon', ragged=True,
                                chunk=RAGGED_CHUNK)
        p_all = rd.get_all_pairs()
        pairs = rd.get_original_pairs(p_all, r0=[0.08, 0.18])

    elif name == 'local':
        dset = xr.open_dataset('./data/Particles_local.nc')
        rd = RelativeDispersion(dset, xpos='xpos', ypos='ypos',
                                uvel='uvel', vvel='vvel',
                                time='time', ID='ID',
                                coord='cartesian', ragged=False,
                                chunk=NONRAGGED_CHUNK)
        p_all = rd.get_all_pairs()
        pairs = rd.get_original_pairs(p_all, r0=[0, 0.01])

    else:  # nonlocal
        dset = xr.open_dataset('./data/Particles_nonlocal.nc')
        rd = RelativeDispersion(dset, xpos='xpos', ypos='ypos',
                                uvel='uvel', vvel='vvel',
                                time='time', ID='ID',
                                coord='cartesian', ragged=False,
                                chunk=NONRAGGED_CHUNK)
        p_all = rd.get_all_pairs()
        pairs = rd.get_original_pairs(p_all, r0=[0, 0.01])

    return rd, pairs


# =========================================================================== #
#  Building-block profiling                                                    #
# =========================================================================== #
def profile_building_blocks(rd, pairs, verbose=True):
    """Compute and .load() building blocks one by one, recording memory."""
    results = []

    steps = [
        ('separation_measures',  lambda: rd.separation_measures(pairs)),
        ('velocity_measures',    lambda: rd.velocity_measures(pairs)),
        ('acceleration_measures', lambda: rd.acceleration_measures(pairs)),
    ]

    bb = {}
    for name, func in steps:
        gc.collect()
        with MemMonitor() as m:
            t0 = time.perf_counter()
            out = func()
            for v in out:
                if hasattr(v, 'load'):
                    v.load()
            elapsed = time.perf_counter() - t0

        results.append((name, m.baseline, m.peak, m.delta, elapsed))

        if name == 'separation_measures':
            bb['rx'], bb['ry'], bb['rxy'], bb['r'], bb['rpb'] = out
        elif name == 'velocity_measures':
            bb['du'], bb['dv'], bb['dul'], bb['dut'], \
                bb['vmi'], bb['vmj'], bb['uv'] = out
        else:
            bb['dax'], bb['day'], bb['dal'], bb['dat'], \
                bb['ai'], bb['aj'], bb['axy'] = out

        if verbose:
            print(f'    {name:25s}  base={_mb(m.baseline):>8s} MB  '
                  f'peak={_mb(m.peak):>8s} MB  Δ={_mb(m.delta):>8s} MB  '
                  f'({elapsed:.2f}s)')

    return bb, results


# =========================================================================== #
#  Per-measure profiling                                                       #
# =========================================================================== #
def _measure_calls(bb):
    """Return [(name, callable), …] for every measure, using pre-loaded
    building blocks from *bb*."""
    r   = bb['r'];    rpb = bb['rpb']
    rx  = bb['rx'];   ry  = bb['ry'];   rxy = bb['rxy']
    du  = bb['du'];   dv  = bb['dv']
    dul = bb['dul'];  dut = bb['dut']
    uv  = bb['uv'];   vmi = bb['vmi'];  vmj = bb['vmj']
    dax = bb['dax'];  day = bb['day']
    ai  = bb['ai'];   aj  = bb['aj'];   axy = bb['axy']

    ct = dict(mean_at='const-t')
    cr = dict(mean_at='const-r', rbins=rbins)

    return [
        # ---- const-t ----
        ('r2_t',    lambda: relative_dispersion(r, order=2, **ct)),
        ('rpb2_t',  lambda: relative_dispersion(rpb, order=2, **ct)),
        ('S2_t',    lambda: velocity_structure_function(np.hypot(du, dv), r, **ct)),
        ('S2ll_t',  lambda: velocity_structure_function(dul, r, **ct)),
        ('S2tr_t',  lambda: velocity_structure_function(dut, r, **ct)),
        ('S3_t',    lambda: velocity_structure_function(dul*(du**2+dv**2), r, order=1, **ct)),
        ('a2_t',    lambda: velocity_structure_function(np.hypot(dax, day), r, **ct)),
        ('K2_t',    lambda: relative_diffusivity(r, samples='all', **ct)),
        ('K2p_t',   lambda: relative_diffusivity(r, samples='abs', **ct)),
        ('FAGR_t',  lambda: finite_amplitude_growth_rate(r, **ct)),
        ('FAGRp_t', lambda: finite_amplitude_growth_rate(r, samples='positive', **ct)),
        ('imrv_t',  lambda: initial_memory(rx, ry, du, dv, r, **ct)),
        ('imrv2_t', lambda: initial_memory(rx, ry, du, dv, r, order=2, **ct)),
        ('imra_t',  lambda: initial_memory(rx, ry, dax, day, r, **ct)),
        ('aniso_t', lambda: anisotropy(rx, ry, rxy, r, **ct)),
        ('LVC_t',   lambda: lagrangian_velocity_correlation(uv, vmi, vmj, r, **ct)),
        ('LAC_t',   lambda: lagrangian_velocity_correlation(axy, ai, aj, r, **ct)),
        ('ku_t',    lambda: kurtosis(r, **ct)),
        ('kupb_t',  lambda: kurtosis(rpb, **ct)),
        ('CVE_t',   lambda: cencini_vulpiani_exponent(r, **ct)),
        # ---- const-r ----
        ('r2_r',    lambda: relative_dispersion(r, order=2, **cr)),
        ('rpb2_r',  lambda: relative_dispersion(rpb, order=2, **cr)),
        ('S2_r',    lambda: velocity_structure_function(np.hypot(du, dv), r, **cr)),
        ('S2ll_r',  lambda: velocity_structure_function(dul, r, **cr)),
        ('S2tr_r',  lambda: velocity_structure_function(dut, r, **cr)),
        ('S3_r',    lambda: velocity_structure_function(dul*(du**2+dv**2), r, order=1, **cr)),
        ('a2_r',    lambda: velocity_structure_function(np.hypot(dax, day), r, **cr)),
        ('K2_r',    lambda: relative_diffusivity(r, samples='all', **cr)),
        ('K2p_r',   lambda: relative_diffusivity(r, samples='abs', **cr)),
        ('FAGR_r',  lambda: finite_amplitude_growth_rate(r, **cr)),
        ('FAGRp_r', lambda: finite_amplitude_growth_rate(r, samples='positive', **cr)),
        ('imrv_r',  lambda: initial_memory(rx, ry, du, dv, r, **cr)),
        ('imrv2_r', lambda: initial_memory(rx, ry, du, dv, r, order=2, **cr)),
        ('imra_r',  lambda: initial_memory(rx, ry, dax, day, r, **cr)),
        ('aniso_r', lambda: anisotropy(rx, ry, rxy, r, **cr)),
        ('LVC_r',   lambda: lagrangian_velocity_correlation(uv, vmi, vmj, r, **cr)),
        ('LAC_r',   lambda: lagrangian_velocity_correlation(axy, ai, aj, r, **cr)),
        ('CVE_r',   lambda: cencini_vulpiani_exponent(r, **cr)),
        ('FSLE_r',  lambda: finite_size_lyapunov_exponent(r, **cr)),
        ('CIST_r',  lambda: cumulative_inverse_separation_time(r, **cr)),
        # ---- helpers (not in cal_measures) ----
        ('PDF',      lambda: probability_density_function(r, rbins=rbins)),
        ('S2rr_r',   lambda: rotational_divergent_components(
                         velocity_structure_function(dul, r, **cr),
                         velocity_structure_function(dut, r, **cr))[0]),
    ]


def profile_measures(bb, interval=0.002, verbose=True):
    """Profile each measure function individually."""
    calls  = _measure_calls(bb)
    results = []

    for name, func in calls:
        gc.collect()
        with MemMonitor(interval=interval) as m:
            t0 = time.perf_counter()
            result = func()
            _force_load(result)
            elapsed = time.perf_counter() - t0

        results.append((name, m.baseline, m.peak, m.delta, elapsed))

        if verbose:
            print(f'    {name:12s}  base={_mb(m.baseline):>8s} MB  '
                  f'peak={_mb(m.peak):>8s} MB  Δ={_mb(m.delta):>8s} MB  '
                  f'({elapsed:.2f}s)')

    return results


# =========================================================================== #
#  cal_measures (all at once) for comparison                                   #
# =========================================================================== #
def profile_cal_measures(rd, pairs, interval=0.002, verbose=True):
    """Profile cal_measures with all measures at once."""
    gc.collect()
    with MemMonitor(interval=interval) as m:
        t0 = time.perf_counter()
        ds = cal_measures(rd, pairs, rbins, one_by_one=False)
        ds = ds.load()
        elapsed = time.perf_counter() - t0

    if verbose:
        print(f'    {"cal_measures":12s}  base={_mb(m.baseline):>8s} MB  '
              f'peak={_mb(m.peak):>8s} MB  Δ={_mb(m.delta):>8s} MB  '
              f'({elapsed:.2f}s)  [{len(ds.data_vars)} vars]')

    return [('cal_measures(all)', m.baseline, m.peak, m.delta, elapsed)]


# =========================================================================== #
#  Summary table                                                               #
# =========================================================================== #
def print_table(title, rows):
    """Print a formatted table from (name, baseline, peak, delta, time) rows."""
    print(f'\n  {title}')
    print(f'  {"Name":<22s}  {"Baseline":>10s}  {"Peak":>10s}  '
          f'{"Δ Peak":>10s}  {"Time":>8s}')
    print(f'  {"-"*22}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}')

    for name, base, peak, delta, elapsed in rows:
        print(f'  {name:<22s}  {_mb(base):>10s}  {_mb(peak):>10s}  '
              f'{_mb(delta):>10s}  {elapsed:>7.2f}s')

    # Highlight top-3 memory consumers
    sorted_rows = sorted(rows, key=lambda r: r[3], reverse=True)
    print(f'\n  Top-3 memory (Δ):  ', end='')
    print(', '.join(f'{r[0]} ({_mb(r[3])} MB)' for r in sorted_rows[:3]))


# =========================================================================== #
#  Main                                                                        #
# =========================================================================== #
def profile_dataset(name, interval=0.002):
    print(f'\n{"=" * 72}')
    print(f'  Dataset: {name}')
    print(f'{"=" * 72}')

    # --- setup ---
    with MemMonitor(interval=interval) as m:
        t0 = time.perf_counter()
        rd, pairs = setup_dataset(name)
        elapsed = time.perf_counter() - t0

    npairs = len(pairs.pair)
    print(f'\n  Setup (RD + pairs):  {npairs} pairs  '
          f'peak={_mb(m.peak)} MB  Δ={_mb(m.delta)} MB  ({elapsed:.2f}s)')

    # --- building blocks ---
    print(f'\n  Building blocks:')
    bb, bb_rows = profile_building_blocks(rd, pairs)

    # --- per-measure ---
    print(f'\n  Per-measure (building blocks pre-loaded):')
    m_rows = profile_measures(bb, interval=interval)

    # --- cal_measures (all) ---
    print(f'\n  cal_measures (all at once, one_by_one=False):')
    cm_rows = profile_cal_measures(rd, pairs, interval=interval)

    # --- summary tables ---
    print_table('Building blocks summary', bb_rows)
    print_table('Per-measure summary (sorted by Δ peak)', 
                sorted(m_rows, key=lambda r: r[3], reverse=True))
    print_table('cal_measures comparison', cm_rows)

    # cleanup
    del rd, pairs, bb
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Profile peak memory usage for xdispersion measures')
    parser.add_argument('--dataset', default='all',
                        choices=['ragged', 'local', 'nonlocal', 'all'],
                        help='Which dataset to profile (default: all)')
    parser.add_argument('--interval', type=float, default=0.002,
                        help='RSS polling interval in seconds (default: 0.002)')
    args = parser.parse_args()

    datasets = (['ragged', 'local', 'nonlocal']
                if args.dataset == 'all' else [args.dataset])

    for ds in datasets:
        profile_dataset(ds, interval=args.interval)

    print(f'\n{"=" * 72}')
    print('  Done.')
    print(f'{"=" * 72}')


if __name__ == '__main__':
    main()
