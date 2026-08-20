# -*- coding: utf-8 -*-
"""
Created on 2025.02.26

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.

Module: xdispersion.measures
============================

Diagnostic measure functions for two-particle (relative dispersion)
statistics.

**Naming convention**

Each function returns a DataArray whose ``.name`` encodes the measure
and averaging mode:

- Suffix ``_t`` → averaged at **constant time** (const-t):
  pairs start together at ``rtime = 0``; the statistic is a function
  of elapsed time.

- Suffix ``_r`` → averaged at **constant separation** (const-r):
  pairs are binned by current separation; the statistic is a function
  of separation distance.

**Bootstrapping**

When ``ensemble > 0``, each function returns a 3-tuple
``(value, lower_bound, upper_bound)`` where the bounds are obtained
by resampling pairs with replacement and computing the ``CI``
confidence interval.

**Required building blocks**

Most functions take pre-computed building blocks from
:class:`~xdispersion.core.RelativeDispersion`:

- Separation: ``rx, ry, rxy, r, rpb``
- Velocity:   ``du, dv, dul, dut, vmi, vmj, uv``
- Acceleration: ``dax, day, dal, dat, ai, aj, axy``
"""
import numpy as np
import xarray as xr
from typing import Optional, Tuple, Literal, Union, List
from tqdm import tqdm
from xhistogram.xarray import histogram
from .utils import semilog_fit, mean_at_rbin, gen_rbins, bootstrap


default_rbins = gen_rbins(0.01, 1000, alpha=1.2)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Below are functions for separation/velocity measures "
"                                                      "
"   Suffix '_t' means averaged at constant time, and   "
"   suffix '_r' means averaged at constant rbin.       "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def relative_dispersion(
    r: xr.DataArray,
    order: Optional[int] = 2,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    r"""Compute the N-th moment of relative separation.

    .. math::
        \mathrm{rN} = \langle r^n \rangle

    For ``order=2`` this is the classic *relative dispersion*
    (Richardson 1926).  For ``order=1`` it is the mean separation.

    Parameters
    ----------
    r : xarray.DataArray
        Relative separation ``[pair, rtime]`` (from
        :meth:`~xdispersion.core.RelativeDispersion.separation_measures`).
    order : int, default 2
        Moment order :math:`n`.
    rbins : xarray.DataArray
        Separation bins for const-r averaging.
    mean_at : {'const-t', 'const-r'}
        Averaging mode.
    ensemble : int, default 0
        Number of bootstrap resamples (0 disables).
    CI : float, default 0.95
        Confidence interval level.
    nproc : int, default 1
        Number of processes for bootstrap.
    
    Returns
    -------
    rN: xarray.DataArray
        Nth-moment of separation.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    rN = r ** order
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair').astype(r.dtype)
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(rN, r).rename(f'r{order}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [rN, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(rN, r).rename(f'r{order}_{mean_at[-1]}'),\
               lb.rename(f'LBr{order}_{mean_at[-1]}'),\
               ub.rename(f'UBr{order}_{mean_at[-1]}')


def velocity_structure_function(
    du: xr.DataArray,
    r: xr.DataArray,
    order: Optional[int] = 2,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    r"""Compute the velocity structure function.

    .. math::
        \mathrm{S}n = \langle |\delta \mathbf{u}|^n \rangle

    The input ``du`` determines which component is computed:

    - ``np.hypot(du, dv)`` → total S2
    - ``dul``              → longitudinal S2ll
    - ``dut``              → transversal S2tr
    - ``dul*(du**2+dv**2)`` with ``order=1`` → S3

    Parameters
    ----------
    du : xarray.DataArray
        Relative velocity (or any quantity whose moment is desired).
    r : xarray.DataArray
        Relative separation ``[pair, rtime]``.
    order : int, default 2
        Moment order :math:`n`.
    rbins : xarray.DataArray
        Separation bins for const-r averaging.
    mean_at : {'const-t', 'const-r'}, default 'const-r'
        Averaging mode.
    ensemble : int, default 0
        Bootstrap resamples (0 disables).
    CI : float, default 0.95
        Confidence interval level.
    nproc : int, default 1
        Number of processes for bootstrap.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    SN: xr.DataArray
        Nth-order velocity structure funciton.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    SN = du ** order
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair').astype(r.dtype)
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(SN, r).rename(f'S{order}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [SN, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(SN, r).rename(f'S{order}_{mean_at[-1]}'),\
               lb.rename(f'LBS{order}_{mean_at[-1]}'),\
               ub.rename(f'UBS{order}_{mean_at[-1]}')


def relative_diffusivity(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    samples: Literal['all', 'positive', 'negative', 'abs'] = 'all',
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate relative diffusivity
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    samples: str
        Samples over which the average is taken ['all', 'positive', 'negative', 'abs'].
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    K2: xr.DataArray
        Relative diffusivity.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # ffill() to ensure nans will not propagate to
    # contaminate the results via finite differencing
    # This may also cause memory problem if rtime is chunked
    # Make sure that rtime is not chunked or chunk({'rtime':-1})
    if r.chunksizes is not None and 'rtime' in r.chunksizes and len(r.chunksizes['rtime']) > 1:
        # for synthetic drifters that rtime is chunked for long-term tracking
        K2 = (r**2.0).differentiate('rtime') / 2.0
    else:
        # for real drifters that t-lengths are not equal
        K2 = (r**2.0).ffill('rtime').differentiate('rtime') / 2.0
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            if samples == 'all':
                return v.mean('pair').astype(r.dtype)
            elif samples == 'positive':
                return v.where(v>0).mean('pair').astype(r.dtype)
            elif samples == 'negative':
                return v.where(v<0).mean('pair').astype(r.dtype)
            elif samples == 'abs':
                return np.abs(v).mean('pair').astype(r.dtype)
            else:
                raise Exception(f'unsupported samples {samples}, '+
                                f'should be one of [all, positive, negative, abs]')
        elif mean_at == 'const-r':
            if samples == 'all':
                return mean_at_rbin(v, r, rbins).astype(r.dtype)
            elif samples == 'positive':
                return mean_at_rbin(v, r, rbins, cond=v>0).astype(r.dtype)
            elif samples == 'negative':
                return mean_at_rbin(v, r, rbins, cond=v<0).astype(r.dtype)
            elif samples == 'abs':
                return mean_at_rbin(np.abs(v), r, rbins).astype(r.dtype)
            else:
                raise Exception(f'unsupported samples {samples}, '+
                                f'should be one of [all, positive, negative, abs]')
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(K2, r).rename(f'K2_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [K2, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(K2, r).rename(f'K2_{mean_at[-1]}'),\
               lb.rename(f'LBK2_{mean_at[-1]}'),\
               ub.rename(f'UBK2_{mean_at[-1]}')


def finite_amplitude_growth_rate(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    samples: Literal['all', 'positive', 'negative'] = 'positive',
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate finite-amplitude growth rate
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    samples: str
        Samples over which the average is taken.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    FAGR: xr.DataArray
        Finite-amplitude growth rate.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # ffill() to ensure nans will not propagate to
    # contaminate the results via finite differencing
    if r.chunksizes is not None and 'rtime' in r.chunksizes and len(r.chunksizes['rtime']) > 1:
        # for synthetic drifters that rtime is chunked for long-term tracking
        sFAGR = np.log(r).differentiate('rtime')
    else:
        # for real drifters that t-lengths are not equal
        sFAGR = np.log(r).ffill('rtime').differentiate('rtime')
    
    p = 'a' # default: average over all samples

    if samples == 'positive':
        p = 'p'
    if samples == 'negative':
        p = 'n'
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            if samples == 'all':
                return v.mean('pair').astype(r.dtype)
            elif samples == 'positive':
                return v.where(v>0).mean('pair').astype(r.dtype)
            elif samples == 'negative':
                return v.where(v<0).mean('pair').astype(r.dtype)
            else:
                raise Exception(f'unsupported samples {samples}, '+
                                f'should be one of [all, positive, negative]')
        elif mean_at == 'const-r':
            if samples == 'all':
                return mean_at_rbin(v, r, rbins).astype(r.dtype)
            elif samples == 'positive':
                return mean_at_rbin(v, r, rbins, cond=v>0).astype(r.dtype)
            elif samples == 'negative':
                return mean_at_rbin(v, r, rbins, cond=v<0).astype(r.dtype)
            else:
                raise Exception(f'unsupported samples {samples}, '+
                                f'should be one of [all, positive, negative]')
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
    
    if ensemble <= 0:
        return how_to_mean(sFAGR, r).rename(f'FAGR{p}_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [sFAGR, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(sFAGR, r).rename(f'FAGR{p}_{mean_at[-1]}'),\
               lb.rename(f'LBFAGR{p}_{mean_at[-1]}'),\
               ub.rename(f'UBFAGR{p}_{mean_at[-1]}')


def initial_memory(
    rx: xr.DataArray,
    ry: xr.DataArray,
    du: xr.DataArray,
    dv: xr.DataArray,
    r: xr.DataArray,
    order: int = 1,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    r"""Calculate initial memory for initial separation vector r0

    .. math::
        \mathrm{initm} = \langle (\mathbf{r}_0 \cdot \mathbf{v})^{\mathrm{order}} \rangle

    or :math:`\langle (\mathbf{r}_0 \cdot \mathbf{a})^{\mathrm{order}} \rangle`,
    depending on the input.
    
    Parameters
    ----------
    rx: xarray.DataArray
        Zonal component of separation.
    ry: xarray.DataArray
        meridional component of separation.
    du: xarray.DataArray
        Zonal component of relative velocity.
    dv: xarray.DataArray
        Meridional component of relative velocity.
    r: xarray.DataArray
        Relative separation.
    order: int
        Order of the moment
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    initm: xr.DataArray
        initial memory.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    initm = (rx.isel(rtime=0) * du + ry.isel(rtime=0) * dv)**order
    
    # define how to take average
    def how_to_mean(v, r):
        if mean_at == 'const-t':
            return v.mean('pair').astype(r.dtype)
        elif mean_at == 'const-r':
            return mean_at_rbin(v, r, rbins).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(initm, r).rename(f'initm_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [initm, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(initm, r).rename(f'initm_{mean_at[-1]}'),\
               lb.rename(f'LBinitm_{mean_at[-1]}'),\
               ub.rename(f'UBinitm_{mean_at[-1]}')

def anisotropy(
    rx: xr.DataArray,
    ry: xr.DataArray,
    rxy: xr.DataArray,
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate anisotropy
    
    Parameters
    ----------
    rx: xr.DataArray
        Zonal component of dispersion
    ry: xr.DataArray
        Meridional component of dispersion
    rxy: xr.DataArray
        Cross component of dispersion
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    aniso: xr.DataArray
        Anisotropy.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(rx, ry, rxy, r):
        if mean_at == 'const-t':
            rx2m = (rx**2.0).mean('pair')
            ry2m = (ry**2.0).mean('pair')
            rxym = rxy.mean('pair')
            ra2m, rb2m, _ = principle_axis_components(rx2m, ry2m, rxym)
            return np.sqrt(ra2m / rb2m).astype(r.dtype)
        elif mean_at == 'const-r':
            rx2m = mean_at_rbin(rx**2.0, r, rbins)
            ry2m = mean_at_rbin(ry**2.0, r, rbins)
            rxym = mean_at_rbin(rxy, r, rbins)
            ra2m, rb2m, _ = principle_axis_components(rx2m, ry2m, rxym)
            return np.sqrt(ra2m / rb2m).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(rx, ry, rxy, r).rename(f'aniso_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [rx, ry, rxy, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(rx, ry, rxy, r).rename(f'aniso_{mean_at[-1]}'),\
               lb.rename(f'LBaniso_{mean_at[-1]}'),\
               ub.rename(f'UBaniso_{mean_at[-1]}')


def lagrangian_velocity_correlation(
    uv: xr.DataArray,
    vs1: xr.DataArray,
    vs2: xr.DataArray,
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Lagrangian velocity correlation
    
    Parameters
    ----------
    uv: xr.DataArray
        Cross-variation of velocity
    vs1: xr.DataArray
        Velocity magnitude of first particle
    vs2: xr.DataArray
        Velocity magnitude of second particle
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    lvc: xr.DataArray
        Lagrangian velocity correlation, within [-1, 1].
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(uv, v1, v2, r):
        if mean_at == 'const-t':
            uvm = uv.mean('pair')
            v1m = (v1**2.0).mean('pair')
            v2m = (v2**2.0).mean('pair')
            return ((2.0 * uvm) / (v1m + v2m)).astype(r.dtype)
        elif mean_at == 'const-r':
            uvm = mean_at_rbin(uv     , r, rbins)
            v1m = mean_at_rbin(v1**2.0, r, rbins)
            v2m = mean_at_rbin(v2**2.0, r, rbins)
            return ((2.0 * uvm) / (v1m + v2m)).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(uv, vs1, vs2, r).rename(f'lvc_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [uv, vs1, vs2, r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(uv, vs1, vs2, r).rename(f'lvc_{mean_at[-1]}'),\
               lb.rename(f'LBlvc_{mean_at[-1]}'),\
               ub.rename(f'UBlvc_{mean_at[-1]}')


def kurtosis(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Kurtosis
    
    Parameters
    ----------
    r: xr.DataArray
        Relative separation r
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    Ku: xr.DataArray
        Kurtosis.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(r):
        if mean_at == 'const-t':
            r4m = (r**4).mean('pair')
            r2m = (r**2).mean('pair')
            return (r4m / r2m ** 2).astype(r.dtype)
        elif mean_at == 'const-r':
            r4m = mean_at_rbin(r**4, r, rbins)
            r2m = mean_at_rbin(r**2, r, rbins)
            return (r4m / r2m ** 2).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'Ku_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(r).rename(f'Ku_{mean_at[-1]}'),\
               lb.rename(f'LBKu_{mean_at[-1]}'),\
               ub.rename(f'UBKu_{mean_at[-1]}')


def cencini_vulpiani_exponent(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate Cencini-Vulpiani exponent or proxy FSLE

    This should be similar to FAGR.
    
    Parameters
    ----------
    r: xr.DataArray
        Relative separation r
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    CVE: xr.DataArray
        Cencini-Vulpiani exponent.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    # define how to take average
    def how_to_mean(r):
        r2 = (r ** 2.0)
        K2 = r2.differentiate('rtime') / 2.0
        
        if mean_at == 'const-t':
            return (K2.mean('pair') / r2.mean('pair')).astype(r.dtype)
        elif mean_at == 'const-r':
            return (mean_at_rbin(K2, r, rbins) / rbins**2.0).astype(r.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be one of [const-t, const-r]')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'CVE_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(r).rename(f'CVE_{mean_at[-1]}'),\
               lb.rename(f'LBCVE_{mean_at[-1]}'),\
               ub.rename(f'UBCVE_{mean_at[-1]}')


def finite_size_lyapunov_exponent_bak(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    interpT: Optional[int] = 1,
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate finite-size Lyapunov exponent
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    interpT: int (> 0)
        Interpolate along time dimension so that FSLE can be resolved
        at the smallest scales.  1 means no interpolation.
    
    Returns
    -------
    FSLE: float
        Finite-Size Lyapunov Exponent.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    if mean_at == 'const-t':
        raise Exception(f'unsupported mean_at string {mean_at}, '+
                        f'should be only const-r')
    
    def get_Td(r_single, rbins):
        if interpT > 1:
            rtime = r_single.rtime
            timeInt = np.linspace(rtime[0], rtime[-1], int((len(rtime)-1)*interpT+1))
            rinterp = r_single.interp(rtime=timeInt)
        elif interpT == 1:
            rinterp = r_single.load()
        else:
            raise Exception(f'invalid interpT {interpT}, should be larger than 0')
        
        rd = rinterp[rinterp.argmin().values:]
        
        return xr.where(rd > rbins, 1, np.nan).idxmax('rtime')
    
    alpha = rbins.values[-1] / rbins.values[-2] # ratio of neighbouring bins

    # loop over each pair to get Td, but too slow!!!!!!
    Td = []
    for i in tqdm(range(len(r['pair'])), ncols=80):
        Td.append(get_Td(r.isel(pair=i), rbins))

    Td = xr.concat(Td, dim='pair')
    
    FSLE = Td.diff('rbin')
    FSLE = (np.log(alpha) / FSLE.where(FSLE != 0))
    
    # define how to take average
    def how_to_mean(v):
        return v.mean('pair').astype(r.dtype)
        
    if ensemble <= 0:
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [FSLE], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}'),\
               lb.rename(f'LBFSLE_{mean_at[-1]}'),\
               ub.rename(f'UBFSLE_{mean_at[-1]}')


def finite_size_lyapunov_exponent(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    mean_at: Literal['const-t', 'const-r'] = 'const-r',
    interpT: Optional[int] = 1,
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate finite-size Lyapunov exponent
    
    Parameters
    ----------
    r: xarray.DataArray
        Relative separation.
    rbins: xr.DataArray
        A given set of separation bins used to average.
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    interpT: int (> 0)
        Interpolate along time dimension so that FSLE can be resolved
        at the smallest scales.  1 means no interpolation.
    
    Returns
    -------
    FSLE: float
        Finite-Size Lyapunov Exponent.
    lb: xarray.DataArray
        Lower-bound of confidence interval
    ub: xarray.DataArray
        Upper-bound of confidence interval
    """
    if mean_at == 'const-t':
        raise Exception(f'unsupported mean_at string {mean_at}, '+
                        f'should be only const-r')
    
    if interpT > 1:
        rtime = r.rtime
        timeInt = np.linspace(rtime[0], rtime[-1], int((len(rtime)-1)*interpT+1))
        rinterp = r.interp(rtime=timeInt)
    elif interpT == 1:
        rinterp = r
    else:
        raise Exception(f'invalid interpT {interpT}, should be an integer larger than 0')
    
    def get_Td(r_single, rbins, rtime):
        # r_single: shape (rtime,)
        # rbins:   shape (rbin,)
        # rtime:   shape (rtime,)
        
        # find index of minimum separation
        minidx = np.argmin(r_single)
        rd = r_single[minidx:]    # starts from minidx
        rtime_rd = rtime[minidx:] # starts from minidx
        
        # Vectorized: compare all time steps × all rbins at once
        # mask shape: (len(rd), len(rbins))
        mask = rd[:, None] > rbins[None, :]
        
        # First True index along time axis (argmax returns 0 for all-False)
        first_idx = np.argmax(mask, axis=0)
        has_crossed = np.any(mask, axis=0)
        
        Td = np.where(has_crossed, rtime_rd[first_idx], np.nan)
        return Td

    Td = xr.apply_ufunc(
        get_Td,
        rinterp.chunk({'rtime':-1}) if rinterp.chunks else rinterp,
        rbins,
        rinterp['rtime'],
        input_core_dims=[['rtime'], ['rbin'], ['rtime']],
        output_core_dims=[['rbin']],
        vectorize=True,
        dask='parallelized' if rinterp.chunks else False,
        output_dtypes=[r.dtype]
    )

    Td = Td.assign_coords(pair=r['pair'], rbin=rbins)
    
    alpha = rbins.values[-1] / rbins.values[-2] # ratio of neighbouring bins
    
    FSLE = Td.diff('rbin')
    FSLE = (np.log(alpha) / FSLE.where(FSLE != 0))
    
    # define how to take average
    def how_to_mean(v):
        return v.mean('pair').astype(r.dtype)
        
    if ensemble <= 0:
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [FSLE], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(FSLE).rename(f'FSLE_{mean_at[-1]}'),\
               lb.rename(f'LBFSLE_{mean_at[-1]}'),\
               ub.rename(f'UBFSLE_{mean_at[-1]}')


def cumulative_inverse_separation_time(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
    lower: Optional[float] = 0.1,
    upper: Optional[float] = 0.9,
    mean_at: Literal['const-t', 'const-r'] = 'const-t',
    maskout: List[float] = [1e-8, 5e4],
    ensemble: Optional[int] = 0,
    CI: Optional[float] = 0.95,
    nproc: int = 1
) -> Union[xr.DataArray,
           Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Calculate cumulative inverse separation time

    This is a diagnostic proposed by LaCasce and Meunier (2022, JFM), similar to FSLE.
    
    Parameters
    ----------
    r: xarray.DataArray
        Pair separations, typically as a function of ['pair', 'time'].
    rbins: numpy.array
        Separation bins which is uniform in a logarithm scale.
    lower: float
        Lower bound of CDF.
    upper: float
        Upper bound of CDF.
    maskout: list of float
        A range of valid results e.g., [minvalue, maxvalue].
    mean_at: str
        Condition of average. Should be one of ['const-t', 'const-r'],
        indicating average at constant time or separation.
    ensemble: int
        Times to bootstrapping.  0 for no bootstrapping
    CI: float
        Confidence interval for bootstrapping
    
    Returns
    -------
    CIST: xarray.DataArray
        Cumulative inverse separation time (unit of inverse time of CDF).
    """
    # define how to take average
    def how_to_mean(v):
        if mean_at == 'const-r':
            CDF = cumulative_density_function(probability_density_function(v, rbins), rbins)
            CDFrng = CDF.where(np.logical_and(CDF>lower, CDF<upper)).chunk({'rtime':-1})
            
            slope, inter = xr.apply_ufunc(semilog_fit, CDFrng['rtime'], CDFrng,
                                          dask='parallelized',
                                          input_core_dims=[['rtime'], ['rtime']],
                                          output_core_dims=[[], []],
                                        #   output_dtypes=[float, float],
                                          vectorize=True)
            
            # time for separation goes from 0 to r when CDF = 0.5
            fitted = np.exp((0.5 - inter) / slope)
            diff = fitted.diff('rbin')
            CIST  = 1.0 / diff
            
            if maskout:
                CIST = CIST.where(np.logical_and(CIST>maskout[0], CIST<maskout[1]))
            
            return CIST.astype(v.dtype)
        else:
            raise Exception(f'unsupported mean_at string {mean_at}, '+
                            f'should be only const-r')
        
    if ensemble <= 0:
        return how_to_mean(r).rename(f'CIST_{mean_at[-1]}')
    else:
        lb, ub = bootstrap(how_to_mean, [r], {},
                           ensemble=ensemble, CI=CI, nproc=nproc)
        
        return how_to_mean(r).rename(f'CIST_{mean_at[-1]}'),\
               lb.rename(f'LBCIST_{mean_at[-1]}'),\
               ub.rename(f'UBCIST_{mean_at[-1]}')


def probability_density_function(
    r: xr.DataArray,
    rbins: Optional[xr.DataArray] = default_rbins,
) -> xr.DataArray:
    """Calculate probability density function of pair separation r
    
    Parameters
    ----------
    r: xarray.DataArray
        Pair separations, typically as a function of ['pair', 'time'].
    rbins: numpy.array
        Separation bins which is uniform in a logarithm scale.
    
    Returns
    -------
    PDF: xarray.DataArray
        Probability density function.
    """
    tmp = rbins.values if isinstance(rbins, xr.DataArray) else rbins
    PDF = histogram(r.rename('r'), bins=tmp, dim=['pair'], density=True).rename('PDF')
    PDF['r_bin'] = tmp[1:]
    
    return PDF.rename({'r_bin':'rbin'}).astype(r.dtype)


def cumulative_density_function(
    PDF: xr.DataArray,
    bin_edges: Union[xr.DataArray, np.array] = None,
) -> xr.DataArray:
    """Calculate cumulative density function of pair separation r
    
    Parameters
    ----------
    PDF: xarray.DataArray
        Probability density function of pair separations.
    bin_edges: numpy.array
        1D array of bin edges (N+1 length).
    
    Returns
    -------
    CDF: xarray.DataArray
        Cumulative density function.
    """
    if bin_edges is None:
        values = PDF['rbin'].diff('rbin').values.astype(PDF.dtype)
        values = np.insert(values, 0, values[0])
        bin_width = xr.DataArray(values, dims='rbin', coords={'rbin':PDF['rbin'].values})
    else:
        bin_width = xr.DataArray(np.diff(bin_edges).astype(PDF.dtype), dims='rbin',
                                 coords={'rbin':PDF['rbin'].values})
    
    return (PDF * bin_width).cumsum('rbin').rename('CDF').astype(PDF.dtype)


"""
Helper methods are defined below
"""
def principle_axis_components(
    rx2m: xr.DataArray,
    ry2m: xr.DataArray,
    rxym: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculate principle axis components
    
    Parameters
    ----------
    rx2m: xr.DataArray
        Zonal component of dispersion <rx^2>
    ry2m: xr.DataArray
        Meridional component of dispersion <ry^2>
    rxym: xr.DataArray
        Cross component of dispersion <rxy>
    
    Returns
    -------
    ra: xr.DataArray
        Major component of separation.
    rb: xr.DataArray
        Minor component of separation.
    ang: xr.DataArray
        Angle between major and zonal components.
    """
    ra2 = ((rx2m + ry2m + np.sqrt((rx2m - ry2m)**2 + 4 * rxym**2)) / 2.0).astype(rx2m.dtype)
    rb2 = (rx2m + ry2m - ra2).astype(rx2m.dtype)
    ang = np.arctan2(ra2 - rx2m, rxym).astype(rx2m.dtype)
    
    return ra2, rb2, ang


def rotational_divergent_components(
    S2ll: xr.DataArray,
    S2tr: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculate rotational and divergent components of velocity structure function
    
    Parameters
    ----------
    S2ll: xr.DataArray
        Longitudinal component of 2nd-order velocity structure function.
    S2tr: xr.DataArray
        Transversal component of 2nd-order velocity structure function.
    
    Returns
    -------
    S2rr: xr.DataArray
        Rotational component of 2nd-order velocity structure function.
    S2dd: xr.DataArray
        Divergent component of 2nd-order velocity structure function.
    """
    rr = S2ll.rbin
    
    S2rr = (S2tr + ((S2tr - S2ll)/rr*rr.diff('rbin')).cumsum('rbin')).astype(S2ll.dtype)
    S2dd = (S2ll - ((S2tr - S2ll)/rr*rr.diff('rbin')).cumsum('rbin')).astype(S2ll.dtype)
    
    return S2rr.rename('S2rr'), S2dd.rename('S2dd')
