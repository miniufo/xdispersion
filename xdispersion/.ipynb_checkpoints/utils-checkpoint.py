# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from typing import Union, Optional, List, Dict, Tuple, Callable
from xhistogram.xarray import histogram
import multiprocessing
import concurrent.futures


"""
Core classes are defined below
"""
# @nb.jit(nopython=True, cache=False)
def geodist(
    lon1: Union[xr.DataArray, np.ndarray],
    lon2: Union[xr.DataArray, np.ndarray],
    lat1: Union[xr.DataArray, np.ndarray],
    lat2: Union[xr.DataArray, np.ndarray]
) -> Union[xr.DataArray, np.ndarray]:
    """Calculate great-circle distance on a unit sphere.
    
    Parameters
    ----------
    lon1: float or numpy.ndarray
        longitude of particle 1 (in radian).
    lon2: float or numpy.ndarray
        longitude of particle 2 (in radian).
    lat1: float or numpy.ndarray
        latitude of particle 1 (in radian).
    lat2: float or numpy.ndarray
        latitude of particle 2 (in radian).
    
    Returns
    -------
    dis: float or numpy.ndarray
        Great-circle distance.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2.0 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2.0
    
    dis = 2.0 * np.arcsin(np.sqrt(a))
    
    return dis

def bootstrap(
    func: Callable,
    args: List[xr.DataArray],
    kwargs: Dict,
    ensemble: int = 1000,
    CI: int = 0.95,
    nproc: int = 1,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculate standard error and confidence interval

    This is designed as repeating 'measure = func(params)' ensemble times.

    Both standard error and CI are obtained through bootstrapping.

    Reference:
    https://www.dummies.com/article/academics-the-arts/science/biology/the-bootstrap-method-for-standard-errors-and-confidence-intervals-164614/
    https://www.stat.cmu.edu/~ryantibs/advmethods/notes/bootstrap.pdf
    https://www.schmidheiny.name/teaching/bootstrap2up.pdf
    
    Parameters
    ----------
    func: function
        A function that transform a group of samples into a metric.
    args: list of xarray.DataArray
        A set of given parameters for func.
    kwargs: dict
        A set of keyword parameters for func.
    ensemble: int
        How many times bootstrapping is done.
    CI: float
        Confidence interval.
    nproc: int
        Number of process.
    
    Returns
    -------
    CIL: xr.DataArray
        Lower bound of confidence intervals.
    CIU: xr.DataArray
        Upper bound of confidence intervals.
    """
    if not isinstance(args, list):
        raise Exception('args should be a list of xr.DataArray')
    
    size = len(args[0]['pair'])

    if nproc == 1:
        tmp = []
        for i in range(ensemble):
            indices  = np.random.choice(range(size), size=size, replace=True)
            resample = [arg.isel({'pair':indices}) for arg in args]
            metrics  = func(*resample, **kwargs)
            tmp.append(metrics)
    elif nproc > 1:
        def _bootstrap_task():
            indices  = np.random.choice(range(size), size=size, replace=True)
            resample = [arg.isel({'pair':indices}) for arg in args]
            metric   = func(*resample, **kwargs)
            return metric
        
        # with multiprocessing.Pool(processes=nproc) as pool:
        #     tmp = [pool.apply(_bootstrap_task) for i in range(ensemble)]
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:
            # Submit tasks to the thread pool
            futures = [executor.submit(_bootstrap_task) for i in range(ensemble)]
            
            # Retrieve results as they complete
            tmp = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        raise Exception(f'invalid nproc={nproc}, should be a positive integer')
    
    re = xr.concat(tmp, dim='_ensem')
    re['_ensem'] = np.arange(ensemble)
    
    half = (1.0 - CI) / 2.0
    qntl = re.quantile([half, 1.0 - half], dim='_ensem', skipna=True)
    #stde = re.std('_ensem')
    #emn  = re.mean('_ensem')
    #CIL  = (2.0 * emn - qntl.isel(quantile=1)).drop_vars('quantile')
    #CIU  = (2.0 * emn - qntl.isel(quantile=0)).drop_vars('quantile')
    CIL  = qntl.isel(quantile=0).drop_vars('quantile')
    CIU  = qntl.isel(quantile=1).drop_vars('quantile')
    
    return CIL, CIU

def gen_rbins(
    r_lower: float,
    r_upper: float,
    alpha: float = 1.2,
    thre: Optional[float] = None,
    incre: Optional[float] = None
) -> xr.DataArray:
    """Generate a set of separation bins
    
    rbins(n) = r_lower ^ alpha**n, if only log-scale;
    
    If thre, incre are provide, a set of linear-scale bins will
    be generated after thre, with an increment of incre.
    
    Parameters
    ----------
    r_lower: float
        Lower bound of separation.
    r_upper: float
        Upper bound of separation.
    alpha: float
        Increment of neighbouring bins.
    thre: float
        Threshold separates the log and linear scales.
    incre: float
        Increment of neighbouring bins.
    
    Returns
    -------
    rbins: numpy.array
        Separation bins which is uniform in a log scale.
    """
    if thre == None:
        num = np.log(r_upper/r_lower) / np.log(alpha)
        n   = np.arange(1, np.floor(num)+1)
        rbins = r_lower * alpha**n
        rbins = np.insert(rbins, 0, r_lower)
        rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})

        assert (rbins[1:].values / rbins[:-1].values != alpha).any()
        
    else:
        num = np.log(thre/r_lower) / np.log(alpha)
        n   = np.arange(1, np.floor(num)+1)
        rbins = r_lower * alpha**n
        rbins = np.hstack([np.insert(rbins, 0, r_lower), np.linspace(thre, r_upper, int((r_upper-thre)/incre))])
        rbins = xr.DataArray(rbins, dims='rbin', coords={'rbin':rbins})
    
    return rbins.rename('rbins')


def semilog_fit(
    t: np.array,
    y: np.array
) -> Tuple[float, float, float]:
    """semi-log fit of a timeseries y=f(t)
    
    Parameters
    ----------
    t: numpy.ndarray
        Relative time axis.
    y: numpy.ndarray
        Values at each time.
    
    Returns
    -------
    slope: numpy.ndarray
        Slope of the semi-log fit.
    inter: numpy.ndarray
        Intersection of the fit.
    rmse: numpy.ndarray
        Root mean squared error of the fit.
    """
    idx = ~np.isnan(y)

    # select non-nan points to fit
    yy = y[idx]
    tt = t[idx]
    
    if len(yy) > 1:
        try:
            if tt[0] == 0:
                tt[0] = 1e-20
            slope, inter = np.polyfit(np.log(tt), yy, 1)
        except:
            print('polyfit error')
            return np.nan, np.nan, np.nan
        fitted = slope * tt + inter;
        rmse = np.sqrt(np.sum((fitted - yy) ** 2.0) / len(tt));
        
        return slope, inter, rmse
    else:
        return np.nan, np.nan, np.nan


def mean_at_rbin(
    var: xr.DataArray,
    r: xr.DataArray,
    rbins: xr.DataArray,
    cond: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Average at constant separation bin (rbin)
    
    Parameters
    ----------
    var: xr.DataArray
        A given variable.
    r: xr.DataArray
        Observed pair separation.
    rbin: xr.DataArray or numpy.array
        A set of prescribed separation bins
    cond: xr.DataArray
        An extra condition for selecting samples
    """
    ones = var - var + 1
    
    if cond is None:
        wei = ones
    else:
        wei = xr.where(cond, 1, 0)

    rbv = rbins.values if type(rbins) is xr.DataArray else rbins
    
    mean = histogram(r.rename('rtmp'), bins=rbv, weights=(var *wei).rename('vtmp'), block_size=1) \
         / histogram(r.rename('rtmp'), bins=rbv, weights=(ones*wei).rename('otmp'), block_size=1)
    
    mean['rtmp_bin'] = rbv[:-1]
    
    return mean.rename({'rtmp_bin':'rbin'})


def sum_at_rbin(
    var: xr.DataArray,
    r: xr.DataArray,
    rbins: xr.DataArray,
    cond: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Sum-up at constant separation bin (rbin)
    
    Parameters
    ----------
    var: xr.DataArray
        A given variable.
    r: xr.DataArray
        Observed pair separation.
    rbin: xr.DataArray or numpy.array
        A set of prescribed separation bins
    cond: xr.DataArray
        An extra condition for selecting samples
    """
    ones = var - var + 1
    
    if cond is None:
        wei = ones
    else:
        wei = xr.where(cond, 1, 0)

    rbv = rbins.values if type(rbins) is xr.DataArray else rbins
    
    total = histogram(r.rename('rtmp'), bins=rbv, weights=(var *wei).rename('vtmp'), block_size=1)
    
    total['rtmp_bin'] = rbv[:-1]
    
    return total.rename({'rtmp_bin':'rbin'})


def get_overlap_indices(
    times1: np.array,
    times2: np.array
) -> Tuple[int, int, int, int]:
    """get overlapping indices.
    
    Parameters
    ----------
    times1: numpy.array
        times of trajectory 1.
    times2: numpy.array
        times of trajectory 2.
    
    Returns
    -------
    i1, i2, j1, j2: int
        i1 and i2 are start and end indices for trajectory 1;
        j1 and j2 are start and end indices for trajectory 2
        one can slice using times[i1:i2], so the i2 is exclusive!!!
    """
    # cases for non-overlap
    if times2[0] > times1[-1]:
        return None, None, None, None

    # cases for non-overlap
    if times2[-1] < times1[0]:
        return None, None, None, None

    lts, sts = times1, times2

    switch = False
    if len(times1) < len(times2):
        switch = True
        lts = times2 # longer
        sts = times1 # shorter
    
    strIdx = np.where(lts == sts[ 0])[0]
    endIdx = np.where(lts == sts[-1])[0]

    hasStr = False
    hasEnd = False

    if len(strIdx) != 0:
        hasStr = True
        strIdx = strIdx[0]

    if len(endIdx) != 0:
        hasEnd = True
        endIdx = endIdx[0]

    if hasStr and hasEnd:
        if switch:
            i1, i2, j1, j2 = 0, len(sts), strIdx, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = strIdx, endIdx+1, 0, len(sts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
    
    if (not hasStr) and hasEnd:
        strIdx = np.where(sts == lts[ 0])[0][0]
        if switch:
            i1, i2, j1, j2 = strIdx, len(sts), 0, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = 0, endIdx+1, strIdx, len(sts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2

    if hasStr and (not hasEnd):
        endIdx = np.where(sts == lts[-1])[0][0]
        if switch:
            i1, i2, j1, j2 = 0, endIdx+1, strIdx, len(lts)
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2
        else:
            i1, i2, j1, j2 = strIdx, len(lts), 0, endIdx+1
            _check_indices(i1, i2, j1, j2)
            return i1, i2, j1, j2

    raise Exception(f'should not reach here: {hasStr}, {hasEnd}, {switch}')


"""
Helper (private) methods are defined below
"""
def _check_indices(
    i1: int,
    i2: int,
    j1: int,
    j2: int
):
    if i2 - i1 != j2 - j1:
        raise Exception('invalid indices')


