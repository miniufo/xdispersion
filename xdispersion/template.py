# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.

Module: xdispersion.template
============================

High-level orchestration for computing all (or a subset of) diagnostic
measures in one call.

**MEASURES registry**

A global list ``MEASURES`` maps measure names (e.g. ``'r2_t'``,
``'FSLE_r'``) to callables that take building-block arrays and
keyword arguments.  :func:`cal_measures` iterates over this registry,
calls each measure function, and assembles the results into a single
:class:`xarray.Dataset`.

**cal_measures options**

- ``one_by_one=True``:  compute and ``.load()`` each measure
  immediately, minimising peak memory but recomputing building blocks
  for each measure (use with ``chunk > 0`` and dask-backed data).
- ``one_by_one=False``:  build the full dask graph for all measures
  and compute at once (faster, higher peak memory).
- ``measures=[...]``:  compute only the named measures.
- ``ensemble > 0``:  bootstrap confidence intervals.
"""
import numpy as np
import xarray as xr
import warnings
from tqdm import tqdm
from .measures import relative_dispersion, velocity_structure_function,\
    relative_diffusivity, finite_amplitude_growth_rate, initial_memory, anisotropy,\
    lagrangian_velocity_correlation, kurtosis, cencini_vulpiani_exponent, finite_size_lyapunov_exponent,\
    cumulative_inverse_separation_time, probability_density_function, cumulative_density_function,\
    rotational_divergent_components
from .utils import mean_at_rbin, sum_at_rbin

"""
A template for calculating all/some available measures.
Supports:
- Using global measure registry (default)
- Passing custom measure lists (dict format)
- Passing variable name (string) lists (e.g., ["r2_t", "S2_r"])
"""

# ==================================================
# Measure registry (ONLY edit here for new measures)
# ==================================================
MEASURES = []

def add_measure(name, func, comment, *, kind="const-t"):
    MEASURES.append({
        "name": name,
        "func": func,
        "comment": comment,
        "kind": kind,          # const-t / const-r / other
    })


# ======= const-t measures =======
add_measure("r2_t",   lambda r, **kw: relative_dispersion(r, **kw),
         "relative dispersion averaged at constant time")
add_measure("rpb2_t", lambda r, rpb, **kw: relative_dispersion(rpb, **kw),
         "perturbation dispersion averaged at constant time")
add_measure("S2_t", lambda r, du, dv, **kw: velocity_structure_function(np.hypot(du, dv), r, **kw),
         "2nd-order velocity structure function averaged at constant time")
add_measure("S2ll_t", lambda r, dul, **kw: velocity_structure_function(dul, r, **kw),
         "2nd-order longitudinal structure function averaged at constant time")
add_measure("S2tr_t", lambda r, dut, **kw: velocity_structure_function(dut, r, **kw),
         "2nd-order transversal structure function averaged at constant time")
add_measure("S3_t", lambda r, du, dv, dul, **kw: velocity_structure_function(dul*(du**2+dv**2), r, **kw, order=1),
         "3rd-order velocity structure function averaged at constant time")
add_measure("a2_t", lambda r, dax, day, **kw: velocity_structure_function(np.hypot(dax, day), r, **kw),
         "2nd-order acceleration structure function averaged at constant time", kind="const-t")
add_measure("K2_t", lambda r, **kw: relative_diffusivity(r, **kw, samples='all'),
         "relative diffusivity averaged at constant time")
add_measure("K2p_t", lambda r, **kw: relative_diffusivity(r, **kw, samples='abs'),
         "absolute value of relative diffusivity averaged at constant time")
add_measure("FAGR_t", lambda r, **kw: finite_amplitude_growth_rate(r, **kw, samples='all'),
         "finite-amplitude growth rate averaged at constant time")
add_measure("FAGRp_t", lambda r, **kw: finite_amplitude_growth_rate(r, **kw, samples='positive'),
         "positive finite-amplitude growth rate averaged at constant time")
add_measure("imrv_t", lambda r, rx, ry, du, dv, **kw: initial_memory(rx, ry, du, dv, r, **kw),
         "initial memory <rv> averaged at constant time")
add_measure("imrv2_t", lambda r, rx, ry, du, dv, **kw: initial_memory(rx, ry, du, dv, r, **kw, order=2),
         "initial memory <(rv)^2> averaged at constant time")
add_measure("imra_t", lambda r, rx, ry, dax, day, **kw: initial_memory(rx, ry, dax, day, r, **kw),
         "initial memory <ra> averaged at constant time")
add_measure("aniso_t", lambda r, rx, ry, rxy, **kw: anisotropy(rx, ry, rxy, r, **kw),
         "anisotropy averaged at constant time")
add_measure("LVC_t", lambda r, uv, vmi, vmj, **kw: lagrangian_velocity_correlation(uv, vmi, vmj, r, **kw),
         "Lagrangian velocity correlation averaged at constant time")
add_measure("LAC_t", lambda r, axy, ai, aj, **kw: lagrangian_velocity_correlation(axy, ai, aj, r, **kw),
         "Lagrangian acceleration correlation averaged at constant time")
add_measure("ku_t", lambda r, **kw: kurtosis(r, **kw),
         "kurtosis averaged at constant time")
add_measure("kupb_t", lambda r, rpb, **kw: kurtosis(rpb, **kw),
         "perturbation kurtosis averaged at constant time")
add_measure("CVE_t", lambda r, **kw: cencini_vulpiani_exponent(r, **kw),
         "Cencini-Vulpiani exponent averaged at constant time")

# ======= const-r measures =======
add_measure("r2_r", lambda r, **kw: relative_dispersion(r, **kw, order=2),
         "relative dispersion averaged at constant separation", kind="const-r")
add_measure("rpb2_r", lambda r, rpb, **kw: relative_dispersion(rpb, **kw, order=2),
         "perturbation dispersion averaged at constant separation", kind="const-r")
add_measure("S2_r", lambda r, du, dv, **kw: velocity_structure_function(np.hypot(du, dv), r, **kw),
         "2nd-order velocity structure function averaged at constant separation", kind="const-r")
add_measure("S2ll_r", lambda r, dul, **kw: velocity_structure_function(dul, r, **kw),
         "2nd-order longitudinal structure function averaged at constant separation", kind="const-r")
add_measure("S2tr_r", lambda r, dut, **kw: velocity_structure_function(dut, r, **kw),
         "2nd-order transversal structure function averaged at constant separation", kind="const-r")
add_measure("S3_r", lambda r, du, dv, dul, **kw: velocity_structure_function(dul*(du**2+dv**2), r, **kw, order=1),
         "3rd-order velocity structure function averaged at constant separation", kind="const-r")
add_measure("a2_r", lambda r, dax, day, **kw: velocity_structure_function(np.hypot(dax, day), r, **kw),
         "2nd-order acceleration structure function averaged at constant separation", kind="const-r")
add_measure("K2_r", lambda r, **kw: relative_diffusivity(r, **kw, samples='all'),
         "relative diffusivity averaged at constant separation", kind="const-r")
add_measure("K2p_r", lambda r, **kw: relative_diffusivity(r, **kw, samples='abs'),
         "absolute value of relative diffusivity averaged at constant separation", kind="const-r")
add_measure("FAGR_r", lambda r, **kw: finite_amplitude_growth_rate(r, **kw, samples='all'),
         "finite-amplitude growth rate averaged at constant separation", kind="const-r")
add_measure("FAGRp_r", lambda r, **kw: finite_amplitude_growth_rate(r, **kw, samples='positive'),
         "positive finite-amplitude growth rate averaged at constant separation", kind="const-r")
add_measure("imrv_r", lambda r, rx, ry, du, dv, **kw: initial_memory(rx, ry, du, dv, r, **kw),
         "initial memory <rv> averaged at constant separation", kind="const-r")
add_measure("imrv2_r", lambda r, rx, ry, du, dv, **kw: initial_memory(rx, ry, du, dv, r, **kw, order=2),
         "initial memory <(rv)^2> averaged at constant separation", kind="const-r")
add_measure("imra_r", lambda r, rx, ry, dax, day, **kw: initial_memory(rx, ry, dax, day, r, **kw),
         "initial memory <ra> averaged at constant separation", kind="const-r")
add_measure("aniso_r", lambda r, rx, ry, rxy, **kw: anisotropy(rx, ry, rxy, r, **kw),
         "anisotropy averaged at constant separation", kind="const-r")
add_measure("LVC_r", lambda r, uv, vmi, vmj, **kw: lagrangian_velocity_correlation(uv, vmi, vmj, r, **kw),
         "Lagrangian velocity correlation averaged at constant separation", kind="const-r")
add_measure("LAC_r", lambda r, axy, ai, aj, **kw: lagrangian_velocity_correlation(axy, ai, aj, r, **kw),
         "Lagrangian acceleration correlation averaged at constant separation", kind="const-r")
add_measure("CVE_r", lambda r, **kw: cencini_vulpiani_exponent(r, **kw),
         "Cencini-Vulpiani exponent averaged at constant separation", kind="const-r")
add_measure("FSLE_r", lambda r, **kw: finite_size_lyapunov_exponent(r, **kw),
         "finite-size Lyapunov exponent averaged at constant separation", kind="const-r")
add_measure("CIST_r", lambda r, **kw: cumulative_inverse_separation_time(r, **kw),
         "cumulative inverse separation time averaged at constant separation", kind="const-r")

# ======= derived measures =======
MEASURES.append({
    "name": "S2rr_r",
    "comment": "rotational component of 2nd-order structure function",
    "derived": True
})
MEASURES.append({
    "name": "S2dd_r",
    "comment": "divergent component of 2nd-order structure function",
    "derived": True
})


def cal_measures(rd, pairs, rbins, ensemble=0, nproc=1, one_by_one=False, measures=None, debug=False):
    """
    Calculate all/some measures.
    
    Parameters
    ----------
    rd: RelativeDispersion instance
    pairs: list of two xr.Datasets
    rbins: xr.DataArray
    ensemble: int, optional
        Number of bootstrap ensembles (0 = no bootstrap, no CIs)
    nproc: int, optional
        Number of processes for parallel computation
    one_by_one: bool, optional
        Whether to compute each measure immediately (reduces memory usage)
    measures: list, optional
        - None: use all global MEASURES (default)
        - list of strings: variable names (e.g., ["r2_t", "S2_r"])
        - list of dicts: custom measure definitions (same format as MEASURES)
    
    Returns
    -------
    dset: xr.Dataset
        All calculated measures
    """
    # Process measures parameter
    if measures is None:
        diags_to_use = MEASURES
    else:
        if all(isinstance(m, str) for m in measures):
            # Pass variable name list to filter global measures
            # Check for unknown measure names
            known_names = {m["name"] for m in MEASURES}
            unknown_names = [m for m in measures if m not in known_names]
            if unknown_names:
                warnings.warn(
                    f"Unknown measure(s) ignored: {unknown_names}. "
                    f"Available measures are: {sorted(known_names)}",
                    UserWarning
                )

            # preseving user order while filtering
            diags_to_use = []
            for name in measures:
                if name in known_names:
                    # find the measure in MEASURES
                    measure = next(m for m in MEASURES if m["name"] == name)
                    diags_to_use.append(measure)
        else:
            # Pass custom measure list -- use directly
            diags_to_use = measures

    # Get basic data (local variables).
    # When building blocks are dask-backed (e.g. ragged with chunk or
    # non-ragged), .persist() computes them once and caches the result
    # in dask memory so that subsequent per-measure .load() calls do not
    # re-evaluate the entire dask graph.
    rx, ry, rxy, r, rpb = rd.separation_measures(pairs)
    du, dv, dul, dut, vmi, vmj, uv = rd.velocity_measures(pairs)
    dax, day, dal, dat, ai, aj, axy = rd.acceleration_measures(pairs)

    _bb = [rx, ry, rxy, r, rpb, du, dv, dul, dut, vmi, vmj, uv,
           dax, day, dal, dat, ai, aj, axy]
    if any(hasattr(v, 'data') and hasattr(v.data, 'dask') for v in _bb):
        rx, ry, rxy, r, rpb, du, dv, dul, dut, vmi, vmj, uv, \
            dax, day, dal, dat, ai, aj, axy = [v.persist() for v in _bb]

    results = []

    with tqdm(total=len(diags_to_use), ncols=80) as pbar:
        for measure in diags_to_use:
            if debug:
                print(measure)
            # Handle derived measures (e.g., S2rr_r/S2dd_r)
            if measure.get("derived"):
                if measure["name"] == "S2rr_r":
                    S2ll = next(r for r in results if r.name == "S2ll_r")
                    S2tr = next(r for r in results if r.name == "S2tr_r")
                    da = rotational_divergent_components(S2ll, S2tr)[0]
                elif measure["name"] == "S2dd_r":
                    S2ll = next(r for r in results if r.name == "S2ll_r")
                    S2tr = next(r for r in results if r.name == "S2tr_r")
                    da = rotational_divergent_components(S2ll, S2tr)[1]
                else:
                    continue
                da.name = measure["name"]
                da.attrs["comment"] = measure["comment"]
                results.append(da)
                pbar.update(1)
                continue

            # Prepare keyword arguments for the measure function
            kwargs = dict(mean_at=measure["kind"], ensemble=ensemble, nproc=nproc)
            if measure["kind"] == "const-r":
                kwargs["rbins"] = rbins
            elif measure["kind"] != "const-t":
                raise Exception(f"unsupported kind {measure['kind']}, should be one of [const-t, const-r]")

            # Call the measure function with explicit arguments
            # Each measure's lambda function defines exactly what parameters it needs
            if measure["name"] in ["r2_t", "K2_t", "K2p_t", "FAGR_t", "FAGRp_t", "ku_t", "CVE_t", 
                                  "r2_r", "K2_r", "K2p_r", "FAGR_r", "FAGRp_r", "CVE_r", "FSLE_r", "CIST_r"]:
                out = measure["func"](r, **kwargs)
            elif measure["name"] in ["rpb2_t", "kupb_t", "rpb2_r"]:
                out = measure["func"](r, rpb, **kwargs)
            elif measure["name"] in ["S2_t", "S2_r"]:
                out = measure["func"](r, du, dv, **kwargs)
            elif measure["name"] in ["S2ll_t", "S2ll_r"]:
                out = measure["func"](r, dul, **kwargs)
            elif measure["name"] in ["S2tr_t", "S2tr_r"]:
                out = measure["func"](r, dut, **kwargs)
            elif measure["name"] in ["S3_t", "S3_r"]:
                out = measure["func"](r, du, dv, dul, **kwargs)
            elif measure["name"] in ["a2_t", "a2_r"]:
                out = measure["func"](r, dax, day, **kwargs)
            elif measure["name"] in ["imrv_t", "imrv_r"]:
                out = measure["func"](r, rx, ry, du, dv, **kwargs)
            elif measure["name"] in ["imrv2_t", "imrv2_r"]:
                out = measure["func"](r, rx, ry, du, dv, **kwargs)
            elif measure["name"] in ["imra_t", "imra_r"]:
                out = measure["func"](r, rx, ry, dax, day, **kwargs)
            elif measure["name"] in ["aniso_t", "aniso_r"]:
                out = measure["func"](r, rx, ry, rxy, **kwargs)
            elif measure["name"] in ["LVC_t", "LVC_r"]:
                out = measure["func"](r, uv, vmi, vmj, **kwargs)
            elif measure["name"] in ["LAC_t", "LAC_r"]:
                out = measure["func"](r, axy, ai, aj, **kwargs)
            else:
                # Fallback for any other measures
                out = measure["func"](r, **kwargs)

            # Process results based on actual output type
            if isinstance(out, tuple) and len(out) == 3:
                da, cil, ciu = out
                da.name = measure["name"]
                da.attrs["comment"] = measure["comment"]
                cil.name = f"CIL{measure['name']}"
                ciu.name = f"CIU{measure['name']}"
                results.extend([da, cil, ciu])
            else:
                da = out
                da.name = measure["name"]
                da.attrs["comment"] = measure["comment"]
                results.append(da)

            if one_by_one:
                da.load()
            pbar.update(1)

    return xr.merge(results)



"""
A template for calculating all the available
measures given a set of pair particles, and
return all as a xarray.Dataset that can be
easily output to a file.
"""

def cal_all_measures_bak(rd, pairs, rbins, ensemble=0, nproc=1, one_by_one=False):
    """Calculate all available measures.  Users can add their own measures.
    
    Parameters
    ----------
    rd: RelativeDispersion instance
        Relative dispersion application
    pairs: list of two xr.Datasets
        Pair particles
    rbins: xr.DataArray
        A specified separation bins for r-based measures
    one_by_one: boolean
        Whether calculating the measures one by one.
        If False, a dask dataset is returned quickly but later .compute() may use more memory but is faster;
        If True, each measure will be .compute() before the next one, which use more time but less memory.
    
    Returns
    -------
    dset: xr.Dataset
        All measures in a single xr.Dataset.
    """
    #-------------------- get building-blocks for different measures -----------------#
    rx, ry, rxy, r, rpb = rd.separation_measures(pairs)
    du, dv, dul, dut, vmi, vmj, uv = rd.velocity_measures(pairs)
    dax, day, dal, dat, ai, aj, axy = rd.acceleration_measures(pairs)
    
    with tqdm(total=40, ncols=80) as pbar:
        if ensemble > 0:
            #----------------------- measures averaged at constant time --------------------#
            mean_at = 'const-t'
            r2_t, CILr2_t, CIUr2_t = relative_dispersion(r, order=2,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: r2_t.load()
            pbar.update(1)
            rpb2_t, CILrpb2_t, CIUrpb2_t = relative_dispersion(rpb, order=2,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: rpb2_t.load()
            pbar.update(1)
            S2_t, CILS2_t, CIUS2_t = velocity_structure_function(np.hypot(du, dv), r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2_t.load()
            pbar.update(1)
            S2ll_t, CILS2ll_t, CIUS2ll_t = velocity_structure_function(dul, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2ll_t.load()
            pbar.update(1)
            S2tr_t, CILS2tr_t, CIUS2tr_t = velocity_structure_function(dut, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2tr_t.load()
            pbar.update(1)
            S3_t, CILS3_t, CIUS3_t = velocity_structure_function(dul*(du**2+dv**2), r, order=1,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S3_t.load()
            pbar.update(1)
            K2_t, CILK2_t, CIUK2_t = relative_diffusivity(r, samples='all',
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: K2_t.load()
            pbar.update(1)
            K2p_t, CILK2p_t, CIUK2p_t = relative_diffusivity(r, samples='abs',
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: K2p_t.load()
            pbar.update(1)
            FAGR_t, CILFAGR_t, CIUFAGR_t = finite_amplitude_growth_rate(r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: FAGR_t.load()
            pbar.update(1)
            imrv_t, CILimrv_t, CIUimrv_t = initial_memory(rx, ry, du, dv, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imrv_t.load()
            pbar.update(1)
            imrv2_t, CILimrv2_t, CIUimrv2_t = initial_memory(rx, ry, du, dv, r, order=2,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imrv2_t.load()
            pbar.update(1)
            imra_t, CILimra_t, CIUimra_t = initial_memory(rx, ry, dax, day, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imra_t.load()
            pbar.update(1)
            aniso_t, CILaniso_t, CIUaniso_t = anisotropy(rx, ry, rxy, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: aniso_t.load()
            pbar.update(1)
            LVC_t, CILLVC_t, CIULVC_t = lagrangian_velocity_correlation(uv, vmi, vmj, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: LVC_t.load()
            pbar.update(1)
            LAC_t, CILLAC_t, CIULAC_t = lagrangian_velocity_correlation(axy, ai, aj, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: LAC_t.load()
            pbar.update(1)
            ku_t, CILku_t, CIUku_t = kurtosis(r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: ku_t.load()
            pbar.update(1)
            kupb_t, CILkupb_t, CIUkupb_t = kurtosis(rpb,
                                                    mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: kupb_t.load()
            pbar.update(1)
            CVE_t, CILCVE_t, CIUCVE_t = cencini_vulpiani_exponent(r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: CVE_t.load()
            pbar.update(1)
            num_t = xr.where(np.isnan(r), 0, 1).sum('pair')
            if one_by_one: num_t.load()
        
            #--------------------- measures average at constant separation ------------------#
            mean_at = 'const-r'
            r2_r, CILr2_r, CIUr2_r = relative_dispersion(r, order=2, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: r2_r.load()
            pbar.update(1)
            rpb2_r, CILrpb2_r, CIUrpb2_r = relative_dispersion(rpb, order=2, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: rpb2_r.load()
            pbar.update(1)
            S2_r, CILS2_r, CIUS2_r = velocity_structure_function(np.hypot(du, dv), r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2_r.load()
            pbar.update(1)
            S2ll_r, CILS2ll_r, CIUS2ll_r = velocity_structure_function(dul, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2ll_r.load()
            pbar.update(1)
            S2tr_r, CILS2tr_r, CIUS2tr_r = velocity_structure_function(dut, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S2tr_r.load()
            pbar.update(1)
            S3_r, CILS3_r, CIUS3_r = velocity_structure_function(dul*(du**2+dv**2), r, order=1, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: S3_r.load()
            pbar.update(1)
            K2_r, CILK2_r, CIUK2_r = relative_diffusivity(r, rbins=rbins, samples='all',
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: K2_r.load()
            pbar.update(1)
            K2p_r, CILK2p_r, CIUK2p_r = relative_diffusivity(r, rbins=rbins, samples='abs',
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: K2p_r.load()
            pbar.update(1)
            FAGR_r, CILFAGR_r, CIUFAGR_r = finite_amplitude_growth_rate(r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: FAGR_r.load()
            pbar.update(1)
            imrv_r, CILimrv_r, CIUimrv_r = initial_memory(rx, ry, du, dv, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imrv_r.load()
            pbar.update(1)
            imrv2_r, CILimrv2_r, CIUimrv2_r = initial_memory(rx, ry, du, dv, r, order=2, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imrv2_r.load()
            pbar.update(1)
            imra_r, CILimra_r, CIUimra_r = initial_memory(rx, ry, dax, day, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: imra_r.load()
            pbar.update(1)
            aniso_r, CILaniso_r, CIUaniso_r = anisotropy(rx, ry, rxy, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: aniso_r.load()
            pbar.update(1)
            LVC_r, CILLVC_r, CIULVC_r = lagrangian_velocity_correlation(uv, vmi, vmj, r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: LVC_r.load()
            pbar.update(1)
            LAC_r, CILLAC_r, CIULAC_r = lagrangian_velocity_correlation(axy, ai, aj, r,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: LAC_r.load()
            pbar.update(1)
            CVE_r, CILCVE_r, CIUCVE_r = cencini_vulpiani_exponent(r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: CVE_r.load()
            pbar.update(1)
            FSLE_r, CILFSLE_r, CIUFSLE_r = finite_size_lyapunov_exponent(r, rbins=rbins,
                                              mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: FSLE_r.load()
            pbar.update(1)
            CIST_r, CILCIST_r, CIUCIST_r = cumulative_inverse_separation_time(r, rbins=rbins, lower=0.10, upper=0.90,
                                              maskout=[1e-8, 5e3], mean_at=mean_at, ensemble=ensemble, nproc=nproc)
            if one_by_one: CIST_r.load()
            pbar.update(1)
            num_r = sum_at_rbin(xr.where(np.isnan(r), 0, 1), r, rbins=rbins)
            if one_by_one: num_r.load()
            pbar.update(1)
            
            #----------------------------- other measures --------------------------#
            S2rr_r, S2dd_r = rotational_divergent_components(S2ll_r, S2tr_r)
            if one_by_one: S2rr_r.load(); S2dd_r.load()
            pbar.update(1)
            PDF = probability_density_function(r, rbins=rbins)
            if one_by_one: PDF.load()
            pbar.update(1)
            CDF = cumulative_density_function(PDF)
            if one_by_one: CDF.load()
            pbar.update(1)
        
            #------------------------------- output list ----------------------------#
            vs = [r2_t   , CILr2_t   , CIUr2_t   , rpb2_t , CILrpb2_t , CIUrpb2_t ,
                  S2_t   , CILS2_t   , CIUS2_t   , K2_t   , CILK2_t   , CIUK2_t   ,
                  S2ll_t , CILS2ll_t , CIUS2ll_t , S2tr_t , CILS2tr_t , CIUS2tr_t ,
                  S3_t   , CILS3_t   , CIUS3_t   , K2p_t  , CILK2p_t  , CIUK2p_t  ,
                  FAGR_t , CILFAGR_t , CIUFAGR_t , imrv_t , CILimrv_t , CIUimrv_t ,
                  imrv2_t, CILimrv2_t, CIUimrv2_t, imra_t , CILimra_t , CIUimra_t ,
                  aniso_t, CILaniso_t, CIUaniso_t, LVC_t  , CILLVC_t  , CIULVC_t  ,
                  LAC_t  , CILLAC_t  , CIULAC_t  ,
                  ku_t   , CILku_t   , CIUku_t   , kupb_t , CILkupb_t , CIUkupb_t ,
                  CVE_t  , CILCVE_t  , CIUCVE_t  , num_t  ,
                  r2_r   , CILr2_r   , CIUr2_r   , rpb2_r , CILrpb2_r , CIUrpb2_r ,
                  S2_r   , CILS2_r   , CIUS2_r   , K2_r   , CILK2_r   , CIUK2_r   ,
                  S2ll_r , CILS2ll_r , CIUS2ll_r , S2tr_r , CILS2tr_r , CIUS2tr_r ,
                  S3_r   , CILS3_r   , CIUS3_r   , K2p_r  , CILK2p_r  , CIUK2p_r  ,
                  FAGR_r , CILFAGR_r , CIUFAGR_r , imrv_r , CILimrv_r , CIUimrv_r ,
                  imrv2_r, CILimrv2_r, CIUimrv2_r, imra_r , CILimra_r , CIUimra_r ,
                  aniso_r, CILaniso_r, CIUaniso_r, LVC_r  , CILLVC_r  , CIULVC_r  ,
                  LAC_r  , CILLAC_r  , CIULAC_r  , CVE_r  , CILCVE_r  , CIUCVE_r  , num_r  ,
                  FSLE_r , CILFSLE_r , CIUFSLE_r , CIST_r , CILCIST_r , CIUCIST_r ,
                  S2rr_r , S2dd_r    , PDF       , CDF    ,
                  ]
            
            #------------------------------- output names ----------------------------#
            names = ['r2_t'   , 'CILr2_t'   , 'CIUr2_t'   , 'rpb2_t' , 'CILrpb2_t' , 'CIUrpb2_t' ,
                     'S2_t'   , 'CILS2_t'   , 'CIUS2_t'   , 'K2_t'   , 'CILK2_t'   , 'CIUK2_t'   ,
                     'S2ll_t' , 'CILS2ll_t' , 'CIUS2ll_t' , 'S2tr_t' , 'CILS2tr_t' , 'CIUS2tr_t' ,
                     'S3_t'   , 'CILS3_t'   , 'CIUS3_t'   , 'K2p_t'  , 'CILK2p_t'  , 'CIUK2p_t'  ,
                     'FAGR_t' , 'CILFAGR_t' , 'CIUFAGR_t' , 'imrv_t' , 'CILimrv_t' , 'CIUimrv_t' ,
                     'imrv2_t', 'CILimrv2_t', 'CIUimrv2_t', 'imra_t' , 'CILimra_t' , 'CIUimra_t' ,
                     'aniso_t', 'CILaniso_t', 'CIUaniso_t', 'LVC_t'  , 'CILLVC_t'  , 'CIULVC_t'  ,
                     'LAC_t'  , 'CILLAC_t'  , 'CIULAC_t'  ,
                     'ku_t'   , 'CILku_t'   , 'CIUku_t'   , 'kupb_t' , 'CILkupb_t' , 'CIUkupb_t' ,
                     'CVE_t'  , 'CILCVE_t'  , 'CIUCVE_t'  , 'num_t'  ,
                     'r2_r'   , 'CILr2_r'   , 'CIUr2_r'   , 'rpb2_r' , 'CILrpb2_r' , 'CIUrpb2_r' ,
                     'S2_r'   , 'CILS2_r'   , 'CIUS2_r'   , 'K2_r'   , 'CILK2_r'   , 'CIUK2_r'   ,
                     'S2ll_r' , 'CILS2ll_r' , 'CIUS2ll_r' , 'S2tr_r' , 'CILS2tr_r' , 'CIUS2tr_r' ,
                     'S3_r'   , 'CILS3_r'   , 'CIUS3_r'   , 'K2p_r'  , 'CILK2p_r'  , 'CIUK2p_r'  ,
                     'FAGR_r' , 'CILFAGR_r' , 'CIUFAGR_r' , 'imrv_r' , 'CILimrv_r' , 'CIUimrv_r' ,
                     'imrv2_r', 'CILimrv2_r', 'CIUimrv2_r', 'imra_r' , 'CILimra_r' , 'CIUimra_r' ,
                     'aniso_r', 'CILaniso_r', 'CIUaniso_r', 'LVC_r'  , 'CILLVC_r'  , 'CIULVC_r'  ,
                     'LAC_r'  , 'CILLAC_r'  , 'CIULAC_r'  , 'CVE_r'  , 'CILCVE_r'  , 'CIUCVE_r'  , 'num_r'  ,
                     'FSLE_r' , 'CILFSLE_r' , 'CIUFSLE_r' , 'CIST_r' , 'CILCIST_r' , 'CIUCIST_r' ,
                     'S2rr_r' , 'S2dd_r'    , 'PDF'       , 'CDF'    ,]
            
            #------------------------------- output comments ----------------------------#
            comments = ['relative dispersion averaged at constant time',
                        'lower bound for relative dispersion',
                        'upper bound for relative dispersion',
                        'perturbation dispersion averaged at constant time',
                        'lower bound for perturbation dispersion',
                        'upper bound for perturbation dispersion',
                        '2nd-order structure function averaged at constant time',
                        'lower bound for 2nd-order structure function',
                        'upper bound for 2nd-order structure function',
                        'relative diffusivity averaged at constant time',
                        'lower bound for relative diffusivity',
                        'upper bound for relative diffusivity',
                        '2nd-order longitudinal structure function averaged at constant time',
                        'lower bound for 2nd-order longitudinal structure function',
                        'upper bound for 2nd-order longitudinal structure function',
                        '2nd-order transversal structure function averaged at constant time',
                        'lower bound for 2nd-order transversal structure function',
                        'upper bound for 2nd-order transversal structure function',
                        '3rd-order structure function averaged at constant time',
                        'lower bound for 3rd-order structure function',
                        'upper bound for 3rd-order structure function',
                        'absolute relative diffusivity averaged at constant time',
                        'lower bound for absolute relative diffusivity',
                        'upper bound for absolute relative diffusivity',
                        'finite-amplitude growth rate averaged at constant time',
                        'lower bound for finite-amplitude growth rate',
                        'upper bound for finite-amplitude growth rate',
                        'initial memory rv averaged at constant time',
                        'lower bound for initial memory rv',
                        'upper bound for initial memory rv',
                        'initial memory rv2 averaged at constant time',
                        'lower bound for initial memory rv2',
                        'upper bound for initial memory rv2',
                        'initial memory ra averaged at constant time',
                        'lower bound for initial memory ra',
                        'upper bound for initial memory ra',
                        'anisotropy averaged at constant time',
                        'lower bound for anisotropy',
                        'upper bound for anisotropy',
                        'Lagrangian velocity correlation averaged at constant time',
                        'lower bound for Lagrangian velocity correlation',
                        'upper bound for Lagrangian velocity correlation',
                        'Lagrangian acceleration correlation averaged at constant time',
                        'lower bound for Lagrangian acceleration correlation',
                        'upper bound for Lagrangian acceleration correlation',
                        'kurtosis averaged at constant time',
                        'lower bound for kurtosis',
                        'upper bound for kurtosis',
                        'perturbation kurtosis averaged at constant time',
                        'lower bound for perturbation kurtosis',
                        'upper bound for perturbation kurtosis',
                        'Cencini-Vulpiani exponent averaged at constant time',
                        'lower bound for Cencini-Vulpiani exponent',
                        'upper bound for Cencini-Vulpiani exponent',
                        'number of samples at constant time',
                        #
                        'relative dispersion averaged at constant separation',
                        'lower bound for relative dispersion',
                        'upper bound for relative dispersion',
                        'perturbation dispersion averaged at constant separation',
                        'lower bound for perturbation dispersion',
                        'upper bound for perturbation dispersion',
                        '2nd-order structure function averaged at constant separation',
                        'lower bound for 2nd-order structure function',
                        'upper bound for 2nd-order structure function',
                        'relative diffusivity averaged at constant separation',
                        'lower bound for relative diffusivity',
                        'upper bound for relative diffusivity',
                        '2nd-order longitudinal structure function averaged at constant separation',
                        'lower bound for 2nd-order longitudinal structure function',
                        'upper bound for 2nd-order longitudinal structure function',
                        '2nd-order transversal structure function averaged at constant separation',
                        'lower bound for 2nd-order transversal structure function',
                        'upper bound for 2nd-order transversal structure function',
                        '3rd-order structure function averaged at constant separation',
                        'lower bound for 3rd-order transversal structure function',
                        'upper bound for 3rd-order transversal structure function',
                        'absolute relative diffusivity averaged at constant separation',
                        'lower bound for absolute relative diffusivity',
                        'upper bound for absolute relative diffusivity',
                        'finite-amplitude growth rate averaged at constant separation',
                        'lower bound for finite-amplitude growth rate',
                        'upper bound for finite-amplitude growth rate',
                        'initial memory rv averaged at constant separation',
                        'lower bound for initial memory rv',
                        'upper bound for initial memory rv',
                        'initial memory rv2 averaged at constant separation',
                        'lower bound for initial memory rv2',
                        'upper bound for initial memory rv2',
                        'initial memory ra averaged at constant separation',
                        'lower bound for initial memory ra',
                        'upper bound for initial memory ra',
                        'anisotropy averaged at constant separation',
                        'lower bound for anisotropy',
                        'upper bound for anisotropy',
                        'Lagrangian velocity correlation averaged at constant separation',
                        'lower bound for Lagrangian velocity correlation',
                        'upper bound for Lagrangian velocity correlation',
                        'Lagrangian acceleration correlation averaged at constant separation',
                        'lower bound for Lagrangian acceleration correlation',
                        'upper bound for Lagrangian acceleration correlation',
                        'Cencini-Vulpiani exponent averaged at constant separation',
                        'lower bound for Cencini-Vulpiani exponent',
                        'upper bound for Cencini-Vulpiani exponent',
                        'number of samples at constant separation',
                        'finite-size Lyapunov exponent averaged at constant separation',
                        'lower bound for finite-size Lyapunov exponent',
                        'upper bound for finite-size Lyapunov exponent',
                        'cumulative inverse separation time averaged at constant separation',
                        'lower bound for cumulative inverse separation time',
                        'upper bound for cumulative inverse separation time',
                        #
                        'rotational component of 2nd-order structure function',
                        'divergent component of 2nd-order structure function',
                        'probability density function of separation',
                        'cumulative density function of separation']
        
        else: # no bootstrapping and confidence levels
            #----------------------- measures averaged at constant time --------------------#
            mean_at = 'const-t'
            r2_t = relative_dispersion(r, order=2, mean_at=mean_at, nproc=nproc)
            if one_by_one: r2_t.load()
            pbar.update(1)
            rpb2_t = relative_dispersion(rpb, order=2, mean_at=mean_at, nproc=nproc)
            if one_by_one: rpb2_t.load()
            pbar.update(1)
            S2_t = velocity_structure_function(np.hypot(du, dv), r, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2_t.load()
            pbar.update(1)
            S2ll_t = velocity_structure_function(dul, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2ll_t.load()
            pbar.update(1)
            S2tr_t = velocity_structure_function(dut, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2tr_t.load()
            pbar.update(1)
            S3_t = velocity_structure_function(dul*(du**2+dv**2), r, order=1, mean_at=mean_at, nproc=nproc)
            if one_by_one: S3_t.load()
            pbar.update(1)
            K2_t = relative_diffusivity(r, samples='all', mean_at=mean_at, nproc=nproc)
            if one_by_one: K2_t.load()
            pbar.update(1)
            K2p_t = relative_diffusivity(r, samples='abs', mean_at=mean_at, nproc=nproc)
            if one_by_one: K2p_t.load()
            pbar.update(1)
            FAGR_t = finite_amplitude_growth_rate(r, mean_at=mean_at, nproc=nproc)
            if one_by_one: FAGR_t.load()
            pbar.update(1)
            imrv_t = initial_memory(rx, ry, du, dv, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: imrv_t.load()
            pbar.update(1)
            imrv2_t = initial_memory(rx, ry, du, dv, r, order=2, mean_at=mean_at, nproc=nproc)
            if one_by_one: imrv2_t.load()
            pbar.update(1)
            imra_t = initial_memory(rx, ry, dax, day, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: imra_t.load()
            pbar.update(1)
            aniso_t = anisotropy(rx, ry, rxy, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: aniso_t.load()
            pbar.update(1)
            LVC_t = lagrangian_velocity_correlation(uv, vmi, vmj, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: LVC_t.load()
            pbar.update(1)
            LAC_t = lagrangian_velocity_correlation(axy, ai, aj, r, mean_at=mean_at, nproc=nproc)
            if one_by_one: LAC_t.load()
            pbar.update(1)
            ku_t = kurtosis(r, mean_at=mean_at, nproc=nproc)
            if one_by_one: ku_t.load()
            pbar.update(1)
            kupb_t = kurtosis(rpb, mean_at=mean_at, nproc=nproc)
            if one_by_one: kupb_t.load()
            pbar.update(1)
            CVE_t = cencini_vulpiani_exponent(r, mean_at=mean_at, nproc=nproc)
            if one_by_one: CVE_t.load()
            pbar.update(1)
            num_t = xr.where(np.isnan(r), 0, 1).sum('pair')
            if one_by_one: num_t.load()
            
            #--------------------- measures average at constant separation ------------------#
            mean_at = 'const-r'
            r2_r = relative_dispersion(r, order=2, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: r2_r.load()
            pbar.update(1)
            rpb2_r = relative_dispersion(rpb, order=2, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: rpb2_r.load()
            pbar.update(1)
            S2_r = velocity_structure_function(np.hypot(du, dv), r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2_r.load()
            pbar.update(1)
            S2ll_r = velocity_structure_function(dul, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2ll_r.load()
            pbar.update(1)
            S2tr_r = velocity_structure_function(dut, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: S2tr_r.load()
            pbar.update(1)
            S3_r = velocity_structure_function(dul*(du**2+dv**2), r, order=1, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: S3_r.load()
            pbar.update(1)
            K2_r = relative_diffusivity(r, rbins=rbins, samples='all', mean_at=mean_at, nproc=nproc)
            if one_by_one: K2_r.load()
            pbar.update(1)
            K2p_r = relative_diffusivity(r, rbins=rbins, samples='abs', mean_at=mean_at, nproc=nproc)
            if one_by_one: K2p_r.load()
            pbar.update(1)
            FAGR_r = finite_amplitude_growth_rate(r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: FAGR_r.load()
            pbar.update(1)
            imrv_r = initial_memory(rx, ry, du, dv, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: imrv_r.load()
            pbar.update(1)
            imrv2_r = initial_memory(rx, ry, du, dv, r, order=2, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: imrv2_r.load()
            pbar.update(1)
            imra_r = initial_memory(rx, ry, dax, day, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: imra_r.load()
            pbar.update(1)
            aniso_r = anisotropy(rx, ry, rxy, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: aniso_r.load()
            pbar.update(1)
            LVC_r = lagrangian_velocity_correlation(uv, vmi, vmj, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: LVC_r.load()
            pbar.update(1)
            LAC_r = lagrangian_velocity_correlation(axy, ai, aj, r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: LAC_r.load()
            pbar.update(1)
            CVE_r = cencini_vulpiani_exponent(r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: CVE_r.load()
            pbar.update(1)
            FSLE_r = finite_size_lyapunov_exponent(r, rbins=rbins, mean_at=mean_at, nproc=nproc)
            if one_by_one: FSLE_r.load()
            pbar.update(1)
            CIST_r = cumulative_inverse_separation_time(r, rbins=rbins, lower=0.10, upper=0.90, maskout=[1e-8, 5e3], mean_at=mean_at, nproc=nproc)
            if one_by_one: CIST_r.load()
            pbar.update(1)
            num_r = sum_at_rbin(xr.where(np.isnan(r), 0, 1), r, rbins=rbins)
            if one_by_one: num_r.load()
            pbar.update(1)
            
            #----------------------------- other measures --------------------------#
            S2rr_r, S2dd_r = rotational_divergent_components(S2ll_r, S2tr_r)
            if one_by_one: S2rr_r.load(); S2dd_r.load()
            pbar.update(1)
            # PDF = probability_density_function(r, rbins=rbins)
            # if one_by_one: PDF.load()
            # pbar.update(1)
            # CDF = cumulative_density_function(PDF)
            # if one_by_one: CDF.load()
            # pbar.update(1)
            
            #------------------------------- output list ----------------------------#
            vs = [r2_t  , rpb2_t , S2_t   , K2_t  , S2ll_t, S2tr_t, S3_t , K2p_t,
                  FAGR_t, imrv_t , imrv2_t, imra_t, aniso_t, LVC_t, LAC_t, ku_t ,
                  kupb_t, CVE_t  , num_t,
                  r2_r  , rpb2_r , S2_r   , K2_r  , S2ll_r, S2tr_r, S3_r , K2p_r,
                  FAGR_r, imrv_r , imrv2_r, imra_r, aniso_r, LVC_r, LAC_r, CVE_r, num_r,
                  FSLE_r, CIST_r , S2rr_r , S2dd_r#, PDF   , CDF   ,
                  ]
            
            #------------------------------- output names ----------------------------#
            names = ['r2_t'  , 'rpb2_t' , 'S2_t'   , 'K2_t'  , 'S2ll_t', 'S2tr_t', 'S3_t' , 'K2p_t',
                     'FAGR_t', 'imrv_t' , 'imrv2_t', 'imra_t', 'aniso_t', 'LVC_t', 'LAC_t', 'ku_t' ,
                     'kupb_t', 'CVE_t', 'num_t',
                     'r2_r'  , 'rpb2_r' , 'S2_r'   , 'K2_r'  , 'S2ll_r', 'S2tr_r', 'S3_r' , 'K2p_r',
                     'FAGR_r', 'imrv_r' , 'imrv2_r', 'imra_r', 'aniso_r', 'LVC_r', 'LAC_r', 'CVE_r', 'num_r',
                     'FSLE_r', 'CIST_r' , 'S2rr_r' , 'S2dd_r'#, 'PDF'   , 'CDF'   ,
                    ]
            
            #------------------------------- output comments ----------------------------#
            comments = ['relative dispersion averaged at constant time',
                        'perturbation dispersion averaged at constant time',
                        '2nd-order structure function averaged at constant time',
                        'relative diffusivity averaged at constant time',
                        '2nd-order longitudinal structure function averaged at constant time',
                        '2nd-order transversal structure function averaged at constant time',
                        '3rd-order structure function averaged at constant time',
                        'absolute relative diffusivity averaged at constant time',
                        'finite-amplitude growth rate averaged at constant time',
                        'initial memory rv averaged at constant time',
                        'initial memory rv2 averaged at constant time',
                        'initial memory ra averaged at constant time',
                        'anisotropy averaged at constant time',
                        'Lagrangian velocity correlation averaged at constant time',
                        'Lagrangian acceleration correlation averaged at constant time',
                        'kurtosis averaged at constant time',
                        'perturbation kurtosis averaged at constant time',
                        'Cencini-Vulpiani exponent averaged at constant time',
                        'number of samples at constant time',
                        #
                        'relative dispersion averaged at constant separation',
                        'perturbation dispersion averaged at constant separation',
                        '2nd-order structure function averaged at constant separation',
                        'relative diffusivity averaged at constant separation',
                        '2nd-order longitudinal structure function averaged at constant separation',
                        '2nd-order transversal structure function averaged at constant separation',
                        '3rd-order structure function averaged at constant separation',
                        'absolute relative diffusivity averaged at constant separation',
                        'finite-amplitude growth rate averaged at constant separation',
                        'initial memory rv averaged at constant separation',
                        'initial memory rv2 averaged at constant separation',
                        'initial memory ra averaged at constant separation',
                        'anisotropy averaged at constant separation',
                        'Lagrangian velocity correlation averaged at constant separation',
                        'Lagrangian acceleration correlation averaged at constant separation',
                        'Cencini-Vulpiani exponent averaged at constant separation',
                        'number of samples at constant separation',
                        'finite-size Lyapunov exponent averaged at constant separation',
                        'cumulative inverse separation time averaged at constant separation',
                        #
                        'rotational component of 2nd-order structure function',
                        'divergent component of 2nd-order structure function',
                        #'probability density function of separation',
                        #'cumulative density function of separation',
                       ]
    
    if len(vs) != len(names) or len(vs) != len(comments):
        raise Exception(f'invalid lengths: {len(vs)}, {len(names)}, {len(comments)}')
    
    tmp = []
    for v, n, c in zip(vs, names, comments):
        v = v.rename(n)
        v.attrs['comment'] = c
        tmp.append(v)

    return xr.merge(tmp)
    




"""
Below are functions used for bootstrapping
"""
