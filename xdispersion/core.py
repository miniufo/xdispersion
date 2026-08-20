# -*- coding: utf-8 -*-
"""
Created on 2025.02.26

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.

Module: xdispersion.core
========================

This module defines the :class:`RelativeDispersion` class, which is the
main entry point for two-particle (relative dispersion) analysis.

**Key concepts:**

- **Ragged vs. non-ragged**:  ragged datasets (e.g. from
  `clouddrift <https://github.com/Cloud-Drift/clouddrift>`__) store
  trajectories of unequal length in a flat obs-dimension indexed by a
  ``rowsize`` array.  Non-ragged datasets have a uniform time dimension
  shared by all particles.

- **Original vs. chance pairs**:  *original* pairs are those whose
  **initial** separation falls within a prescribed range ``r0``.
  *Chance* pairs are identified when the separation **first** drops
  into the range at some later time.

- **``chunk`` parameter**:  controls **pair-data** chunking (not
  drifter/particle chunking).  When set, :meth:`load_variable` returns
  dask-backed arrays so that peak memory is proportional to ``chunk``
  rather than the total pair count.  Set to ``None`` for in-memory
  (numpy) processing when the pair count is small.

**Typical workflow**::

    rd = RelativeDispersion(ds, xpos='lon', ypos='lat',
                            uvel='u', vvel='v', time='time',
                            ID='traj', coord='latlon',
                            ragged=True, maxtlen=2880)
    p_all = rd.get_all_pairs()
    p_ori = rd.get_original_pairs(p_all, r0=[1, 10])
    rx, ry, rxy, r, rpb = rd.separation_measures(p_ori)
"""
import numpy as np
import xarray as xr
import itertools
import dask.array as dsa
from tqdm import tqdm
from dask import delayed
from dask.array import histogram as dsa_histogram
from typing import Optional, List, Dict, Tuple, Literal
from .utils import geodist, get_overlap_indices
from .measures import rotational_divergent_components

"""
Core classes are defined below
"""

class RelativeDispersion(object):
    """
    Core class for relative dispersion (two-particle) analysis.

    Supports both **ragged** (unequal-length trajectories, e.g. GLAD
    drifters) and **non-ragged** (uniform-length, e.g. synthetic
    particles) datasets, in either ``latlon`` or ``cartesian``
    coordinates.

    The ``chunk`` parameter is the unified memory-control knob: it
    controls how many **pairs** are processed at a time when loading
    pair data via :meth:`load_variable`.  Smaller chunks → lower peak
    memory but higher overhead; ``None`` → everything in numpy (fastest
    for small pair counts).
    """
    def __init__(self,
        ds_traj: xr.Dataset,
        xpos: str,
        ypos: str,
        uvel: str,
        vvel: str,
        time: str,
        coord: Literal['cartesian', 'latlon'],
        ID: str,
        maxtlen: Optional[int] = -1 ,
        Rearth: Optional[float] = 6371.2,
        ragged: Optional[bool] = False,
        chunk: Optional[int] = None,
    ) -> None:
        """
        Construct a RelativeDispersion class

        Parameters
        ----------
        ds_traj: xarray.Dataset
            A ragged trajectory dataset.
        xpos: str
            x-position name e.g., lon or longitude
        ypos: str
            y-position name e.g., lat or latitude
        uvel: str
            x-velocity name e.g., u or uvel
        vvel: str
            y-velocity name e.g., v or vvel
        time: str
            Name of time dimension
        coord: str
            Type of coordinates in ['cartesian' 'latlon'].
        ID: str
            Dimension name for particle IDs.
        maxtlen: int
            Set the maximum length of rtime (relative time).
        Rearth: float
            The radius of the earth, either in m or in km, which determine
            the units of later statistics if coord is latlon.
        ragged: boolean
            Whether the dataset is a ragged one.  Default is False so that
            each drifter is of the same length.
        chunk: int or None
            Unified memory-control knob for pair-related operations:
            - None: use in-memory path (no pair chunking);
            - positive int: chunk size for pair/block processing.
        """
        self.xpos    = xpos
        self.ypos    = ypos
        self.uvel    = uvel
        self.vvel    = vvel
        self.time    = time
        self.coord   = coord
        self.ID      = ID
        self.ragged  = ragged
        self.Rearth  = Rearth
        self.deg2m   = np.deg2rad(1.0) * Rearth
        self.ds_traj = ds_traj
        self.dtype   = ds_traj[uvel].dtype
        self.maxtlen = maxtlen
        self.chunk   = chunk

        # Keep chunk semantics explicit for both ragged and non-ragged paths.
        if self.chunk is not None and self.chunk < 1:
            raise Exception('chunk should be a positive integer or None')

        times = ds_traj[time]
        if np.issubdtype(times.dtype, np.datetime64):
            # change unit to day
            self.dt = (times[1] - times[0]).astype('int').values / 1e9 / 86400
        else:
            self.dt = (times[1] - times[0]).values

        if ragged and maxtlen < 0:
            raise Exception('maxtlen should be positive when trajectories are a ragged dataset')

        if not ragged and maxtlen < 0:
            maxtlen = len(times)
        
        if coord not in ['cartesian', 'latlon']:
            raise Exception(f'invalid coord {coord}, should be [cartesian, latlon]')
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    "       Below are particle-related functions.        "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_all_pairs(self) -> xr.Dataset:
        """Generate all possible pairs from the trajectory dataset.

        For *ragged* data, every pair of trajectories that has temporal
        overlap is included.  For *non-ragged* data, all
        :math:`C(n,2) = n(n-1)/2` pairs are generated.

        The returned Dataset contains:

        - ``pID``   – (pair, 2) particle IDs for each pair
        - ``tlen``  – (pair,) overlapping time length (in steps)
        - ``stim``  – (pair,) start time of the overlap
        - ``r0``    – (pair,) initial separation distance
        - ``xpos0`` – (pair, 2) initial x-positions
        - ``ypos0`` – (pair, 2) initial y-positions
        - ``idxI``  – (pair, 2) global start/end indices into the ragged
          array for particle *i*  (ragged only)
        - ``idxJ``  – (pair, 2) same for particle *j*  (ragged only)

        Parameters
        ----------
        (none — uses ``self.ds_traj``)

        Returns
        -------
        pairs : xarray.Dataset
            Pair metadata with dimension ``pair`` (and ``particle``
            for the two-particle arrays).
        """
        dset = self.ds_traj
        
        if self.ragged: # for ragged drifter dataset
            # Always use _get_all_chunked (pre-allocated arrays, no
            # list+concatenate overhead).  When chunk is None, pass
            # ntraj so all trajectories are processed in a single block.
            ntraj_r = len(dset[self.ID])
            _chunk = self.chunk if self.chunk is not None else ntraj_r
            pID , tlen, stim, r0, xpos0, ypos0,\
            idxI, idxJ = self._get_all_chunked(dset[self.ID].values,
                                               dset['rowsize'].values,
                                               dset[self.xpos].values,
                                               dset[self.ypos].values,
                                               dset[self.time].values,
                                               _chunk)
            
            # pair index, starts from 0 to the total number of pairs
            pair = np.arange(len(r0)  , dtype=np.int32)
            # particle index in a single pair
            particle = np.array([0, 1], dtype=np.int32)
        
            tlen = xr.DataArray(tlen, name='tlen', dims='pair', coords={'pair':pair})
            r0   = xr.DataArray(r0  , name='r0'  , dims='pair', coords={'pair':pair})
            stim = xr.DataArray(stim, name='stim', dims='pair', coords={'pair':pair})
            
            pID   = xr.DataArray(pID , name='pID', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            xpos0 = xr.DataArray(xpos0, name='xpos0', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            ypos0 = xr.DataArray(ypos0, name='ypos0', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            idxI  = xr.DataArray(idxI, name='idxI', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            idxJ  = xr.DataArray(idxJ, name='idxJ', dims=['pair','particle'],
                                 coords={'pair':pair, 'particle':particle})
            
            print(f'there are {len(r0)} pairs of particles')
            
            return xr.merge([tlen, stim, r0, pID, xpos0, ypos0, idxI, idxJ])
            
        else:
            ntraj = len(dset[self.ID])
            # pair_idx = xr.DataArray(np.array(list(itertools.combinations(range(ntraj), 2))),
            #                         dims=("pair", "particle"), name='pID',
            #                         coords={'pair':np.arange(ntraj*(ntraj-1)/2, dtype='int32'),
            #                                 'particle':np.array([0,1], dtype='int32')})
            # xpos0 = dset[self.xpos].isel({self.time:0, self.ID:pair_idx}).drop_vars([self.time,self.ID]).rename('xpos0')
            # ypos0 = dset[self.ypos].isel({self.time:0, self.ID:pair_idx}).drop_vars([self.time,self.ID]).rename('ypos0')
            # pID   = dset[self.ID].isel({self.ID:pair_idx}).drop_vars(self.ID).rename('pID')
            # tlen  = (xpos0 - xpos0).isel(particle=0).rename('tlen') + len(dset.time)
            # stime = (dset.time.isel({self.time:0}, drop=True) + (xpos0 - xpos0).isel({'particle':0})).rename('stime')
            # r0    = np.hypot(xpos0.isel(particle=0) - xpos0.isel(particle=1),
            #                  ypos0.isel(particle=0) - ypos0.isel(particle=1)).rename('r0')
            if self.chunk is None:
                if ntraj <= 5000:
                    self.chunk = ntraj * (ntraj - 1) // 2
                else:
                    self.chunk = ntraj * 2

            chunk = self.chunk
            pair_idx = self._get_pair_index(ntraj, chunk)
            xpos0 = self._select_pair_var(dset[self.xpos], pair_idx).rename('xpos0')
            ypos0 = self._select_pair_var(dset[self.ypos], pair_idx).rename('ypos0')
            pID   = self._select_pair_var(dset[self.ID],   pair_idx).rename('pID')
            tlen  = (xpos0 - xpos0).isel(particle=0).rename('tlen') + len(dset.time)
            stime = (dset.time.isel({self.time:0}, drop=True) + 
                     (xpos0 - xpos0).isel({'particle':0})).rename('stime')
            r0    = np.hypot(xpos0.isel(particle=0) - xpos0.isel(particle=1),
                             ypos0.isel(particle=0) - ypos0.isel(particle=1)).rename('r0')
            
            tlen  = (xpos0 - xpos0).isel(particle=0).rename('tlen') + len(dset.time)
            stime = (dset.time.isel({self.time:0}, drop=True) +
                     (xpos0 - xpos0).isel({'particle':0})).rename('stime')
            r0    = np.hypot(xpos0.isel(particle=0) - xpos0.isel(particle=1),
                             ypos0.isel(particle=0) - ypos0.isel(particle=1)).rename('r0')
            
            return xr.merge([tlen, stime, r0, pID, xpos0, ypos0])
    
    def get_original_pairs(self,
        pairs: xr.Dataset,
        r0: List[float]
    ) -> xr.Dataset:
        """Select *original* pairs whose initial separation is within ``r0``.

        Original pairs are those where the **initial** separation
        :math:`r(t=0)` falls in ``[r0[0], r0[1])``.  These are the
        pairs used for constant-time (const-t) statistics, where all
        pairs start at the same relative time ``rtime = 0``.

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs`.
        r0 : float or list of float
            If a single float ``r``, it is interpreted as ``[0, r]``.
            If a list ``[rmin, rmax]``, pairs with ``rmin <= r0 < rmax``
            are selected.

        Returns
        -------
        pair_o : xarray.Dataset
            Subset of *pairs* satisfying the initial-separation criterion.
        """
        if isinstance(r0, float):
            if r0 <= 0:
                raise Exception('r0 should be larger than 0')
            r0 = [0, r0]
        
        if isinstance(r0, list):
            rmin, rmax = r0
        else:
            raise Exception(f'unsupported r0 {r0}, should be a list ' +
                            f'of two floats or a single float')
            
        cond = np.logical_and(pairs.r0>=rmin, pairs.r0<rmax).load()
        
        return pairs.where(cond, drop=True).astype(pairs.dtypes)
    
    
    def get_chance_pairs(self,
        pairs: xr.Dataset,
        r0: List[float]
    ) -> xr.Dataset:
        """Select *chance* pairs whose minimum separation enters ``r0``.

        For each pair, the time series of separation is scanned for its
        **minimum**.  If that minimum falls within ``[rmin, rmax)`` and
        occurs at a non-zero relative time, the pair is re-labeled as a
        chance pair starting from that minimum-separation time.  The
        ``stim``, ``tlen``, ``r0``, ``xpos0``, ``ypos0``, ``idxI`` and
        ``idxJ`` fields are adjusted accordingly.

        Chance pairs are used for constant-separation (const-r)
        statistics such as FSLE and CIST.

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs`.
        r0 : float or list of float
            Same interpretation as in :meth:`get_original_pairs`.

        Returns
        -------
        pair_c : xarray.Dataset
            Subset of *pairs* re-labeled as chance pairs, with updated
            start times and indices.
        """
        if isinstance(r0, float):
            if r0 <= 0:
                raise Exception('r0 should be larger than 0')
            r0 = [0, r0]
        
        if isinstance(r0, list):
            rmin, rmax = r0
        else:
            raise Exception(f'unsupported r0 {r0}, should be a list ' +
                            f'of two floats or a single float')
        
        ds = pairs.copy(deep=True) # make a copy to modify
        idxI = pairs.idxI.values
        idxJ = pairs.idxJ.values
        
        xpos = self.ds_traj[self.xpos].values
        ypos = self.ds_traj[self.ypos].values
        
        for i in range(len(pairs['pair'])):
            idxIS, idxIE = idxI[i] # the first particle
            idxJS, idxJE = idxJ[i] # the second particle

            if self.coord == 'latlon':
                xi = np.deg2rad(xpos[idxIS:idxIE])
                xj = np.deg2rad(xpos[idxJS:idxJE])
                yi = np.deg2rad(ypos[idxIS:idxIE])
                yj = np.deg2rad(ypos[idxJS:idxJE])
    
                r = geodist(xi, xj, yi, yj) * self.Rearth
            else:
                r = np.hypot(xpos[idxIS:idxIE] - xpos[idxJS:idxJE],
                             ypos[idxIS:idxIE] - ypos[idxJS:idxJE])
            
            idx = r.argmin() # relative index of (the first) minimum separation
            rm  = r.min()    # minimum separation

            # idx == 0 means an original pair
            if idx > 0 and (rmin <= rm < rmax):
            #if (rmin <= rm <= rmax):
                ds.tlen[i] = ds.tlen[i] - idx
                ds.stim[i] = self.ds_traj[self.time][idxIS + idx]
                ds.r0[i]   = rm
                ds.xpos0[i,0] = xpos[idxIS + idx]
                ds.xpos0[i,1] = xpos[idxJS + idx]
                ds.ypos0[i,0] = ypos[idxIS + idx]
                ds.ypos0[i,1] = ypos[idxJS + idx]
                ds.idxI[i, 0] = idxIS + idx # only change start index
                ds.idxJ[i, 0] = idxJS + idx # only change start index
            else:
                ds.r0[i] = np.nan # assign nan so that we could drop them
        
        return ds.dropna(dim='pair').astype(pairs.dtypes)
    
    def load_variable(self,
        pairs: xr.Dataset,
        vname: str,
    ) -> xr.DataArray:
        """Load a trajectory variable for all pairs as a 3-D DataArray.

        The returned array has dimensions ``[pair, particle, rtime]``
        where:

        - ``pair``     – pair index (0 .. N-1)
        - ``particle`` – 0 or 1 (the two particles in a pair)
        - ``rtime``    – relative time from pair start, in units of
          ``self.dt``

        **Ragged path**:  extracts ragged-array slices using
        ``idxI`` / ``idxJ`` from *pairs*.  When ``self.chunk`` is set,
        each chunk is loaded lazily via ``dask.array.from_delayed``,
        keeping peak memory proportional to ``chunk`` rather than ``N``.
        When ``chunk`` is ``None``, the full array is materialised in
        numpy (fastest for small N).

        **Non-ragged path**:  selects values by particle ID using
        xarray ``.sel()`` and, when ``chunk`` is set, re-chunks along
        the ``pair`` dimension.

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs` (or a filtered subset).
        vname : str
            One of the variable names passed to the constructor
            (``xpos``, ``ypos``, ``uvel``, ``vvel``).

        Returns
        -------
        re : xarray.DataArray
            3-D array ``[pair, particle, rtime]``.  Padded with NaN
            where the overlap is shorter than ``maxtlen`` (ragged only).
        """
        N = len(pairs['pair'])
        v = self.ds_traj[vname]
        
        if self.ragged:
            maxtlen = self.maxtlen
            
            idxI = pairs.idxI.values
            idxJ = pairs.idxJ.values
            
            # Extract the underlying numpy array once to avoid repeated
            # DataArray indexing overhead inside the loop (10×+ speedup
            # for large pair counts).
            v_vals = np.asarray(v)
            
            # Closure that loads a contiguous range of pairs into a numpy
            # array of shape (end-start, 2, maxtlen).  When self.chunk is
            # smaller than N, this function is wrapped in dask.from_delayed
            # so that each chunk is loaded lazily — only when the dask
            # scheduler actually needs the data.  This keeps peak memory
            # proportional to chunk_size rather than N.
            def _load_chunk(start, end):
                n = end - start
                re = np.full((n, 2, maxtlen), np.nan, dtype=self.dtype)
                for i in range(start, end):
                    idxIS, idxIE = idxI[i]
                    idxJS, idxJE = idxJ[i]
                    size = idxIE - idxIS
                    
                    if size <= maxtlen:
                        re[i - start, 0, :size] = v_vals[idxIS:idxIE]
                        re[i - start, 1, :size] = v_vals[idxJS:idxJE]
                    else:
                        re[i - start, 0, :maxtlen] = v_vals[idxIS:idxIS + maxtlen]
                        re[i - start, 1, :maxtlen] = v_vals[idxJS:idxJS + maxtlen]
                return re
            
            chunk_size = self.chunk if self.chunk is not None else N
            
            if chunk_size >= N:
                # No chunking — load everything at once (original behaviour)
                re = _load_chunk(0, N)
            else:
                # Chunked — build a lazy dask array via from_delayed so that
                # each chunk is materialised only when needed.
                delayed_chunks = []
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    delayed_chunks.append(
                        dsa.from_delayed(
                            delayed(_load_chunk)(start, end),
                            shape=(end - start, 2, maxtlen),
                            dtype=self.dtype,
                        )
                    )
                re = dsa.concatenate(delayed_chunks, axis=0)
            
            return xr.DataArray(re, dims=['pair', 'particle', 'rtime'],
                                coords={'pair':pairs.pair, 'particle':[0,1],
                                        'rtime':np.arange(maxtlen)*self.dt})
        else:
            if pairs.pID.chunks is not None:
                def load_chunk(pair_slice):
                    return v.sel({self.ID:pairs.pID.isel({'pair':pair_slice})}).data
                
                slices = [slice(i, min(i+self.chunk, N)) for i in range(0, N, self.chunk)]
                
                if isinstance(v.data, dsa.core.Array):
                    x_data = dsa.concatenate([load_chunk(slc) for slc in slices], axis=v.get_axis_num(self.ID))
                elif isinstance(v.data, np.ndarray):
                    x_data = np.concatenate([np.asarray(load_chunk(slc)) for slc in slices], axis=v.get_axis_num(self.ID))
                else:
                    raise Exception(f'unsupported type of {type(v.data)}')
                
                if v.get_axis_num('time') == 0:
                    dims = ['rtime', 'pair', 'particle']
                else:
                    dims = ['pair', 'particle', 'rtime']
                
                return xr.DataArray(x_data, name=vname, dims=dims,
                                    coords={'pair': pairs.pair,
                                            'particle': pairs.particle,
                                            'rtime': np.arange(len(v[self.time]))*self.dt})
            else:
                tmp = v.sel({self.ID:pairs.pID}).drop_vars(self.ID).rename({self.time:'rtime'})
                if self.chunk is not None:
                    tmp = tmp.chunk({'pair':self.chunk})
                tmp['rtime'] = np.arange(len(v[self.time])) * self.dt
                return tmp
    

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    "          Below are the helper functions.           "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    def separation_measures(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray]:
        """Compute separation-related building blocks.

        Calls :meth:`load_variable` for ``xpos`` and ``ypos``, then
        derives:

        - ``rx``, ``ry``  – zonal / meridional separation components
        - ``rxy``         – cross term ``rx * ry``
        - ``r``           – total separation ``|r|`
        - ``rpb``         – perturbation separation (deviation from
          initial position), i.e. how far the *change* in separation
          has grown relative to ``t=0``

        For ``latlon`` coordinates, geographic distances are computed
        using the great-circle formula with ``self.Rearth``.

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs` (or a filtered subset).

        Returns
        -------
        rx, ry, rxy, r, rpb : xarray.DataArray
            Each has dimensions ``[pair, rtime]``.
        """
        xpos = self.load_variable(pairs, self.xpos)
        ypos = self.load_variable(pairs, self.ypos)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)
        
        if self.coord == 'latlon':
            xi = np.deg2rad(xi)
            yi = np.deg2rad(yi)
            xj = np.deg2rad(xj)
            yj = np.deg2rad(yj)
            
            rx  = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry  = (yi - yj) * self.Rearth
            rxy = rx * ry
            r   = geodist(xi, xj, yi, yj) * self.Rearth
            rxp = ((xi-xi.isel(rtime=0)) - (xj-xj.isel(rtime=0))) * np.cos((yi + yj)/2.0)
            ryp = ((yi-yi.isel(rtime=0)) - (yj-yj.isel(rtime=0)))
            rpb = np.hypot(rxp, ryp) * self.Rearth
        else:
            rx = xi - xj
            ry = yi - yj
            rxy = rx * ry
            r   = np.hypot(rx, ry)
            rpb = np.hypot((xi-xi.isel(rtime=0)) - (xj-xj.isel(rtime=0)),
                           (yi-yi.isel(rtime=0)) - (yj-yj.isel(rtime=0)))
        
        return rx, ry, rxy, r, rpb

    
    def velocity_measures(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray]:
        """Compute velocity-related building blocks.

        Loads ``xpos``, ``ypos``, ``uvel``, ``vvel`` and derives:

        - ``du``, ``dv``  - velocity difference (particle i - particle j)
        - ``dul``         - longitudinal velocity difference
          (projection along the separation vector)
        - ``dut``         - transversal velocity difference
          (projection perpendicular to the separation vector)
        - ``vmi``         - velocity magnitude of particle i
        - ``vmj``         - velocity magnitude of particle j
        - ``uv``          - inner product of the two velocities
          (used for Lagrangian velocity correlation)

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs` (or a filtered subset).

        Returns
        -------
        du, dv, dul, dut, vmi, vmj, uv : xarray.DataArray
            Each has dimensions ``[pair, rtime]``.
        """
        xpos = self.load_variable(pairs, self.xpos)
        ypos = self.load_variable(pairs, self.ypos)
        uvel = self.load_variable(pairs, self.uvel)
        vvel = self.load_variable(pairs, self.vvel)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)

        ui = uvel.isel(particle=0)
        uj = uvel.isel(particle=1)
        vi = vvel.isel(particle=0)
        vj = vvel.isel(particle=1)
        
        if self.coord == 'latlon': # need to convert degree to unit of Rearth
            xi = np.deg2rad(xi)
            xj = np.deg2rad(xj)
            yi = np.deg2rad(yi)
            yj = np.deg2rad(yj)
            
            du  = ui - uj
            dv  = vi - vj
            vsi = np.hypot(ui, vi)
            vsj = np.hypot(uj, vj)
            uv  = ui * uj + vi * vj
            
            # convert radial to meter
            rx = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry = (yi - yj) * self.Rearth
            r  = geodist(xi, xj, yi, yj) * self.Rearth
            
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        else: # cartesian coordinate
            du  = ui - uj
            dv  = vi - vj
            vsi = np.hypot(ui, vi)
            vsj = np.hypot(uj, vj)
            uv  = ui * uj + vi * vj
            
            rx = xi - xj
            ry = yi - yj
            r  = np.hypot(rx, ry)

            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        return du, dv, dul, dut, vsi, vsj, uv

    
    def acceleration_measures2(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray]:
        """[Deprecated] Calculate acceleration by differentiating positions twice.

        This is the position-differentiation version.  Use
        :meth:`acceleration_measures` instead, which differentiates
        velocities (more accurate when velocities are directly observed).
        """
        dt = self.dt
        
        xpos = self.load_variable(pairs, self.xpos)
        ypos = self.load_variable(pairs, self.ypos)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)
        
        if self.coord == 'latlon': # need to convert degree to unit of Rearth
            xi = np.deg2rad(xi)
            xj = np.deg2rad(xj)
            yi = np.deg2rad(yi)
            yj = np.deg2rad(yj)
            
            axi = xi.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            axj = xj.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            ayi = yi.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            ayj = yj.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            
            # convert radial to meter
            axi = axi * np.cos((yi + yj)/2.0) * self.Rearth
            axj = axj * np.cos((yi + yj)/2.0) * self.Rearth
            ayi = ayi * self.Rearth
            ayj = ayj * self.Rearth
            
            dax = axi - axj
            day = ayi - ayj
            ai  = np.hypot(axi, ayi)
            aj  = np.hypot(axj, ayj)
            axy = axi * axj + ayi * ayj
            
            # convert radial to meter
            rx = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry = (yi - yj) * self.Rearth
            r  = geodist(xi, xj, yi, yj) * self.Rearth
            
            dal = (rx * dax + ry * day) / r # longitudinal acceleration
            dat = (rx * day - ry * dax) / r # transversal  acceleration
            
        else: # cartesian coordinate
            axi = xi.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            axj = xj.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            ayi = yi.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            ayj = yj.pad({'rtime':1}, mode='edge').diff('rtime', label='lower').diff('rtime', label='upper') / dt**2.0
            
            dax = axi - axj
            day = ayi - ayj
            ai  = np.hypot(axi, ayi)
            aj  = np.hypot(axj, ayj)
            axy = axi * axj + ayi * ayj
            
            rx = xi - xj
            ry = yi - yj
            r  = np.hypot(rx, ry)

            dal = (rx * dax + ry * day) / r # longitudinal acceleration
            dat = (rx * day - ry * dax) / r # transversal  acceleration
            
        return dax, day, dal, dat, ai, aj, axy

    
    def acceleration_measures(self,
        pairs: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray]:
        """Compute acceleration-related building blocks.

        Loads ``xpos``, ``ypos``, ``uvel``, ``vvel`` and differentiates
        velocities along ``rtime`` to obtain accelerations, then derives:

        - ``dax``, ``day``  – acceleration difference (i − j)
        - ``dal``           – longitudinal acceleration difference
        - ``dat``           – transversal acceleration difference
        - ``ai``, ``aj``    – acceleration magnitude of each particle
        - ``axy``           – inner product of accelerations (used for
          Lagrangian acceleration correlation)

        Parameters
        ----------
        pairs : xarray.Dataset
            Output of :meth:`get_all_pairs` (or a filtered subset).

        Returns
        -------
        dax, day, dal, dat, ai, aj, axy : xarray.DataArray
            Each has dimensions ``[pair, rtime]``.
        """
        dt = self.dt
        
        xpos = self.load_variable(pairs, self.xpos)
        ypos = self.load_variable(pairs, self.ypos)
        uvel = self.load_variable(pairs, self.uvel)
        vvel = self.load_variable(pairs, self.vvel)
        
        xi = xpos.isel(particle=0)
        xj = xpos.isel(particle=1)
        yi = ypos.isel(particle=0)
        yj = ypos.isel(particle=1)

        ui = uvel.isel(particle=0)
        uj = uvel.isel(particle=1)
        vi = vvel.isel(particle=0)
        vj = vvel.isel(particle=1)
        
        if self.coord == 'latlon': # need to convert degree to unit of Rearth
            xi = np.deg2rad(xi)
            xj = np.deg2rad(xj)
            yi = np.deg2rad(yi)
            yj = np.deg2rad(yj)
            
            axi = ui.differentiate('rtime')
            axj = uj.differentiate('rtime')
            ayi = vi.differentiate('rtime')
            ayj = vj.differentiate('rtime')
            
            dax = axi - axj
            day = ayi - ayj
            ai  = np.hypot(axi, ayi)
            aj  = np.hypot(axj, ayj)
            axy = axi * axj + ayi * ayj
            
            # convert radial to meter
            rx = (xi - xj) * np.cos((yi + yj)/2.0) * self.Rearth
            ry = (yi - yj) * self.Rearth
            r  = geodist(xi, xj, yi, yj) * self.Rearth
            
            dal = (rx * dax + ry * day) / r # longitudinal acceleration
            dat = (rx * day - ry * dax) / r # transversal  acceleration
            
        else: # cartesian coordinate
            axi = ui.differentiate('rtime')
            axj = uj.differentiate('rtime')
            ayi = vi.differentiate('rtime')
            ayj = vj.differentiate('rtime')
            
            dax = axi - axj
            day = ayi - ayj
            ai  = np.hypot(axi, ayi)
            aj  = np.hypot(axj, ayj)
            axy = axi * axj + ayi * ayj
            
            rx = xi - xj
            ry = yi - yj
            r  = np.hypot(rx, ry)

            dal = (rx * dax + ry * day) / r # longitudinal acceleration
            dat = (rx * day - ry * dax) / r # transversal  acceleration
            
        return dax, day, dal, dat, ai, aj, axy
    
    
    def r_based_measures_bak(self,
        pairs: xr.Dataset,
        alpha: float,
        rbins: xr.DataArray,
        interpT: Optional[int] = 4
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """Calculate r-based measures using all available pairs
        
        r-based measures includes K2, S2, S2L, S2T, S3, FSLE, FAGR, FAGRp.
        No bootstrapping is done here, as one may use all pairs instead
        of only original pairs.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            A given pairs.
        alpha: float
            ratio between neighbouring separation bin (for FSLE).
        rbins: xarray.DataArray
            A given separation bins.
        
        Returns
        -------
        K2: xarray.DataArray
            Relative diffusivity.
        S2: xarray.DataArray
            2nd-order velocity structure funciton.
        S2L: xarray.DataArray
            2nd-order longitudinal velocity structure funciton.
        S2T: xarray.DataArray
            2nd-order transversal velocity structure funciton.
        S3: xarray.DataArray
            3rd-order velocity structure funciton.
        FSLEO: xarray.DataArray
            Finite-size Lyapunov exponent using original time interval.
        FSLEI: xarray.DataArray
            Finite-size Lyapunov exponent using interpolated time interval (see interpT).
        FAGR: xarray.DataArray
            Finite-amplitude growth rate.
        FAGRp: xarray.DataArray
            Positive finite-amplitude growth rate (equivalent to FSLE).
        numS: xarray.DataArray
            Number of observations for K2, S2, S2L, S2T, S3, FAGR.
        numP: xarray.DataArray
            Number of observations for positive FAGR.
        numF: xarray.DataArray
            Number of observations for FSLE.
        """
        N = len(pairs['pair'])
        
        xpos = self.ds_traj[self.xpos]
        ypos = self.ds_traj[self.ypos]
        uvel = self.ds_traj[self.uvel]
        vvel = self.ds_traj[self.vvel]
        
        K2    = (rbins - rbins).rename('K2')
        S2    = (rbins - rbins).rename('S2')
        S2L   = (rbins - rbins).rename('S2L')
        S2T   = (rbins - rbins).rename('S2T')
        S3    = (rbins - rbins).rename('S3')
        FSLEO = (rbins - rbins).rename('FSLEO')
        FSLEI = (rbins - rbins).rename('FSLEI')
        FAGR  = (rbins - rbins).rename('FAGR')
        FAGRp = (rbins - rbins).rename('FAGRp')
        numS  = (rbins - rbins).rename('num_S2')
        numP  = (rbins - rbins).rename('num_FAGRp')
        numF  = (rbins - rbins).rename('num_FSLE')
        
        k2  =    K2.values[:-1]
        s2  =    S2.values[:-1]
        s2l =   S2L.values[:-1]
        s2t =   S2T.values[:-1]
        s3  =    S3.values[:-1]
        fslo= FSLEO.values[:-1]
        fsli= FSLEI.values[:-1]
        fag =  FAGR.values[:-1]
        fap = FAGRp.values[:-1]
        nvs =  numS.values[:-1]
        nvp =  numP.values[:-1]
        nvf =  numF.values[:-1]
        
        idxI = pairs.idxI.values # mmm
        idxJ = pairs.idxJ.values # mmm
        tlen = pairs.tlen.values

        Rearth = self.Rearth
        _histo = np.histogram
        rbinv  = rbins.values
        deltaT = self.dt
        dtype  = self.dtype
        
        for i in tqdm(range(N), ncols=80):
            size = int(tlen[i])
            
            #########   allocate variables   ########
            xx = np.zeros((2, size), dtype=dtype) + np.nan
            yy = np.zeros((2, size), dtype=dtype) + np.nan
            uu = np.zeros((2, size), dtype=dtype) + np.nan
            vv = np.zeros((2, size), dtype=dtype) + np.nan
            
            idxIS, idxIE = idxI[i] # the first particle # mmm
            idxJS, idxJE = idxJ[i] # the second particle # mmm
            
            #########      fill in data      ########
            xx[0, :] = xpos[idxIS:idxIE]
            xx[1, :] = xpos[idxJS:idxJE]
            yy[0, :] = ypos[idxIS:idxIE]
            yy[1, :] = ypos[idxJS:idxJE]
            uu[0, :] = uvel[idxIS:idxIE]
            uu[1, :] = uvel[idxJS:idxJE]
            vv[0, :] = vvel[idxIS:idxIE]
            vv[1, :] = vvel[idxJS:idxJE]
            
            # xx[:, :] = xpos.sel({self.ID: pairs.pID[i]}).load()
            # yy[:, :] = ypos.sel({self.ID: pairs.pID[i]}).load()
            # uu[:, :] = uvel.sel({self.ID: pairs.pID[i]}).load()
            # vv[:, :] = vvel.sel({self.ID: pairs.pID[i]}).load()
            
            #########   start calculations   ########
            if self.coord == 'latlon':
                xx  = np.deg2rad(xx)
                yy  = np.deg2rad(yy)
                
                rx  = (xx[0] - xx[1]) * np.cos((yy[0] + yy[1])/2.0) * Rearth
                ry  = (yy[0] - yy[1]) * Rearth
                r   = geodist(xx[0], xx[1], yy[0], yy[1]) * Rearth
                
                du  = uu[0] - uu[1]
                dv  = vv[0] - vv[1]
                dul = (rx * du + ry * dv) / r # longitudinal velocity
                dut = (rx * dv - ry * du) / r # transversal  velocity
                
            else:
                rx  = xx[0] - xx[1]
                ry  = yy[0] - yy[1]
                r   = np.hypot(rx, ry)

                du  = uu[0] - uu[1]
                dv  = vv[0] - vv[1]
                dul = (rx * du + ry * dv) / r # longitudinal velocity
                dut = (rx * dv - ry * du) / r # transversal  velocity
            
            ######### wrap r into DataArray #########
            r_or = xr.DataArray(r, dims='time',
                                coords={'time':np.arange(size) * deltaT})
            if interpT > 1:
                timeInt = np.linspace(0, r_or.time.values[-1], int((size-1)*interpT+1))
                r_da = r_or.interp(time=timeInt)
            else:
                r_da = r_or
            
            #########       for FSLE, FAGR     #########
            rd = r_or[r_or.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('time')
            fsle = Td.diff('rbin')
            fsleO= (np.log(alpha) / fsle.where(fsle != 0))
            
            rd = r_da[r_da.argmin().values:]
            Td = xr.where(rd > rbins, 1, np.nan).idxmax('time')
            fsle = Td.diff('rbin')
            fsleI= (np.log(alpha) / fsle.where(fsle != 0))
            
            fagr = np.log(r_or).differentiate('time').values
            
            #########  accumulated within bins  #########
            tmp_K2 , _ = _histo(r, bins=rbinv, weights=(r_or**2).differentiate('time').values/2)
            tmp_S2 , _ = _histo(r, bins=rbinv, weights=du**2+dv**2)
            tmp_S2L, _ = _histo(r, bins=rbinv, weights=dul**2)
            tmp_S2T, _ = _histo(r, bins=rbinv, weights=dut**2)
            tmp_S3 , _ = _histo(r, bins=rbinv, weights=dul*(du**2+dv**2))
            tmp_FG , _ = _histo(r, bins=rbinv, weights=fagr)
            tmp_FGp, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, fagr, 0))
            tmp_noS, _ = _histo(r, bins=rbinv, weights=du-du+1)
            tmp_noP, _ = _histo(r, bins=rbinv, weights=np.where(fagr>0, 1, 0))
            tmp_noF    = np.where(np.isnan(fsleO), 0, 1)
            
            k2  += np.where(np.isnan(tmp_K2 ), 0, tmp_K2 )
            s2  += np.where(np.isnan(tmp_S2 ), 0, tmp_S2 )
            s2l += np.where(np.isnan(tmp_S2L), 0, tmp_S2L)
            s2t += np.where(np.isnan(tmp_S2T), 0, tmp_S2T)
            s3  += np.where(np.isnan(tmp_S3 ), 0, tmp_S3 )
            fslo+= np.where(np.isnan(fsleO  ), 0, fsleO  )
            fsli+= np.where(np.isnan(fsleI  ), 0, fsleI  )
            fag += np.where(np.isnan(tmp_FG ), 0, tmp_FG )
            fap += np.where(np.isnan(tmp_FGp), 0, tmp_FGp)
            nvs += tmp_noS
            nvp += tmp_noP
            nvf += tmp_noF
        
        K2    /= numS
        S2    /= numS
        S2L   /= numS
        S2T   /= numS
        S3    /= numS
        FAGR  /= numS
        FAGRp /= numP
        FSLEO /= numF
        FSLEI /= numF
        
        return K2, S2, S2L, S2T, S3, FSLEO, FSLEI, FAGR, FAGRp, numS, numP, numF

    
    def r_based_measures(self,
        pairs: xr.Dataset,
        alpha: float,
        rbins: xr.DataArray,
        interpT: Optional[int] = 4
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
               xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """Calculate r-based measures using all available pairs
        
        r-based measures includes K2, S2, S2L, S2T, S3, FSLE, FAGR, FAGRp.
        No bootstrapping is done here, as one may use all pairs instead
        of only original pairs.
        
        Parameters
        ----------
        pairs: xarray.Dataset
            A given pairs.
        alpha: float
            ratio between neighbouring separation bin (for FSLE).
        rbins: xarray.DataArray
            A given separation bins.
        
        Returns
        -------
        K2: xarray.DataArray
            Relative diffusivity.
        S2: xarray.DataArray
            2nd-order velocity structure funciton.
        S2L: xarray.DataArray
            2nd-order longitudinal velocity structure funciton.
        S2T: xarray.DataArray
            2nd-order transversal velocity structure funciton.
        S2rr: xarray.DataArray
            rotational component of velocity structure function.
        S2dd: xarray.DataArray
            divergent component of velocity structure function.
        S3: xarray.DataArray
            3rd-order velocity structure funciton.
        FSLEO: xarray.DataArray
            Finite-size Lyapunov exponent using original time interval.
        FSLEI: xarray.DataArray
            Finite-size Lyapunov exponent using interpolated time interval (see interpT).
        FAGR: xarray.DataArray
            Finite-amplitude growth rate.
        FAGRp: xarray.DataArray
            Positive finite-amplitude growth rate (equivalent to FSLE).
        numS: xarray.DataArray
            Number of observations for K2, S2, S2L, S2T, S3, FAGR.
        numP: xarray.DataArray
            Number of observations for positive FAGR.
        numFO: xarray.DataArray
            Number of observations for original FSLE.
        numFI: xarray.DataArray
            Number of observations for interpolated FSLE (see interpT).
        """
        N = len(pairs['pair'])
        
        xpos = self.load_variable(pairs, self.xpos)
        ypos = self.load_variable(pairs, self.ypos)
        uvel = self.load_variable(pairs, self.uvel)
        vvel = self.load_variable(pairs, self.vvel)

        Rearth = self.Rearth
        rbinv  = rbins.values
        deltaT = self.dt
        dtype  = self.dtype
        
        #########   start calculations   ########
        if self.coord == 'latlon':
            xx  = np.deg2rad(xpos)
            yy  = np.deg2rad(ypos)
            uu  = uvel
            vv  = vvel

            xx1 = xx.isel(particle=0)
            xx2 = xx.isel(particle=1)
            yy1 = yy.isel(particle=0)
            yy2 = yy.isel(particle=1)
            
            rx  = (xx1 - xx2) * np.cos((yy1 + yy2)/2.0) * Rearth
            ry  = (yy1 - yy2) * Rearth
            r   = geodist(xx1, xx2, yy1, yy2) * Rearth
            
            du  = uu.isel(particle=0) - uu.isel(particle=1)
            dv  = vv.isel(particle=0) - vv.isel(particle=1)
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
            
        else:
            xx  = xpos
            yy  = ypos
            uu  = uvel
            vv  = vvel
            
            rx  = xx.isel(particle=0) - xx.isel(particle=1)
            ry  = yy.isel(particle=0) - yy.isel(particle=1)
            r   = np.hypot(rx, ry)

            du  = uu.isel(particle=0) - uu.isel(particle=1)
            dv  = vv.isel(particle=0) - vv.isel(particle=1)
            dul = (rx * du + ry * dv) / r # longitudinal velocity
            dut = (rx * dv - ry * du) / r # transversal  velocity
        
        ######### interpolation for FSLE #########
        if interpT > 1:
            r_da = r.interp(rtime=np.linspace(
                            0, r['rtime'].values[-1],
                            int((len(r['rtime'].values)-1)*interpT+1)))
        else:
            r_da = r
        
        #########       for FSLE     #########
        def get_Td(r_single, rbins, rtime):
            # r_single: shape (rtime,)
            # rbins: shape (rbin,)
            # rtime: shape (rtime,)
            
            # find indices of minimum separation
            minidx = np.argmin(r_single)
            rd = r_single[minidx:]    # starts from minidx
            rtime_rd = rtime[minidx:] # starts from minidx
            
            # find the first time indices when separation is larger than rbins
            Td = np.full_like(rbins, np.nan)
            for i, rb in enumerate(rbins):
                mask = rd > rb
                if np.any(mask):
                    Td[i] = rtime_rd[np.argmax(mask)]
            
            return Td
        
        alpha = rbins.values[-1] / rbins.values[-2] # ratio of neighbouring bins

        ###### original r for FSLEO ######
        Td = xr.apply_ufunc(
            get_Td,
            r.chunk({'rtime':-1}) if r.chunks else r,
            rbins,
            r['rtime'],
            input_core_dims=[['rtime'], ['rbin'], ['rtime']],
            output_core_dims=[['rbin']],
            vectorize=True,
            dask='parallelized' if r.chunks else False,
            output_dtypes=[r.dtype]
        )

        Td = Td.assign_coords(pair=r['pair'], rbin=rbins)
        FSLEO = Td.diff('rbin')
        numFO = xr.where(np.isnan(FSLEO), 0, 1).sum('pair').load()
        FSLEO = (np.log(alpha) / FSLEO.where(FSLEO != 0)).mean('pair').load()
        FSLEO['rbin'] = rbinv[:-1]

        ###### interpolated r for FSLEI ######
        Td = xr.apply_ufunc(
            get_Td,
            r_da.chunk({'rtime':-1}) if r_da.chunks else r_da,
            rbins,
            r_da['rtime'],
            input_core_dims=[['rtime'], ['rbin'], ['rtime']],
            output_core_dims=[['rbin']],
            vectorize=True,
            dask='parallelized' if r_da.chunks else False,
            output_dtypes=[r.dtype]
        )
        
        Td = Td.assign_coords(pair=r['pair'], rbin=rbins)
        FSLEI = Td.diff('rbin')
        numFI = xr.where(np.isnan(FSLEI), 0, 1).sum('pair').load()
        FSLEI = (np.log(alpha) / FSLEI.where(FSLEI != 0)).mean('pair').load()
        FSLEI['rbin'] = rbinv[:-1]
        
        fagr = np.log(r).differentiate('rtime')
        
        # Determine whether inputs are chunked (Dask arrays) or NumPy arrays,
        # and select the appropriate histogram function.
        def run_histogram(a_da, weights_da, bins):
            a_arr = a_da.data
            w_arr = weights_da.data
            if isinstance(a_arr, dsa.core.Array):
                if hasattr(w_arr, 'chunks') and w_arr.chunks != a_arr.chunks:
                    w_arr = w_arr.rechunk(a_arr.chunks)

                # Drop invalid samples by zeroing their weights and forcing
                # coordinates into a valid numeric value.
                valid = dsa.isfinite(a_arr) & dsa.isfinite(w_arr)
                a_clean = dsa.where(valid, a_arr, bins[0])
                w_clean = dsa.where(valid, w_arr, 0.0)

                return dsa_histogram(a_clean, bins=bins, weights=w_clean)
            else:
                if hasattr(a_arr, 'compute'):
                    a_arr = a_arr.compute()
                if hasattr(w_arr, 'compute'):
                    w_arr = w_arr.compute()

                valid = np.isfinite(a_arr) & np.isfinite(w_arr)
                a_clean = a_arr[valid]
                w_clean = w_arr[valid]

                return np.histogram(a_clean, bins=bins, weights=w_clean)

        #########  accumulated within bins  #########
        tmp_K2 , _ = run_histogram(r, (r**2).differentiate('rtime')/2, rbinv)
        tmp_S2 , _ = run_histogram(r, du**2+dv**2, rbinv)
        tmp_S2L, _ = run_histogram(r, dul**2, rbinv)
        tmp_S2T, _ = run_histogram(r, dut**2, rbinv)
        tmp_S3 , _ = run_histogram(r, dul*(du**2+dv**2), rbinv)
        tmp_FG , _ = run_histogram(r, fagr, rbinv)
        tmp_FGp, _ = run_histogram(r, xr.where(fagr>0, fagr, 0), rbinv)
        tmp_noS, _ = run_histogram(r, du-du+1, rbinv)
        tmp_noP, _ = run_histogram(r, xr.where(fagr>0, 1, 0), rbinv)
        
        tmp_K2  = xr.where(np.isnan(tmp_K2 ), 0, tmp_K2 )
        tmp_S2  = xr.where(np.isnan(tmp_S2 ), 0, tmp_S2 )
        tmp_S2L = xr.where(np.isnan(tmp_S2L), 0, tmp_S2L)
        tmp_S2T = xr.where(np.isnan(tmp_S2T), 0, tmp_S2T)
        tmp_S3  = xr.where(np.isnan(tmp_S3 ), 0, tmp_S3 )
        tmp_FG  = xr.where(np.isnan(tmp_FG ), 0, tmp_FG )
        tmp_FGp = xr.where(np.isnan(tmp_FGp), 0, tmp_FGp)
        
        K2    = xr.DataArray(tmp_K2 , dims='rbin', coords={'rbin':rbins[:-1]}).load()
        S2    = xr.DataArray(tmp_S2 , dims='rbin', coords={'rbin':rbins[:-1]}).load()
        S2L   = xr.DataArray(tmp_S2L, dims='rbin', coords={'rbin':rbins[:-1]}).load()
        S2T   = xr.DataArray(tmp_S2T, dims='rbin', coords={'rbin':rbins[:-1]}).load()
        S3    = xr.DataArray(tmp_S3 , dims='rbin', coords={'rbin':rbins[:-1]}).load()
        FAGR  = xr.DataArray(tmp_FG , dims='rbin', coords={'rbin':rbins[:-1]}).load()
        FAGRp = xr.DataArray(tmp_FGp, dims='rbin', coords={'rbin':rbins[:-1]}).load()
        numS  = xr.DataArray(tmp_noS, dims='rbin', coords={'rbin':rbins[:-1]}).load()
        numP  = xr.DataArray(tmp_noP, dims='rbin', coords={'rbin':rbins[:-1]}).load()
        
        K2    /= numS
        S2    /= numS
        S2L   /= numS
        S2T   /= numS
        S3    /= numS
        FAGR  /= numS
        FAGRp /= numP

        S2rr, S2dd = rotational_divergent_components(S2L, S2T)
        
        return xr.merge([K2.rename('K2'), S2.rename('S2'),
                         S2L.rename('S2L'), S2T.rename('S2T'),
                         S2rr.rename('S2r'), S2dd.rename('S2d'), S3.rename('S3'),
                         FSLEO.rename('FSLEO'), FSLEI.rename('FSLEI'),
                         FAGR.rename('FAGR'), FAGRp.rename('FAGRp'),
                         numS.rename('num_S2'), numP.rename('num_FAGRp'),
                         numFO.rename('num_FSLEO'), numFI.rename('num_FSLEI')])
    
    
    def _get_all_chunked(self,
        ID: np.array,
        rowsize: np.array,
        xpos: np.array,
        ypos: np.array,
        times: np.array,
        chunk: int,
    ) -> Tuple[np.array, np.array, np.array, np.array,
               np.array, np.array, np.array, np.array]:
        """Generate all ragged pair information block-by-block.

        Notes
        -----
        Uses **pre-allocated arrays** filled block-by-block instead of
        accumulating Python lists and concatenating at the end.  This
        avoids the 2× peak memory of the list + ``np.concatenate``
        pattern.

        The ``chunk`` parameter controls how many outer-loop trajectories
        are processed before results are flushed into the pre-allocated
        arrays.  A smaller chunk does not change peak memory for the
        *pair metadata* (which is small), but it is kept for API
        consistency with the non-ragged path.

        To actually control peak memory for the much larger
        ``load_variable`` arrays, see ``load_variable`` which honours
        ``self.chunk`` via dask ``from_delayed``.
        """
        ntraj = len(ID)
        dtype = xpos.dtype

        # verify that each trajectory has at least one record
        nvalid = 0
        for r in rowsize:
            if r > 0:
                nvalid += 1
        if nvalid != ntraj:
            raise Exception(f'there are {ntraj - nvalid} empty trajectories')

        if chunk < 1:
            raise Exception('chunk should be >= 1 for ragged chunked processing')

        # Upper bound on the number of pairs (all trajectory combinations).
        # Pre-allocating this many slots is cheap because pair metadata
        # is tiny compared to load_variable arrays.
        npair_max = ntraj * (ntraj - 1) // 2

        pID  = np.empty((npair_max, 2), dtype=np.int32)
        tlen = np.empty((npair_max,),   dtype=np.int32)
        stim = np.empty((npair_max,),   dtype=times.dtype)
        r0   = np.empty((npair_max,),   dtype=dtype)
        xp   = np.empty((npair_max, 2), dtype=dtype)
        yp   = np.empty((npair_max, 2), dtype=dtype)
        idx1 = np.empty((npair_max, 2), dtype=np.int32)
        idx2 = np.empty((npair_max, 2), dtype=np.int32)

        # global start index for each trajectory in flattened ragged arrays
        idx = np.roll(rowsize.cumsum(), 1)
        idx[0] = 0

        pos = 0  # current fill position in pre-allocated arrays

        for ibeg in range(0, ntraj, chunk):
            iend = min(ibeg + chunk, ntraj)

            for i in range(ibeg, iend):
                for j in range(i + 1, ntraj):
                    idxI, idxJ = idx[i], idx[j]

                    tsI = times[idxI:idxI + rowsize[i]]
                    tsJ = times[idxJ:idxJ + rowsize[j]]

                    i1, i2, j1, j2 = get_overlap_indices(tsI, tsJ)

                    if i1 is None:
                        continue

                    x1, x2 = xpos[idxI + i1], xpos[idxJ + j1]
                    y1, y2 = ypos[idxI + i1], ypos[idxJ + j1]

                    pID[pos]  = [ID[i], ID[j]]
                    xp[pos]   = [x1, x2]
                    yp[pos]   = [y1, y2]
                    stim[pos] = times[idxI + i1]
                    tlen[pos] = i2 - i1

                    if self.coord == 'latlon':
                        xp1, xp2 = np.deg2rad([x1, x2])
                        yp1, yp2 = np.deg2rad([y1, y2])
                        r0[pos] = geodist(xp1, xp2, yp1, yp2)
                    else:
                        r0[pos] = np.hypot(x1 - x2, y1 - y2)

                    idx1[pos] = [idxI + i1, idxI + i2]
                    idx2[pos] = [idxJ + j1, idxJ + j2]
                    pos += 1

        # Truncate to the actual number of valid pairs
        npair = pos
        pID  = pID[:npair]
        tlen = tlen[:npair]
        stim = stim[:npair]
        r0   = r0[:npair]
        xp   = xp[:npair]
        yp   = yp[:npair]
        idx1 = idx1[:npair]
        idx2 = idx2[:npair]

        if self.coord == 'latlon':
            r0 = r0 * self.Rearth

        return pID, tlen, stim, r0, xp, yp, idx1, idx2
    
    def _select_pair_var(self,
        var: xr.DataArray,
        pair_idx: xr.DataArray
    ) -> xr.DataArray:
        """replace isel as isel cannot handle dask array

        Select values from `var` for index pairs in `pair_idx` (dask-backed).
        Returns xr.DataArray with dims ('pair','particle').
        Works for var that has a particle index dimension named self.ID (and possibly time).
    
        Parameters
        ----------
        var : xr.DataArray
            Input data array to select pairs from.
        pair_idx : xr.DataArray
            Data array containing the pair indices.  Should have dims ('pair','particle')
        """
        # select time=0 if var has time dimension
        if self.time in var.dims:
            v0 = var.isel({self.time: 0})
        else:
            v0 = var

        # underlying data as dask array
        data = v0.data
        if not isinstance(data, dsa.core.Array):
            # wrap numpy into dask to ensure consistent behavior
            chunks = (self.chunk if self.chunk > 0 else min(1_000_000, v0.shape[0]))
            data = dsa.from_array(data, chunks=chunks)

        # axis along which to take indices (should correspond to self.ID)
        try:
            id_axis = v0.get_axis_num(self.ID)
        except Exception:
            id_axis = 0

        # pair_idx.data is a dask array shape (npair,2)
        idx0 = pair_idx.data[:, 0]
        idx1 = pair_idx.data[:, 1]

        # use dask.take to gather values; result shape (npair,)
        val0 = dsa.take(data, idx0, axis=id_axis)
        val1 = dsa.take(data, idx1, axis=id_axis)

        # stack into (npair, 2)
        stacked = dsa.stack([val0, val1], axis=1)

        # pair length (integer)
        npair = int(pair_idx.sizes['pair'])

        pair_coord = np.arange(npair, dtype='int64')
        particle_coord = np.array([0, 1], dtype='int32')

        return xr.DataArray(stacked, dims=('pair', 'particle'),
                            coords={'pair': pair_coord, 'particle': particle_coord},
                            name=var.name)  

    def _get_pair_index(self,
        ntraj: int,
        chunk: int = 1000,
    ) -> xr.DataArray:
        """
        Return a dask-backed xr.DataArray of all unique pairs for `ntraj` items.
        The returned DataArray has dims ('pair','particle') where 'pair' is chunked.
    
        Parameters
        ----------
        ntraj : int
            Number of particles. Number of npair = ntraj*(ntraj-1)//2.
        chunk : int or None
            Chunk size for the dask array along the pair dimension. If None, a default
            is chosen (min(1_000_000, npair) to avoid too-fine chunking).
        """
        if ntraj < 2:
            raise Exception('ntraj should be >= 2')

        if chunk < 1:
            raise Exception('chunk should be >= 1')
    
        npair = ntraj * (ntraj - 1) // 2
        chunk = min(npair, chunk)
    
        # k: linear index 0..npair-1 (dask-backed)
        k = dsa.arange(npair, chunks=chunk, dtype='int64')
    
        # use float64 for sqrt to avoid integer overflow & ensure safe sqrt
        two_n_minus1 = float(2 * ntraj - 1)
    
        # i = floor(((2n-1) - sqrt((2n-1)^2 - 8*k)) / 2)
        # cast k to float64 for sqrt calculation
        i_float = dsa.floor((two_n_minus1 -
                             dsa.sqrt(two_n_minus1**2 - 8.0 * k.astype('float64'))) / 2.0)
        i = i_float.astype('int64')
    
        # cumulative count up to row i: cum_i = i*n - (i*(i+1))//2
        # use integer operations on dask arrays
        cum_i = i * ntraj - (i * (i + 1) // 2)
    
        offset = k - cum_i  # offset within row i
        j = (i.astype('int64') + 1 + offset).astype('int64')
    
        pair_idx = dsa.stack([i, j], axis=1)
        pair_crd = dsa.arange(npair, chunks=chunk, dtype='int64')
    
        return xr.DataArray(pair_idx, name='pID', dims=('pair', 'particle'),
                            coords={'pair': pair_crd,
                                    'particle': np.array([0, 1], dtype='int32')})
    
    
    def __repr__(self) -> str:
        """Print this class as a string"""
        if np.issubdtype(self.ds_traj[self.time].dtype, np.datetime64):
            suffix = ' (days)'
        else:
            suffix = ''
        
        return \
            f' RelativeDispersion class with:\n'\
            f'   xpos: {self.xpos} \n'\
            f'   ypos: {self.ypos} \n'\
            f'   uvel: {self.uvel} \n'\
            f'   vvel: {self.vvel} \n'\
            f'   time: {self.time} \n'\
            f'  coord: {self.coord}\n'\
            f'  delta: {self.dt:6.3f}{suffix}\n'\
            f'maxtlen: {self.maxtlen}\n'\



"""
Helper (private) methods are defined below
"""




