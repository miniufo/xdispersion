# -*- coding: utf-8 -*-
"""
Created on 2025.04.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.

Module: xdispersion.fpe
========================

Numerical integration of the Fokker-Planck equation for the PDF of
pair separations.

Given an initial PDF and a scale-dependent diffusivity
:math:`\\kappa(r)`, the Fokker-Planck equation

.. math::
    \\frac{\\partial P}{\\partial t} =
    -\\frac{\\partial}{\\partial r}\\bigl[\\kappa(r)\\, P\\bigr]
    + \\frac{1}{2}\\frac{\\partial^2}{\\partial r^2}
    \\bigl[\\kappa(r)^2\\, P\\bigr]

is integrated forward in time using RK4 / RK2 / Euler schemes.
This allows comparison between observed and theoretically-predicted
PDF evolution.
"""
import numpy as np
import numba as nb
import xarray as xr
from typing import Optional, Tuple, Literal, Union, List
from tqdm import tqdm


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Here we use numerical scheme to forward the Fokker-Planck  "
" equation for the PDF of pair separations.                  "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def integrate(
    init: xr.DataArray,
    kappa: xr.DataArray,
    time: Union[List, np.ndarray],
    dt: float,
    scheme: Literal['RK4', 'RK2', 'Euler'] = 'RK2',
) -> xr.DataArray:
    """Forwarding the Fokker-Planck equation by N steps using dt.
    
    Parameters
    ----------
    init: xr.DataArray
        A given (initial) state of the PDF.
    kappa: xr.DataArray
        Scale-dependent diffusivity.
    time: list
        A list of time (floats) to be integrated and stored.
        Note that these should be multiples of dt.
    dt: float
        One time step of the forwarding.
    
    Returns
    -------
    PDFnew: numpy.ndarray
        New state of the PDF after a time step of dt.
    """
    re  = []
    r   = init['r']
    CFL = CFLcondition(kappa, dt)

    if scheme == 'RK4':
        steper = RungeKutta4
    elif scheme == 'RK2':
        steper = RungeKutta2
    elif scheme == 'Euler':
        steper = Euler
    else:
        raise Exception(f'invalid scheme {scheme}')

    time = np.array(time)
    Nout = np.round(time / dt).astype(np.int32)
    
    tmp1 = init.values
    for i in tqdm(range(Nout.max()+1), ncols=80):
        tmp1 = steper(tmp1, kappa.values, r.values, dt)
        
        if i in Nout:
            re.append(tmp1)
    
    PDFs = xr.DataArray(np.array(re), dims=['time', 'r'],
                        coords={'r':r, 'time':time})

    return PDFs


"""""""""""""""""""""""
" Helper method below "
"""""""""""""""""""""""

@nb.jit(nopython=True, cache=False)
def right_hand_side(
    PDF: np.array,
    kappa: np.array,
    r: np.array,
) -> np.array:
    """Calculate R.H.S. of the Fokker-Planck equation.

    The 2D Fokker-Planck equation in cylindric coordinates is:

    dP   1 d ⌈        dP⌉
    -- = - --|kappa r --|
    dt   r dr⌊        dr⌋

    where kappa(r) is a scale-dependent diffusivity.  Then RHS is
    discretized using a centered finite-difference scheme.  The scheme
    takes into account the non-uniform r-grids.
    
    Parameters
    ----------
    PDF: numpy.ndarray
        A given state of PDF.
    kappa: numpy.ndarray
        Scale-dependent diffusivity.
    r: numpy.ndarray
        A given separations on which PDF and kappa are defined.
    
    Returns
    -------
    rhs: numpy.ndarray
        Right-hand-side of the diffusion equation.
    """
    rhs = np.zeros_like(PDF) # allocate output
    dr  = np.diff(r)
    
    for i in range(1, len(PDF)-1): # loop over interior
        kPH = (kappa[i+1] + kappa[i]) / 2.0
        kNH = (kappa[i-1] + kappa[i]) / 2.0
        rPH = (r[i+1] + r[i]) /2.0
        rNH = (r[i-1] + r[i]) /2.0
        
        rhs[i] = ((PDF[i+1] - PDF[i  ]) * kPH * rPH / dr[i  ] -
                  (PDF[i  ] - PDF[i-1]) * kNH * rNH / dr[i-1]
                 ) / ((dr[i-1] + dr[i]) / 2) / r[i]
    
    # add tendency at r=0, which makes the solution better
    rhs[ 0] = rhs[ 1] * 0.99
    #rhs[-1] = rhs[-2] * 1.1
        
    return rhs


@nb.jit(nopython=True, cache=False)
def RungeKutta4(
    PDF: np.array,
    kappa: np.array,
    r: np.array,
    dt: float,
) -> np.array:
    """4th-order Runge-Kutta method for forwarding the Fokker-Planck equation.
    
    Parameters
    ----------
    PDF: numpy.ndarray
        A given state of PDF.
    kappa: numpy.ndarray
        Scale-dependent diffusivity.
    r: numpy.ndarray
        A given separations on which PDF and kappa are defined.
    dt: float
        One time step of the forwarding.
    
    Returns
    -------
    PDFnew: numpy.ndarray
        New state of the PDF after a time step of dt.
    """
    PDFnew = np.zeros_like(PDF) # allocate output
    
    k1 = right_hand_side(PDF, kappa, r)
    k2 = right_hand_side(PDF + dt/2.0 * k1, kappa, r)
    k3 = right_hand_side(PDF + dt/2.0 * k2, kappa, r)
    k4 = right_hand_side(PDF + dt     * k3, kappa, r)
    
    PDFnew = PDF + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    return PDFnew


@nb.jit(nopython=True, cache=False)
def RungeKutta2(
    PDF: np.array,
    kappa: np.array,
    r: np.array,
    dt: float
) -> np.array:
    """2nd-order Runge-Kutta method for forwarding the Fokker-Planck equation.
    
    Parameters
    ----------
    PDF: numpy.ndarray
        A given state of PDF.
    kappa: numpy.ndarray
        Scale-dependent diffusivity.
    r: numpy.ndarray
        A given separations on which PDF and kappa are defined.
    dt: float
        One time step of the forwarding.
    
    Returns
    -------
    PDFnew: numpy.ndarray
        New state of the PDF after a time step of dt.
    """
    PDFnew = np.zeros_like(PDF) # allocate output
    
    k1 = right_hand_side(PDF, kappa, r)
    k2 = right_hand_side(PDF + dt * k1, kappa, r)
    
    PDFnew = PDF + dt/2.0 * (k1 + k2)
    
    return PDFnew


@nb.jit(nopython=True, cache=False)
def Euler(
    PDF: np.array,
    kappa: np.array,
    r: np.array,
    dt: float
) -> np.array:
    """Euler method for forwarding the Fokker-Planck equation.
    
    Parameters
    ----------
    PDF: numpy.ndarray
        A given state of PDF.
    kappa: numpy.ndarray
        Scale-dependent diffusivity.
    r: numpy.ndarray
        A given separations on which PDF and kappa are defined.
    dt: float
        One time step of the forwarding.
    
    Returns
    -------
    PDFnew: numpy.ndarray
        New state of the PDF after a time step of dt.
    """
    PDFnew = np.zeros_like(PDF) # allocate output
    
    k1 = right_hand_side(PDF, kappa, r)
    
    PDFnew = PDF + dt * k1
    
    return PDFnew


def CFLcondition(
    kappa: xr.DataArray,
    dt: float,
) -> xr.DataArray:
    """CFL condition for the Fokker-Planck equation.

    CFL = 2 * kappa * dt / dr ** 2.0 <= 1
    
    Parameters
    ----------
    kappa: numpy.ndarray
        Scale-dependent diffusivity.
    dt: float
        A given time step.
    
    Returns
    -------
    CFL: numpy.ndarray
        CFL condition.
    """
    dr = kappa['r'] - kappa['r']
    dr[1:] = kappa['r'].diff('r').values
    
    CFL = 2.0 * kappa * dt / dr ** 2.0
    
    if (CFL[1:] > 1).any(): # CFL[0] == inf
        print('Warning: CFL condition is violated')

    return CFL

