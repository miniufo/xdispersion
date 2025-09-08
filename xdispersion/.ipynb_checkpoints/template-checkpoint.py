# -*- coding: utf-8 -*-
"""
Created on 2024.11.20

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from tqdm import tqdm
from .measures import rel_disp, vel_struct_func,\
     rel_diff, famp_growth_rate, init_memory, anisotropy,\
     lagr_vel_corr, kurtosis, cen_vul_exp, fsize_lyap_exp,\
     cumul_inv_sep_time, prob_dens_func, cumul_dens_func
from .utils import mean_at_rbin, sum_at_rbin


"""
A template for calculating all the available
measures given a set of pair particles, and
return all as a xarray.Dataset that can be
easily output to a file.
"""

def cal_all_measures(rd, pairs, rbins, ensemble=0, nproc=1):
    """Calculate all available measures.  Users can add their own measures.
    
    Parameters
    ----------
    rd: RelativeDispersion instance
        Relative dispersion application
    pairs: list of two xr.Datasets
        Pair particles
    rbins: xr.DataArray
        A specified separation bins for r-based measures
    
    Returns
    -------
    dset: xr.Dataset
        All measures in a single xr.Dataset.
    """
    #-------------------- get building-blocks for different measures -----------------#
    rx, ry, rxy, r, rpb = rd.separation_measures(pairs)
    du, dv, dul, dut, vmi, vmj, uv = rd.velocity_measures(pairs)
    
    #----------------------- measures averaged at constant time --------------------#
    with tqdm(total=29, ncols=80) as pbar:
        mean_at = 'const-t'
        r2_t, CILr2_t, CIUr2_t = rel_disp(r, order=2,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        rpb2_t, CILrpb2_t, CIUrpb2_t = rel_disp(rpb, order=2,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2_t, CILS2_t, CIUS2_t = vel_struct_func(np.hypot(du, dv), r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2ll_t, CILS2ll_t, CIUS2ll_t = vel_struct_func(dul, r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2tr_t, CILS2tr_t, CIUS2tr_t = vel_struct_func(dut, r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S3_t, CILS3_t, CIUS3_t = vel_struct_func(dul*(du**2+dv**2), r, order=1,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        K2_t, CILK2_t, CIUK2_t = rel_diff(r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        FAGR_t, CILFAGR_t, CIUFAGR_t = famp_growth_rate(r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        initm_t, CILinitm_t, CIUinitm_t = init_memory(rx, ry, du, dv, r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        aniso_t, CILaniso_t, CIUaniso_t = anisotropy(rx, ry, rxy, r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        LVC_t, CILLVC_t, CIULVC_t = lagr_vel_corr(uv, vmi, vmj, r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        ku_t, CILku_t, CIUku_t = kurtosis(r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        CVE_t, CILCVE_t, CIUCVE_t = cen_vul_exp(r,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        num_t = xr.where(np.isnan(r), 0, 1).sum('pair')
    
    #--------------------- measures average at constant separation ------------------#
        mean_at = 'const-r'
        r2_r, CILr2_r, CIUr2_r = rel_disp(r, order=2, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        rpb2_r, CILrpb2_r, CIUrpb2_r = rel_disp(rpb, order=2, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2_r, CILS2_r, CIUS2_r = vel_struct_func(np.hypot(du, dv), r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2ll_r, CILS2ll_r, CIUS2ll_r = vel_struct_func(dul, r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S2tr_r, CILS2tr_r, CIUS2tr_r = vel_struct_func(dut, r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        S3_r, CILS3_r, CIUS3_r = vel_struct_func(dul*(du**2+dv**2), r, order=1, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        K2_r, CILK2_r, CIUK2_r = rel_diff(r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        FAGR_r, CILFAGR_r, CIUFAGR_r = famp_growth_rate(r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        initm_r, CILinitm_r, CIUinitm_r = init_memory(rx, ry, du, dv, r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        aniso_r, CILaniso_r, CIUaniso_r = anisotropy(rx, ry, rxy, r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        LVC_r, CILLVC_r, CIULVC_r = lagr_vel_corr(uv, vmi, vmj, r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        ku_r, CILku_r, CIUku_r = kurtosis(r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        CVE_r, CILCVE_r, CIUCVE_r = cen_vul_exp(r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        FSLE_r, CILFSLE_r, CIUFSLE_r = fsize_lyap_exp(r, rbins=rbins,
                                          mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        CIST_r, CILCIST_r, CIUCIST_r = cumul_inv_sep_time(r, rbins=rbins, lower=0.10, upper=0.90,
                                          maskout=[1e-8, 5e3], mean_at=mean_at, ensemble=ensemble, nproc=nproc)
        pbar.update(1)
        num_r = sum_at_rbin(xr.where(np.isnan(r), 0, 1), r, rbins=rbins)
        
    #----------------------------- other measures --------------------------#
        pbar.update(1)
        PDF = prob_dens_func(r, rbins=rbins)
        CDF = cumul_dens_func(PDF, rbins)
    
    
    #------------------------------- output list ----------------------------#
    vs = [r2_t   , CILr2_t   , CIUr2_t   , rpb2_t , CILrpb2_t , CIUrpb2_t ,
          S2_t   , CILS2_t   , CIUS2_t   ,
          S2ll_t , CILS2ll_t , CIUS2ll_t , S2tr_t , CILS2tr_t , CIUS2tr_t ,
          S3_t   , CILS3_t   , CIUS3_t   , K2_t   , CILK2_t   , CIUK2_t   ,
          FAGR_t , CILFAGR_t , CIUFAGR_t , initm_t, CILinitm_t, CIUinitm_t,
          aniso_t, CILaniso_t, CIUaniso_t, LVC_t  , CILLVC_t  , CIULVC_t  ,
          ku_t   , CILku_t   , CIUku_t   , CVE_t  , CILCVE_t  , CIUCVE_t  , num_t  ,
          r2_r   , CILr2_r   , CIUr2_r   , rpb2_r , CILrpb2_r , CIUrpb2_r ,
          S2_r   , CILS2_r   , CIUS2_r   ,
          S2ll_r , CILS2ll_r , CIUS2ll_r , S2tr_r , CILS2tr_r , CIUS2tr_r ,
          S3_r   , CILS3_r   , CIUS3_r   , K2_r   , CILK2_r   , CIUK2_r   ,
          FAGR_r , CILFAGR_r , CIUFAGR_r , initm_r, CILinitm_r, CIUinitm_r,
          aniso_r, CILaniso_r, CIUaniso_r, LVC_r  , CILLVC_r  , CIULVC_r  ,
          ku_r   , CILku_r   , CIUku_r   , CVE_r  , CILCVE_r  , CIUCVE_r  , num_r  ,
          FSLE_r , CILFSLE_r , CIUFSLE_r , CIST_r , CILCIST_r , CIUCIST_r ,
          PDF    , CDF       ,
          ]
    
    #------------------------------- output names ----------------------------#
    names = ['r2_t'   , 'CILr2_t'   , 'CIUr2_t'   , 'rpb2_t' , 'CILrpb2_t' , 'CIUrpb2_t' ,
             'S2_t'   , 'CILS2_t'   , 'CIUS2_t'   ,
             'S2ll_t' , 'CILS2ll_t' , 'CIUS2ll_t' , 'S2tr_t' , 'CILS2tr_t' , 'CIUS2tr_t' ,
             'S3_t'   , 'CILS3_t'   , 'CIUS3_t'   , 'K2_t'   , 'CILK2_t'   , 'CIUK2_t'   ,
             'FAGR_t' , 'CILFAGR_t' , 'CIUFAGR_t' , 'initm_t', 'CILinitm_t', 'CIUinitm_t',
             'aniso_t', 'CILaniso_t', 'CIUaniso_t', 'LVC_t'  , 'CILLVC_t'  , 'CIULVC_t'  ,
             'ku_t'   , 'CILku_t'   , 'CIUku_t'   , 'CVE_t'  , 'CILCVE_t'  , 'CIUCVE_t'  , 'num_t'  ,
             'r2_r'   , 'CILr2_r'   , 'CIUr2_r'   , 'rpb2_r' , 'CILrpb2_r' , 'CIUrpb2_r' ,
             'S2_r'   , 'CILS2_r'   , 'CIUS2_r'   ,
             'S2ll_r' , 'CILS2ll_r' , 'CIUS2ll_r' , 'S2tr_r' , 'CILS2tr_r' , 'CIUS2tr_r' ,
             'S3_r'   , 'CILS3_r'   , 'CIUS3_r'   , 'K2_r'   , 'CILK2_r'   , 'CIUK2_r'   ,
             'FAGR_r' , 'CILFAGR_r' , 'CIUFAGR_r' , 'initm_r', 'CILinitm_r', 'CIUinitm_r',
             'aniso_r', 'CILaniso_r', 'CIUaniso_r', 'LVC_r'  , 'CILLVC_r'  , 'CIULVC_r'  ,
             'ku_r'   , 'CILku_r'   , 'CIUku_r'   , 'CVE_r'  , 'CILCVE_r'  , 'CIUCVE_r'  , 'num_r'  ,
             'FSLE_r' , 'CILFSLE_r' , 'CIUFSLE_r' , 'CIST_r' , 'CILCIST_r' , 'CIUCIST_r',
             'PDF'    , 'CDF'       ,]
    
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
                '2nd-order longitudinal structure function averaged at constant time',
                'lower bound for 2nd-order longitudinal structure function',
                'upper bound for 2nd-order longitudinal structure function',
                '2nd-order transversal structure function averaged at constant time',
                'lower bound for 2nd-order transversal structure function',
                'upper bound for 2nd-order transversal structure function',
                '3rd-order structure function averaged at constant time',
                'lower bound for 3rd-order structure function',
                'upper bound for 3rd-order structure function',
                'relative diffusivity averaged at constant time',
                'lower bound for relative diffusivity',
                'upper bound for relative diffusivity',
                'finite-amplitude growth rate averaged at constant time',
                'lower bound for finite-amplitude growth rate',
                'upper bound for finite-amplitude growth rate',
                'initial memory averaged at constant time',
                'lower bound for initial memory',
                'upper bound for initial memory',
                'anisotropy averaged at constant time',
                'lower bound for anisotropy',
                'upper bound for anisotropy',
                'Lagrangian velocity correlation averaged at constant time',
                'lower bound for Lagrangian velocity correlation',
                'upper bound for Lagrangian velocity correlation',
                'kurtosis averaged at constant time',
                'lower bound for kurtosis',
                'upper bound for kurtosis',
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
                '2nd-order longitudinal structure function averaged at constant separation',
                'lower bound for 2nd-order longitudinal structure function',
                'upper bound for 2nd-order longitudinal structure function',
                '2nd-order transversal structure function averaged at constant separation',
                'lower bound for 2nd-order transversal structure function',
                'upper bound for 2nd-order transversal structure function',
                '3rd-order structure function averaged at constant separation',
                'lower bound for 3rd-order transversal structure function',
                'upper bound for 3rd-order transversal structure function',
                'relative diffusivity averaged at constant separation',
                'lower bound for relative diffusivity',
                'upper bound for relative diffusivity',
                'finite-amplitude growth rate averaged at constant separation',
                'lower bound for finite-amplitude growth rate',
                'upper bound for finite-amplitude growth rate',
                'initial memory averaged at constant separation',
                'lower bound for initial memory',
                'upper bound for initial memory',
                'anisotropy averaged at constant separation',
                'lower bound for anisotropy',
                'upper bound for anisotropy',
                'Lagrangian velocity correlation averaged at constant separation',
                'lower bound for Lagrangian velocity correlation',
                'upper bound for Lagrangian velocity correlation',
                'kurtosis averaged at constant separation',
                'lower bound for kurtosis',
                'upper bound for kurtosis',
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
                'probability density function of separation',
                'cumulative density function of separation']

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
