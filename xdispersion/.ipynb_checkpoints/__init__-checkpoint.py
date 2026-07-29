# -*- coding: utf-8 -*-

from .analytics import ana_r2, ana_Ku, ana_K2, ana_CIST,\
                       ana_S2, ana_S3, ana_PDF,\
                       num_r2, num_Ku, num_K2, num_CIST
from .core import RelativeDispersion
from .measures import rel_disp, vel_struct_func, rel_diff,\
                      famp_growth_rate, init_memory, anisotropy,\
                      lagr_vel_corr, kurtosis, cen_vul_exp,\
                      fsize_lyap_exp, cumul_inv_sep_time,\
                      prob_dens_func, cumul_dens_func,\
                      principle_axis_components,\
                      rotational_divergent_components
from .fpe import integrate, CFLcondition
from .plot import panel
from .template import cal_all_measures
from .utils import geodist, gen_rbins, sum_at_rbin, mean_at_rbin

__version__ = "0.0.1"
