# -*- coding: utf-8 -*-
"""
xdispersion: relative dispersion of Lagrangian particle pairs.

Built on xarray and dask, this package provides analytical and numerical
dispersion statistics (relative dispersion, diffusivity, FSLE, CIST, etc.)
for two-particle Lagrangian analysis, along with forward-time integration
and plotting helpers.
"""

from .analytics import ana_r2, ana_Ku, ana_K2, ana_CIST,\
                       ana_S2, ana_S3, ana_PDF,\
                       num_r2, num_Ku, num_K2, num_CIST
from .core import RelativeDispersion
from .measures import relative_dispersion, velocity_structure_function,\
                      relative_diffusivity, finite_amplitude_growth_rate,\
                      initial_memory, anisotropy,\
                      lagrangian_velocity_correlation, kurtosis, cencini_vulpiani_exponent,\
                      finite_size_lyapunov_exponent, cumulative_inverse_separation_time,\
                      probability_density_function, cumulative_density_function,\
                      principle_axis_components,\
                      rotational_divergent_components
from .fpe import integrate, CFLcondition
from .plot import panel
from .template import cal_measures
from .utils import geodist, gen_rbins, sum_at_rbin, mean_at_rbin, loglog_fit, semilog_fit

__version__ = "0.0.6"
