# -*- coding: utf-8 -*-
"""
Plot module of xdispersion: convenience plotting for relative-dispersion results.

:func:`panel` creates a single matplotlib axis showing one or more
empirical curves (with optional error bars) together with one or more
analytic predictions.  The function handles log-log axes, multi-curve
legends, and twin-axis layouts.
"""
import numpy as np
from .utils import mean_at_rbin
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Some plot functions are defined below
"""
def panel(ax, vs, avs, title, yscale, xscale, ylim, xlim, ylabel, xlabel=None, loc='ll',
          fontsize=12, timebased=True, ncols=2, thre=None, size=1.6, rebins=None):
    """Plot a panel of measures, with analytic predictions
    
    Parameters
    ----------
    ax: axe
        An axe for panel plot
    vs: list of dict
        A simple data-structure for a single curve
    avs: list of dict
        A simple data-structure for an analytic curve
    title: str
        title
    yscale: str
        "log" or "linear"
    xscale: str
        "log" or "semilog" or "linear"
    ylim: list of two float
        range of y-axis
    xlim: list of two float
        range of x-axis
    ylabel: str
        y-axis label
    xlabel: str
        x-axis label
    loc: str
        location of the legend
    fontsize: int
        font size
    timebased: boolean
        Whether the x-axis is time or separation
    ncols: int
        columns of 2
    thre: float
        x-value where in the semilog case the scale changes
    size: int
        The relative size of linear range to the log range (in semilog case)
    rebins: numpy.array or xarray.DataArray
        1D array of separations to resample the original one
    
    Returns
    -------
    ax: axe
        The log axe for this plot.
    ax2: axe
        The linear axe for this plot.
    lgd: list
        List of legends
    """
    ax, lgd = _panel_vars(ax, vs, avs, timebased=timebased, rebins=rebins)
    ax, ax2, lgd = _add_axes(ax, lgd, title, yscale, xscale, ylim, xlim, ylabel, xlabel,
                             loc=loc, fontsize=fontsize, timebased=timebased,
                             ncols=ncols, thre=thre, size=size)
    
    if ax2 != None:
        ax, _ = _panel_vars(ax2, vs, avs, timebased=timebased, rebins=rebins)
    
    return ax, ax2, lgd


"""
Helper (private) methods are defined below
"""

def _panel_vars(ax, vs, avs, timebased=True, rebins=None):
    """Plot a single panel
    
    Parameters
    ----------
    ax: axe
        An axe for panel plot
    vs: list of dict
        A simple data-structure for a single curve
    avs: list of dict
        A simple data-structure for an analytic curve
    timebased: boolean
        Whether the x-axis is time or separation
    rebins: numpy.array or xarray.DataArray
        1D array of separations to resample the original one
    
    Returns
    -------
    ax: axe
        The axe for this plot.
    lgd: list
        List of legends
    """
    lgd = []
    
    for v in vs:
        v      = v.copy()
        r      = v.pop('r', None)
        var    = v.pop('var')
        label  = v.pop('label')
        method = v.pop('method')
        CIs    = v.pop('CIs') if 'CIs' in v else None
        xaxis  = var['rtime'] if timebased else r
        color  = v['color'] if 'color' in v else None
        if color == None and 'edgecolor' in v:
            color = v['edgecolor']
        #color = 'gray'
        alpha = 0.15
        
        if rebins is not None:
            if 'r' in var.dims:
                var = mean_at_rbin(var, var['r'], rebins.values)
            elif 'rbin' in var.dims:
                var = mean_at_rbin(var, var['rbin'], rebins.values)
            else:
                var = mean_at_rbin(var, xaxis, rebins.values)
        
        tmp = var.name
        var = var.rename('')

        if method == 'plot':
            if rebins is None:
                if 'rbin' in var.dims:
                    lgd.append(ax.plot(var['rbin'], var, label=label, **v))
                elif 'r' in var.dims:
                    lgd.append(ax.plot(var['r'], var, label=label, **v))
                else:
                    lgd.append(ax.plot(xaxis, var, label=label, **v))
            else:
                if 'rbin' in var.dims:
                    lgd.append(ax.plot(var['rbin'], var, label=label, **v))
                else:
                    lgd.append(ax.plot(var['r'], var, label=label, **v))

            if CIs is not None:
                if rebins is None:
                    ax.fill_between(xaxis, CIs[0], CIs[1], alpha=alpha, color=color, zorder=-5)
                else:
                    CIL = mean_at_rbin(CIs[0], xaxis, rebins.values)
                    CIU = mean_at_rbin(CIs[1], xaxis, rebins.values)
                    ax.fill_between(var['rbin'], CIL, CIU, alpha=alpha, color=color, zorder=-5)
            
        elif method == 'scatter':
            if rebins is None:
                if 'rbin' in var.dims:
                    lgd.append(ax.scatter(var['rbin'], var, label=label, **v))
                elif 'r' in var.dims:
                    lgd.append(ax.scatter(var['r'], var, label=label, **v))
                else:
                    lgd.append(ax.scatter(xaxis, var, label=label, **v))
            else:
                if 'rbin' in var.dims:
                    lgd.append(ax.scatter(var['rbin'], var, label=label, **v))
                else:
                    lgd.append(ax.scatter(var['r'], var, label=label, **v))

            if CIs is not None:
                if rebins is None:
                    ax.fill_between(xaxis, CIs[0], CIs[1], alpha=alpha, color=color, zorder=-5)
                else:
                    CIL = mean_at_rbin(CIs[0], xaxis, rebins.values)
                    CIU = mean_at_rbin(CIs[1], xaxis, rebins.values)
                    ax.fill_between(var['rbin'], CIL, CIU, alpha=alpha, color=color, zorder=-5)
            
        else:
            raise Exception(f'invalid method {method}')

        var = var.rename(tmp)
    
    for v in avs:
        v      = v.copy()
        r      = v.pop('r', None)
        var    = v.pop('var')
        label  = v.pop('label')
        method = v.pop('method')
        xaxis  = var['rtime'] if timebased else r
        
        tmp = var.name
        var = var.rename('')

        if method == 'plot':
            lgd.append(ax.plot(xaxis, var, label=label, **v))
            
        elif method == 'scatter':
            lgd.append(ax.scatter(xaxis, var, label=label, **v))
            
        else:
            raise Exception(f'invalid method {method}')
        
        var = var.rename(tmp)
    
    return ax, lgd


def _add_axes(ax, lgd, title, yscale, xscale, ylim, xlim, ylabel, xlabel=None, loc='ll',
              fontsize=12, timebased=True, ncols=2, thre=None, size=1.6):
    """Add an axes in a single panel but use different x-scales
    
    Parameters
    ----------
    ax: axe
        An axe for panel plot
    lgd: list
        A list of legends
    title: str
        title
    yscale: str
        "log" or "linear"
    xscale: str
        "log" or "semilog" or "linear"
    ylim: list of two float
        range of y-axis
    xlim: list of two float
        range of x-axis
    ylabel: str
        y-axis label
    xlabel: str
        x-axis label
    loc: str
        location of the legend
    fontsize: int
        font size
    timebased: boolean
        Whether the x-axis is time or separation
    ncols: int
        columns of 2
    thre: float
        x-value where in the semilog case the scale changes
    size: int
        The relative size of linear range to the log range (in semilog case)
    
    Returns
    -------
    ax: axe
        The log axe for this plot.
    ax2: axe
        The linear axe for this plot.
    lgd: list
        List of legends
    """
    ax2 = None

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize-1)
    else:
        #if timebased:
        #    ax.set_xlabel('time', fontsize=fontsize-1)
        #else:
        #    ax.set_xlabel('r', fontsize=fontsize-1)
        ax.set_xlabel('', fontsize=fontsize-1)
    
    if xscale == 'log':
        ax.set_xscale('log')
        # ax.set_xticks(_ticks)
        # ax.set_xticklabels(_labels, fontsize=fontsize-1)
        ax.set_xlim(xlim)
        #ax.tick_params(axis='x', labelsize=fontsize-1)

        if yscale == 'log':
            ax.set_yscale('log')
            # ax.set_yticks(_ticks)
            # ax.set_yticklabels(_labels, fontsize=fontsize-1)
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
        elif yscale == 'linear':
            ax.set_yscale('linear')
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
            ax.tick_params(axis='y', labelsize=fontsize-1)
        else:
            raise Exception('unsupported yscale: '+yscale)
    
    elif xscale == 'linear':
        ax.set_xscale('linear')
        ax.set_xlim(xlim)
        ax.tick_params(axis='x', labelsize=fontsize-1)

        if yscale == 'log':
            ax.set_yscale('log')
            # ax.set_yticks(_ticks)
            # ax.set_yticklabels(_labels, fontsize=fontsize-1)
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
        elif yscale == 'linear':
            ax.set_yscale('linear')
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
        else:
            raise Exception('unsupported yscale: '+yscale)
    
    elif xscale == 'semilog':
        if thre is None:
            thre = (xlim[1] - xlim[0]) / 4.0
        
        ax.set_xscale('log')
        # ax.set_xticks(_ticks)
        # ax.set_xticklabels(_labels, fontsize=fontsize-1)
        ax.set_xlim((xlim[0], thre))
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(xlabel, fontsize=fontsize-1)
        ax.tick_params(axis='x', labelsize=fontsize-1)
        
        if yscale == 'log':
            ax.set_yscale('log')
            # ax.set_yticks(_ticks)
            # ax.set_yticklabels(_labels, fontsize=fontsize-1)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
            ax.set_ylim(ylim)
        elif yscale =='linear':
            ax.set_yscale('linear')
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel, fontsize=fontsize-1)
            ax.tick_params(axis='y', labelsize=fontsize-1)
        else:
            raise Exception('unsupported yscale: '+yscale)

        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes('right', size=size, pad=0)
        ax2.set_xscale('linear')
        ax2.set_xlim((thre, xlim[1]))
        ax2.tick_params(axis='x', labelsize=fontsize-1)

        if yscale == 'log':
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', labelsize=fontsize-1, colors='#00000000')
            ax2.set_ylim(ylim)
        elif yscale == 'linear':
            ax2.set_yscale('linear')
            ax2.tick_params(axis='y', labelsize=fontsize-1, colors='#00000000')
            ax2.set_ylim(ylim)
        else:
            raise Exception('unsupported yscale: '+yscale)

        ax2.set_ylabel('', fontsize=fontsize-1)
        ax2.set_yticks([], minor=True)
        ax2.set_yticklabels([], fontsize=0)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlabel('', fontsize=fontsize-1)
    else:
        raise Exception('unsupported xscale: '+xscale)

    ax.set_title(title, fontsize=fontsize)

    if loc != None:
        if xscale == 'semilog':
            if 'l' == loc[1]:
                ax.legend(lgd, loc=loc, prop={'size': fontsize-1}, ncols=ncols)
            elif 'r' == loc[1]:
                ax2.legend(lgd, loc=loc, prop={'size': fontsize-1}, ncols=ncols)
            else:
                raise Exception('invalid location for legend')
        else:
            ax.legend(lgd, loc=loc, prop={'size': fontsize-1}, ncols=ncols)
    
    return ax, ax2, lgd
