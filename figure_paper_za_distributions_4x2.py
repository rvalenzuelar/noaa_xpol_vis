# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:19:34 2016

@author: raul
"""
import matplotlib.pyplot as plt
import xpol_tta_analysis as xta
from rv_utilities import discrete_cmap

from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15


params = dict(wdir_surf=130,wdir_wprof=170,
              rain_czd=0.25,nhours=2)

try:
    x08
except NameError:
    x08=xta.process(case=[8],params=params)

try:
    x09
except NameError:
    x09=xta.process(case=[9],params=params)

try:
    x10
except NameError:
    x10=xta.process(case=[10],params=params)

try:
    x11
except NameError:
    x11=xta.process(case=[11],params=params)
    
try:
    x12
except NameError:
    x12=xta.process(case=[12],params=params)

try:
    x13
except NameError:
    x13=xta.process(case=[13],params=params)

try:
    x14
except NameError:
    x14=xta.process(case=[14],params=params)

try:
    xall
except NameError:
    xall=xta.process(case=[8, 9, 10, 11, 12, 13, 14],
                     params=params)
    
cmap = discrete_cmap(7, base_cmap='Set1')

datelabs = ('12-14Jan03',
            '21-23Jan03',
            '15-16Feb03',
            '09Jan03',
            '02Feb04',
            '16-18Feb04',
            '25Feb04',
            'All')

cases = (x08,x09,x10,x11,x12,x13,x14,xall)
modes = ('ppi','rhi')

    
for mode in modes:
    fig,ax = plt.subplots(4,2,figsize=(8,11.5),
                          sharey=True,sharex=True)
    ax = ax.flatten()
    for case,n,lab in zip(cases,range(8),datelabs):
        if mode == 'ppi':
            try:
                size_tta = case.ppi_tta.index.size
            except AttributeError:
                size_tta = None
            size_ntta = case.ppi_ntta.index.size
        elif mode == 'rhi':
            try:
                size_tta = case.rhi_tta.index.size
            except AttributeError:
                size_tta = None
            size_ntta = case.rhi_ntta.index.size
            
        if n in [2,6]:
            h2 = case.plot_dist(ax=ax[n],mode=mode,
                                tta=False, color=cmap(1))
        else:
            h1 = case.plot_dist(ax=ax[n],mode=mode,
                                tta=True, color=cmap(0))
            h2 = case.plot_dist(ax=ax[n],mode=mode,
                                tta=False, color=cmap(1))
            
        ax[n].text(-15,0.9,lab,weight='bold',fontsize=13)
        ax[n].text(-12,0.78,'TTA (n={})'.format(size_tta),
                    fontsize=12)        
        ax[n].text(-12,0.62,'NO-TTA (n={})'.format(size_ntta),
                    fontsize=12)        
        
    ax[0].text(49,1.1,mode.upper(),weight='bold',fontsize=15)
    ax[4].text(-35,1.5,'Cumulative frequency',fontsize=15,
                rotation=90)
    ax[6].text(35,-0.3,'Reflectivity [dBZ]',fontsize=15)
    legend_loc=[-0.05,0.1,0.2,0.8]
    ax[0].legend([h1,h2],
                 ['',''],
                 prop={'size': 12},
                 numpoints=1,
                 handletextpad=0.1,
                 framealpha=0,
                 handlelength=1,
                 bbox_to_anchor=legend_loc)
    plt.subplots_adjust(wspace=0.1,hspace=0.15)
    
    fname='/home/raul/Desktop/fig_za_cdf_{}.png'.format(mode)
    plt.savefig(fname, dpi=300, format='png',papertype='letter',
                bbox_inches='tight')

#plt.show()

