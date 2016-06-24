# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:07:32 2016

@author: raul
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp


''' 
    if instances do not exist in iPython namespace
    then create them
'''
try:
    x08
except NameError:
    x08=xta.process(case=[8])

try:
    x10
except NameError:
    x10=xta.process(case=[10])
    
try:
    x14
except NameError:
    x14=xta.process(case=[14])

scale=1.6
plt.figure(figsize=(8*scale, 3*scale))
gs0 = gridspec.GridSpec(1, 2,
                        top=0.99, bottom=0.01,
                        left=0.15, right=0.85,
                        wspace=0.05)

gs00 = gssp(3, 1,
            subplot_spec=gs0[0],
            wspace=0, hspace=0)

gs01 = gssp(3, 1,
            subplot_spec=gs0[1],
            wspace=0, hspace=0)

ax0 = plt.subplot(gs00[0],gid='(a)')
ax1 = plt.subplot(gs01[0],gid='(b)')
ax2 = plt.subplot(gs00[1],gid='(c)')
ax3 = plt.subplot(gs01[1],gid='(d)')
ax4 = plt.subplot(gs00[2],gid='(e)')
ax5 = plt.subplot(gs01[2],gid='(f)')

axes = [ax0, ax1, ax2, ax3, ax4, ax5]

ticklabsize = 14
cbarlabsize = 14

x08.plot(ax=ax0,name='contourf',mode='rhi',target='vr',
         cbar=dict(loc='top',size='4%',units='[m/s]',fontsize=cbarlabsize),
         xticklabs=False,
         tta=False,
         ticklabsize=ticklabsize)

x08.plot(ax=ax1,name='contourf',mode='rhi',target='z',
         cbar=dict(loc='top',size='4%',units='[%]',fontsize=cbarlabsize),
         with_distr=True,
         yticklabs=False,
         xticklabs=False,
         tta=False)

x10.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
         yticklabs=False,
         xticklabs=False,
         tta=False)

x10.plot(ax=ax3,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False,
         tta=False)

x14.plot(ax=ax4,name='contourf',mode='rhi',target='vr',
         yticklabs=False,
         ticklabsize=ticklabsize,
         tta=False)

x14.plot(ax=ax5,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         ticklabsize=ticklabsize,
         tta=False)

for ax in axes:
    ax.text(0.9,0.85,ax.get_gid(),size=14,
            weight='bold',transform=ax.transAxes,
            ha='left')

plt.show()
