# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:53:15 2016

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
    x09
except NameError:
    x09=xta.process(case=[9])
    
try:
    x12
except NameError:
    x12=xta.process(case=[12])

try:
    x13
except NameError:
    x13=xta.process(case=[13])

scale=1.6
plt.figure(figsize=(8*scale, 4*scale))
gs0 = gridspec.GridSpec(1, 2,
                        top=0.99, bottom=0.01,
                        left=0.15, right=0.85,
                        wspace=0.05)

gs00 = gssp(4, 1,
            subplot_spec=gs0[0],
            wspace=0, hspace=0)

gs01 = gssp(4, 1,
            subplot_spec=gs0[1],
            wspace=0, hspace=0)

ax0 = plt.subplot(gs00[0],gid='(a)')
ax1 = plt.subplot(gs01[0],gid='(b)')
ax2 = plt.subplot(gs00[1],gid='(c)')
ax3 = plt.subplot(gs01[1],gid='(d)')
ax4 = plt.subplot(gs00[2],gid='(e)')
ax5 = plt.subplot(gs01[2],gid='(f)')
ax6 = plt.subplot(gs00[3],gid='(g)')
ax7 = plt.subplot(gs01[3],gid='(h)')

axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

ticklabsize = 16
cbarlabsize = 16

x08.plot(ax=ax0,name='contourf',mode='rhi',target='vr',
         cbar=dict(loc='top',size='4%',label='[m/s]',fontsize=cbarlabsize),
         xticklabs=False,
         casename='12-14Jan03',
         ticklabsize=ticklabsize)

x08.plot(ax=ax1,name='contourf',mode='rhi',target='z',
         cbar=dict(loc='top',size='4%',label='[%]',fontsize=cbarlabsize),
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

x09.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
         yticklabs=False,
         casename='21-23Jan03',
         xticklabs=False)

x09.plot(ax=ax3,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

x12.plot(ax=ax4,name='contourf',mode='rhi',target='vr',
         yticklabs=False,
         casename='02Feb04',
         xticklabs=False)

x12.plot(ax=ax5,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

x13.plot(ax=ax6,name='contourf',mode='rhi',target='vr',
         yticklabs=False,
         casename='16-18Feb04',
         ticklabsize=ticklabsize)

x13.plot(ax=ax7,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         ticklabsize=ticklabsize)

for ax in axes:
    ax.text(0.9,0.85,ax.get_gid(),size=14,
            weight='bold',transform=ax.transAxes,
            ha='left')

#plt.show()

fname='/home/raul/Desktop/tta_rhi_singlestorm.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
