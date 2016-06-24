# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:50:10 2016

@author: raul
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp

try:
    xall
except NameError:
    xall=xta.process(case=[8, 9, 10, 11, 13, 14])

scale=1.2
plt.figure(figsize=(7.5*scale, 11*scale))
gs0 = gridspec.GridSpec(1, 2,
                        wspace=0.01)

height_ratios = [2.5,1,2.5,1]
width_ratios = [1,1,1,0.8]

gs00 = gssp(4, 1,
            subplot_spec=gs0[0],
            height_ratios=height_ratios,
            hspace=0.1)

gs01 = gssp(4, 1,
            subplot_spec=gs0[1],
            height_ratios=height_ratios,
            hspace=0.1)

ax0 = plt.subplot(gs00[0],gid='(a)')
ax1 = plt.subplot(gs01[0],gid='(b)')
ax2 = plt.subplot(gs00[1],gid='(c)')
ax3 = plt.subplot(gs01[1],gid='(d)')
ax4 = plt.subplot(gs00[2],gid='(e)')
ax5 = plt.subplot(gs01[2],gid='(f)')
ax6 = plt.subplot(gs00[3],gid='(g)')
ax7 = plt.subplot(gs01[3],gid='(h)')


axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

cvalues1 = range(-25,28,3)
cvalues2 = range(0,26,3)

xall.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues1)

xall.plot(ax=ax1,name='contourf',mode='ppi',target='vr',
          cbar=dict(loc='right',label='[m/s]'),
          cvalues=cvalues1,
          tta=False)

xall.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues2,
          xticklabs=False)

xall.plot(ax=ax3,name='contourf',mode='rhi',target='vr',
          cbar=dict(loc='right',label='[m/s]',labelpad=13),
          cvalues=cvalues2,
          xticklabs=False,
          yticklabs=False,
          tta=False)

xall.plot(ax=ax4,name='contourf',mode='ppi',target='z',
          cbar=dict(loc='right',invisible=True),
          with_distr=True,
          cvalues=cvalues1)

xall.plot(ax=ax5,name='contourf',mode='ppi',target='z',
          cbar=dict(loc='right',label='[%]'),
          cvalues=cvalues1,
          with_distr=True,
          tta=False)

xall.plot(ax=ax6,name='contourf',mode='rhi',target='z',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues2,
          with_distr=True)

xall.plot(ax=ax7,name='contourf',mode='rhi',target='z',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues2,
          yticklabs=False,
          with_distr=True,
          tta=False)

for ax in axes:
    gid = ax.get_gid()
    if gid in ['(a)','(b)','(e)','(f)']:
        ax.text(0.9,0.93,gid,size=14,
                weight='bold',
                transform=ax.transAxes)
    else:
        ax.text(0.9,0.82,gid,size=14,
                weight='bold',
                transform=ax.transAxes)        

plt.show()
