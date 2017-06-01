# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:07:32 2016

@author: raul
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
import mpl_toolkits.axisartist as AA
import matplotlib as mpl
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from rv_utilities import add_colorbar
mpl.rcParams['font.size']=15

''' 
    if instances do not exist in iPython namespace
    then create them
'''
params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )
try:
    x08
except NameError:
    x08=xta.process(case=[8],params=params)

try:
    x10
except NameError:
    x10=xta.process(case=[10],params=params)
    
try:
    x14
except NameError:
    x14=xta.process(case=[14],params=params)

scale=1.6
fig = plt.figure(figsize=(8*scale, 3*scale))

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

ax0.text(0.95, 1.05 ,'NO-TTA',transform=ax0.transAxes,
         fontsize=15,weight='bold')

x08.plot(ax=ax0,name='contourf',mode='rhi',target='vr',
         xticklabs=False,
         tta=False,
         ticklabsize=ticklabsize)

x08.plot(ax=ax1,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False,
         tta=False)

x10.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
         ylabel=False,
         xticklabs=False,
         tta=False)

x10.plot(ax=ax3,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False,
         tta=False)

hvr = x14.plot(ax=ax4,name='contourf',mode='rhi',target='vr',
         ylabel=False,
         ticklabsize=ticklabsize,
         tta=False)

hz = x14.plot(ax=ax5,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         ticklabsize=ticklabsize,
         tta=False)

''' add vertical date labels '''
ax1.text(1,0.5,'12-14Jan03',fontsize=15,va='center',
         transform=ax1.transAxes,rotation=-90)
ax3.text(1,0.5,'15-16Feb03',fontsize=15,va='center',
         transform=ax3.transAxes,rotation=-90)
ax5.text(1,0.5,'25Feb04',fontsize=15,va='center',
         transform=ax5.transAxes,rotation=-90)

''' make floating axis colorbar for vr y z '''
#                  [left, bott, wid, hgt]
cbVr = AA.Axes(fig,[0.15, -0.13, 0.34, 0.75])
cbZ  = AA.Axes(fig,[0.51, -0.13, 0.34, 0.75])
add_colorbar(cbVr,hvr,label='[m/s]',loc='bottom',
             ticks=range(0, 32, 2),
             ticklabels=range(0, 34, 4))
add_colorbar(cbZ,hz,label='[%]',loc='bottom',
             ticks=range(20, 110, 10),
             ticklabels=range(20, 120, 20))
fig.add_axes(cbVr)
fig.add_axes(cbZ)
cbVr.remove()  # leave only colorbar
cbZ.remove()  # leave only colorbar

''' add axis id '''
for ax in axes:
    ax.text(0.9, 0.85, ax.get_gid(), size=14,
            weight='bold', transform=ax.transAxes,
            ha='left')

plt.show()

# fname='/Users/raulvalenzuela/Documents/ntta_rhi_singlestorm.png'
# plt.savefig(fname, dpi=300, format='png',papertype='letter',
#             bbox_inches='tight')
