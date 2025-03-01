# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:53:15 2016

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
    use:
        %run -i figure_paper_tta_rhi_singlestorms
        
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
fig = plt.figure(figsize=(8*scale, 4*scale))

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

ax0.text(0.95, 1.05 ,'TTA',transform=ax0.transAxes,
         fontsize=15,weight='bold')

x08.plot(ax=ax0,name='contourf',mode='rhi',target='vr',
         xticklabs=False,
         ticklabsize=ticklabsize)

x08.plot(ax=ax1,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

x09.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
         ylabel = False,
         xticklabs=False)

x09.plot(ax=ax3,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

x12.plot(ax=ax4,name='contourf',mode='rhi',target='vr',
         ylabel=False,
         xticklabs=False)

x12.plot(ax=ax5,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         xticklabs=False)

hvr = x13.plot(ax=ax6,name='contourf',mode='rhi',target='vr',
         ylabel=False,
         ticklabsize=ticklabsize)

hz = x13.plot(ax=ax7,name='contourf',mode='rhi',target='z',
         with_distr=True,
         yticklabs=False,
         ticklabsize=ticklabsize)

''' add vertical date labels '''
ax1.text(1,0.5,'12-14Jan03',fontsize=15,va='center',
         transform=ax1.transAxes,rotation=-90)
ax3.text(1,0.5,'21-23Jan03',fontsize=15,va='center',
         transform=ax3.transAxes,rotation=-90)
ax5.text(1,0.5,'02Feb04',fontsize=15,va='center',
         transform=ax5.transAxes,rotation=-90)
ax7.text(1,0.5,'16-18Feb04',fontsize=15,va='center',
         transform=ax7.transAxes,rotation=-90)

''' make floating axis colorbar for vr y z '''
#                  [left, bott, wid, hgt]
cbVr = AA.Axes(fig,[0.15, -0.1, 0.34, 0.6])
cbZ  = AA.Axes(fig,[0.51, -0.1, 0.34, 0.6])
add_colorbar(cbVr,hvr,label='[m/s]',loc='bottom',
             ticks=range(0,32,2),
             ticklabels=range(0,34,4))
add_colorbar(cbZ,hz,label='[%]',loc='bottom',
             ticks=range(20,110,10),
             ticklabels=range(20,120,20))
fig.add_axes(cbVr)
fig.add_axes(cbZ)
cbVr.remove() # leave only colorbar
cbZ.remove() # leave only colorbar

''' add axis id '''
for ax in axes:
    ax.text(0.9,0.85,ax.get_gid(),size=14,
            weight='bold',transform=ax.transAxes,
            ha='left')

''' add RHI arrow '''
arrows=( {'c0':(70,3),'c1':(180,15),'rad':-0.05},
         {'c0':(25,7),'c1':(180,20),'rad':-0.05},
         {'c0':(30,5),'c1':(180,25),'rad':-0.07},
        )
            
scale = 4.1 # use for output figure
#scale = 1.0 # use for iPython figure
axes = (axes[0],axes[2],axes[6])
for arr,ax in zip(arrows,axes):
    c0 = tuple(v*scale for v in arr['c0'])
    c1 = tuple(v*scale for v in arr['c1'])
    rad = arr['rad']
    ax.annotate("",
                xy = c1,
                xytext = c0,
                xycoords='axes pixels', 
                textcoords='axes pixels',
                arrowprops=dict(
                                shrinkA=5,
                                shrinkB=5,
                                fc="w", ec="k",
                                connectionstyle="arc3,rad={}".format(rad),
                                )
                )


#plt.show()

fname='/home/raul/Desktop/tta_rhi_singlestorm.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
