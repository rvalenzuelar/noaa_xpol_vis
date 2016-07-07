# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:56:12 2016

@author: raul
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
#from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp


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

scale=1.2
fig=plt.figure(figsize=(8*scale, 8.5*scale))

gs0 = gridspec.GridSpec(3, 2,
                        top=0.99, bottom=0.01,
                        left=0.15, right=0.85,
                        wspace=0.05,hspace=0)

ax0 = plt.subplot(gs0[0],gid='(a)')
ax1 = plt.subplot(gs0[1],gid='(b)')
ax2 = plt.subplot(gs0[2],gid='(c)')
ax3 = plt.subplot(gs0[3],gid='(d)')
ax4 = plt.subplot(gs0[4],gid='(e)')
ax5 = plt.subplot(gs0[5],gid='(f)')

axes = [ax0, ax1, ax2, ax3, ax4, ax5]

cvalues = range(-28,31,3)
bmap = True
terrain = True
qc = True

x08.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
         cbar=dict(loc='top',label='[m/s]'),
         cvalues=cvalues,
         terrain=terrain,bmap=bmap,
         qc=qc,casename='12-14Jan03',
         tta=False)

x08.plot(ax=ax1,name='contourf',mode='ppi',target='z',
         cbar=dict(loc='top',label='[%]'),
         terrain=terrain,bmap=bmap,
         qc=qc,
         sector=range(130,160),
         tta=False)

x10.plot(ax=ax2,name='contourf',mode='ppi',target='vr',
         cvalues=cvalues,
         terrain=terrain,bmap=bmap,
         qc=qc,casename='15-16Feb03',
         tta=False)

x10.plot(ax=ax3,name='contourf',mode='ppi',target='z',
         terrain=terrain,bmap=bmap,
         sector=range(140,170),
         qc=qc,
         tta=False)

x14.plot(ax=ax4,name='contourf',mode='ppi',target='vr',
         cvalues=cvalues,
         terrain=terrain,bmap=bmap,
         qc=qc,casename='25-26Feb04',
         tta=False)

x14.plot(ax=ax5,name='contourf',mode='ppi',target='z',
         terrain=terrain,bmap=bmap,
         sector=range(150,180),
         qc=qc,
         tta=False)


for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=14,
            weight='bold',transform=ax.transAxes)

#plt.show()

fname='/home/raul/Desktop/ntta_ppi_singlestorm.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

