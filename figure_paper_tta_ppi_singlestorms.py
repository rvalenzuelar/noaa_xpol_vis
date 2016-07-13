# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:28:16 2016

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

plt.figure(figsize=(8.5, 11))

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

cvalues = range(-26,28,4)

x08.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
         cbar=dict(loc='top',label='[m/s]'),
         cvalues=cvalues,terrain=True,bmap=True,
         qc=True,casename='12-14Jan03')

x08.plot(ax=ax1,name='contourf',mode='ppi',target='z',
         cbar=dict(loc='top',label='[%]'),
         terrain=True,bmap=True,
         sector=range(130,160),
         qc=True)

x09.plot(ax=ax2,name='contourf',mode='ppi',target='vr',
         cvalues=cvalues,terrain=True,bmap=True,
         qc=True,casename='21-23Jan03')

x09.plot(ax=ax3,name='contourf',mode='ppi',target='z',
         terrain=True,bmap=True,
         sector=range(130,160),
         qc=True)

x12.plot(ax=ax4,name='contourf',mode='ppi',target='vr',
         cvalues=cvalues,terrain=True,bmap=True,
         qc=True,casename='02Feb04')

x12.plot(ax=ax5,name='contourf',mode='ppi',target='z',
         terrain=True,bmap=True,
         sector=range(150,180),
         qc=True)

x13.plot(ax=ax6,name='contourf',mode='ppi',target='vr',
         cvalues=cvalues,terrain=True,bmap=True,
         qc=True,casename='16-18Feb04')

x13.plot(ax=ax7,name='contourf',mode='ppi',target='z',
         terrain=True,bmap=True,
         sector=range(150,180),
         qc=True)

for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=14,
            weight='bold',transform=ax.transAxes)

#plt.show()

fname='/home/raul/Desktop/tta_ppi_singlestorm.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')


