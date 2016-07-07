# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:17:12 2016

@author: raul
"""
import matplotlib.pyplot as plt
import xpol_tta_analysis as xta
from rv_utilities import discrete_cmap

from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15

''' have tta and no-tta '''
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


''' only have no-tta '''
try:
    x10
except NameError:
    x10=xta.process(case=[10])

try:
    x11
except NameError:
    x11=xta.process(case=[11])

try:
    x14
except NameError:
    x14=xta.process(case=[14])
    
    
fig,ax = plt.subplots(2,2,figsize=(8,8),
                      sharey=True)
ax=ax.flatten()


cmap=discrete_cmap(7, base_cmap='Set1')

x08.plot_dist(ax=ax[0],mode='ppi',tta=True, colores=cmap(0))
x09.plot_dist(ax=ax[0],mode='ppi',tta=True, colores=cmap(1))
x12.plot_dist(ax=ax[0],mode='ppi',tta=True, colores=cmap(2))
x13.plot_dist(ax=ax[0],mode='ppi',tta=True, colores=cmap(3))

h1=x08.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(0))
h2=x09.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(1))
h5=x12.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(2))
h6=x13.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(3))
h3=x10.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(4))
h4=x11.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(5))
h7=x14.plot_dist(ax=ax[1],mode='ppi',tta=False, colores=cmap(6))

x08.plot_dist(ax=ax[2],mode='rhi',tta=True, colores=cmap(0))
x09.plot_dist(ax=ax[2],mode='rhi',tta=True, colores=cmap(1))
x12.plot_dist(ax=ax[2],mode='rhi',tta=True, colores=cmap(2))
x13.plot_dist(ax=ax[2],mode='rhi',tta=True, colores=cmap(3))

x08.plot_dist(ax=ax[3],mode='rhi',tta=False, colores=cmap(0))
x09.plot_dist(ax=ax[3],mode='rhi',tta=False, colores=cmap(1))
x12.plot_dist(ax=ax[3],mode='rhi',tta=False, colores=cmap(2))
x13.plot_dist(ax=ax[3],mode='rhi',tta=False, colores=cmap(3))
x10.plot_dist(ax=ax[3],mode='ppi',tta=False, colores=cmap(4))
x11.plot_dist(ax=ax[3],mode='ppi',tta=False, colores=cmap(5))
x14.plot_dist(ax=ax[3],mode='ppi',tta=False, colores=cmap(6))

ax[0].set_ylabel('PPI\nCFD')
ax[2].set_ylabel('RHI\nCFD')
ax[2].set_xlabel('dBZ')
ax[3].set_xlabel('dBZ')
ax[0].text(0.5,1.1,'TTA',size=15,transform=ax[0].transAxes,
            ha='center',va='top')
ax[1].text(0.5,1.1,'NO-TTA',size=15,transform=ax[1].transAxes,
            ha='center',va='top')

datelabs=['12-14Jan03',
          '21-23Jan03',
          '15-16Feb03',
          '09-10Jan03',
          '02-03Feb04',
          '16-18Feb04',
          '25-26Feb04']


legend_loc=[0.3,0.1,0.2,0.8]
ax[1].legend([h1,h2,h3,h4,h5,h6,h7],
             datelabs,
             prop={'size': 12},
             numpoints=1,
             handletextpad=0.1,
             framealpha=1,
             handlelength=1,
             bbox_to_anchor=legend_loc)


plt.subplots_adjust(wspace=0.15,hspace=0.15)

#plt.show()

fname='/home/raul/Desktop/za_cdf.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')