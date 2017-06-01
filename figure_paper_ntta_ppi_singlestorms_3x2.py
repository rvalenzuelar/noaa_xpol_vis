# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:56:12 2016

@author: raul
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
import mpl_toolkits.axisartist as AA
import matplotlib as mpl
import numpy as np
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

def main():

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
    
    cvalues = range(-30,34,4)
    bmap = True
    terrain = True
    qc = True
    
    ax0.text(0.9, 1.05 ,'NO-TTA',transform=ax0.transAxes,
             fontsize=15,weight='bold')
    
    x08.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,
             terrain=terrain,bmap=bmap,
             qc=qc,
             tta=False)
    
    x08.plot(ax=ax1,name='contourf',mode='ppi',target='z',
             terrain=terrain,bmap=bmap,
             qc=qc,
             sector=range(130,160),
             tta=False)
    
    x10.plot(ax=ax2,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,
             terrain=terrain,bmap=bmap,
             qc=qc,
             tta=False)
    
    x10.plot(ax=ax3,name='contourf',mode='ppi',target='z',
             terrain=terrain,bmap=bmap,
             sector=range(140,170),
             qc=qc,
             tta=False)
    
    x14.plot(ax=ax4,name='contourf',mode='ppi',target='vr',
             cbar=dict(loc='bottom',label='[m/s]'),
             cvalues=cvalues,
             terrain=terrain,bmap=bmap,
             qc=qc,
             tta=False)
    
    hdtm = x14.plot(ax=ax5,name='contourf',mode='ppi',target='z',
             cbar=dict(loc='bottom',label='[%]'),
             terrain=terrain,bmap=bmap,
             sector=range(150,180),
             qc=qc,
             tta=False)
    
    
    ''' add vertical date labels '''
    ax1.text(1,0.6,'12-14Jan03',fontsize=15,
             transform=ax1.transAxes,rotation=-90)
    ax3.text(1,0.6,'15-16Feb03',fontsize=15,
             transform=ax3.transAxes,rotation=-90)
    ax5.text(1,0.6,'25Feb04',fontsize=15,
             transform=ax5.transAxes,rotation=-90)
    
    ''' make floating axis colorbar for terrain '''
    #                  [left, bott, wid, hgt]
    axaa = AA.Axes(fig,[-0.36,0.83,0.5,0.1])
    axaa.tick_params(labelsize=25)
    add_colorbar(axaa,hdtm,label='',
                 ticks=range(0,1001,1000),
                 ticklabels=['0','1.0'])
    fig.add_axes(axaa)
    axaa.remove() # leave only colorbar
    ax0.text(-0.18, 0.93,'[km]',transform=ax0.transAxes)
    
    ''' add axis id '''
    for ax in axes:
        ax.text(0.01,0.93,ax.get_gid(),size=14,
                weight='bold',transform=ax.transAxes)

    ''' add PPI arrows '''
    def arrow_end(st_co,r,az):
        en_co=[st_co[0],st_co[1]]
        en_co[0]+=r*np.sin(np.radians(az))
        en_co[1]+=r*np.cos(np.radians(az))
        return (en_co[0],en_co[1])
    arrows1={'arrow1':{'c0':(130,115),'az':330},
             'arrow2':{'c0':(80,112),'az':355},
             'arrow3':{'c0':(30,130),'az':10},
            }
    arrows2={'arrow1':{'c0':(130,115),'az':350},
             'arrow2':{'c0':(80,130),'az':10},
             'arrow3':{'c0':(30,160),'az':25},
            }
    arrows3={'arrow1':{'c0':(130,115),'az':330},
             'arrow2':{'c0':(80,100),'az':340},
             'arrow3':{'c0':(30,90),'az':345},
            }
      
    scale = 4.1 # use for output figure
#    scale = 1.0 # use for iPython figure
    length = 30
    arrows=(arrows1,arrows2,arrows3)
    axes = (axes[0],axes[2],axes[4])
    for ax,arrow in zip(axes,arrows):
        for _,arr in arrow.iteritems():
            c0 = tuple(v*scale for v in arr['c0'])
            az = arr['az']
            ax.annotate("",
                        xy         = arrow_end(c0,length*scale,az),
                        xytext     = c0,
                        xycoords   = 'axes pixels',
                        textcoords = 'axes pixels',
                        arrowprops = dict(
                                          shrinkA=7,
                                          shrinkB=7,
                                          fc='w',
                                          ec='k',
                                          lw=1),
                        zorder=1,
                        )



main()

plt.show()

# fname='/home/raul/Desktop/fig_ntta_ppi_singlestorm.png'
# plt.savefig(fname, dpi=300, format='png',papertype='letter',
#             bbox_inches='tight')

