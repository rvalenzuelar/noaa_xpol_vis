# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:28:16 2016

@author: raul
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
import mpl_toolkits.axisartist as AA
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from rv_utilities import add_colorbar
mpl.rcParams['font.size']=15

''' 
    use:
        %run -i figure_paper_tta_ppi_singlestorms
        
    if instances do not exist in iPython namespace
    then create them
'''

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

def main():

    fig = plt.figure(figsize=(8.5, 14))
    
    gs0 = gridspec.GridSpec(1, 2,
                            top=0.99, bottom=0.01,
                            left=0.15, right=0.85,
                            wspace=0.05)
    gs00 = gssp(5, 1,
                subplot_spec=gs0[0],
                wspace=0, hspace=0)
    gs01 = gssp(5, 1,
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
    ax8 = plt.subplot(gs00[4],gid='(i)')
    ax9 = plt.subplot(gs01[4],gid='(j)')
    
    
    cvalues = range(-30,34,4)
    
    ax0.text(0.95, 1.05 ,'TTA',transform=ax0.transAxes,
             fontsize=15,weight='bold')
    
    x08.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,terrain=True,bmap=True,
             qc=True)
    
    hdtm = x08.plot(ax=ax1,name='contourf',mode='ppi',target='z',
             terrain=True,bmap=True,
             sector=range(130,160),
             qc=True)
    
    x09.plot(ax=ax2,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,terrain=True,bmap=True,
             qc=True)
    
    x09.plot(ax=ax3,name='contourf',mode='ppi',target='z',
             terrain=True,bmap=True,
             sector=range(130,160),
             qc=True)

    x11.plot(ax=ax4,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,terrain=True,bmap=True,
             qc=True)
    
    x11.plot(ax=ax5,name='contourf',mode='ppi',target='z',
             terrain=True,bmap=True,
             sector=range(150,180),
             qc=True)
    
    x12.plot(ax=ax6,name='contourf',mode='ppi',target='vr',
             cvalues=cvalues,terrain=True,bmap=True,
             qc=True)
    
    x12.plot(ax=ax7,name='contourf',mode='ppi',target='z',
             terrain=True,bmap=True,
             sector=range(150,180),
             qc=True)
    
    x13.plot(ax=ax8,name='contourf',mode='ppi',target='vr',
             cbar=dict(loc='bottom',label='[m/s]',size='3%'),
             cvalues=cvalues,
             terrain=True, bmap=True,
             qc=True)
    
    x13.plot(ax=ax9,name='contourf',mode='ppi',target='z',
             cbar=dict(loc='bottom',label='[%]',size='3%'),
             terrain=True, bmap=True,
             sector=range(150,180),
             qc=True)
    
    ''' add vertical date labels '''
    labs = ('12-14Jan03','21-23Jan03','09Jan04','02Feb04','16-18Feb04')
    axes = (ax1,ax3,ax5,ax7,ax9)
    for ax,lab in zip(axes,labs):
        ax1.text(1,0.6,lab,fontsize=15,va='center',
             transform=ax.transAxes,rotation=-90)    
    
    
    ''' make floating axis colorbar for terrain '''
    #                  [left, bott, wid, hgt]
    axaa = AA.Axes(fig,[-0.36,0.85,0.5,0.1])
    axaa.tick_params(labelsize=25)
    add_colorbar(axaa,hdtm,label='',
                 ticks=range(0,1001,1000),
                 ticklabels=['0','1.0'])
    fig.add_axes(axaa)
    axaa.remove() # leave only colorbar
    ax0.text(-0.18, 0.93,'[km]',transform=ax0.transAxes)
    
    ''' add axis id '''
    axes = (ax0, ax1, ax2, ax3, ax4,
            ax5, ax6, ax7, ax8, ax9)
    for ax in axes:
        ax.text(0.01,0.93,ax.get_gid(),size=14,
                weight='bold',transform=ax.transAxes)
    
    ''' add PPI arrows '''
    def arrow_end(st_co,r,az):
        en_co=[st_co[0],st_co[1]]
        en_co[0]+=r*np.sin(np.radians(az))
        en_co[1]+=r*np.cos(np.radians(az))
        return (en_co[0],en_co[1])
    arrows1={'arrow1':{'c0':(120,98),'az':310},
             'arrow2':{'c0':(80,85),'az':330},
             'arrow3':{'c0':(20,90),'az':350},
            }
    arrows2={'arrow1':{'c0':(120,98),'az':310},
             'arrow2':{'c0':(80,78),'az':330},
             'arrow3':{'c0':(20,105),'az':0},
            }
    arrows3={'arrow1':{'c0':(120,99),'az':320},
             'arrow2':{'c0':(80,85),'az':340},
             'arrow3':{'c0':(20,90),'az':350},
            }             
    arrows4={'arrow1':{'c0':(120,98),'az':300},
             'arrow2':{'c0':(80,90),'az':350},
             'arrow3':{'c0':(20,110),'az':10},
            } 
    arrows5={'arrow1':{'c0':(120,98),'az':300},
             'arrow2':{'c0':(80,78),'az':330},
             'arrow3':{'c0':(20,90),'az':350},
            }              
    scale = 4.1 # use for output figure
#    scale = 1.0 # use for iPython figure
    length = 30
    arrows=(arrows1,arrows2,arrows3,arrows4,arrows5)
    axes = (axes[0],axes[2],axes[4],axes[6],axes[8])
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

#plt.show()

fname='/home/raul/Desktop/tta_ppi_singlestorm.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')


