# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:50:10 2016

@author: raul
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xpol_tta_analysis as xta
import numpy as np
import mpl_toolkits.axisartist as AA
import matplotlib as mpl
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from rv_utilities import add_colorbar
mpl.rcParams['font.size']=15

''' 
    use:
        %run -i figure_paper_xpol_composites.py
        
    if instances do not exist in iPython namespace
    then create them
'''

try:
    xall
except NameError:
    xall=xta.process(case=[8, 9, 10, 11, 12, 13, 14])

scale=1.2
fig = plt.figure(figsize=(7.5*scale, 11*scale))

gs0 = gridspec.GridSpec(1, 2,
                        wspace=0.01)

height_ratios = [2.5,1,2.5,1]

gs00 = gssp(4, 1,
            subplot_spec=gs0[0],
            height_ratios=height_ratios,
            hspace=0)

gs01 = gssp(4, 1,
            subplot_spec=gs0[1],
            height_ratios=height_ratios,
            hspace=0)

ax0 = plt.subplot(gs00[0],gid='(a)')
ax1 = plt.subplot(gs01[0],gid='(b)')
ax2 = plt.subplot(gs00[1],gid='(c)')
ax3 = plt.subplot(gs01[1],gid='(d)')
ax4 = plt.subplot(gs00[2],gid='(e)')
ax5 = plt.subplot(gs01[2],gid='(f)')
ax6 = plt.subplot(gs00[3],gid='(g)')
ax7 = plt.subplot(gs01[3],gid='(h)')


axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

cvalues1 = range(-30,34,4)
cvalues2 = range(0,32,2)


ax0.text(0.5, 1.05 ,'TTA',transform=ax0.transAxes,
         fontsize=15,weight='bold')

ax1.text(0.5, 1.05 ,'NO-TTA',transform=ax1.transAxes,
         ha='center',fontsize=15,weight='bold')

xall.plot(ax=ax0,name='contourf',mode='ppi',target='vr',
          cbar=dict(loc='right',invisible=True),
          terrain=True,bmap=True,qc=True,
          cvalues=cvalues1)

xall.plot(ax=ax1,name='contourf',mode='ppi',target='vr',
          cbar=dict(loc='right',label='[m/s]'),
          cvalues=cvalues1,
          terrain=True,bmap=True,qc=True,
          tta=False)

xall.plot(ax=ax2,name='contourf',mode='rhi',target='vr',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues2,
          qc=True,
          xticklabs=False)

xall.plot(ax=ax3,name='contourf',mode='rhi',target='vr',
          cbar=dict(loc='right',label='[m/s]',labelpad=13),
          cvalues=cvalues2,
          xticklabs=False,
          yticklabs=False,
          qc=True,
          tta=False)

xall.plot(ax=ax4,name='contourf',mode='ppi',target='z',
          cbar=dict(loc='right',invisible=True),
          terrain=True,bmap=True,qc=True,
          sector=range(135,180),
          cvalues=cvalues1)

hdtm = xall.plot(ax=ax5,name='contourf',mode='ppi',target='z',
          cbar=dict(loc='right',label='[%]'),
          cvalues=cvalues1,
          terrain=True,bmap=True,qc=True,
          sector=range(135,180),
          tta=False)

xall.plot(ax=ax6,name='contourf',mode='rhi',target='z',
          cbar=dict(loc='right',invisible=True),
          qc=True,
          cvalues=cvalues2)

xall.plot(ax=ax7,name='contourf',mode='rhi',target='z',
          cbar=dict(loc='right',invisible=True),
          cvalues=cvalues2,
          yticklabs=False,
          qc=True,
          tta=False)

''' add axis id '''
for ax in axes:
    gid = ax.get_gid()
    if gid in ['(a)','(b)','(e)','(f)']:
        ax.text(0.9,0.93,gid,size=14,
                weight='bold',
                transform=ax.transAxes,
                color='w')
    else:
        ax.text(0.9,0.82,gid,size=14,
                weight='bold',
                transform=ax.transAxes)        

    if gid in ['(c)','(d)']:
        ax.set_xlabel('')

''' make floating axis colorbar for terrain '''
#                  [left, bott, wid, hgt]
axaa = AA.Axes(fig,[-0.38,0.74,0.5,0.1])
axaa.tick_params(labelsize=25)
add_colorbar(axaa,hdtm,label='',
             ticks=range(0,1001,1000),
             ticklabels=['0','1.0'])
fig.add_axes(axaa)
axaa.remove() # leave only colorbar
ax0.text(-0.15, 0.93,'[km]',transform=ax0.transAxes)


''' add PPI arrows '''
def arrow_end(st_co,r,az):
    en_co=[st_co[0],st_co[1]]
    en_co[0]+=r*np.sin(np.radians(az))
    en_co[1]+=r*np.cos(np.radians(az))
    return (en_co[0],en_co[1])
tta_arrows={'arrow1':{'c0':(140,115),'az':300},
            'arrow2':{'c0':(120,98),'az':325},
            'arrow3':{'c0':(90,93),'az':340},
            'arrow4':{'c0':(60,98),'az':350},
            'arrow5':{'c0':(35,105),'az':355},
            'arrow6':{'c0':(15,115),'az':5},
            }
ntta_arrows={'arrow1':{'c0':(130,115),'az':335},
            'arrow2':{'c0':(105,110),'az':350},
            'arrow3':{'c0':(80,110),'az':355},
            'arrow4':{'c0':(55,115),'az':10},
            'arrow5':{'c0':(35,120),'az':10},
            'arrow6':{'c0':(15,130),'az':10},
            }
scale = 4.1 # use for output figure
#scale = 1.0 # use for iPython figure
length = 30
arrows=[tta_arrows,ntta_arrows]
axes = [axes[0],axes[1]]
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
                                      shrinkA=6,
                                      shrinkB=6,
                                      fc='w',
                                      ec='k',
                                      lw=1),
                    zorder=1,
                    )


''' add RHI arrow '''
ax2.annotate("",
             xy = (150*scale, 25*scale),
             xytext = (25*scale,3*scale),
             xycoords='axes pixels', 
             textcoords='axes pixels',
             arrowprops=dict(
                             shrinkA=5,
                             shrinkB=5,
                             fc="w", ec="k",
                             connectionstyle="arc3,rad=-0.1",
                             )
             )




#plt.show()

fname='/home/raul/Desktop/xpol_composite.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

