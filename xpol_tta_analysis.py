# -*- coding: utf-8 -*-
"""
    Created on Fri Jun 17 17:33:45 2016

    Raul Valenzuela
    raul.valenzuela@colorado.edu
    
    Make XPOL partition composites based on
    TTA analysis

    Full path pointing to data location need
    to be exported as environment variable
    (e.g. export XPOL_PATH='/full/path/to/files')
    
    Example:
    
    import xpol_tta_analysis as xta    
    x09 = xta.process(case=[9])
    
    xall = xta.process(case=range(9,15))

"""

import xpol
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from tta_analysis import tta_analysis
from rv_utilities import add_colorbar,add_subplot_axes


class process:


    def __init__(self,case=[]):

        self.case = case

        self.rhi_tta = None
        self.rhi_ntta = None
        self.ppi_tta = None
        self.ppi_ntta = None

        rhi_df = make_dataframe(mode='rhi',case=case)
        ppi_df = make_dataframe(mode='ppi',case=case)

        years = []
        for c in case:
            if c in [8,9,10] and 2003 not in years:
                years.append(2003)
                
            if c in [11,12,13,14] and 2004 not in years:
                years.append(2004)
                
        tta_dates = get_tta_dates(years)

        self.process(rhi_df, tta_dates, mode='rhi')
        self.process(ppi_df, tta_dates, mode='ppi')

        self.statistics()


    def process(self, xpol_df, tta_dates, mode=None):
           

        ' makes hourly groups of xpol dataframe'
        grp = pd.TimeGrouper('1H')
        xpol_groups = xpol_df.groupby(grp)

        TTAdf = None
        NTTAdf = None
        for date_grp,_ in xpol_groups.groups.iteritems():

            ''' some dates where created by the pandas
                grouping method to fill time gaps, so 
                we use try to avoid the KeyError
            '''
            try:
                grpvalue = xpol_groups.get_group(date_grp)
                if date_grp in tta_dates:
                    if TTAdf is None:
                        TTAdf = grpvalue
                    else:
                        TTAdf = TTAdf.append(grpvalue)
                else:
                    if NTTAdf is None:
                        NTTAdf = grpvalue
                    else:
                        NTTAdf = NTTAdf.append(grpvalue)
            except KeyError:
                pass


        if mode == 'rhi':
            self.rhi_tta = TTAdf
            self.rhi_ntta = NTTAdf
        elif mode == 'ppi':
            self.ppi_tta = TTAdf
            self.ppi_ntta = NTTAdf


    def statistics(self):
        
        print('...calculating statistics')        
        
        if len(self.case)>1:
            rhi_tta_za = xpol.convert_to_common_grid(self.rhi_tta['ZA'])
            rhi_tta_vr = xpol.convert_to_common_grid(self.rhi_tta['VR'])
            rhi_ntta_za = xpol.convert_to_common_grid(self.rhi_ntta['ZA'])
            rhi_ntta_vr = xpol.convert_to_common_grid(self.rhi_ntta['VR'])
        else:
            rhi_tta_za = self.rhi_tta['ZA']
            rhi_tta_vr = self.rhi_tta['VR']
            rhi_ntta_za = self.rhi_ntta['ZA']
            rhi_ntta_vr = self.rhi_ntta['VR']
            
        ' use try for cases with tta=None'
        try:
            out = xpol.get_dbz_freq(rhi_tta_za, percentile=50)
            dbz_freq, thres, distrz, binsz = out
            out = xpol.get_mean(rhi_tta_vr, name='VR')        
            vr_mean, good = out
            distr={'csum':distrz,'bins':binsz}
        except TypeError:
            vr_mean,dbz_freq,thres,distr = [None,None,None,None]            
            
        self.rhi_tta_vr = vr_mean
        self.rhi_tta_z = dbz_freq
        self.rhi_tta_thres = thres
        self.rhi_tta_distrz = distr

        ' use try for cases with tta=None'
        try:
            out = xpol.get_dbz_freq(self.ppi_tta['ZA'], percentile=50)
            dbz_freq, thres, distrz, binsz = out
            out = xpol.get_mean(self.ppi_tta['VR'], name='VR')        
            vr_mean, good = out
            distr={'csum':distrz,'bins':binsz}
        except TypeError:
            vr_mean,dbz_freq,thres, distr = [None,None,None,None]
            
        self.ppi_tta_vr = vr_mean
        self.ppi_tta_z = dbz_freq
        self.ppi_tta_thres = thres
        self.ppi_tta_distrz = distr

        out = xpol.get_dbz_freq(rhi_ntta_za, percentile=50)
        dbz_freq, thres, distrz, binsz = out
        out = xpol.get_mean(rhi_ntta_vr, name='VR')        
        vr_mean, good = out        
        distr={'csum':distrz,'bins':binsz}
        
        self.rhi_ntta_vr = vr_mean
        self.rhi_ntta_z = dbz_freq
        self.rhi_ntta_thres = thres
        self.rhi_ntta_distrz = distr

        out = xpol.get_dbz_freq(self.ppi_ntta['ZA'], percentile=50)
        dbz_freq, thres, distrz, binsz = out
        out = xpol.get_mean(self.ppi_ntta['VR'], name='VR')        
        vr_mean, good = out        
        distr={'csum':distrz,'bins':binsz}
        
        self.ppi_ntta_vr = vr_mean
        self.ppi_ntta_z = dbz_freq
        self.ppi_ntta_thres = thres
        self.ppi_ntta_distrz = distr

    def plot(self,name=None,ax=None,mode=None,target=None,tta=True,
             show=False,with_distr=False, cbar=None, cvalues=None,
             yticklabs=True, xticklabs=True, ticklabsize=16):
        
        import numpy.ma as ma
#        import cartopy.crs as ccrs
#        import cartopy.feature as cfeature
        
        if tta is True:
            if mode == 'rhi' and target == 'z':
                array = self.rhi_tta_z
            elif mode == 'rhi' and target == 'vr':
                array = self.rhi_tta_vr
            elif mode == 'ppi' and target == 'z':
                array = self.ppi_tta_z
            elif mode == 'ppi' and target == 'vr':
                array = self.ppi_tta_vr
        else:
            if mode == 'rhi' and target == 'z':
                array = self.rhi_ntta_z
            elif mode == 'rhi' and target == 'vr':
                array = self.rhi_ntta_vr
            elif mode == 'ppi' and target == 'z':
                array = self.ppi_ntta_z
            elif mode == 'ppi' and target == 'vr':
                array = self.ppi_ntta_vr
            
        ''' set grid values '''
        case = self.case
        if mode == 'ppi':
            x = np.arange(-58, 45 +0.5, 0.5)        
            y = np.arange(-58, 33 +0.5, 0.5)
        elif mode == 'rhi':
            if len(case) == 1 and case[0] in [11,13,14]:
                x = np.arange(-30, 30.14 , 0.14)        
            elif len(case) == 1 and case[0] == 12:
                x = np.arange(-22, 30 , 0.14)  
            elif len(case) == 1:
                x = np.arange(-40, 20 , 0.14) 
            else:
                x = np.arange(-40, 30.6 , 0.14)  # for common grid
            y = np.arange(  0, 12 +0.20, 0.20)        
        
        ''' set contour values '''
        if target == 'z':        
            cvalues = np.arange(20,110,10)        
        elif target == 'vr':
            if cvalues is None:
                if mode == 'ppi':
                    cvalues = np.arange(-19,21,2)    
                elif mode == 'rhi':
                    cvalues = np.arange(0,32,2)
                
        ''' set masked values '''
        if array is None:
            array = np.zeros((y.size,x.size))+np.nan
        arraym = ma.masked_where(np.isnan(array),array)
        
        ''' axis '''
        if ax is None:
                fig,ax = plt.subplots()
                self.fig = fig
        else:
            self.fig = None
        self.ax = ax
        
        ''' make plot '''
        cmap = get_colormap(mode=mode,target=target)
        if name == 'pcolor':        
            p = ax.pcolormesh(x,y,arraym)
        elif name =='contourf':
            X,Y = np.meshgrid(x,y)
            p = ax.contourf(X,Y,arraym,cvalues,cmap=cmap)

        
        ''' add colorbar '''
        if cbar is not None:
            add_colorbar(ax,p,**cbar)


        ''' terrain profile '''
        if mode == 'rhi':
            add_terrain_prof(ax,self.case[0])

            
        ''' configure plot '''
        make_pretty_plot(self,mode=mode,target=target,tta=tta,
                         yticklabs=yticklabs,xticklabs=xticklabs,
                         ticklabsize=ticklabsize)  

        ''' add cummulative distribution plot '''
        if with_distr is True:
            if mode == 'ppi':
                scale=1.
                rect = [0.65, 0.6, 0.3*scale, 0.3*scale]
            elif mode == 'rhi':
                scale = 1.7
                rect = [0.75, 0.25, 0.12*scale, 0.3*scale]
            subax = add_subplot_axes(ax, rect)
            self.plot_dist(ax=subax,mode=mode,tta=tta)

        
        if show is True:
            plt.show()

   
    def plot_dist(self,ax=None,mode=None,tta=True,show=False):


        if tta is True:
            if mode == 'rhi':
                array = self.rhi_tta_distrz['csum']
                bins = self.rhi_tta_distrz['bins']
                thres = self.rhi_tta_thres
            elif mode == 'ppi':
                array = self.ppi_tta_distrz['csum']
                bins = self.ppi_tta_distrz['bins']
                thres = self.ppi_tta_thres
        else:
            if mode == 'rhi':
                array = self.rhi_ntta_distrz['csum']
                bins = self.rhi_ntta_distrz['bins']
                thres = self.rhi_ntta_thres
            elif mode == 'ppi':
                array = self.ppi_ntta_distrz['csum']
                bins = self.ppi_ntta_distrz['bins']
                thres = self.ppi_ntta_thres

        if ax is None:
            fig,ax = plt.subplots()
            
        ax.plot(bins[:-1],array,lw=3,color='b')
        plt.axvline(x=thres,lw=2,color='r')
        plt.axvline(x=0,lw=2,color='k',linestyle=':')
        ax.set_xlim([-20,50])
        ax.set_ylim([0,1])        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if show is True:
            plt.show()
                
        
    def plot_ppi_rhi(self,with_distr=False):


        import matplotlib.gridspec as gsp   

        scale = 1.8
        plt.figure(figsize=(5*scale,6.5*scale))
        gs1 = gsp.GridSpec(4, 1)
#        gs1.update(left=0.05, right=0.48, wspace=0.05)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])

        self.plot(ax=ax1,name='contourf',mode='ppi',
                  target='z',with_distr=with_distr,
                  cbar=dict(loc='right'))
        self.plot(ax=ax2,name='contourf',mode='rhi',
                  target='z',with_distr=with_distr,
                  cbar=dict(loc='right'))

        plt.show()
        

def add_coastline(ax,x,y):
    pass

def get_colormap(mode=None,target=None):
    

    if mode == 'ppi':
        if target == 'z':
            cmap = cm.get_cmap('inferno')
        elif target == 'vr':
            cmap = xpol.custom_cmap('ppi_vr1')
            plt.register_cmap(cmap=cmap)

    if mode == 'rhi':
        if target == 'z':
            cmap = 'inferno'
        elif target == 'vr':
            cmap = xpol.custom_cmap('rhi_vr1')  
            plt.register_cmap(cmap=cmap)

    return cmap



def make_pretty_plot(self,mode=None,target=None,tta=True,
                     yticklabs=True,xticklabs=True,
                     ticklabsize=12):
    
    ax = self.ax
    fig = self.fig
    dbztxt = '{:2.1f} dBZ'
    vrtxt = 'N={}'
    txtsize = 12  # good size for letter size fig
    
    if mode == 'ppi':
        ax.set_xlim([-60,40])
        ax.set_ylim([-60,40])
        if fig is not None:
            fig.set_figheight(6.7) # in inches
            fig.set_figwidth(8.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if target == 'z':
            if tta is True:
                thr = self.ppi_tta_thres    
            else:
                thr = self.ppi_ntta_thres
            txt1 = dbztxt.format(thr)
            ax.text(0.85,0.55,txt1, size=txtsize,
                transform=ax.transAxes, ha='center',
                color=(1,0,0),weight='bold')             
        elif target == 'vr':    
            if tta is True:
                cnt = self.ppi_tta.index.size
            else:
                cnt = self.ppi_ntta.index.size
            txt1 = vrtxt.format(cnt)    
            ax.text(0.01,0.01,txt1, size=txtsize,
                transform=ax.transAxes, ha='left',
                color=(0,0,0),weight='bold') 
        xpol.add_rings(ax, alpha=0.5, txtsize=txtsize)
        xpol.add_azimuths(ax, alpha=0.3)
               

        
    if mode == 'rhi':
        ax.set_xlim([-40,40])            
        ax.set_ylim([0,5]) 
        if fig is not None:
            fig.set_figheight(2.0) # in inches
            fig.set_figwidth(8.0)
        if target == 'z':
            if tta is True:
                thr = self.rhi_tta_thres    
            else:
                thr = self.rhi_ntta_thres    
            txt1 = dbztxt.format(thr)
            ax.text(0.98,0.13,txt1,size=txtsize,
                    transform=ax.transAxes,ha='right',
                    color=(1,0,0), weight='bold')            
        elif target == 'vr':    
            if tta is True:
                cnt = self.rhi_tta.index.size
            else:
                cnt = self.rhi_ntta.index.size
            txt1 = vrtxt.format(cnt)
            ax.text(0.98,0.13,txt1,size=txtsize,
                    transform=ax.transAxes,ha='right',
                    color=(0,0,0), weight='bold')


        if yticklabs is False:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Altitude [km] MSL',
                          fontdict=dict(size=ticklabsize-1))
            
        if xticklabs is False:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Distance from the radar [km]',
                          fontdict=dict(size=ticklabsize-1))

        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        xticks[-1].label1.set_visible(False)
        ax.xaxis.set_tick_params(labelsize=ticklabsize)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[-1].label1.set_visible(False)
        ax.yaxis.set_tick_params(labelsize=ticklabsize)
        
        
        
def add_terrain_prof(ax,case):

# # use this is prof1.npy or prof2.npy are absent
#    import radar_kml    
#
#    p0 = (38.51,-123.24)
#    p1 = (39.05,-123.24)
#    p2 = (39.05,-123.16)
#    
#    if case == 12:
#        prof = radar_kml.get_elev_profile(p0,p1,npoints=120)['elev']
#    else:
#        prof = radar_kml.get_elev_profile(p0,p2,npoints=120)['elev']            

    if case == 12:
        prof = np.load('prof1.npy')
    else:
        prof = np.load('prof2.npy')
    
    x = np.arange(0.5,60.5,0.5)
    y = prof/1000.
    ax.fill_between(x, 0, y,facecolor='gray')
    

def get_tta_dates(years):
    
        params = dict(wdir_surf=125,wdir_wprof=170,
                      rain_czd=0.25,nhours=2)

        tta_dates03 = []
        tta_dates04 = []
        
        if 2003 in years:
            print('...getting 2003 tta dates')
            tta = tta_analysis(2003)
            tta.start_df(**params)
            tta_dates03 = tta.tta_dates.tolist()
        
        if 2004 in years:
            print('...getting 2004 tta dates')
            tta = tta_analysis(2004)
            tta.start_df(**params)
            tta_dates04 = tta.tta_dates.tolist()
        
        return tta_dates03 + tta_dates04


def make_dataframe(mode=None,case=None):
    
    '''
        make xpol dataframes appending cases
    '''
    
    print('...making {} dataframe'.format(mode))
    
    setcase = {8: [0.5, 180, 10, 100],
               9: [0.5, 180, 13, 100],
               10: [0.5, 180, 30, 100],
               11: [0.5, 180, 20, 100],
               12: [0.5,   6, 15, 100],
               13: [0.5, 180, 25, 100],
               14: [0.5, 180, 30, 100]}

    first = True
    for c in case:

        elevation, azimuth, _, maxv = setcase[c]

        if mode == 'rhi':
            if first is True:
                df = xpol.get_data(c, 'RHI', azimuth)
                first = False
            else:
                df = df.append(xpol.get_data(c, 'RHI', azimuth))
        elif mode == 'ppi':
            if first is True:
                df = xpol.get_data(c, 'PPI', elevation)
                first = False
            else:
                df = df.append(xpol.get_data(c, 'PPI', elevation))

    return df





'''
  *** check if following funcs are needed ***
'''

def make_plot_tta(xpol_dict,mode=None):

    tta_vr = xpol_dict['tta_vr']
    tta_z = xpol_dict['tta_z']
    data = tta_vr+tta_z

    # axes = get_axes_grid((4, 2))
    # axes = get_axes_subplots((4, 2))
    axes = get_axes_gridspec4x2()

    if mode == 'ppi':
        for n, ax in enumerate(axes):
            if n <= 3:
                xpol.plot(data[n], ax=ax, name='VR',
                          smode='ppi', colorbar=False)
            else:
                xpol.plot(data[n], ax=ax, name='freq', smode='ppi',
                          colorbar=False, vmax=100)
    elif mode == 'rhi':

        for n, ax in enumerate(axes):
            d = xpol.convert_to_common_grid(data[n])
            if n <= 3:
                xpol.plot(d, ax=ax, name='VR',
                          smode='rhi', colorbar=False,
                          add_yticklabs=True)
            else:
                xpol.plot(d, ax=ax, name='freq',
                          smode='rhi', colorbar=False, vmax=100)

            if n not in [3, 7]:
                ax.set_xticklabels([])

            if n > 0:
                ax.set_yticklabels([])

def make_plot_notta(xpol_dict, select,mode=None):

    notta_vr = [xpol_dict['notta_vr'][n] for n in select]
    notta_z = [xpol_dict['notta_z'][n] for n in select]
    data = notta_vr + notta_z

    # ' reorder to fit plot panels'
    # new_order = [0, 3, 1, 4, 2, 5]
    # data = [data[i] for i in new_order]

    # axes = get_axes_grid((3, 2))
    # axes = get_axes_subplots((3, 2))
    axes = get_axes_gridspec3x2()

    if mode == 'ppi':
        for n, ax in enumerate(axes):
            # if np.mod(n, 2) == 0:
            if n < 3:
                xpol.plot(data[n], ax=ax, name='VR',
                          smode='ppi', colorbar=False)
            else:
                xpol.plot(data[n], ax=ax, name='freq', smode='ppi',
                          colorbar=False, vmax=100)

    elif mode == 'rhi':
        for n, ax in enumerate(axes):
            d = xpol.convert_to_common_grid(data[n])
            if n < 3:
                xpol.plot(d, ax=ax, name='VR',
                          smode='rhi', colorbar=False,
                          add_yticklabs=True)
            else:
                xpol.plot(d, ax=ax, name='freq',
                          smode='rhi', colorbar=False, vmax=100)

            if n not in [2, 5]:
                ax.set_xticklabels([])

            if n > 0:
                ax.set_yticklabels([])


def get_axes_grid(nrows_ncols):

    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(8.5, 11))
    axes = ImageGrid(fig, 111,
                     nrows_ncols=nrows_ncols,
                     axes_pad=0,
                     add_all=True,
                     share_all=False,
                     label_mode="L",
                     cbar_location="top",
                     cbar_mode="single",
                     cbar_size='2%',
                     aspect=True)

    return axes


def get_axes_subplots(nrows_ncols):

    nrows, ncols = nrows_ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8.5, 11),
                             sharex=True,
                             sharey=True)
    axes = axes.flatten()

    return axes


def get_axes_gridspec4x2():

    import matplotlib.gridspec as gridspec

    f = plt.figure(figsize=(8.5, 11))
    gs0 = gridspec.GridSpec(1, 2,
                            top=0.99, bottom=0.01,
                            left=0.15, right=0.85,
                            wspace=0.05)

    gs00 = gridspec.GridSpecFromSubplotSpec(4, 1,
                                            subplot_spec=gs0[0],
                                            wspace=0, hspace=0)
    ax0 = plt.Subplot(f, gs00[0])
    f.add_subplot(ax0)
    ax1 = plt.Subplot(f, gs00[1])
    f.add_subplot(ax1)
    ax2 = plt.Subplot(f, gs00[2])
    f.add_subplot(ax2)
    ax3 = plt.Subplot(f, gs00[3])
    f.add_subplot(ax3)

    gs01 = gridspec.GridSpecFromSubplotSpec(4, 1,
                                            subplot_spec=gs0[1],
                                            wspace=0, hspace=0)
    ax4 = plt.Subplot(f, gs01[0])
    f.add_subplot(ax4)
    ax5 = plt.Subplot(f, gs01[1])
    f.add_subplot(ax5)
    ax6 = plt.Subplot(f, gs01[2])
    f.add_subplot(ax6)
    ax7 = plt.Subplot(f, gs01[3])
    f.add_subplot(ax7)


    return np.array([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7])


def get_axes_gridspec3x2():

    import matplotlib.gridspec as gridspec

    f = plt.figure(figsize=(8.5, 11))
    gs0 = gridspec.GridSpec(1, 2,
                            top=0.9, bottom=0.1,
                            left=0.1, right=0.92,
                            hspace=0.1)

    gs00 = gridspec.GridSpecFromSubplotSpec(3, 1,
                                            subplot_spec=gs0[0],
                                            wspace=0, hspace=0)
    ax0 = plt.Subplot(f, gs00[0])
    f.add_subplot(ax0)
    ax1 = plt.Subplot(f, gs00[1])
    f.add_subplot(ax1)
    ax2 = plt.Subplot(f, gs00[2])
    f.add_subplot(ax2)

    gs01 = gridspec.GridSpecFromSubplotSpec(3, 1,
                                            subplot_spec=gs0[1],
                                            wspace=0, hspace=0)
    ax3 = plt.Subplot(f, gs01[0])
    f.add_subplot(ax3)
    ax4 = plt.Subplot(f, gs01[1])
    f.add_subplot(ax4)
    ax5 = plt.Subplot(f, gs01[2])
    f.add_subplot(ax5)

    return np.array([ax0, ax1, ax2, ax3, ax4, ax5])


def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)




