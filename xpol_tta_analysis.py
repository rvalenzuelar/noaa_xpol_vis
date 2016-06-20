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

"""

# import Windprof2 as wp
import xpol
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tta_analysis import tta_analysis


class process:


    def __init__(self):

        self.rhi_tta = None
        self.rhi_ntta = None
        self.ppi_tta = None
        self.ppi_ntta = None

        rhi_df = make_dataframe(mode='rhi')
        ppi_df = make_dataframe(mode='ppi')

        # print rhi_df.index.size
        # print ppi_df.index.size

        self.rhi_df = rhi_df
        self.ppi_df = ppi_df

        # self.process(rhi_df, mode='rhi')
        # self.process(ppi_df, mode='ppi')


    def process(self, xpol_df, mode=None):
        
        tta_vr = []
        tta_z = []
        tta_thres = []
        notta_vr = []
        notta_z = []
        notta_thres = []    

        params = dict(wdir_surf=125,wdir_wprof=170,
                      rain_czd=0.25,nhours=2)

        ' makes hourly groups of xpol dataframe'
        grp = pd.TimeGrouper('1H')
        xpol_grp = xpol_df.groupby(grp)

        ' add column to create booleans'
        # xpol_grp.apply(new_column)

        y = None
        TTAdf = None
        NTTAdf = None

        for date_grp, _ in xpol_grp.groups.iteritems():

            year = date_grp.year

            ' gets new tta_dates if year changes'
            if y != year:
                tta = tta_analysis(year)
                tta.start_df(**params)
                tta_dates = tta.tta_dates
                y = year

            if date_grp in tta_dates:
                if TTAdf is None:
                    TTAdf = xpol_grp.get_group(date_grp)
                else:
                    TTAdf = TTAdf.append(xpol_grp.get_group(date_grp))
            else:
                if NTTAdf is None:
                    NTTAdf = xpol_grp.get_group(date_grp)
                else:
                    NTTAdf = NTTAdf.append(xpol_grp.get_group(date_grp))

        if mode == 'rhi':
            self.rhi_tta = TTAdf
            self.rhi_ntta = NTTAdf
        elif mode == 'ppi':
            self.ppi_tta = TTAdf
            self.ppi_ntta = NTTAdf





        # return {'tta_vr': tta_vr, 'tta_z': tta_z, 'tta_thres': tta_thres,
        #         'notta_vr': notta_vr, 'notta_z': notta_z, 'notta_thres': notta_thres}

def process_ppi(ppi_list):


    for ppis, case in zip(ppi_list, np.arange(len(ppi_list))+8):

        tta_times = wp.get_tta_times(case=str(case), homedir=homedir)
        print(tta_times)

        tta_idxs = np.asarray([], dtype=int)
        for time in tta_times:
            idx = np.where((ppis.index.day == time.day) &
                           (ppis.index.hour == time.hour))[0]
            if idx.size > 0:
                tta_idxs = np.append(tta_idxs, idx)

        notta_idxs = np.delete(np.arange(len(ppis.index)), tta_idxs)
        ppis_tta = ppis.iloc[tta_idxs]
        ppis_notta = ppis.iloc[notta_idxs]

        if ppis_tta.size > 0:
            print('TTA')
            dbz_freq, thres, csum = xpol.get_dbz_freq(ppis_tta['ZA'],
                                                      percentile=50)
            tta_z.append(dbz_freq)
            tta_thres.append(thres)
            ppi_tta_mean, good = xpol.get_mean(ppis_tta['VR'], name='VR')
            tta_vr.append(ppi_tta_mean)

        if ppis_notta.size > 0:
            print('NO TTA')
            dbz_freq, thres, csum = xpol.get_dbz_freq(ppis_notta['ZA'],
                                                      percentile=50)
            notta_z.append(dbz_freq)
            notta_thres.append(thres)

            ppi_notta_mean, good = xpol.get_mean(ppis_notta['VR'], name='VR')
            notta_vr.append(ppi_notta_mean)

    return {'tta_vr': tta_vr, 'tta_z': tta_z, 'tta_thres': tta_thres,
            'notta_vr': notta_vr, 'notta_z': notta_z,
            'notta_thres': notta_thres}


def process_rhi(rhi_list):


    for rhis, case in zip(rhi_list, np.arange(len(rhi_list))+8):

        tta_times = wp.get_tta_times(case=str(case), homedir=homedir)
        print(tta_times)

        tta_idxs = np.asarray([], dtype=int)
        for time in tta_times:
            idx = np.where((rhis.index.day == time.day) &
                           (rhis.index.hour == time.hour))[0]
            if idx.size > 0:
                tta_idxs = np.append(tta_idxs, idx)

        notta_idxs = np.delete(np.arange(len(rhis.index)), tta_idxs)
        rhis_tta = rhis.iloc[tta_idxs]
        rhis_notta = rhis.iloc[notta_idxs]

        if rhis_tta.size > 0:
            print('TTA')
            dbz_freq, thres, csum = xpol.get_dbz_freq(rhis_tta['ZA'],
                                                      percentile=50)
            tta_z.append(dbz_freq)
            tta_thres.append(thres)
            rhi_tta_mean, good = xpol.get_mean(rhis_tta['VR'], name='VR')
            tta_vr.append(rhi_tta_mean)

        if rhis_notta.size > 0:
            print('NO TTA')
            dbz_freq, thres, csum = xpol.get_dbz_freq(rhis_notta['ZA'],
                                                      percentile=50)
            notta_z.append(dbz_freq)
            notta_thres.append(thres)
            rhi_notta_mean, good = xpol.get_mean(rhis_notta['VR'], name='VR')
            notta_vr.append(rhi_notta_mean)

    return {'tta_vr': tta_vr, 'tta_z': tta_z, 'tta_thres': tta_thres,
            'notta_vr': notta_vr, 'notta_z': notta_z,
            'notta_thres': notta_thres}       




def make_dataframe(mode=None):

    '''
        make xpol dataframes appending cases
    '''
    setcase = {8: [0.5, 180, 10, 100],
               9: [0.5, 180, 13, 100],
               10: [0.5, 180, 30, 100],
               11: [0.5, 180, 20, 100],
               12: [0.5,   6, 15, 100],
               13: [0.5, 180, 25, 100],
               14: [0.5, 180, 30, 100]}

    # out_list = []
    first = True
    for case in range(8, 15):

        elevation, azimuth, _, maxv = setcase[case]

        if mode == 'rhi':
            if first is True:
                df = xpol.get_data(case, 'RHI', azimuth)
                first = False
            else:
                df.append(xpol.get_data(case, 'RHI', azimuth))
        elif mode == 'ppi':
            if first is True:
                df = xpol.get_data(case, 'PPI', elevation)
                first = False
            else:
                df.append(xpol.get_data(case, 'PPI', elevation))

    return df


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

def new_column(x):
    x['isTTA']=np.nan
    return x



