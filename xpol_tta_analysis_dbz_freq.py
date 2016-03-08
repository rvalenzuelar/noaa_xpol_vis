import Windprof2 as wp
import xpol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from subplot_axes import add_subplot_axes


sns.reset_orig()

setcase = {8: [0.5, 180, 10, 100],
           9: [0.5, 180, 13, 100],
           10: [0.5, 180, 30, 100],
           11: [0.5, 180, 20, 100],
           12: [0.5,   6, 15, 100],
           13: [0.5, 180, 25, 100],
           14: [0.5, 180, 30, 100]}

closeall = True

homedir = os.path.expanduser('~')
# homedir = '/localdata/'

for case in range(8, 15):

    elevation, azimuth, _, maxv = setcase[case]
    tta_times = wp.get_tta_times(case=str(case), homedir=homedir)
    print(tta_times)

    ' PPIs'
    '**********************************************************************'
    ppis = xpol.get_data(case, 'PPI', elevation, homedir=homedir)

    tta_idxs = np.asarray([], dtype=int)
    for time in tta_times:
        idx = np.where((ppis.index.day == time.day) &
                       (ppis.index.hour == time.hour))[0]
        if idx.size > 0:
            tta_idxs = np.append(tta_idxs, idx)
    notta_idxs = np.delete(np.arange(len(ppis.index)), tta_idxs)

    fig, axes = plt.subplots(2, 2, figsize=(
        11, 10.5), sharex=True, sharey=True)
    ax = axes.flatten()

    ppis_tta = ppis.iloc[tta_idxs]

    if ppis_tta.size > 0:

        print('TTA')
        dbz_freq, thres, csum = xpol.get_dbz_freq(ppis_tta['ZA'])
        xpol.plot(dbz_freq, ax=ax[2], name='freq', smode='ppi',
                  colorbar=False, case=case, vmax=maxv,
                  textbox='dBZ Threshold:{:2.1f}'.format(thres))
        rect = [0.6, -0.1, 0.4, 0.4]
        subax = add_subplot_axes(ax[2], rect)
        xpol.dbz_hist(ppis_tta['ZA'], ax=subax, plot=True)

        rect = [0.6, 0.59, 0.4, 0.4]
        subax = add_subplot_axes(ax[2], rect)
        xpol.make_hist(csum, ax=subax, hist_type='count')

        ppi_tta_mean, good = xpol.get_mean(ppis_tta['VR'], name='VR')
        xpol.plot(ppi_tta_mean, ax=ax[0], name='VR',
                  smode='ppi', colorbar=False, case=case)

        rect = [0.6, 0.59, 0.4, 0.4]
        subax = add_subplot_axes(ax[0], rect)
        xpol.make_hist(good, ax=subax, hist_type='vr')

        n = ' (n={})'.format(ppis_tta.index.size)
    else:
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].text(0.5, 0.5, 'NO DATA', transform=ax[
                   0].transAxes, ha='center', weight='bold')
        ax[2].text(0.5, 0.5, 'NO DATA', transform=ax[
                   2].transAxes, ha='center', weight='bold')
        n = ''

    ax[0].set_title('TTA ' + n)

    ppis_notta = ppis.iloc[notta_idxs]

    if ppis_notta.size > 0:
        print('NO TTA')

        dbz_freq, thres, csum = xpol.get_dbz_freq(ppis_notta['ZA'])
        xpol.plot(dbz_freq, ax=ax[3], name='freq', smode='ppi',
                  colorbar=True, case=case, vmax=maxv,
                  textbox='dBZ Threshold:{:2.1f}'.format(thres))
        rect = [0.7, -0.1, 0.4, 0.4]
        subax1 = add_subplot_axes(ax[3], rect)
        xpol.dbz_hist(ppis_notta['ZA'], ax=subax1, plot=True)

        rect = [0.7, 0.59, 0.4, 0.4]
        subax2 = add_subplot_axes(ax[3], rect)
        xpol.make_hist(csum, ax=subax2, hist_type='count')

        ppi_notta_mean, good = xpol.get_mean(ppis_notta['VR'], name='VR')
        xpol.plot(ppi_notta_mean, ax=ax[1],  name='VR',
                  smode='ppi', colorbar=True, case=case)

        rect = [0.7, 0.59, 0.4, 0.4]
        subax = add_subplot_axes(ax[1], rect)
        xpol.make_hist(good, ax=subax, hist_type='vr')

        n = ' (n={})'.format(ppis_notta.index.size)

    ax[1].set_title('NO-TTA' + n)

    t = 'XPOL Time Average - C{} {} - {} - Nscan={}\nBeg: {} - End: {} (UTC)'
    ym = ppis.index[0].strftime('%Y-%b')
    beg = ppis.index[0].strftime('%d %H:%M')
    end = ppis.index[-1].strftime('%d %H:%M')
    title = t.format(str(case).zfill(2), ym, ppis.index.name,
                     ppis.index.size, beg, end)
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=0.05, wspace=0.1,
                        left=0.05, right=0.95, bottom=0.05)

    if closeall:
        o = 'c{}_xpol_tta_average_{}_{}.pdf'
        saveas = o.format(str(case).zfill(2), 'ppi',
                          str(elevation * 10).zfill(3))
        xpolpdf = PdfPages(saveas)
        xpolpdf.savefig()
        xpolpdf.close()

    ' RHIs'
    '*************************************************************************'

    rhis = xpol.get_data(case, 'RHI', azimuth, homedir=homedir)
    elev = xpol.get_max(rhis['EL'])

    tta_idxs = np.asarray([], dtype=int)
    for time in tta_times:
        idx = np.where((rhis.index.day == time.day) &
                       (rhis.index.hour == time.hour))[0]
        if idx.size > 0:
            tta_idxs = np.append(tta_idxs, idx)
    notta_idxs = np.delete(np.arange(len(rhis.index)), tta_idxs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    ax = axes.flatten()

    rhis_notta = rhis.iloc[notta_idxs]
    if rhis_notta.size > 0:
        print('NO TTA')

        dbz_freq, thres, csum = xpol.get_dbz_freq(rhis_notta['ZA'])
        xpol.plot(dbz_freq, ax=ax[3], name='freq', smode='rhi',
                  colorbar=True, case=case, vmax=maxv,
                  textbox='dBZ Threshold:{:2.1f}'.format(thres))

        rect = [0.8, 0.7, 0.35, 0.35]
        subax = add_subplot_axes(ax[3], rect)
        xpol.dbz_hist(rhis_notta['ZA'], ax=subax, plot=True)

        rect = [0.0, 0.7, 0.35, 0.35]
        subax = add_subplot_axes(ax[3], rect)
        xpol.make_hist(csum, ax=subax, hist_type='count')

        rhi_notta_mean, good = xpol.get_mean(rhis_notta['VR'], name='VR')
        xpol.plot(rhi_notta_mean, ax=ax[1],  name='VR', smode='rhi',
                  colorbar=True, case=case, elev=elev)

        rect = [0.0, 0.62, 0.35, 0.35]
        subax = add_subplot_axes(ax[1], rect)
        xpol.make_hist(good, ax=subax, hist_type='vr')

        n = ' (n={})'.format(rhis_notta.index.size)
        ax[1].set_title('NO-TTA ' + n)

    ax[0].set_ylabel('Altitude MSL [km]', fontsize=14)
    ax[2].set_xlabel('Distance from the radar [km]', fontsize=14)

    rhis_tta = rhis.iloc[tta_idxs]
    if rhis_tta.size > 0:
        print('TTA')

        dbz_freq, thres, csum = xpol.get_dbz_freq(rhis_tta['ZA'])
        xpol.plot(dbz_freq, ax=ax[2], name='freq', smode='rhi',
                  colorbar=False, case=case, vmax=maxv,
                  textbox='dBZ Threshold:{:2.1f}'.format(thres))

        rect = [0.7, 0.7, 0.35, 0.35]
        subax = add_subplot_axes(ax[2], rect)
        xpol.dbz_hist(rhis_tta['ZA'], ax=subax, plot=True)

        rect = [-0.15, 0.7, 0.35, 0.35]
        subax = add_subplot_axes(ax[2], rect)
        xpol.make_hist(csum, ax=subax, hist_type='count')

        rhi_tta_mean, good = xpol.get_mean(rhis_tta['VR'], name='VR')
        xpol.plot(rhi_tta_mean, ax=ax[0], name='VR', smode='rhi',
                  colorbar=False, case=case,
                  elev=elev, add_yticklabs=True)

        rect = [-0.15, 0.62, 0.35, 0.35]
        subax = add_subplot_axes(ax[0], rect)
        xpol.make_hist(good, ax=subax, hist_type='vr')

        n = ' (n={})'.format(rhis_tta.index.size)
        ax[0].set_title('TTA ' + n)

    else:
        xticks = ax[1].get_xticks()
        yticks = ax[1].get_yticks()
        ax[0].set_xticks(xticks)
        ax[0].set_yticks(yticks)
        ax[0].set_xticklabels([str(int(x)) for x in xticks])
        ax[0].set_yticklabels([str(y) for y in yticks])
        ax[0].text(0.5, 0.5, 'NO DATA', transform=ax[
                   0].transAxes, ha='center', weight='bold')
        ax[2].text(0.5, 0.5, 'NO DATA', transform=ax[
                   2].transAxes, ha='center', weight='bold')
        n = ''

    t = 'XPOL Time Average - C{} {} - {} - Nscan={}\nBeg: {} - End: {} (UTC)'
    ym = rhis.index[0].strftime('%Y-%b')
    beg = rhis.index[0].strftime('%d %H:%M')
    end = rhis.index[-1].strftime('%d %H:%M')
    title = t.format(str(case).zfill(2), ym, rhis.index.name,
                     rhis.index.size, beg, end)
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.97)

    if closeall:
        o = 'c{}_xpol_tta_average_{}_{}.pdf'
        saveas = o.format(str(case).zfill(2), 'rhi', str(azimuth).zfill(3))
        xpolpdf = PdfPages(saveas)
        xpolpdf.savefig()
        xpolpdf.close()
        plt.close('all')
    else:
        plt.show(block=False)
