'''
    Processing of NOAA XPOL in cartesian coordinates

    Raul Valenzuela
    January, 2016
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Circlem

from netCDF4 import Dataset
from glob import glob
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from rv_utilities import add_colorbar

#from mpl_toolkits.basemap import Basemap


def plot(array, ax=None, show=True, name=None, smode=None,
         date=None, title=None, add_azline=None,
         colorbar=True, extent=None, second_date=None, case=None,
         add_yticklabs=False, vmax=None, textbox=None):

    if ax is None:
        fig, ax = plt.subplots()

    if extent is None:
        if smode == 'rhi':
            extent = [-40, 30, 0.05, 10]
        elif smode == 'ppi':
            extent = [-58, 45, -58, 33]

    if name == 'ZA':
        vmin, vmax = [-10, 45]
        cmap = 'jet'
        cbar_ticks = range(vmin, vmax + 5, 5)
    elif name == 'VR':
        if smode == 'ppi':
            vmin, vmax = [-20, 20]
            cmap = custom_cmap('ppi_vr1')
            cbar_ticks = range(vmin, vmax + 5, 5)
        else:
            vmin, vmax = [0, 40]
            cmap = custom_cmap('rhi_vr1')
            cbar_ticks = range(vmin, vmax + 10, 10)
    elif name == 'count':
        vmin, vmax = [0, 180]
        cmap = 'viridis'
        cbar_ticks = range(vmin, vmax + 10, 10)
    elif name == 'percent':
        vmin, vmax = [0, 100]
        cmap = 'inferno'
        cbar_ticks = range(vmin, vmax + 10, 10)
    elif name == 'freq':
        vmin = 0
        cmap = 'inferno'
        cbar_ticks = range(0, 110, 10)
    elif name == 'ratio':
        vmin = 0
        cmap = 'viridis'
        cbar_ticks = np.arange(0, vmax + 0.2, 0.2)
    elif name == 'rainrate':
        vmin = 0
        cmap = 'viridis_r'
        cbar_ticks = np.arange(0, vmax + 10, 10)

    im = ax.imshow(array, interpolation='none', origin='lower',
                   vmin=vmin, vmax=vmax,
                   cmap=cmap,
                   extent=extent,
                   aspect='auto')

    if smode == 'ppi':
        add_rings(ax, space_km=10, color='k')
        add_azimuths(ax, space_deg=30)
        add_ring(ax, radius=57.8,
                 color=(0.85, 0.85, 0.85),
                 lw=10)  # masking outer ring
        add_background_circle(ax,
                              radius=60, color=(0.85, 0.85, 0.85))
        if add_azline:
            x = np.arange(-40, 40)
            if add_azline in [0, 180]:
                y = x
                x = y - x
            elif add_azline in [90, 270]:
                y = x * np.sin(np.radians(add_azline))
            else:
                y = x * np.cos(np.radians(add_azline - 180))
            ax.plot(x, y, '-', color='red')
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        # ax.set_axis_bgcolor((0.5,0.5,0.5))
        if textbox:
            props = dict(facecolor='white')
            plt.text(0.05, 0.05, textbox, transform=ax.transAxes,
                     bbox=props)
    elif smode == 'rhi':
        # ax.set_xlim([-40, 20])
        # ax.set_xlim([extent[0], extent[1]+2])
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([0, 11])
        yticks = np.arange(0.5, 11.5, 1.0)
        ax.set_yticks(yticks)
        if add_yticklabs:
            yticklabels = np.array([str(y) for y in yticks])
            # yticklabels[::2]=''
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticklabels([''])
        ax.grid(True)
        if textbox:
            props = dict(facecolor='white')
            plt.text(0.73, 0.05, textbox, transform=ax.transAxes,
                     bbox=props)
    plt.draw()

    if date:
        plt.text(0.0, 1.01, date, ha='left', transform=ax.transAxes)
    if second_date:
        plt.text(1.0, 1.01, second_date, ha='right', transform=ax.transAxes)
    if title:
        plt.text(0.4, 1.01, title, ha='left', transform=ax.transAxes)
    if colorbar:
        add_colorbar(ax, im, cbar_ticks)

    return ax


def plotm(array, ax=None, show=True, name=None):

#    if not ax:
#        fig, ax = plt.subplots()
#
#    if name == 'ZA':
#        vmin, vmax = [-10, 45]
#        cmap = 'jet'
#        cbar_ticks = range(vmin, vmax + 5, 5)
#    elif name == 'VR':
#        vmin, vmax = [-20, 20]
#        cmap = custom_cmap()
#        cbar_ticks = range(vmin, vmax + 5, 5)
#
#    radarx = np.arange(-58, 45.5, 0.5)
#    radary = np.arange(-58, 33.5, 0.5)
#    radarloc = (38.505260, -123.229607)  # at FRS
#    radarlon = cart2geo(radarx, 'WE', radarloc)
#    radarlat = cart2geo(radary, 'NS', radarloc)
#
#    m = Basemap(projection='merc',
#                llcrnrlat=min(radarlat),
#                urcrnrlat=max(radarlat),
#                llcrnrlon=min(radarlon),
#                urcrnrlon=max(radarlon),
#                lat_0=radarloc[0],
#                lon_0=radarloc[1],
#                resolution='i')
#
#    m.drawcoastlines()
#    im = m.imshow(array, interpolation='none', origin='lower',
#                  vmin=vmin, vmax=vmax, cmap=cmap)
#    add_rings(ax, space_km=10, color='k', mapping=[
#              m, radarloc[0], radarloc[1]])
#    add_locations(ax, m, 'all')
#
#    add_colorbar(ax, im, cbar_ticks)
#
#    plt.show(block=False)
    
    pass

def plot_mean(means, dates, name, smode, elev=None, title=None, colorbar=True):

    xpolmean = PdfPages('xpol_meanscan_' + name.lower() + '.pdf')
    ntimes, _, _ = means.shape
    for n in range(ntimes):
        if smode in ['ppi', 'PPI']:
            fig, ax = plt.subplots()
            plotm(means[n, :, :], ax=ax, show=False, name=name)
        elif smode in ['rhi', 'RHI']:
            fig, ax = plt.subplots(figsize=(10, 5))
            if name == 'ZA':
                plot(means[n, :, :], ax=ax, show=False, name=name, smode='rhi',
                     date=dates[n], title=title, colorbar=colorbar)
            elif name == 'VR':
                plot(means[n, :, :], ax=ax, show=False, name=name, smode='rhi',
                     date=dates[n], elev=elev[n, :, :], title=title,
                     colorbar=colorbar)
            ax.set_xlabel('Distance from radar [km]')
            ax.set_ylabel('Altitude AGL [km]')
        print(dates[n])
        xpolmean.savefig()
        plt.close('all')
    xpolmean.close()


def plot_single(xpol_dataframe, name=None, smode=None,
                colorbar=True, case=None, saveas=None,
                convert=None, vmax=None):

    xpolsingle = PdfPages(saveas)
    dates = xpol_dataframe.index
    single = xpol_dataframe[name]
    ntimes = single.shape[0]
    title = 'C' + str(case).zfill(2) + ' ' + \
        xpol_dataframe.index.name + ' ' + name
    print(title)
    if 'time_az2' in xpol_dataframe:
        second_dates = xpol_dataframe['time_az2']
    else:
        second_dates = [''] * ntimes

    if case in [13, 14]:
        rhi_extent = [-30, 30, 0.05, 11]
    elif case == 12:
        rhi_extent = [-22, 30, 0.05, 11]
    else:
        rhi_extent = [-40, 20, 0.05, 11]
    ppi_extent = [-58, 45, -58, 33]

    for n in range(ntimes):
        if smode in ['ppi', 'PPI']:
            fig, ax = plt.subplots(figsize=(7, 6))
            ar = single.ix[n]
            valid = ar.size - np.sum(np.isnan(ar))  # number of non-nan points
            if convert == 'to_rainrate':
                ar = get_rainrate(ar, rate='hour')
                plot(ar, ax=ax, show=False, name='rainrate', smode='ppi',
                     date=dates[n], colorbar=colorbar,
                     extent=ppi_extent, title=title + ' points=' + str(valid),
                     vmax=vmax)
            else:
                plot(ar, ax=ax, show=False, name=name, smode='ppi',
                     date=dates[n], colorbar=colorbar,
                     extent=ppi_extent, title=title + ' points=' + str(valid))
            plt.subplots_adjust(left=0.05, right=0.93, bottom=0.05, top=0.95)

        if smode in ['rhi', 'RHI']:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_gid('ax1')
            ar = single.ix[n]
            valid = ar.size - np.sum(np.isnan(ar))  # number of non-nan points

            if name == 'VR':
                elev_angle = get_max(xpol_dataframe['EL'])
                plot(ar, ax=ax, show=False, name=name, smode='rhi',
                     date=dates[n], elev=elev_angle, colorbar=colorbar,
                     extent=rhi_extent,  second_date=second_dates[n],
                     title=title + ' points=' + str(valid))
            elif name == 'ZA':
                if second_dates is not None:
                    plot(ar, ax=ax, show=False, name=name, smode='rhi',
                         date=dates[n], colorbar=colorbar, extent=rhi_extent,
                         second_date=second_dates[n],
                         title=title + ' points=' + str(valid))
                else:
                    plot(ar, ax=ax, show=False, name=name, smode='rhi',
                         date=dates[n], colorbar=colorbar, extent=rhi_extent,
                         title=title + ' points=' + str(valid))

            ax.set_xlabel('Distance from radar [km]')
            ax.set_ylabel('Altitude AGL [km]')
            plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12)

        o = 'Plotting {} {}'
        print(o.format(smode, dates[n].strftime('%Y-%b-%d %H:%M:%S')))

        xpolsingle.savefig()
        plt.close('all')
    xpolsingle.close()


def get_data(case, scanmode, angle, datadir=None, index=None):

    '''
        datadir is full path of directory containing 
        subdirectories with netcdf files separated by case
    '''

    import os

    if datadir is None:
        try:
            datadir = os.environ['XPOL_PATH']
        except KeyError:
            print('*** Need to provide datadir or export XPOL_PATH ***')

    if scanmode == 'PPI':
        basestr = datadir + '/c{0}/PPI/elev{1}/'
        angle = angle * 10
    elif scanmode == 'RHI':
        basestr = datadir + '/c{0}/RHI/az{1}/'

    basedir = basestr.format(str(case).zfill(2), str(int(angle)).zfill(3))
    cdf_files = glob(basedir + '*.cdf')
    cdf_files.sort()
    if index is not None:
        cdf_files = [cdf_files[index]]

    za_arrays = []  # attenuation corrected dBZ
    vr_arrays = []  # radial velocity
    el_arrays = []  # elevation angle
    time_index = []
    time_index2 = []  # for complementary azimuths

    for n, f in enumerate(cdf_files):
        data = Dataset(f, 'r')
        scale = 64.
        if scanmode == 'PPI':
            VR = np.squeeze(data.variables['VR'][0, 1, :, :]) / scale
            ZA = np.squeeze(data.variables['ZA'][0, 1, :, :]) / scale
            EL = np.squeeze(data.variables['EL'][0, 1, :, :]) / scale
        elif scanmode == 'RHI':
            VR = np.squeeze(data.variables['VR'][0, :, 1, :]) / scale
            ZA = np.squeeze(data.variables['ZA'][0, :, 1, :]) / scale
            EL = np.squeeze(data.variables['EL'][0, :, 1, :]) / scale

        ' add second complemental azimuth in RHI '
        if case in [11, 13, 14] and scanmode == 'RHI':
            basedir = basestr.format(str(case).zfill(2), '360')
            cdf_files2 = glob(basedir + '*.cdf')
            cdf_files2.sort()
            if index is not None:
                cdf_files2 = [cdf_files2[index]]
            data2 = Dataset(cdf_files2[n])
            VR2 = np.squeeze(data2.variables['VR'][0, :, 1, :]) / scale
            ZA2 = np.squeeze(data2.variables['ZA'][0, :, 1, :]) / scale
            EL2 = np.squeeze(data2.variables['EL'][0, :, 1, :]) / scale
            VR = np.hstack((VR, VR2))
            ZA = np.hstack((ZA, ZA2))
            EL = np.hstack((EL, EL2))
            sd = data2.variables['start_date'][:]
            st = data2.variables['start_time'][:]
            raw_date = parse_rawdate(sd, st)
            raw_fmt = '%m/%d/%Y %H:%M:%S'
            dt2 = datetime.strptime(raw_date, raw_fmt)
            time_index2.append(dt2)
            data2.close()

        VR = np.ma.filled(VR, fill_value=np.nan)
        ZA = np.ma.filled(ZA, fill_value=np.nan)
        EL = np.ma.filled(EL, fill_value=np.nan)

        ' when merging is not getting rid of fill values so this fix it'
        VR[VR == -32768.0] = np.nan
        ZA[ZA == -32768.0] = np.nan

        if scanmode == 'RHI':
            ''' compute wind component and remove
            center cone in radial velocity '''
            VR = VR / np.cos(np.radians(EL))
            idx = np.where((EL >= 65) & (EL <= 115))
            VR[idx] = np.nan
            ''' abs value represent southerly wind in 180-360
            RHIs '''
            VR = np.abs(VR)
            if case == 12:
                ''' c12 uses 186-6 azim RHIs so compute meridional'''
                VR = VR/np.cos(np.radians(6))

        vr_arrays.append(VR)
        za_arrays.append(ZA)
        el_arrays.append(EL)

        sd = data.variables['start_date'][:]
        st = data.variables['start_time'][:]
        raw_date = parse_rawdate(sd, st)
        raw_fmt = '%m/%d/%Y %H:%M:%S'
        dt = datetime.strptime(raw_date, raw_fmt)
        time_index.append(dt)
        data.close()

    df = pd.DataFrame(index=time_index, columns=['ZA', 'VR', 'EL'])
    df['ZA'] = za_arrays
    df['VR'] = vr_arrays
    df['EL'] = el_arrays

    # -- this block messes with index retreival --
    # if scanmode == 'PPI':
    #     name = '{0} {1}'
    #     df.index.name = name.format(scanmode, str(angle / 10.))
    # else:
    #     name = '{0} {1}-{2}'
    #     if angle <= 180 and angle > 90:
    #         df.index.name = name.format(scanmode, str(angle), str(angle + 180))
    #     elif angle <= 90 and angle > 0:
    #         df.index.name = name.format(scanmode, str(angle + 180), str(angle))
    #     else:
    #         df.index.name = name.format(scanmode, str(angle), str(angle - 180))

    if len(time_index2) > 0:
        df['time_az2'] = time_index2

    return df


def get_axis(axisname, case, scanmode):

    import os

    try:
        datadir = os.environ['XPOL_PATH']
    except KeyError:
        print('*** Need to provide datadir or export XPOL_PATH ***')

    if scanmode in ['PPI','ppi']:
        basestr = datadir + '/c{0}/PPI/elev{1}/'
        basedir = [basestr.format(str(case).zfill(2),'005')]
    elif scanmode in ['RHI','rhi']:
        basestr = datadir + '/c{0}/RHI/az{1}/'
        if case in [11, 13, 14]:
            ang1, ang2 = ['180', '360']
            basedir = [basestr.format(str(case).zfill(2), ang1),
                       basestr.format(str(case).zfill(2), ang2)]
        elif case in [8, 9, 10]:
            basedir = [basestr.format(str(case).zfill(2), '180')]
        elif case == 12:
            basedir = [basestr.format(str(case).zfill(2), '006')]

    if len(basedir) == 1:
        cdf_files = glob(basedir[0] + '*.cdf')
        data = Dataset(cdf_files[0], 'r')
        return data.variables[axisname][:]
    else:
        if axisname in ['Z','z']:
            cdf_files = glob(basedir[0] + '*.cdf')
            data = Dataset(cdf_files[0], 'r')
            return data.variables[axisname][:]
        else:
            cdf_files0 = glob(basedir[0] + '*.cdf')
            cdf_files1 = glob(basedir[1] + '*.cdf')
            data1 = Dataset(cdf_files0[0], 'r')
            data2 = Dataset(cdf_files1[0], 'r')
            ax1 = data1.variables[axisname][:]
            ax2 = data2.variables[axisname][:]
            return np.concatenate((ax1, ax2), axis=0)

def parse_rawdate(start_date=None, start_time=None, datestring=None):

    if start_date is not None and start_time is not None:
        date = ''.join(start_date).replace('/01', '/20')
        time = ''.join(start_time)
        raw_date = date + ' ' + time
        return raw_date


def get_mean(arrays, minutes=None, name=None, good_thres=1000):
    '''
    good_thres is the minimum # of pixels (gates, cells) that
    a sweep needs to have to be considered good
    '''
    if minutes:
        g = pd.TimeGrouper(str(minutes) + 'T')
        G = arrays.groupby(g)

        gindex = G.indices.items()
        gindex.sort()
        mean = []
        dates = []
        for gx in gindex:
            gr = arrays.ix[gx[1]].values
            a = gr[0]

            if name == 'ZA':
                a = np.power(10., a / 10.)

            for g in gr[1:]:
                a = np.dstack((a, g))

            if a.ndim == 2:
                ''' in case the hour has only one sweep
                creates a singleton (nrows, ncols, 1) '''
                a = np.expand_dims(a, axis=2)
            m = np.nanmean(a, axis=2)

            if name == 'ZA':
                m = 10 * np.log10(m)
            mean.append(m)
            dates.append(gx[0])

            return dates, np.array(mean)

    else:
        narrays = arrays.shape[0]
        good_array = np.array([])
        for n in range(0, narrays):
            if n == 0:
                A = arrays.iloc[[n]].values[0]  # first value
                # number of non-nan points
                valid = A.size - np.sum(np.isnan(A))
                # good_array = np.append(good_array, valid)
            else:
                a = arrays.iloc[[n]].values[0]
                valid = a.size - np.sum(np.isnan(a))
                # good_array = np.append(good_array, valid)
                A = np.dstack((A, a))
            good_array = np.append(good_array, valid)

        if name == 'ZA':
            A = toLinear(A)

        mean = np.nanmean(A, axis=2)

        if name == 'ZA':
            mean = toLog10(mean)

        return mean, good_array

def get_median(arrays, minutes=None, name=None, good_thres=1000):
    '''
    good_thres is the minimum # of pixels (gates, cells) that
    a sweep needs to have to be considered good
    '''
    if minutes:
        g = pd.TimeGrouper(str(minutes) + 'T')
        G = arrays.groupby(g)

        gindex = G.indices.items()
        gindex.sort()
        median = []
        dates = []
        for gx in gindex:
            gr = arrays.ix[gx[1]].values
            a = gr[0]

            if name == 'ZA':
                a = np.power(10., a / 10.)

            for g in gr[1:]:
                a = np.dstack((a, g))

            if a.ndim == 2:
                ''' in case the hour has only one sweep
                creates a singleton (nrows, ncols, 1) '''
                a = np.expand_dims(a, axis=2)
            m = np.nanmedian(a, axis=2)

            if name == 'ZA':
                m = 10 * np.log10(m)
            median.append(m)
            dates.append(gx[0])

            return dates, np.array(median)

    else:
        narrays = arrays.shape[0]
        good_array = np.array([])
        for n in range(0, narrays):
            if n == 0:
                A = arrays.iloc[[n]].values[0]  # first value
                # number of non-nan points
                valid = A.size - np.sum(np.isnan(A))
                # good_array = np.append(good_array, valid)
            else:
                a = arrays.iloc[[n]].values[0]
                valid = a.size - np.sum(np.isnan(a))
                # good_array = np.append(good_array, valid)
                A = np.dstack((A, a))
            good_array = np.append(good_array, valid)

        if name == 'ZA':
            A = toLinear(A)

        median = np.nanmedian(A, axis=2)

        if name == 'ZA':
            median = toLog10(median)

        return median, good_array


def get_dbz_freq(arrays, percentile=None, constant=None):
    
    
    from rv_utilities import pandas2stack
    
    narrays = arrays.shape[0]
    X = pandas2stack(arrays)
    Z = X[~np.isnan(X)].flatten()
    
    ' gets cummulative distribution of Z '
    freqz, binsz = np.histogram(Z,
                                bins=np.arange(-15,50),
                                density=True)    
    distrz = np.cumsum(freqz)    

    ''' gets percentile '''
    if percentile is not None:
        thres = np.percentile(Z, percentile)
    elif constant is not None:
        thres = constant
    
    a = arrays.iloc[[0]].values[0]
    COND = np.zeros(a.shape)
    for n in range(narrays):
        # print('processing array # {}'.format(n))
        a = arrays.iloc[[n]].values[0]
        mask = make_mask(a)
        a[mask] = np.nan  # removes artifacts along edge
        cond = (a >= thres).astype(int)
        COND = np.dstack((COND, cond))

#    mean, _ = get_mean(arrays, name='ZA')
    method = 2
    if method == 1:
        csum = np.sum(COND, axis=2),
        freq = (csum / narrays) * 100.
    elif method == 2:
        csum = np.sum(COND, axis=2)

        ''' QC: remove isolated grid points along sweep edge '''
        summax = int(np.nanmax(csum))
        hist, bins = np.histogram(csum, bins=range(summax+2))
        idx = np.where(hist == 1)[0]
        if idx.size > 0:
            for i in idx:
                csum[csum == i] = np.nan
        
        ''' normalized frequency '''
        narrays = np.nanmax(csum)
        freq = (csum / narrays) * 100.

    ''' removes missing obs '''
    freq[freq == 0] = np.nan

    return freq, thres, distrz, binsz
    # return csum, thres, mean


def convert_to_common_grid(input_array):

    """
    :param input_array: rhi array 
    :return:  pandas df with common grid dimensions to be
            statistically processed 
    """

    if type(input_array) == pd.core.frame.DataFrame:
        newdf = input_array.copy()
        for n, a in enumerate(input_array):
            if a.shape == (61, 429):
                midp = 286
            elif a.shape == (61, 430):
                midp = 215
            elif a.shape == (61, 372):
                midp = 157
            offset = 290-midp
            size = a.shape[1]
            end = size+offset
            common = np.empty((61, 505))
            common[:] = np.nan
            common[:, offset:end] = a
            newdf.iloc[n] = common
        return newdf
        
    elif type(input_array) == pd.core.series.Series:
        newdf = input_array.copy()
        n = 0
        for i, a in input_array.iteritems():
            if a.shape == (61, 429):
                midp = 286
            elif a.shape == (61, 430):
                midp = 215
            elif a.shape == (61, 372):
                midp = 157
            offset = 290-midp
            size = a.shape[1]
            end = size+offset
            common = np.empty((61, 505))
            common[:] = np.nan
            common[:, offset:end] = a
            newdf.iloc[n] = common
            n += 1
        return newdf      
        
    else:
        if input_array.shape == (61, 429):
            midp = 286
        elif input_array.shape == (61, 430):
            midp = 215
        elif input_array.shape == (61, 372):
            midp = 157
        offset = 290-midp
        size = input_array.shape[1]
        end = size+offset
        common = np.empty((61, 505))
        common[:] = np.nan
        common[:, offset:end] = input_array
        return common


def make_mask(array):
    ''' creates mask of border to eliminate artifacts'''
    mask = np.zeros(array.shape)
    mina = 245
    maxa = 310
    filter_circle1 = [polar2cart(116, z, center=(116, 116))
                      for z in range(mina, maxa)]
    filter_circle2 = [polar2cart(115, z, center=(116, 116))
                      for z in range(mina, maxa)]
    filter_circle3 = [polar2cart(114, z, center=(116, 116))
                      for z in range(mina, maxa)]
    filter_circle4 = [polar2cart(113, z, center=(116, 116))
                      for z in range(mina, maxa)]

    for f in filter_circle4:
        try:
            mask[f] = 1
        except IndexError:
            pass
    for f in filter_circle3:
        try:
            mask[f] = 1
        except IndexError:
            pass
    for f in filter_circle2:
        try:
            mask[f] = 1
        except IndexError:
            pass
    for f in filter_circle1:
        try:
            mask[f] = 1
        except IndexError:
            pass

    for c in range(mask.shape[1]):
        col = mask[:, c]

        idx = np.where(col)[0]
        try:
            col[idx+1] = True
        except IndexError:
            col[idx[:-2]+1] = True

        idx = np.where(col)[0]
        try:
            col[idx+1] = True
        except IndexError:
            col[idx[:-2]+1] = True

        mask[:, c] = col

    return mask.astype(bool)


def polar2cart(ro, phi, center=(0, 0)):
    x = int(ro*np.cos(np.radians(phi))+center[0])
    y = int(ro*np.sin(np.radians(phi))+center[1])
    return(x, y)


def dbz_hist(arrays, ax=None, plot=False):

    import matplotlib.mlab as mlab
    from rv_utilities import pandas2stack

    A = pandas2stack(arrays)
    mu = np.nanmean(A)
    sigma = np.nanstd(A)

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        hist, bins, _ = ax.hist(A.flatten(), normed=1,
                                bins=range(-10, 60, 5), histtype='step')
        x = range(-20, 60)
        y = mlab.normpdf(x, mu, sigma)
        plt.plot(x, y, 'r-', linewidth=2)
        ax.text(0.6, 0.85, r'$\mu={:3.1f}$'.format(mu),
                transform=ax.transAxes, weight='bold')
        ax.text(0.6, 0.75, r'$\sigma={:3.1f}$'.format(sigma),
                transform=ax.transAxes, weight='bold')
        ax.set_yticklabels('')
        plt.show(block=False)

    return mu, sigma


def make_hist(inarray, ax=None, hist_type=None):

    if ax is None:
        fig, ax = plt.subplots()

    if hist_type == 'count':
        top = int(np.nanmax(inarray))
        bins = range(1, top, 1)
        xticks = None
        yticks = None
        # text = False
    elif hist_type == 'vr':
        top = int(np.ceil(np.max(inarray)/1000.)*1000.)
        bins = range(0, top+1000, 1000)
        if top == 5000:
            xticks = range(0, 5000, 1000)
        else:
            xticks = bins[::4]
        xtlab = [int(i/1000.) for i in xticks]
        yticks = True
        # text = True

    hist, bins, _ = ax.hist(inarray.flatten(),
                            bins=bins, histtype='step')

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtlab)
    # if text:
        # ax.text(0.5, 0.9, ' <1000:{:3.0f}'.format(np.sum(hist[0])),
        #         transform=ax.transAxes)
        # ax.text(0.5, 0.8, '>=1000:{:3.0f}'.format(np.sum(hist[1:])),
        # transform=ax.transAxes)

    if yticks is not None:
        pass
    else:
        ax.set_yticks([])

    plt.show(block=False)


def filter_sum(array_sum, dbz_mean=None):
    '''
    Removes artifacts in sum due to
    artifacts in reflectivity
    '''
    arraysum = array_sum.copy()
    topv = int(np.ceil(np.nanmax(arraysum)/10.)*10.)
    if topv == 30:
        delr = 1
    else:
        delr = 5
    hist, bins = np.histogram(arraysum,
                              bins=range(0, topv, delr))
    count_thres = 5
    idx = np.where(hist < count_thres)[0]
    if (dbz_mean is None) and (idx.size > 0):
        target_count = bins[idx[0]]
        arraysum[arraysum >= target_count] = np.nan
    elif (idx.size > 0):
        target_count = bins[idx[0]]
        cond1 = (arraysum >= target_count)
        mean = np.nanmean(dbz_mean)
        std = np.nanstd(dbz_mean)
        botc = mean - std
        topc = mean + std
        cond2 = (dbz_mean <= botc)
        cond3 = (dbz_mean >= topc)
        arraysum[cond1 & (cond2 | cond3)] = np.nan

        # target_count = bins[idx[0]]
        # cond1 = (arraysum >= target_count)
        # mean = np.nanmean(dbz_mean)
        # std = np.nanstd(dbz_mean)
        # coef = 0.5
        # nsum = 999
        # while nsum > 0:
        #     print(nsum)
        #     botc = mean - coef*std
        #     topc = mean + coef*std
        #     cond2 = (dbz_mean <= botc) | (dbz_mean >= topc)
        #     arraysum[cond1 & cond2] = np.nan
        #     hist, bins = np.histogram(arraysum, bins=range(0, topv))
        #     nsum = np.sum(hist[hist < count_thres])
        #     coef += 0.5

    return arraysum


def get_dbz_precip_accum(arrays):

    narrays = arrays.shape[0]

    for n in range(0, narrays):
        a = arrays.iloc[[n]].values[0]
        pp = get_rainrate(a, rate='minute')  # [mm min^-1]
        timestamp = arrays.iloc[[n]].index[0]
        mins = timestamp.minute
        if n == 0:
            pp = pp * mins  # [mm]
            pparray = np.zeros(pp.shape)
            pparray = np.dstack((pparray, pp))
            min0 = mins
        else:
            mins2 = np.abs(mins - min0)
            pp = pp * mins2  # [mm]
            pparray = np.dstack((pparray, pp))
            min0 = mins

    pp_acum = np.nansum(pparray, axis=2)
    mean, _ = get_mean(arrays, name='ZA')
    pp_acum[np.isnan(mean)] = np.nan
    return pp_acum


def get_rainrate(array, rate=None):
    ' uses Matrosov et al (2005) eq 10'

    linear = toLinear(array)  # [mm^6 m^-3]
    rainrate = np.power(linear / 180., 10 / 14.)  # [mm h^-1]
    if rate == 'hour':
        return rainrate
    elif rate == 'minute':
        return rainrate / 60.  # [mm min^-1]


def toLinear(x):

    return np.power(10., x / 10.)


def toLog10(x):

    return 10. * np.log10(x)


def get_count(arrays):

    a = np.isnan(arrays.ix[[0]].values[0])
    narrays = arrays.shape[0]
    for n in range(1, narrays):
        rr = np.isnan(arrays.ix[[n]].values[0])
        a = np.dstack((a, rr))

    shape = arrays.ix[[0]].values[0].shape
    total = np.zeros(shape) + narrays
    return total - np.sum(a, axis=2)


def get_percent(arrays):

    a = np.isnan(arrays.ix[[0]].values[0])
    narrays = arrays.shape[0]
    for n in range(1, narrays):
        rr = np.isnan(arrays.ix[[n]].values[0])
        a = np.dstack((a, rr))

    shape = arrays.ix[[0]].values[0].shape
    total = np.zeros(shape) + narrays
    return 100. * (total - np.sum(a, axis=2)) / total


def get_max(arrays):

    a = arrays.ix[[0]].values[0]
    narrays = arrays.shape[0]
    for n in range(1, narrays):
        rr = arrays.ix[[n]].values[0]
        a = np.dstack((a, rr))

    return np.nanmax(a, axis=2)


def add_rings(ax, space_km=10, color='k', mapping=False,
              center=(0,0), alpha=1.0, txtsize=12):


    for r in range(0, 60 + space_km, space_km):

        ring = add_ring(ax=ax, radius=r, mapping=mapping,
                        color=color, center=center, alpha=alpha)
        vert = ring.get_path().vertices
#        print x,y
        if r == 60:
            txt = str(r)+'\nkm'
            va = 'top'
            ha = 'center'
        else:
            txt = str(r)
            va = 'top'
            ha = 'center'
        if mapping:
            textdirection=240
            x, y = vert[textdirection]
            ax.text(x, y, txt, ha=ha, va=va,
                    bbox=dict(fc='none', ec='none', pad=2.),
                    clip_on=True, size=txtsize,linespacing=0.7)            
        else:
            textdirection = -5
            x, y = vert[textdirection]
            ax.text(x * r, y * r, txt, ha=ha, va=va,
                    bbox=dict(fc='none', ec='none', pad=2.),
                    clip_on=True, size=txtsize,linespacing=0.7)
        

def add_ring(ax=None, radius=None, mapping=None, color=None,
             lw=1, center=None, alpha=1.0):

    from shapely.geometry import Polygon
    from descartes import PolygonPatch

    if mapping is not None:
        m = mapping[0]
        olat = mapping[1]
        olon = mapping[2]
        c = Circlem.circle(m, olat, olon, radius * 1000.)
        circraw = Polygon(c)
        circ = PolygonPatch(circraw, fc='none',
                            ec=color, linewidth=lw, alpha=alpha)
    else:
        circ = plt.Circle(center, radius, fill=False,
                          color=color, linewidth=lw, alpha=alpha)

    ax.add_patch(circ)

    return circ


def add_sector(ax=None, radius=None, mapping=None, color=None,
             lw=2, center=None, alpha=1.0, sector=None,
             label=None):
    
    from shapely.geometry import Polygon
    from descartes import PolygonPatch

    if mapping is not None:
        m = mapping[0]
        olat = mapping[1]
        olon = mapping[2]
        s = Circlem.sector(m, olat, olon, radius * 1000., sector)
        
        circraw = Polygon(s)
        circ = PolygonPatch(circraw, fc='none',
                            ec=color, linewidth=lw, alpha=alpha)
    else:
        circ = plt.Circle(center, radius, fill=False,
                          color=color, linewidth=lw, alpha=alpha)

    ax.add_patch(circ)

    median = np.median(sector)
    txlon,txlat=Circlem.pick_circle_point(olon, olat,
                              median, 50*1000.)

    x,y=m(txlon,txlat)
    ax.text(x,y,label,ha='center',va='center',
            color=color,size=14,weight='bold',rotation=180-median)

    return circ

def add_background_circle(ax=None, radius=None, color=None):

    circ = plt.Circle((0, 0), radius, fill=True, color=color, zorder=0)
    ax.add_patch(circ)


def add_azimuths(ax, space_deg=30, lw=2, alpha=1.0, mapping=False):


    if mapping:
        m = mapping[0]        
        olat = mapping[1]
        olon = mapping[2]
        d = 5  #deg

        lats=[olat-d, olat+d]
        lons=[olon,olon]
        x,y = m(lons,lats)
        m.plot(x, y,linewidth=lw, color='k', alpha=alpha)

        lats=[olat, olat]
        lons=[olon-d,olon+d]
        x,y = m(lons,lats)
        m.plot(x, y,linewidth=lw, color='k', alpha=alpha)

    else:
        ax.plot([0, 0], [-60, 60], linewidth=lw, color='k', alpha=alpha)
        ax.plot([-60, 60], [0,0], linewidth=lw, color='k', alpha=alpha)


def add_zisodop_arrow(ax=None, vr_array=None):
    '''
    source:
    http://stackoverflow.com/questions/7878398/
    how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array

    Not implemented. Need to resolve how coord systems are
    interpreted in map_coordinate
    '''
    from scipy.ndimage import map_coordinates

    vr_array[np.isnan(vr_array)] = 0.
    vr_array_geo = np.flipud(vr_array.T)
    x0, y0 = [207, 183]
    x1, y1 = [0, 0]
    num = 100
    y, x = np.linspace(x1, x0, num), np.linspace(y1, y0, num)
    z = map_coordinates(vr_array_geo, np.vstack((x, y)))

    ff, aa = plt.subplots()
    aa.plot(z)
    plt.show(block=False)

    ff, aa = plt.subplots()
    im = aa.imshow(np.flipud(vr_array), interpolation='none')
    plt.colorbar(im)
    aa.plot([x0, x1], [y0, y1], color='b')
    aa.plot(x0, y0, 'go')
    aa.plot(x1, y1, 'ro')
    plt.show(block=False)


def custom_cmap(cmap_set=None):

    import matplotlib.colors as mcolors

    if cmap_set == 'ppi_vr1':
        colors1b = plt.cm.Purples_r(np.linspace(0.0, 0.7, 32))
        colors1a = plt.cm.RdPu_r(np.linspace(0.1, 0.8, 32))
        colors1c = plt.cm.Blues_r(np.linspace(0.0, 0.7, 31))
        colors1d = plt.cm.Greens_r(np.linspace(0.0, 0.7, 31))
        colors2 = plt.cm.jet(np.linspace(0.6, 0.9, 126))
        white = np.array([1, 1, 1, 1])
        colors = np.vstack((colors1a, colors1b, colors1c, colors1d,
                            white, white, white, white,
                            colors2))
    elif cmap_set == 'ppi_vr2':
        colors1a = plt.cm.cubehelix(np.linspace(0.0, 0.2, 32))
        colors1b = plt.cm.cubehelix(np.linspace(0.4, 0.6, 32))
        colors1c = plt.cm.cubehelix(np.linspace(0.7, 0.8, 31))
        colors1d = plt.cm.cubehelix(np.linspace(0.9, 1.0, 31))
        colors2 = plt.cm.jet(np.linspace(0.6, 0.9, 126))
        white = np.array([1, 1, 1, 1])
        colors = np.vstack((colors1a, colors1b, colors1c, colors1d,
                            white, white, white, white,
                            colors2))
    elif cmap_set == 'rhi_vr1':
        colors1 = plt.cm.Greens_r(np.linspace(0.0, 0.7, 64))
        colors2 = plt.cm.Blues_r(np.linspace(0.0, 0.7, 64))
        colors4 = plt.cm.RdPu_r(np.linspace(0.1, 0.8, 64))
        colors3 = plt.cm.Purples_r(np.linspace(0, 0.7, 64))
        colors = np.vstack((colors1, colors2, colors3, colors4))
    elif cmap_set == 'rhi_vr2':
        colors1 = plt.cm.BrBG_r(np.linspace(0, 0.4, 64))
        colors2 = plt.cm.PiYG_r(np.linspace(0, 0.4, 64))
        colors3 = plt.cm.BrBG(np.linspace(0, 0.4, 64))
        colors4 = plt.cm.RdBu(np.linspace(0., 0.4, 64))
        colors = np.vstack((colors1, colors2, colors3, colors4))

    newcmap = mcolors.LinearSegmentedColormap.from_list(cmap_set, colors)

    return newcmap


def cart2geo(cartLine, orientation, radarLoc):

    from geographiclib.geodesic import Geodesic

    out = []
    for p in cartLine:
        if p >= 0:
            if orientation == 'WE':
                az = 90
            else:
                az = 360
        else:
            if orientation == 'WE':
                az = 270
            else:
                az = 180
        gd = Geodesic.WGS84.Direct(radarLoc[0], radarLoc[
            1], az,  np.abs(p) * 1000.)
        if orientation == 'WE':
            out.append(gd['lon2'])
        else:
            out.append(gd['lat2'])
    return out


def add_locations(ax, m, names):

    locs = {'FRS': (38.505260, -123.229607),
            'BBY': (38.3167, -123.0667),
            'CZD': (38.6181, -123.228),
            'Petaluma': (38.232, -122.636),
            'B13': (38.233505, -123.30429)}

    if names == 'all':
        names = locs.keys()

    for n in names:
        x, y = m(*locs[n][::-1])
        m.plot(x, y, marker='o', color='k')
