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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.basemap import Basemap


def plot(array, ax=None, show=True, name=None, smode=None,
         date=None, elev=None, title=None, add_azline=None,
         colorbar=True, extent=None, second_date=None, case=None,
         add_yticklabs=False, vmax=None, textbox=None):

    if ax is None:
        fig, ax = plt.subplots()

    if extent is None:
        if smode == 'rhi':
            if case in [11, 13, 14]:
                extent = [-30, 30, 0.05, 11]
            elif case == 12:
                extent = [-30, 20, 0.05, 11]
            else:
                extent = [-40, 20, 0.05, 11]
        elif smode == 'ppi':
            extent = [-58, 45, -58, 33]

    if name == 'ZA':
        vmin, vmax = [-10, 45]
        cmap = 'jet'
        cbar_ticks = range(vmin, vmax + 5, 5)
    elif name == 'VR':
        vmin, vmax = [-20, 20]
        cmap = custom_cmap('ppi_vr1')
        cbar_ticks = range(vmin, vmax + 5, 5)
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

    if elev is not None and name == 'VR':
        array = array / np.cos(np.radians(elev))
        idx = np.where((elev >= 65) & (elev <= 115))
        array[idx] = np.nan
        vmin, vmax = [0, 40]
        cmap = custom_cmap('rhi_vr1')
        cbar_ticks = range(vmin, vmax + 10, 10)

    im = ax.imshow(array, interpolation='none', origin='lower',
                   vmin=vmin, vmax=vmax,
                   cmap=cmap,
                   extent=extent,
                   aspect='auto')

    if smode == 'ppi':
        add_rings(ax, space_km=10, color='k')
        add_azimuths(ax, space_deg=30)
        add_ring(ax, radius=57.8, color=(0.85, 0.85, 0.85), lw=10)
        add_circle(ax, radius=60, color=(0.85, 0.85, 0.85))
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
    elif smode == 'rhi':
        # ax.set_xlim([-40, 20])
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

    plt.draw()

    if date:
        plt.text(0.0, 1.01, date, ha='left', transform=ax.transAxes)
    if second_date:
        plt.text(1.0, 1.01, second_date, ha='right', transform=ax.transAxes)
    if title:
        plt.text(0.4, 1.01, title, ha='left', transform=ax.transAxes)
    if colorbar:
        add_colorbar(ax, im, cbar_ticks)
    if textbox:
        props = dict(facecolor='white')
        plt.text(0.6, 0.9, textbox, transform=ax.transAxes, bbox=props)

    # if show:
        # plt.show(block=False)


def plotm(array, ax=None, show=True, name=None):

    if not ax:
        fig, ax = plt.subplots()

    if name == 'ZA':
        vmin, vmax = [-10, 45]
        cmap = 'jet'
        cbar_ticks = range(vmin, vmax + 5, 5)
    elif name == 'VR':
        vmin, vmax = [-20, 20]
        cmap = custom_cmap()
        cbar_ticks = range(vmin, vmax + 5, 5)

    radarx = np.arange(-58, 45.5, 0.5)
    radary = np.arange(-58, 33.5, 0.5)
    radarloc = (38.505260, -123.229607)  # at FRS
    radarlon = cart2geo(radarx, 'WE', radarloc)
    radarlat = cart2geo(radary, 'NS', radarloc)

    m = Basemap(projection='merc',
                llcrnrlat=min(radarlat),
                urcrnrlat=max(radarlat),
                llcrnrlon=min(radarlon),
                urcrnrlon=max(radarlon),
                lat_0=radarloc[0],
                lon_0=radarloc[1],
                resolution='i')

    m.drawcoastlines()
    im = m.imshow(array, interpolation='none', origin='lower',
                  vmin=vmin, vmax=vmax, cmap=cmap)
    add_rings(ax, space_km=10, color='k', mapping=[
              m, radarloc[0], radarloc[1]])
    add_locations(ax, m, 'all')

    add_colorbar(ax, im, cbar_ticks)

    plt.show(block=False)


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


def get_data(case, scanmode, angle, homedir=None):

    if scanmode == 'PPI':
        basestr = homedir + '/XPOL/netcdf/c{0}/PPI/elev{1}/'
        angle = angle * 10
    elif scanmode == 'RHI':
        basestr = homedir + '/XPOL/netcdf/c{0}/RHI/az{1}/'

    basedir = basestr.format(str(case).zfill(2), str(int(angle)).zfill(3))
    cdf_files = glob(basedir + '*.cdf')
    cdf_files.sort()

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
            ' invert sign to represent northward wind from southward azimuths'
            if case == 12:
                if angle == 147:
                    VR = VR * -1
            else:
                VR = VR * -1

        ' add second complemental azimuth in RHI '
        if case in [11, 13, 14] and scanmode == 'RHI':
            basedir = basestr.format(str(case).zfill(2), '360')
            cdf_files2 = glob(basedir + '*.cdf')
            cdf_files2.sort()
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

    if scanmode == 'PPI':
        df.index.name = scanmode + ' ' + str(angle / 10.)
    else:
        if angle <= 180 and angle > 90:
            df.index.name = scanmode + ' ' + \
                str(angle) + '-' + str(angle + 180)
        elif angle <= 90 and angle > 0:
            df.index.name = scanmode + ' ' + \
                str(angle + 180) + '-' + str(angle)
        else:
            df.index.name = scanmode + ' ' + \
                str(angle) + '-' + str(angle - 180)

    if len(time_index2) > 0:
        df['time_az2'] = time_index2

    return df


def parse_rawdate(start_date=None, start_time=None, datestring=None):

    if start_date is not None and start_time is not None:
        date = ''.join(start_date).replace('/01', '/20')
        time = ''.join(start_time)
        raw_date = date + ' ' + time
        return raw_date


def get_mean(arrays, minutes=None, name=None, good_thres=1000):

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
        good = 0
        for n in range(0, narrays):
            if n == 0:
                a = arrays.iloc[[n]].values[0]  # first value
                # number of non-nan points
                valid = a.size - np.sum(np.isnan(a))
                if valid > good_thres:
                    good += 1
            else:
                rr = arrays.iloc[[n]].values[0]
                # number of non-nan points
                valid = rr.size - np.sum(np.isnan(rr))
                if valid > good_thres:
                    good += 1
                a = np.dstack((a, rr))

        if name == 'ZA':
            a = toLinear(a)

        mean = np.nanmean(a, axis=2)

        if name == 'ZA':
            mean = toLog10(mean)

        return mean, good


def get_dbz_freq(arrays, thres=None):

    narrays = arrays.shape[0]
    first = True
    for n in range(0, narrays):
        a = arrays.iloc[[n]].values[0]
        cond = (a >= thres) + 0  # convert to Int boolean
        if first:
            COND = np.zeros(a.shape)
            COND = np.dstack((COND, cond))
            first = False
        else:
            COND = np.dstack((COND, cond))

    freq = (np.sum(COND, axis=2) / narrays) * 100.
    mean, _ = get_mean(arrays, name='ZA')
    freq[np.isnan(mean)] = np.nan

    return freq


def get_dbz_threshold(arrays):

    import matplotlib.mlab as mlab

    narrays = arrays.shape[0]
    first = True
    for n in range(0, narrays):
        a = arrays.iloc[[n]].values[0]
        if first:
            A = a.copy()
            first = False
        else:
            A = np.dstack((A, a))

    mu = np.nanmean(A)
    sigma = np.nanstd(A)

    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(A.flatten(), normed=1, bins=range(-10, 60, 5))
    x = range(-10, 60)
    y = mlab.normpdf(x, mu, sigma)
    plt.plot(x, y, 'r--', linewidth=2)
    ax.text(range(-10, 60))
    plt.show(block=False)

    print(hist)
    print(bins)
    print(mu)
    print(sigma)


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


def add_rings(ax, space_km=10, color='k', mapping=False):

    from shapely.geometry import Polygon
    from descartes import PolygonPatch

    # textdirection=225
    textdirection = -5

    for r in range(0, 60 + space_km, space_km):

        ring = add_ring(ax=ax, radius=r, mapping=mapping, color=color)
        vert = ring.get_path().vertices
        x, y = vert[textdirection]
        ax.text(x * r, y * r, str(r), ha='center', va='center',
                bbox=dict(fc='none', ec='none', pad=2.),
                clip_on=True)


def add_ring(ax=None, radius=None, mapping=False, color=None, lw=1):

    if mapping:
        m = mapping[0]
        olat = mapping[1]
        olon = mapping[2]
        c = Circlem.circle(m, olat, olon, radius * 1000.)
        circraw = Polygon(c, linestyle=':')
        circ = PolygonPatch(circraw, fc='none', ec='b')
        # foo=1
    else:
        circ = plt.Circle((0, 0), radius, fill=False,
                          color=color, linewidth=lw)
        # foo=radius

    ax.add_patch(circ)

    return circ


def add_circle(ax=None, radius=None, color=None):

    circ = plt.Circle((0, 0), radius, fill=True, color=color, zorder=0)
    ax.add_patch(circ)


def add_azimuths(ax, space_deg=30):

    xaz = ax.get_xlim()
    yaz = ax.get_ylim()
    ax.plot(xaz, [0, 0], color='k')
    ax.plot([0, 0], yaz, color='k')


def add_colorbar(ax, im, cbar_ticks):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=cbar_ticks)
    return cbar


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

    newcmap = mcolors.LinearSegmentedColormap.from_list('newCmap', colors)

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
