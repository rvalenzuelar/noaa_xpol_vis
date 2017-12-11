"""

    Make figure of random composite for paper

    Raul Valenzuela
    2017

"""


import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from xpol_tta_analysis import get_geocoords,get_colormap
import h5py
import numpy.ma as ma
import numpy as np
from cartopy.feature import NaturalEarthFeature
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from geographiclib.geodesic import Geodesic
from xpol import add_sector


''' load data '''
coast = NaturalEarthFeature(category='physical', scale='10m',
                            facecolor='none', name='coastline')

ppi_tta = xr.open_dataarray('random_ppi_freq_tta.nc')
ppi_ntta = xr.open_dataarray('random_ppi_freq_ntta.nc')

ppi_tta.name = 'tta'
ppi_ntta.name  ='ntta'

rhi_tta = xr.open_dataarray('random_rhi_freq_tta.nc')
rhi_ntta = xr.open_dataarray('random_rhi_freq_ntta.nc')

rhi_tta.name = 'tta'
rhi_ntta.name  ='ntta'


''' get radial distance'''
ppi_tta['x'] = np.linspace(-80, 80, ppi_tta.x.size)
ppi_tta['y'] = np.linspace(-80, 80, ppi_tta.y.size)
rho = np.sqrt(ppi_tta.x ** 2 + ppi_tta.y ** 2)  # [km]


''' set geo coordinates '''
lats, lons = get_geocoords(ppi_tta.y.size, ppi_tta.x.size)
ppi_tta['x'] = lons
ppi_tta['y'] = lats
ppi_ntta['x'] = lons
ppi_ntta['y'] = lats

cmap1 = get_colormap(mode='ppi', target='z')
cmap2 = get_colormap(mode='rhi', target='z')


''' figure setup '''
proj = ccrs.PlateCarree()

scale = 1.2
fig = plt.figure(figsize=(8.55,  5.27))

gs0 = gridspec.GridSpec(1, 2, wspace=0.1)

height_ratios = [2.5, 1]

gs00 = gssp(2, 1,
            subplot_spec=gs0[0],
            height_ratios=height_ratios,
            hspace=0.1)

gs01 = gssp(2, 1,
            subplot_spec=gs0[1],
            height_ratios=height_ratios,
            hspace=0.1)

ax0 = plt.subplot(gs00[0], gid='(a)', projection=proj)
ax1 = plt.subplot(gs01[0], gid='(b)', projection=proj)
ax2 = plt.subplot(gs00[1], gid='(c)')
ax3 = plt.subplot(gs01[1], gid='(d)')


''' make PPI plots '''
for ax, ppi, pname in zip([ax0, ax1], [ppi_tta, ppi_ntta],
                   ['(a)', '(b)']):

    ''' plot '''
    p = ppi.plot.contourf(ax=ax, cmap=cmap1,
                      add_colorbar=False,
                      levels=range(20, 110, 10))

    ''' add terrain '''
    f = h5py.File('obs_domain_elevation.h5', 'r')
    dtm = f['/dtm'].value
    f.close()
    dtmm = ma.masked_where(dtm <= -15, dtm)
    X, Y = np.meshgrid(lons, lats)
    hdtm = ax.pcolormesh(X, Y, dtmm, vmin=0, vmax=1000,
                        cmap='gray_r',
                         )

    ''' add coastline '''
    ax.add_feature(coast, edgecolor='black')

    ''' aspect ratio '''
    ax.set_aspect('auto')

    ''' add radials in lat/lon  '''
    origin = (38.505260, -123.229607)  # [FRS]

    for loc in range(10, 70, 10):
        cn = rho.plot.contour(ax=ax, levels=[loc])
        first = True
        for line in cn.allsegs[0]:
            def fun(a):
                az = np.arctan2(a[0], a[1]) * 180 / np.pi  # [deg]
                endp = Geodesic.WGS84.Direct(origin[0], origin[1],
                                             az, loc * 1000)
                return endp['lon2'], endp['lat2']
            line2 = np.apply_along_axis(fun, 1, line)
            ax.plot(line2[:, 0], line2[:, 1], color='k', alpha=0.3)

            if first:
                c2 = Geodesic.WGS84.Direct(origin[0], origin[1],
                                             240, loc * 1000)
                x = c2['lon2']
                y = c2['lat2']
                if loc == 60:
                    ax.text(x, y, '{}\nkm'.format(loc),
                            ha='center', va='top')
                else:
                    ax.text(x, y, '{}'.format(loc), ha='center')
                first = False

    ''' zonal limits '''
    ax.set_xlim([-123.9, -122.72])
    ax.set_ylim([38, 38.8])

    ''' add meridional/zonal lines '''
    ax.plot([-123.9, -122.72], [origin[0], origin[0]],
            color='k',alpha=0.3)
    ax.plot([origin[1], origin[1]], [37, 39],
            color='k', alpha=0.3)

    ''' add colorbar '''
    # copy/paste colorbar from composite figure
    # using image editor (weird results produced
    # here using the code)

    ''' add panel name '''
    ax.text(0.9,0.9,pname,transform=ax.transAxes,
            fontsize=12,weight='bold', color='w')

    ''' add sector '''
    sector = range(135, 180)
    fun = lambda x: Geodesic.WGS84.Direct(origin[0], origin[1], x,
                                          60 * 1000)
    res = map(fun, sector)
    latsec = [r['lat2'] for r in res]
    lonsec = [r['lon2'] for r in res]
    ax.plot(lonsec, latsec, color='g', lw=2)
    ax.plot([origin[1], lonsec[0]], [origin[0], latsec[0]],
            color='g', lw=2)
    ax.plot([origin[1], lonsec[-1]], [origin[0], latsec[-1]],
            color='g', lw=2)
    txt = 'Beam\nBlocked'
    c2 = Geodesic.WGS84.Direct(origin[0], origin[1],
                               155, 50 * 1000)
    x, y = c2['lon2'],c2['lat2']
    ax.text(x, y, txt, ha='center', va='center',
            color='g', size=14, weight='bold', rotation=25)

''' make RHI plots '''
for ax, rhi, pname in zip([ax2, ax3], [rhi_tta, rhi_ntta],
                   ['(c)', '(d)']):
    rhi.plot.contourf(ax=ax, cmap=cmap2,
                      add_colorbar=False,
                      levels=range(20,110,10))

    prof = np.load('prof2.npy')

    x = np.arange(0.5, 60.5, 0.5)
    y = prof / 1000.
    ax.fill_between(x, 0, y, facecolor='gray',
                    edgecolor='k')

    ax.set_xlim([-40,40])
    ax.set_ylim([0, 5])
    ax.set_xticks(range(-30, 40, 10))
    ax.set_yticks([1, 2, 3, 4])
    if rhi.name == 'ntta':
        ax.set_yticklabels('')
        ax.set_ylabel('')
    else:
        ax.set_ylabel('Altitude [km] MSL')
    ax.tick_params(axis='both',direction='in')
    ax.set_xlabel('Distance from the radar [km]')

    ''' add panel name '''
    ax.text(0.9,0.8,pname,transform=ax.transAxes,
            fontsize=12,weight='bold', color='k')