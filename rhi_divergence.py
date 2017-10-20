
import xarray as xr
import xpol_tta_analysis as xta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage.filters import gaussian_filter

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )
# data = xta.process(case=[12], params=params)

y = data.get_axis('x', 'rhi')
z = data.get_axis('z', 'rhi')

ntta = data.rhi_ntta['VR']
tta = data.rhi_tta['VR']

first = True

for grp in [tta, ntta]:
    index = grp.sort_index().index
    values = grp.sort_index().values

    ''' concatenate '''
    concat = values[0]
    concat = concat[:, :, None]
    for v in values[1:]:
        concat = np.concatenate([concat, v[:, :, None]],
                                axis=2)

    ''' apply smooth '''
    # concat = gaussian_filter(concat, 0.6, order=0)

    if first:
        dat = xr.DataArray(data=concat,
                          coords=[z, y, index],
                          dims=['z','y','time'])
        first = False
    else:
        dan = xr.DataArray(data=concat,
                          coords=[z, y, index],
                          dims=['z','y','time'])

da = dan


''' Smooth by averaging along y-axis;
    every 7 elements is about 1 km.
'''
for da in [dat, dan]:
    da_resamp = da.groupby_bins('y', dat.y[::7]).mean(dim='y')
    da_resamp.coords['y_bins'] = da.y[:-3:7].values
    da_resamp = da_resamp.rename({'y_bins':'y'})
    da = da_resamp

    ''' compute divergence '''
    dy = da.y.diff(dim='y').values.mean() * 1000.  # [m]
    dv = da.diff(dim='y')
    dv_dy = dv/dy

    ''' box mean in time '''
    dv_dy_sl = dv_dy.sel(z=slice(0, .6), y=slice(-30, -5))
    dv_dy_box = dv_dy_sl.sum(dim=['y', 'z'])
    print dv_dy_box.median()

''' make dv_dy animation '''
mdata = np.ma.masked_invalid(dv_dy.values)
fig, ax = plt.subplots()
cax = ax.pcolormesh(dv_dy.y, z, mdata[:-1, :-1, 0],
                    vmin=-0.005, vmax=0.005,
                    cmap='RdBu')
fig.colorbar(cax)
fmt = '%Y-%m-%d %H:%M UTC'
ax.set_title(dv_dy.time[0].values)

def animate(i):
    print i
    cax.set_array(mdata[:-1, :-1, i].flatten())
    ax.set_title(dv_dy.time[i].values)

anim = animation.FuncAnimation(fig, animate, frames=34)
plt.show()


''' make doppler animation '''
# mdata = np.ma.masked_invalid(da.values)
# fig, ax = plt.subplots()
# cax = ax.pcolormesh(y, z, mdata[:-1, :-1, 0],
#                     vmin=0, vmax=20)
# fig.colorbar(cax)
# fmt = '%Y-%m-%d %H:%M UTC'
# ax.set_title(da.time[0].values)
#
# def animate(i):
#     print i
#     cax.set_array(mdata[:-1, :-1, i].flatten())
#     ax.set_title(da.time[i].values)
#
# anim = animation.FuncAnimation(fig, animate, frames=71)
# plt.show()


