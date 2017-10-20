



import xpol_tta_analysis as xta
import numpy as np
import matplotlib.pyplot as plt
import xpol
from datetime import datetime
from rv_utilities import add_colorbar
import mpl_toolkits.axisartist as AA
import xarray as xr

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )

try:
    xp08
except NameError:
    xp08 = xta.process(case=[8], params=params)

try:
    xp09
except NameError:
    xp09 = xta.process(case=[9], params=params)

try:
    xp11
except NameError:
    xp11 = xta.process(case=[11], params=params)

try:
    xp12
except NameError:
    xp12 = xta.process(case=[12], params=params)

try:
    xp13
except NameError:
    xp13 = xta.process(case=[13], params=params)


def make_xarray(xp, slices):

    xp.rhi_ntta['VR'].sort_index(inplace=True)
    xp.rhi_ntta['ZA'].sort_index(inplace=True)

    add_dim_vr = map(lambda x:np.expand_dims(x,axis=2),
                  xp.rhi_ntta['VR'].values)

    concat_vr = np.concatenate(add_dim_vr, axis=2)

    add_dim_za = map(lambda x: np.expand_dims(x, axis=2),
                     xp.rhi_ntta['ZA'].values)

    concat_za = np.concatenate(add_dim_za, axis=2)

    time = [i.to_pydatetime() for i in xp.rhi_ntta['VR'].index]
    x = xp.get_axis('x','rhi')
    z = xp.get_axis('z','rhi')

    ds = xr.Dataset(data_vars={'VR':(['z','x','time'],concat_vr),
                               'ZA':(['z','x','time'],concat_za)},
                      coords={'z':z,'x':x,'time':time})

    first = True
    for sl in slices:
        if first:
            ds_sls = ds.sel(time=sl)
            first = False
        else:
            ds_sls = xr.concat([ds_sls, ds.sel(time=sl)],
                               dim='time')

    return ds_sls

y = 2003
slices = [slice(datetime(y,1,12,19,0), datetime(y,1,12,22,0)),
           slice(datetime(y,1,13,0,0), datetime(y,1,13,9,0)),
           slice(datetime(y,1,13,10,0), datetime(y,1,13,11,0)),
           slice(datetime(y,1,13,12,0),datetime(y,1,13,13,0))]
ds08 = make_xarray(xp08, slices)

y = 2003
slices = [slice(datetime(y,1,22,20,0), datetime(y,1,23,2,0))]
ds09 = make_xarray(xp09, slices)

y = 2004
slices = [slice(datetime(y,1,9,13,0), datetime(y,1,9,16,0)),
          slice(datetime(y,1,9,19,0), datetime(y,1,9,21,0)),]
ds11 = make_xarray(xp11, slices)

y = 2004
slices = [slice(datetime(y,2,2,2,0), datetime(y,2,2,5,0)),
          slice(datetime(y,2,2,6,0), datetime(y,2,2,8,0)),
          slice(datetime(y,2,2,12,0), datetime(y,2,2,15,0))]
ds12 = make_xarray(xp12, slices)

y = 2004
slices = [slice(datetime(y,2,16,18,0), datetime(y,2,16,21,0)),
          slice(datetime(y,2,16,22,0), datetime(y,2,17,4,0)),
          slice(datetime(y,2,17,5,0), datetime(y,2,17,6,0)),
          slice(datetime(y,2,17,7,0),datetime(y,2,17,9,0)),
          slice(datetime(y,2,17,10,0), datetime(y,2,17,11,0)),
          slice(datetime(y,2,17,12,0), datetime(y,2,17,15,0)),
          slice(datetime(y,2,17,16,0), datetime(y,2,17,18,0)),
          slice(datetime(y,2,17,19,0),datetime(y,2,18,0,0)),
          slice(datetime(y,2,18,2,0),datetime(y,2,18,3,0)),
          slice(datetime(y,2,18,5,0),datetime(y,2,18,6,0))]
ds13 = make_xarray(xp13, slices)

scale = 1.4
fig, axs = plt.subplots(5, 2, figsize=(4.6*scale,  4.3*scale),
                       sharex=True, sharey=True
                       )
axs = axs.flatten()
dts = ['12-14Jan03', '21-23Jan03','09Jan04','02Feb04', '16-18Feb04']
fignames = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
            '(g)', '(h)', '(i)', '(j)']
ds_group = [ds08, ds09, ds11, ds12, ds13]

''' Doppler velocity '''
cvalues = np.arange(0, 32, 2)
cmap = xpol.custom_cmap('rhi_vr1')
for ds, ax, dt, figname in zip(ds_group,
                               axs[::2], dts,
                               fignames):

    X, Z = np.meshgrid(ds.x.values, ds.z.values)
    cf = ax.contourf(X, Z, ds.median(dim='time')['VR'], cvalues,
                     cmap=cmap)

    ax.set_ylim([0, 5])
    ax.set_xlim([-40, 40])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks(range(-30, 40, 10))

''' Reflectivity '''
cvalues = np.arange(20, 110, 10)
for ds, ax, dt, figname in zip(ds_group,
                               axs[1::2], dts,
                               fignames):

    ds_mask = ds['ZA'].where(ds['ZA'] >= 10)
    ds_bool = ds_mask.notnull()
    ds_sum = ds_bool.sum(dim='time')
    ds_freq = (ds_sum/float(ds_bool.time.size))*100

    X, Z = np.meshgrid(ds.x.values, ds.z.values)
    cf = ax.contourf(X, Z, ds_freq, cvalues,
                     cmap='inferno')

    ax.set_ylim([0, 5])
    ax.set_xlim([-40, 40])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks(range(-30, 40, 10))