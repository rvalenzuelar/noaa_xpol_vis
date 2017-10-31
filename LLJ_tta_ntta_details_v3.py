



import xpol_tta_analysis as xta
import numpy as np
import matplotlib.pyplot as plt
import xpol
from datetime import datetime
from rv_utilities import add_colorbar
import mpl_toolkits.axisartist as AA
from make_xarray import make_xarray


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


y = 2003
slices = [slice(datetime(y,1,12,19,0), datetime(y,1,12,22,0)),
           slice(datetime(y,1,13,0,0), datetime(y,1,13,9,0)),
           slice(datetime(y,1,13,10,0), datetime(y,1,13,11,0)),
           slice(datetime(y,1,13,12,0),datetime(y,1,13,13,0))]
ds08 = make_xarray(xp08, slices=slices)

y = 2003
slices = [slice(datetime(y,1,22,20,0), datetime(y,1,23,2,0))]
ds09 = make_xarray(xp09, slices=slices)

y = 2004
slices = [slice(datetime(y,1,9,13,0), datetime(y,1,9,16,0)),
          slice(datetime(y,1,9,19,0), datetime(y,1,9,21,0)),]
ds11 = make_xarray(xp11, slices=slices)
''' rhi sweep cutted at dropped time '''
ds11 = ds11.drop(datetime(2004,1,9,19,13), dim='time')



y = 2004
slices = [slice(datetime(y,2,2,2,0), datetime(y,2,2,5,0)),
          slice(datetime(y,2,2,6,0), datetime(y,2,2,8,0)),
          slice(datetime(y,2,2,12,0), datetime(y,2,2,15,0))]
ds12 = make_xarray(xp12, slices=slices)

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

scale = 1.6
fig, axs = plt.subplots(5, 2, figsize=(6*scale,  5*scale),
                       sharex=True, sharey=True
                       )
axs = axs.flatten()
ds_group = [ds08, ds09, ds11, ds12, ds13]

''' Doppler velocity '''
cvalues = np.arange(0, 32, 2)
cmap = xpol.custom_cmap('rhi_vr1')
counts = list()
for ds, ax, in zip(ds_group, axs[::2]):

    X, Z = np.meshgrid(ds.x.values, ds.z.values)
    cf_vr = ax.contourf(X, Z, ds.mean(dim='time')['VR'],
                     cvalues, cmap=cmap)
    counts.append(ds.time.size)

''' Reflectivity '''
cvalues = np.arange(20, 110, 10)
dbz = list()
for ds, ax in zip(ds_group, axs[1::2]):

    dbz_thres = 18
    # dbz_thres = ds['ZA'].median().values
    ds_mask = ds['ZA'].where(ds['ZA'] >= dbz_thres)
    ds_bool = ds_mask.notnull()
    ds_sum = ds_bool.sum(dim='time')
    ds_freq = (ds_sum/float(ds_bool.time.size))*100

    X, Z = np.meshgrid(ds.x.values, ds.z.values)
    cf_za = ax.contourf(X, Z, ds_freq, cvalues,
                     cmap='inferno')
    dbz.append(dbz_thres)


''' terrain profile '''
prof = np.load('prof2.npy')
xt = np.arange(0.5, 60.5, 0.5)
yt = prof / 1000.


''' adjust axes'''
dts = ['', '12-14Jan03',
       '', '21-23Jan03',
       '', '09Jan04',
       '', '02Feb04',
       '', '16-18Feb04']
fignames = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
            '(g)', '(h)', '(i)', '(j)']
for i in range(1, 10, 2): counts.insert(i,0)
for i in range(0, 10, 2): dbz.insert(i,0)

for ax, dt, fn, cnt, z in zip(axs, dts, fignames, counts, dbz):

    ax.set_ylim([0, 5])
    ax.set_xlim([-40, 40])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks(range(-30, 40, 10))

    ''' terrain '''
    ax.fill_between(xt, 0, yt, facecolor='gray', edgecolor='k')

    ''' figure name '''
    ax.text(0.98, 0.8, fn, size=12,
            transform=ax.transAxes, ha='right',
            color=(0, 0, 0), weight='bold')

    ''' nobs '''
    if fn in ['(a)', '(c)', '(e)', '(g)', '(i)']:
        txt1 = 'N={}'.format(cnt)
        ax.text(0.98, 0.13, txt1, size=12,
                transform=ax.transAxes, ha='right',
                color=(0, 0, 0), weight='bold')

    ''' dbz thres '''
    if fn in ['(b)', '(d)', '(f)', '(h)', '(j)']:
        txt1 = '{:1.1f} dBZ'.format(float(z))
        ax.text(0.98, 0.13, txt1, size=12,
                transform=ax.transAxes, ha='right',
                color=(0, 0, 0), weight='bold')


    ''' add dates '''
    if fn in ['(b)', '(d)', '(f)', '(h)', '(j)']:
        ax.text(1, 0.5, dt, fontsize=12,
                 transform=ax.transAxes,
                va='center',
                rotation=-90)

    if fn == '(e)':
        ax.set_ylabel('Altitude [km] MSL', fontsize=12)

    if fn == '(a)':
        tit = 'LLJ during NO-TTA conditions'
        ax.set_title(' '*60+tit,
                     fontsize=15)

    if fn in ['(i)', '(j)']:
        ax.set_xlabel('Distance from radar [km]', fontsize=12)

    if fn == '(i)':
        cbax = AA.Axes(fig, [0.125, 0.1, 0.37, 0.75])
        cb = add_colorbar(cbax, cf_vr, label='[m/s]', labelpad=20,
                            loc='bottom')


    if fn == '(j)':
        cbax = AA.Axes(fig, [0.52, 0.1, 0.37, 0.75])
        cb = add_colorbar(cbax, cf_za, label='[%]', labelpad=20,
                            loc='bottom')

    ax.xaxis.set_tick_params(labelsize=12,
                             direction='in')
    ax.yaxis.set_tick_params(labelsize=12,
                             direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

plt.subplots_adjust(top=0.95, bottom=0.2, hspace=0, wspace=0.05)
