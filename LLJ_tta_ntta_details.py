"""

    Raul Valenzuela
    2017

"""
import xpol_tta_analysis as xta
import numpy as np
import matplotlib.pyplot as plt
import xpol
from datetime import datetime
from rv_utilities import add_colorbar

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )

casein = 13

if casein != case:
    xp = xta.process(case=[casein], params=params)
    case = casein

# xp = xta.process(case=[casein], params=params)

if casein == 13:
    sets = dict(ntta={'ini': datetime(2004, 2, 16, 18, 0),
                      'end': datetime(2004, 2, 16, 21, 0)},
                tta={'ini': datetime(2004, 2, 16, 11, 0),
                     'end': datetime(2004, 2, 16, 14, 0)})
elif casein == 9:
    sets = dict(ntta={'ini': datetime(2003, 1, 22, 23, 0),
                      'end': datetime(2003, 1, 23, 2, 0)},
                tta={'ini': datetime(2003, 1, 21, 12, 0),
                     'end': datetime(2003, 1, 21, 15, 0)})


mode = 'tta'
serie_llj = dict()
serie = dict()
serie['tta'] = xp.rhi_tta['VR']
serie['ntta'] = xp.rhi_ntta['VR']
arr_mean = dict()

for mode in ['tta', 'ntta']:
    query = (serie[mode].index > sets[mode]['ini']) & \
            (serie[mode].index < sets[mode]['end'])
    serie_llj[mode] = serie[mode].loc[query]

    arr = np.expand_dims(serie_llj[mode].iloc[0], axis=2)
    n = len(serie_llj[mode])
    for i in range(1, n):
        expand = np.expand_dims(serie_llj[mode].iloc[i], axis=2)
        arr = np.concatenate((arr, expand), axis=2)
    arr_mean[mode] = np.nanmean(arr, axis=2)

x = xp.get_axis('x', 'rhi')
z = xp.get_axis('z', 'rhi')
X, Z = np.meshgrid(x, z)

cmap = xpol.custom_cmap('rhi_vr1')
cvalues = np.arange(0, 32, 4)
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
cf = ax[0].contourf(X, Z, arr_mean['tta'], cvalues, cmap=cmap)
cf = ax[1].contourf(X, Z, arr_mean['ntta'], cvalues, cmap=cmap)
ax[0].set_ylim([0, 3])
ax[1].set_xlim([-30, 15])

# hgt = 0.5
# ax[0].axhline(y=hgt, linestyle='--', color='k')
# ax[1].axhline(y=hgt, linestyle='--', color='k')

add_colorbar(ax[0], cf, label='[m/s]', labelpad=20)
add_colorbar(ax[1], cf, invisible=True)

prof = np.load('prof2.npy')
xt = np.arange(0.5, 60.5, 0.5)
yt = prof / 1000.
ax[0].fill_between(xt, 0, yt, facecolor='gray')
ax[1].fill_between(xt, 0, yt, facecolor='gray')

ax[0].set_title('TTA')
ax[1].set_title('NO-TTA')

ax[1].set_xlabel('Distance from the radar [km]')
ax[1].set_ylabel('Altitude [km] MSL')
ax[0].set_ylabel('Altitude [km] MSL')

t0 = sets['tta']['ini'].strftime('%d-%b-%Y %H%M UTC')
t1 = sets['tta']['end'].strftime('%H%M UTC')
txt1 = '{} to {}'.format(t0, t1)
t0 = sets['ntta']['ini'].strftime('%d-%b-%Y %H%M UTC')
t1 = sets['ntta']['end'].strftime('%H%M UTC')
txt2 = '{} to {}'.format(t0, t1)

tt1 = ax[0].text(-28, 2.5, txt1)
tt2 = ax[1].text(-28, 2.5, txt2)

tt1.set_bbox(dict(color='w', edgecolor='none'))
tt2.set_bbox(dict(color='w', edgecolor='none'))

''' add RHI arrow '''
if case == 13:
    arrows = (
              {'c0': (-10, 0.35), 'c1': (2.5, 0.5)},
              {'c0': (-10, 0.1), 'c1': (2.5, 0.35)},
             )
    arwprops = (
                {'angleA':10, 'angleB':-180,
                 'armA':200, 'armB':100, 'rad':60},
                {'angleA':5, 'angleB':-180,
                 'armA':200, 'armB':100, 'rad':60}
                )
elif case == 9:
    arrows = (
              {'c0': (-10, 0.2), 'c1': (2.5, 0.35)},
              {'c0': (-10, 0.1), 'c1': (2.5, 0.3)},
             )
    arwprops = (
                {'angleA':10, 'angleB':-180,
                 'armA':100, 'armB':100, 'rad':80},
                {'angleA':0, 'angleB':-180,
                 'armA':300, 'armB':100, 'rad':60}
                )

axes = (ax[0], ax[1])
scale = 2.1  # use for output figure
# scale = 1.0 # use for iPython figure
for arr, ax, prop in zip(arrows, axes, arwprops):
    c0 = tuple(v * scale for v in arr['c0'])
    c1 = tuple(v * scale for v in arr['c1'])
    txtprop = ''.join(',{}={}'.format(key, val)
                      for key, val in prop.items())
    ax.annotate("",
                xy=c1,
                xytext=c0,
                xycoords='data',
                textcoords='data',
                arrowprops=dict(
                    arrowstyle='-|>,head_length=0.7,head_width=0.2',
                    linewidth=3,
                    # shrinkA=5,
                    # shrinkB=5,
                    fc="w", ec=(1.000,0.167 ,0.000),
                    # connectionstyle="arc3,rad={}".format(-0.05),
                    connectionstyle="arc"+txtprop,
                    )
                )
plt.tight_layout()
fig.set_size_inches(8.2,8.7)