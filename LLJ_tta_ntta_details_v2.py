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
import mpl_toolkits.axisartist as AA

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
    xp13
except NameError:
    xp13 = xta.process(case=[13], params=params)

xp = dict()
xp[8] = xp08
xp[9] = xp09
xp[13] = xp13

sets = dict()
sets[8] = dict(ini=[datetime(2003, 1, 12, 19, 0),
                    datetime(2003, 1, 13, 0, 0),
                    datetime(2003, 1, 13, 10, 0),
                    datetime(2003, 1, 13, 12, 0),
                    ],
                end=[datetime(2003, 1, 12, 22, 0),
                     datetime(2003, 1, 13, 9, 0),
                     datetime(2003, 1, 13, 11, 0),
                     datetime(2003, 1, 13, 13, 0),
                    ]
                )

sets[9] = dict(ini=[datetime(2003, 1, 22, 20, 0)],
                end=[datetime(2003, 1, 23, 2, 0)],
                )

sets[13] = dict(ini=[datetime(2004, 2, 16, 18, 0),
                     datetime(2004, 2, 16, 22, 0),
                     datetime(2004, 2, 17, 5, 0),
                     datetime(2004, 2, 17, 7, 0),
                     datetime(2004, 2, 17, 10, 0),
                     datetime(2004, 2, 17, 12, 0),
                     datetime(2004, 2, 17, 16, 0),
                     datetime(2004, 2, 17, 19, 0),
                     datetime(2004, 2, 18, 2, 0),
                     datetime(2004, 2, 18, 5, 0),
                    ],
                end=[datetime(2004, 2, 16, 21, 0),
                     datetime(2004, 2, 17, 4, 0),
                     datetime(2004, 2, 17, 6, 0),
                     datetime(2004, 2, 17, 9, 0),
                     datetime(2004, 2, 17, 11, 0),
                     datetime(2004, 2, 17, 15, 0),
                     datetime(2004, 2, 17, 18, 0),
                     datetime(2004, 2, 18, 0, 0),
                     datetime(2004, 2, 18, 3, 0),
                     datetime(2004, 2, 18, 6, 0),
                     ],
                )


serie_llj = dict()
serie = dict()
serie[8] = xp08.rhi_ntta['VR']
serie[9] = xp09.rhi_ntta['VR']
serie[13] = xp13.rhi_ntta['VR']

cmap = xpol.custom_cmap('rhi_vr1')
cvalues = np.arange(0, 32, 2)
prof = np.load('prof2.npy')
xt = np.arange(0.5, 60.5, 0.5)
yt = prof / 1000.
scale = 1.4
fig, axs = plt.subplots(3, 1, figsize=(4.6*scale,  4.3*scale),
                       sharex=True, sharey=True
                       )
dts = ['12-14Jan03', '21-23Jan03', '16-18Feb04']
fignames = ['(a)', '(b)', '(c)']
for c, ax, dt, figname in zip([8, 9, 13], axs, dts, fignames):

    query = list()
    for i in range(len(sets[c]['ini'])):
        query.append((serie[c].index > sets[c]['ini'][i]) &
                     (serie[c].index < sets[c]['end'][i]))
    ''' union operation '''
    query = sum(query).astype(bool)

    serie_llj[c] = serie[c].loc[query]

    arr = np.expand_dims(serie_llj[c].iloc[0], axis=2)
    n = len(serie_llj[c])
    for i in range(1, n):
        expand = np.expand_dims(serie_llj[c].iloc[i], axis=2)
        arr = np.concatenate((arr, expand), axis=2)
    _,_,cnt = arr.shape
    arr_mean = np.nanmean(arr, axis=2)

    x = xp[c].get_axis('x', 'rhi')
    z = xp[c].get_axis('z', 'rhi')
    X, Z = np.meshgrid(x, z)

    cf = ax.contourf(X, Z, arr_mean, cvalues, cmap=cmap)
    ax.set_ylim([0, 5])
    ax.set_xlim([-40, 40])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks(range(-30, 40, 10))
    txt1 = 'N={}'.format(cnt)
    ax.text(0.98, 0.13, txt1, size=12,
            transform=ax.transAxes, ha='right',
            color=(0, 0, 0), weight='bold')
    ax.text(0.98, 0.8, figname, size=12,
            transform=ax.transAxes, ha='right',
            color=(0, 0, 0), weight='bold')

    if c == 13:
        ax.set_xlabel('Distance from radar [km]', fontsize=12)
        cbVr = AA.Axes(fig, [0.16, 0.1, 0.7, 0.75])
        cb = add_colorbar(cbVr, cf, label='[m/s]', labelpad=20,
                            loc='bottom')


    if c == 8:
        ax.set_ylabel('Altitude [km] MSL', fontsize=12)
        ax.set_title('LLJ during NO-TTA conditions', fontsize=15)

    ax.fill_between(xt, 0, yt, facecolor='gray', edgecolor='k')

    ax.text(1, 0.5, dt, fontsize=12,
             transform=ax.transAxes,
            va='center',
            rotation=-90)

    ax.xaxis.set_tick_params(labelsize=12,
                             direction='in')
    ax.yaxis.set_tick_params(labelsize=12,
                             direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # t0 = sets[c]['ini'].strftime('%d-%b %H%M UTC')
    # t1 = sets[c]['end'].strftime('%d-%b %H%M UTC')
    # txt1 = '{}\n to {}'.format(t0, t1)
    # tt1 = ax.text(34, 1, txt1, ha='right')
    # tt1.set_bbox(dict(color='w', edgecolor='none'))


plt.subplots_adjust(top=0.95, bottom=0.2, hspace=0)

# fname='/Users/raulvalenzuela/Desktop/fig_llj_during_ntta.png'
# plt.savefig(fname, dpi=300, format='png', papertype='letter',
#             bbox_inches='tight')