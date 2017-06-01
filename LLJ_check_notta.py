"""

    Raul Valenzuela
    2017

    Note:
        use wind profiler figure as guidance for
        identifying LLJ time period

"""
import xpol_tta_analysis as xta
import numpy as np
import matplotlib.pyplot as plt
import xpol
from datetime import datetime

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )

try:
    x13
except NameError:
    x08 = xta.process(case=[8],params=params)
    x09 = xta.process(case=[9], params=params)
    x10 = xta.process(case=[10], params=params)
    x11 = xta.process(case=[11], params=params)
    x12 = xta.process(case=[12], params=params)
    x13 = xta.process(case=[13], params=params)
    x14 = xta.process(case=[14], params=params)

cases = {
          '08':x08,'09':x09,'10':x10,
         '11':x11,'12':x12,
         '13':x13,
         '14':x14
        }

cnum = '13'
serie = cases[cnum].rhi_tta['VR']
t0 = datetime(2004, 2, 16, 11, 0)
t1 = datetime(2004, 2, 16, 14, 0)
serie = serie[(serie.index>t0) & (serie.index<t1)]


arr = np.expand_dims(serie.iloc[0], axis=2)
n = len(serie)
for i in range(1, n):
    expand = np.expand_dims(serie.iloc[i], axis=2)
    arr = np.concatenate((arr, expand), axis=2)

a,b,_ = arr.shape
x = np.arange(b)
y = np.arange(a)

X, Y = np.meshgrid(x, y)
cmap = xpol.custom_cmap('rhi_vr1')
cvalues = np.arange(0, 32, 2)
arr_mean = np.nanmean(arr, axis=2)
fig, ax = plt.subplots()
cf = ax.contourf(X, Y, arr_mean, cvalues, cmap=cmap)
fig.colorbar(cf)
t0f = t0.strftime('%Y-%m-%d %H%M UTC')
t1f = t1.strftime('%Y-%m-%d %H%M UTC')
fig.suptitle('Case {}: {} to {}'.format(cnum, t0f, t1f))
fig.set_size_inches(9,4)
