"""
    Raul Valenzuela
    2017

"""

import xpol_tta_analysis as xta
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

params = {'nhours': 2, 'rain_czd': 0.25, 'wdir_thres': 150}
try:
    x08
except NameError:
    x08 = xta.process(case=[8], params=params)

idxsort = x08.ppi_tta['ZA'].index.sort_values()
x08sort = x08.ppi_tta['ZA'].loc[idxsort]

x = x08.get_axis('x', 'ppi')
y = x08.get_axis('y', 'ppi')
X, Y = np.meshgrid(x, y)

data = np.expand_dims(x08sort.iloc[0], axis=2)
timestamp = x08sort.index
for arr in x08sort.iloc[1:]:
    arr2 = np.expand_dims(arr, axis=2)
    data = np.concatenate((data, arr2), axis=2)

mdata = np.ma.masked_invalid(data)


''' make animation '''
fig, ax = plt.subplots()
cax = ax.pcolormesh(x, y, mdata[:-1, :-1, 0],
                    vmin=-10, vmax=50)
fig.colorbar(cax)
fmt = '%Y-%m-%d %H:%M UTC'
ax.set_title(timestamp[0].strftime(fmt))

def animate(i):
    cax.set_array(mdata[:-1, :-1, i].flatten())
    ax.set_title(timestamp[i].strftime(fmt))

anim = animation.FuncAnimation(fig, animate, frames=50)
plt.show()
