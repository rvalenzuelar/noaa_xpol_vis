

import matplotlib.pyplot as plt
from make_xarray import make_xarray_ppi
import xpol_tta_analysis as xta

# params = dict(wdir_thres=150,
#               rain_czd=0.25,
#               nhours=2
#               )
#
# xp13 = xta.process(case=[13], params=params)
#
# da = make_xarray_ppi(xp13)
# da = da['ZA']


z = 10**(da/10.)

fig, axs = plt.subplots(3, 1, figsize=[5.25, 9.21])
axs = axs.flatten()


ax = axs[0]
zsum = z.sum(dim='time')
zsum.plot(vmax=3E5, ax=ax)
ax.plot([0,0],[0,-60],color='r')  #180deg
ax.plot([0, 25.3], [0, -54.3], color='r')  # 155deg
ax.plot([0, 4.2], [0, -59.8], color='b')  # 176deg
# ax.set_xlim([-10,45])
# ax.set_ylim([-58,10])
fig.set_size_inches(7.49, 6.63)
ax.text(0.95,0.9,'z Sum',
            transform=ax.transAxes,
            color='w',ha='right')


ax = axs[1]
zmean = z.mean(dim='time')
zmean.plot(vmax=2000, ax=ax)
ax.plot([0,0],[0,-60],color='r')  #180deg
ax.plot([0, 25.3], [0, -54.3],color='r')  # 155deg
ax.plot([0, 4.2], [0, -59.8],color='b')  # 176deg
# ax.set_xlim([-10,45])
# ax.set_ylim([-58,10])
fig.set_size_inches(7.49, 6.63)
ax.text(0.95,0.9,'z Average',
            transform=ax.transAxes,
            ha='right')

ax = axs[2]
zmedian = z.median(dim='time')
zmedian.plot(vmax=1000, ax=ax)
ax.plot([0,0],[0,-60],color='r')  #180deg
ax.plot([0, 25.3], [0, -54.3], color='r')  # 155deg
ax.plot([0, 4.2], [0, -59.8], color='b')  # 176deg
# ax.set_xlim([-10,45])
# ax.set_ylim([-58,10])
fig.set_size_inches(7.49, 6.63)
ax.text(0.95,0.9,'z Median',
            transform=ax.transAxes,
            ha='right')

fig.suptitle('XPOL 16-18Feb storm')
plt.subplots_adjust(top=0.95,bottom=0.05)

z025 = z.quantile(0.25, dim='time')
z075 = z.quantile(0.75, dim='time')
ziqr = z075-z025