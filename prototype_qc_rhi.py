
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from checking_c13_RHIs import get_xpol_fields

za0, vr0 = get_xpol_fields()

i = 60
levels = za0.isel(z=slice(0, 3), time=i)


fig, ax = plt.subplots(4, 1)
levels.plot(ax=ax[0])
hist1 = levels.isel(z=2).plot.hist(ax=ax[1],bins=range(-40, 46))
hist2 = levels.isel(z=1).plot.hist(ax=ax[2],bins=range(-40, 46))
hist3 = levels.isel(z=0).plot.hist(ax=ax[3],bins=range(-40, 46))

tests = list()
tests.append(ks_2samp(levels.isel(z=2), levels.isel(z=1))[1])
tests.append(ks_2samp(levels.isel(z=2), levels.isel(z=0))[1])

''' find mode of third level above surface '''
mode = hist1[1][np.where(hist1[0] == hist1[0].max())]

z = 1
levels_qc = levels.copy()
for test in tests:
    if test < 0.01:
        zqc = levels_qc.isel(z=z)
        zqc = zqc.where(zqc < mode)
        levels_qc[z, :] = zqc
    z -= 1

fig,ax = plt.subplots();levels_qc.plot(ax=ax)