import Windprof2 as wp
import xpol
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

setcase = {8: [0.5, 180],
           9: [0.5, 180],
           10: [0.5, 180],
           11: [0.5, 180],
           12: [0.5,   6],
           13: [0.5, 180],
           14: [0.5, 180]}
# homedir = '/localdata/'
homedir = os.path.expanduser('~')

tta_dframes_za = []
tta_dframes_vr = []
notta_dframes_za = []
notta_dframes_vr = []

for case in range(8, 15):

    tta_times = wp.get_tta_times(case=str(case), homedir=homedir)

    elevation, _ = setcase[case]
    ppis = xpol.get_data(case, 'PPI', elevation, homedir=homedir)

    tta_idxs = np.asarray([], dtype=int)
    for time in tta_times:
        idx = np.where((ppis.index.day == time.day) &
                       (ppis.index.hour == time.hour))[0]
        if idx.size > 0:
            tta_idxs = np.append(tta_idxs, idx)
    notta_idxs = np.delete(np.arange(len(ppis.index)), tta_idxs)

    if len(tta_idxs) > 0:
        df = ppis.iloc[tta_idxs]['ZA']
        tta_dframes_za.append(df)
        df = ppis.iloc[tta_idxs]['VR']
        tta_dframes_vr.append(df)

    if len(notta_idxs) > 0:
        df = ppis.iloc[notta_idxs]['ZA']
        notta_dframes_za.append(df)
        df = ppis.iloc[notta_idxs]['VR']
        notta_dframes_vr.append(df)


tta_frame_za = pd.concat(tta_dframes_za)
tta_frame_vr = pd.concat(tta_dframes_vr)
notta_frame_za = pd.concat(notta_dframes_za)
notta_frame_vr = pd.concat(notta_dframes_vr)

''' PPI composite
------------------------ '''
tta_dbz_freq, tta_thres, _ = xpol.get_dbz_freq(tta_frame_za,
                                               percentile=50)
notta_dbz_freq, notta_thres, _ = xpol.get_dbz_freq(notta_frame_za,
                                                   percentile=50)
tta_vr_mean, _ = xpol.get_mean(tta_frame_vr, name='VR')
notta_vr_mean, _ = xpol.get_mean(notta_frame_vr, name='VR')

''' Plots
---------------------'''
fig, axes = plt.subplots(2, 2, figsize=(
    11, 10.5), sharex=True, sharey=True)
ax = axes.flatten()

xpol.plot(tta_vr_mean, ax=ax[0], name='VR',
          smode='ppi', colorbar=False)
n = 'TTA # of sweeps = {}'.format(len(tta_frame_vr))
ax[0].set_title(n)

xpol.plot(notta_vr_mean, ax=ax[1], name='VR',
          smode='ppi', colorbar=True)
n = 'NO-TTA # of sweeps = {}'.format(len(notta_frame_vr))
ax[1].set_title(n)

xpol.plot(tta_dbz_freq, ax=ax[2], name='freq', smode='ppi',
          colorbar=False,
          textbox='dBZ Threshold:{:2.1f}'.format(tta_thres))

xpol.plot(notta_dbz_freq, ax=ax[3], name='freq', smode='ppi',
          colorbar=True,
          textbox='dBZ Threshold:{:2.1f}'.format(notta_thres))

plt.tight_layout()
plt.show(block=False)
