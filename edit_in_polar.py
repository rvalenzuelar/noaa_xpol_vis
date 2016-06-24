# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:16:38 2016

@author: raul
"""

import Linear2Polar as l2p
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xpol_tta_analysis as xta

try:
  x09
except NameError:
  x09=xta.process(case=[9])

#cx = np.arange(-58, 45.5, 0.5)        
#cy = np.arange(-58, 33.5, 0.5)
cx = np.linspace(-58, 45.5, 207)        
cy = np.linspace(-58, 33.5, 183)


carray = x09.ppi_ntta_z
carraym = ma.masked_where(np.isnan(carray),carray)

x0 = np.where(cx == 0.)[0]  
y0 = np.where(cy == 0.)[0]

parray, theta, r = l2p.project_into_polar(carray,
                                        origin=(116,116),
                                        order=0)


parraym = ma.masked_where(np.isnan(parray),parray)
parraym = ma.masked_where(parraym == 0,parraym)

carray2, cx2, cy2 = l2p.project_into_cart(parray,
                                          r=r,
                                          theta=theta,
                                          origin=(0,0),
                                          order=0)
carray2m = ma.masked_where(np.isnan(carray2),carray2)
carray2m = ma.masked_where(carray2m == 0, carray2m)

fig,ax = plt.subplots(3,1,figsize=(8,11))

#ax[0].pcolormesh(cx,cy,carraym)
ax[0].pcolormesh(range(207),range(183),carraym)
ax[0].scatter(116,116,color='k')
ax[0].set_xlim([0,206])
ax[0].set_ylim([0,182])

ax[1].pcolormesh(range(207),range(183),parraym)
ax[1].set_xlim([0,206])
ax[1].set_ylim([0,182])

ax[2].pcolormesh(range(207), range(183), carray2m)
ax[2].scatter(116,116,color='k')
ax[2].set_xlim([0,206])
ax[2].set_ylim([0,182])

plt.show()



