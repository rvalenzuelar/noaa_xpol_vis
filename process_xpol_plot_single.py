
import xpol
# import gapflow
# import matplotlib.pyplot as plt
# import numpy as np

'''	printing single and mean values
'''
case=11
o='c{}_xpol_singlescan_{}_{}_{}.pdf'


field = 'ZA'

rhis=xpol.get_data(case,'RHI',180)
oname=o.format(str(case).zfill(2), 'rhi', str(180), field)
xpol.plot_single(rhis, name=field, smode='rhi',case=case, saveas=oname)
# xpol.plot_single(rhis.ix[:3,:], name=field, smode='rhi',case=case, saveas=oname)

# ppis=xpol.get_data(case,'PPI',0.5)
# oname=o.format(str(case).zfill(2), 'ppi', str(0.5), field)
# xpol.plot_single(ppis, name=field, smode='ppi', case=case, saveas=oname)
# # # xpol.plot_single(ppis.ix[:3,:], name=field, smode='ppi', case=case)

field = 'VR'

rhis=xpol.get_data(case,'RHI',180)
oname=o.format(str(case).zfill(2), 'rhi', str(180), field)
xpol.plot_single(rhis, name=field, smode='rhi', case=case, saveas=oname)
# xpol.plot_single(rhis.ix[:3,:], name=field, smode='rhi', case=case, saveas=oname)

# ppis=xpol.get_data(case,'PPI',0.5)
# oname=o.format(str(case).zfill(2), 'ppi', str(0.5), field)
# xpol.plot_single(ppis,name=field, smode='ppi', case=case, saveas=oname)
# # xpol.plot_single(ppis.ix[:3,:],name=field, smode='ppi', case=case)
