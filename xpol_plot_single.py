
import xpol
import os

''' printing single and mean values
'''
setcase = {8: [0.5, 180],
           9: [0.5, 180],
           10: [0.5, 180],
           11: [0.5, 180],
           12: [0.5, 6],
           13: [0.5, 180],
           14: [0.5, 180]}

o = 'c{}_xpol_singlescan_{}_{}_{}.pdf'

# homedir = os.path.expanduser('~')
homedir = '/localdata/'


for case in range(12, 13):

    elev, azim = setcase[case]

    field = 'ZA'

    rhis = xpol.get_data(case, 'RHI', azim, homedir=homedir)
    oname = o.format(str(case).zfill(2), 'rhi', str(azim), field)
    xpol.plot_single(rhis, name=field, smode='rhi', case=case, saveas=oname)
    # xpol.plot_single(rhis.ix[:3,:], name=field, smode='rhi',case=case, saveas=oname)

    # ppis=xpol.get_data(case,'PPI',elev)
    # oname=o.format(str(case).zfill(2), 'ppi', str(elev), field)
    # xpol.plot_single(ppis, name=field, smode='ppi', case=case, saveas=oname)

    # ppis = xpol.get_data(case, 'PPI', elev, homedir=homedir)
    # oname = o.format(str(case).zfill(2), 'ppi', str(elev), 'rainrate')
    # xpol.plot_single(ppis, name=field, convert='to_rainrate', smode='ppi',
    #                  case=case, saveas=oname, vmax=40)

    field = 'VR'

    rhis = xpol.get_data(case, 'RHI', azim, homedir=homedir)
    oname = o.format(str(case).zfill(2), 'rhi', str(azim), field)
    xpol.plot_single(rhis, name=field, smode='rhi', case=case, saveas=oname)
    # xpol.plot_single(rhis.ix[:3,:], name=field, smode='rhi', case=case, saveas=oname)

    # # ppis=xpol.get_data(case,'PPI',elev)
    # # oname=o.format(str(case).zfill(2), 'ppi', str(elev), field)
    # # xpol.plot_single(ppis,name=field, smode='ppi', case=case, saveas=oname)
