"""
    Convert xpol analysis (pandas series)
    to xarray

    Raul Valenzuela
    2017
    
"""


import xarray as xr
import numpy as np

def make_xarray_rhi(xp, regime=None, slices=None):

    if regime == 'tta':
        xp_vr = xp.rhi_tta['VR']
        xp_za = xp.rhi_tta['ZA']
    elif regime == 'ntta':
        xp_vr = xp.rhi_ntta['VR']
        xp_za = xp.rhi_ntta['ZA']

    xp_vr.sort_index(inplace=True)
    xp_za.sort_index(inplace=True)

    add_dim_vr = map(lambda x:np.expand_dims(x,axis=2),
                     xp_vr.values)

    concat_vr = np.concatenate(add_dim_vr, axis=2)

    add_dim_za = map(lambda x: np.expand_dims(x, axis=2),
                     xp_za.values)

    concat_za = np.concatenate(add_dim_za, axis=2)

    time = [i.to_pydatetime() for i in xp_vr.index]
    x = xp.get_axis('x','rhi')
    z = xp.get_axis('z','rhi')

    minx = np.min(x)
    maxx = np.max(x)
    xx = np.arange(-40, 30.2, 0.14)  # common grid between cases
    ibeg = find_nearest(xx, minx)
    iend = find_nearest(xx, maxx)

    ds = xr.Dataset(data_vars={'VR': (['z','x','time'], concat_vr),
                               'ZA': (['z','x','time'], concat_za)},
                      coords={'z': z,
                              'x': xx[ibeg:iend+1],
                              'time': time})

    if slices is not None:
        first = True
        for sl in slices:
            if first:
                ds_sls = ds.sel(time=sl)
                first = False
            else:
                ds_sls = xr.concat([ds_sls, ds.sel(time=sl)],
                                   dim='time')

        return ds_sls
    else:
        return ds


def make_xarray_ppi(xp, regime=None, slices=None):

    if regime == 'tta':
        xp_vr = xp.ppi_tta['VR']
        xp_za = xp.ppi_tta['ZA']
    elif regime == 'ntta':
        xp_vr = xp.ppi_ntta['VR']
        xp_za = xp.ppi_ntta['ZA']

    xp_vr.sort_index(inplace=True)
    xp_za.sort_index(inplace=True)

    add_dim_vr = map(lambda x:np.expand_dims(x,axis=2),
                     xp_vr.values)

    concat_vr = np.concatenate(add_dim_vr, axis=2)

    add_dim_za = map(lambda x: np.expand_dims(x, axis=2),
                     xp_za.values)

    concat_za = np.concatenate(add_dim_za, axis=2)

    time = [i.to_pydatetime() for i in xp_vr.index]
    x = xp.get_axis('x','ppi')
    y = xp.get_axis('y','ppi')

    # minx = np.min(x)
    # maxx = np.max(x)
    # xx = np.arange(-40, 30.2, 0.14)  # common grid between cases
    # ibeg = find_nearest(xx, minx)
    # iend = find_nearest(xx, maxx)

    ds = xr.Dataset(data_vars={'VR': (['y','x','time'], concat_vr),
                               'ZA': (['y','x','time'], concat_za)},
                      coords={'y': y,
                              'x': x,
                              'time': time})

    if slices is not None:
        first = True
        for sl in slices:
            if first:
                ds_sls = ds.sel(time=sl)
                first = False
            else:
                ds_sls = xr.concat([ds_sls, ds.sel(time=sl)],
                                   dim='time')

        return ds_sls
    else:
        return ds


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx
