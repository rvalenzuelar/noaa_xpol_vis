"""
    Convert xpol analysis (pandas series)
    to xarray

    Raul Valenzuela
    2017
    
"""


import xarray as xr
import numpy as np

def make_xarray(xp, slices=None):

    xp.rhi_ntta['VR'].sort_index(inplace=True)
    xp.rhi_ntta['ZA'].sort_index(inplace=True)

    add_dim_vr = map(lambda x:np.expand_dims(x,axis=2),
                  xp.rhi_ntta['VR'].values)

    concat_vr = np.concatenate(add_dim_vr, axis=2)

    add_dim_za = map(lambda x: np.expand_dims(x, axis=2),
                     xp.rhi_ntta['ZA'].values)

    concat_za = np.concatenate(add_dim_za, axis=2)

    time = [i.to_pydatetime() for i in xp.rhi_ntta['VR'].index]
    x = xp.get_axis('x','rhi')
    z = xp.get_axis('z','rhi')

    ds = xr.Dataset(data_vars={'VR':(['z','x','time'],concat_vr),
                               'ZA':(['z','x','time'],concat_za)},
                      coords={'z':z,'x':x,'time':time})

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