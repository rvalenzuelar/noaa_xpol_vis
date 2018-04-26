
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xpol
from datetime import datetime
from scipy import stats


def qc_rhi(za, vr):

    """
    QC is based on reflectivity
    but affects both ZA and VR
    """

    za_qc = za.copy()
    vr_qc = vr.copy()

    for grp in zip(za_qc.groupby('time'), vr_qc.groupby('time')):

        sl1 = grp[0][1]  # time slice of za_qc
        sl2 = grp[1][1]  # time slice of vr_qc

        tests = list()
        tests.append(stats.ks_2samp(sl1.isel(z=2), sl1.isel(z=1))[1])
        tests.append(stats.ks_2samp(sl1.isel(z=2), sl1.isel(z=0))[1])

        ''' find mode of third level above surface;
            if multi-modal choose the largest mode
        '''
        # hist1 = np.histogram(sl1.isel(z=2),
        #                      bins=range(-40, 46),
        #                      range=(-40, 46))
        # mode = hist1[1][np.where(hist1[0] == hist1[0].max())][0]
        z = 1
        for test in tests:
            hist1 = np.histogram(sl1.isel(z=z),
                                 bins=range(-40, 46),
                                 range=(-40, 46))
            mode = hist1[1][np.where(hist1[0] == hist1[0].max())][0]
            if test < 0.001:
                good_grids = sl1.isel(z=z) < mode
                sl1[z, :] = sl1.isel(z=z).where(good_grids)
                sl2[z, :] = sl2.isel(z=z).where(good_grids)

            z -= 1

    return za_qc, vr_qc


def get_xpol_fields(azimuths):

    first = True
    for azimuth in azimuths:

        files = glob('/Users/raulvalenzuela/Desktop/'
                    'test_xpol/{}/*.cdf'.format(azimuth))
        files.sort()

        drop_this = [u'radar_or_data_origin',
        u'project_name',
        u'scientist_name',
        u'submitters_name',
        u'data_source',
        u'volume_header',
        # u'start_date',
        u'end_date',
        # u'start_time',
        u'end_time',
        u'cedric_run_date',
        u'cedric_run_time',
        u'reference_Nyquist_vel',
        u'nyquist_velocities',
        u'bad_data_flag',
        u'time_offset',
        u'base_time',
        u'program',
        u'landmark_x_coordinates',
        u'landmark_y_coordinates',
        u'landmark_z_coordinates',
        u'landmark_names',
        u'grid_type',
        u'radar_latitude',
        u'radar_longitude',
        u'radar_altitude',
        u'lat',
        u'lon',
        u'alt',
        u'x_min',
        u'y_min',
        u'x_max',
        u'y_max',
        u'x_spacing',
        u'y_spacing',
        u'el',
        u'z_spacing',
        u'field_names',
        u'TIME',
        u'AZ']

        dss = [xr.open_dataset(ncfile, drop_variables=drop_this)
               for ncfile in files]

        ds = xr.concat(dss, dim='time')

        if first:
            ds1 = ds.copy()
            first = False
        else:
            ds2 = ds.copy()

    ''' parse datatime index '''
    sdates = ds2['start_date'].values.tolist()
    stimes = ds2['start_time'].values.tolist()
    sdt = [d.replace('/01','/') + ' ' + t for d, t in zip(sdates,stimes)]
    timeidx = pd.to_datetime(sdt)

    ''' concat along x-axis and remove y-axis '''
    # ds_a = ds2.isel(time=slice(0, -1), y=1)
    ds_a = ds2.isel(y=1)
    ds_b = ds1.isel(y=1)

    min_size = np.minimum(ds_a.time.size, ds_b.time.size)

    dsc = xr.concat([ds_a.isel(time=slice(0,min_size)),
                     ds_b.isel(time=slice(0,min_size))],
                    dim='x')
    dsc.coords['time'] = timeidx[:min_size]

    ''' netcdf scale '''
    scale = 64.

    ''' RHI elevation angle '''
    EL = dsc['EL'].isel(time=0)/scale

    ''' scale attenuation-corrected reflectivity '''
    za0 = dsc['ZA']/scale
    vr0 = dsc['VR']/scale

    z_round = (za0.z.values*10).astype(np.int)/10.
    za0.coords['z'] = z_round

    return za0, vr0, EL


def get_excedfreq(da):

    da_median = da.median().values.tolist()
    print('median: {:2.1f}'.format(da_median))
    da_grt = da >= da_median
    da_sum = da_grt.sum(dim='time')
    # narray = float(za_tta.time.size)
    narray = float(da_sum.max().values)
    da_freq = 100 * da_sum/narray

    return da_freq


if __name__ == '__main__':

    azimuths = ['az360', 'az180']

    za0, vr0, EL = get_xpol_fields(azimuths)
    za1, vr1 = qc_rhi(za0, vr0)

    ''' copy to new var '''
    za = za0
    vr = vr0

    ''' velocity component '''
    vr = np.abs(vr/np.cos(np.radians(EL)))
    vr = vr.where((EL < 65))

    ''' get component along RHI orientation
        (assumes southerly wind)
    '''
    if 'az360' not in azimuths:
        az = int(azimuths[0][-3:])
        vr = vr/np.cos(np.radians(az))

    ''' get tta dates'''
    # tta_csv = pd.read_csv('tta_dates_2004')
    # tta_str = [v[0] for v in tta_csv.values]
    # tta_dates = pd.to_datetime(tta_str)

    ''' tta and no-tta separation '''
    za_tta = za.sel(time=slice(datetime(2004, 2, 16, 9),
                               datetime(2004, 2, 16, 16))
                    )
    za_ntta = za.drop(za_tta.time.values, dim='time')

    vr_tta = vr.sel(time=slice(datetime(2004, 2, 16, 9),
                               datetime(2004, 2, 16, 16))
                    )
    vr_ntta = vr.drop(vr_tta.time.values, dim='time')

    ''' plot options '''
    cmap = xpol.custom_cmap('rhi_vr1')
    za_ops = dict(cmap='inferno', levels=range(20, 110, 10))
    vr_ops = dict(cmap=cmap, levels=range(0, 52, 2))

    fig, axs = plt.subplots(2,2, figsize=[10.08, 5.47],
                            sharey=True)
    ax = axs.flatten()

    ''' exceedance freq '''
    za_tta_freq = get_excedfreq(za_tta)
    za_tta_freq.where(za_tta_freq > 20).plot.\
        contourf(ax=ax[1], **za_ops)

    za_ntta_freq = get_excedfreq(za_ntta)
    za_ntta_freq.where(za_ntta_freq > 20).plot.\
        contourf(ax=ax[3], **za_ops)

    ''' average doppler '''
    vr_tta_avr = vr_tta.mean(dim='time')
    vr_tta_avr.plot.\
        contourf(ax=ax[0], **vr_ops)

    vr_ntta_avr = vr_ntta.mean(dim='time')
    vr_ntta_avr.plot.\
        contourf(ax=ax[2], **vr_ops)

    ''' final details '''
    ax[0].set_title('TTA: VR mean')
    ax[1].set_title('TTA: ZA exceed freq')
    ax[1].set_ylim(0, 5)
    ax[2].set_title('NTTA: VR mean')
    ax[3].set_title('NTTA: ZA exceed freq')
    plt.tight_layout()
    plt.subplots_adjust(top=0.89)
    title = 'RHI {}-{} orientation'.\
        format(azimuths[1][-3:],azimuths[0][-3:])
    plt.gcf().suptitle(title)