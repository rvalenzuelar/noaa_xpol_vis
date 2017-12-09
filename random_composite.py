
from make_xarray import make_xarray_rhi
from make_xarray import make_xarray_ppi
import xpol_tta_analysis as xta
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random
import xpol


cmap_rhi = xpol.custom_cmap('rhi_vr1')
cmap_ppi = xpol.custom_cmap('ppi_vr1')

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )


def random_sample(xpset,verbose=True):

    max_sample = np.min([x.time.size for x in xpset])

    first = True
    for ds in xpset:
        # rnd = np.random.randint(0, ds.time.size, max_sample)
        rnd = random.sample(range(ds.time.size), max_sample)
        print 'rnd={}'.format(rnd)
        if first:
            conVR = ds['VR'].isel(time=rnd)
            conZA = ds['ZA'].isel(time=rnd)
            first = False
        else:
            conVR = xr.concat([conVR, ds['VR'].isel(time=rnd)],
                            dim='time')
            conZA = xr.concat([conZA, ds['ZA'].isel(time=rnd)],
                            dim='time')

    if verbose:
        print 'max_sample:{}'.format(max_sample)
        print 'total N:{}'.format(conVR.time.size)

    return conVR, conZA


def rnd_rhi_tta():

    ''' case 12 excluded because RHI scan is incomplete '''

    ds08 = make_xarray_rhi(xp08,regime='tta')
    ds09 = make_xarray_rhi(xp09,regime='tta')
    ds11 = make_xarray_rhi(xp11,regime='tta')
    ds13 = make_xarray_rhi(xp13,regime='tta')

    con = xr.concat([ds08, ds09], dim='time')
    con = xr.concat([con, ds11], dim='time')
    con = xr.concat([con, ds13], dim='time')

    rnd = np.random.randint(0, 134, 68)
    sample = con['VR'].isel(time=rnd)
    fig,ax = plt.subplots()
    sample.mean(dim='time').plot(ax=ax, levels=9,
                                 vmin=0,vmax=20)


def composite_rhi(regime=None):

    if regime == 'tta':
        ds08 = make_xarray_rhi(xp08,regime='tta')
        ds09 = make_xarray_rhi(xp09,regime='tta')
        ds11 = make_xarray_rhi(xp11,regime='tta')
        ds13 = make_xarray_rhi(xp13,regime='tta')
        xpset = [ds08, ds09, ds11, ds13]

    elif regime == 'ntta':
        ds08 = make_xarray_rhi(xp08, regime='ntta')
        ds09 = make_xarray_rhi(xp09, regime='ntta')
        ds10 = make_xarray_rhi(xp10, regime='ntta')
        ds11 = make_xarray_rhi(xp11, regime='ntta')
        ds13 = make_xarray_rhi(xp13, regime='ntta')
        ds14 = make_xarray_rhi(xp14, regime='ntta')
        xpset = [ds08, ds09, ds10, ds11, ds13, ds14]

    con = xr.concat(xpset, dim='time')

    mean_vr = con['VR'].mean(dim='time')

    median = con['ZA'].median()
    enh = con['ZA'].where(con['ZA']>median).count(
                    dim='time').astype(float)

    all = con['ZA'].count(dim='time').astype(float)

    print 'median:{}'.format(median.values)
    print 'con time size:{}'.format(con.time.size)
    print 'all.max :{}'.format(all.max().values)


    min_frac = 20

    frac = (enh / all.max())*100

    ''' radial distance for masking '''
    rho = np.sqrt(frac.x ** 2 + frac.z ** 2)  # [km]
    max_range = rho < 28
    min_f = frac > min_frac
    mean_vr_qc = mean_vr.where(max_range)
    frac_qc = frac.where(max_range & min_f)

    fig, ax = plt.subplots(2,1)
    mean_vr_qc.plot.contourf(ax=ax[0],levels=16, vmin=0, vmax=30,
                                add_labels=True,
                                cmap=cmap_rhi)
    plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=100,levels=9)
    frac_qc.plot.contourf(ax=ax[1], **plt_ops)

    ax[0].set_xlim([-40, 40])
    ax[1].set_xlim([-40, 40])
    ax[0].set_ylim([0, 5])
    ax[1].set_ylim([0, 5])

    # frac = (enh / enh.max())*100
    # fig, ax = plt.subplots(3,1,sharex=True)
    # enh.plot.contourf(ax=ax[0],cmap='inferno')
    # all.plot.contourf(ax=ax[1],cmap='inferno')
    # plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=100,levels=9)
    # frac.where(frac>min_frac).plot.contourf(ax=ax[2], **plt_ops)
    # plt.suptitle('Divided by max of enh array')

    # frac = (enh / all.max())*100
    # fig, ax = plt.subplots(3,1,sharex=True)
    # enh.plot.contourf(ax=ax[0],cmap='inferno')
    # all.plot.contourf(ax=ax[1],cmap='inferno')
    # plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=50,levels=7)
    # frac.where(frac>min_frac).plot.contourf(ax=ax[2], **plt_ops)
    # plt.suptitle('Divided by max of all array')
    #
    # frac = (enh / all) * 100
    # fig, ax = plt.subplots(3,1,sharex=True)
    # enh.plot.contourf(ax=ax[0],cmap='inferno')
    # all.plot.contourf(ax=ax[1],cmap='inferno')
    # plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=100,levels=9)
    # frac.where(frac>min_frac).plot.contourf(ax=ax[2],**plt_ops)
    # plt.suptitle('Divided by all array')


def composite_ppi(regime=None):

    if regime == 'tta':

        ds08 = make_xarray_ppi(xp08,regime='tta')
        ds09 = make_xarray_ppi(xp09,regime='tta')
        ds11 = make_xarray_ppi(xp11,regime='tta')
        ds13 = make_xarray_ppi(xp13,regime='tta')
        xpset = [ds08, ds09, ds11, ds13]

    elif regime == 'ntta':

        ds08 = make_xarray_ppi(xp08, regime='ntta')
        ds09 = make_xarray_ppi(xp09, regime='ntta')
        ds10 = make_xarray_ppi(xp10, regime='ntta')
        ds11 = make_xarray_ppi(xp11, regime='ntta')
        ds12 = make_xarray_ppi(xp11, regime='ntta')
        ds13 = make_xarray_ppi(xp13, regime='ntta')
        ds14 = make_xarray_ppi(xp14, regime='ntta')
        xpset = [ds08, ds09, ds10, ds11, ds12, ds13, ds14]


    con = xr.concat(xpset, dim='time')

    mean_vr = con['VR'].mean(dim='time')

    median = con['ZA'].median()
    enh = con['ZA'].where(con['ZA']>median).count(
                    dim='time').astype(float)

    all = con['ZA'].count(dim='time').astype(float)

    print 'median:{}'.format(median.values)
    print 'con time size:{}'.format(con.time.size)
    print 'all.max :{}'.format(all.max().values)

    min_frac = 20

    frac = (enh / all.max())*100

    ''' radial distance for masking '''
    rho = np.sqrt(frac.x ** 2 + frac.y ** 2)  # [km]
    max_range = rho < 55
    min_f = frac > min_frac
    mean_vr_qc = mean_vr.where(max_range)
    frac_qc = frac.where(max_range & min_f)

    fig, ax = plt.subplots(2,1,figsize=[4.38,  7.13]
                                ,sharex=True)

    mean_vr_qc.plot.contourf(ax=ax[0],levels=16, vmin=-30, vmax=30,
                                add_labels=True,
                                cmap=cmap_ppi)
    plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=100,levels=9)
    frac_qc.plot.contourf(ax=ax[1], **plt_ops)


    ''' add radials and meridional/zonal lines '''
    locations = [(-x, -x) for x in range(10, 30, 5) + [35, 40]]
    for a in ax:
        rc = rho.plot.contour(ax=a, levels=range(0, 70, 10),
                              colors='k', alpha=0.6)
        plt.clabel(rc, fmt='%1.0f', manual=locations)

    ax[0].axhline(0, color='k', alpha=0.6)
    ax[0].axvline(0, color='k', alpha=0.6)

    ax[1].axhline(0, color='k', alpha=0.6)
    ax[1].axvline(0, color='k', alpha=0.6)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.01,top=0.95)


def regime_ppi(ppidata, regime=None, vmin=None, vmax=None):

    ds = make_xarray_ppi(ppidata,regime=regime)

    mean_vr = ds['VR'].mean(dim='time')

    median = ds['ZA'].median()
    enh = ds['ZA'].where(ds['ZA']>median).count(
                    dim='time').astype(float)

    all = ds['ZA'].count(dim='time').astype(float)

    print 'median:{}'.format(median.values)
    print 'con time size:{}'.format(ds.time.size)
    print 'all.max :{}'.format(all.max().values)

    min_frac = 20

    frac = (enh / enh.max())*100

    ''' radial distance for masking '''
    rho = np.sqrt(frac.x ** 2 + frac.y ** 2)  # [km]
    max_range = rho < 55
    min_f = frac > min_frac
    mean_vr_qc = mean_vr.where(max_range)
    frac_qc = frac.where(max_range & min_f)

    fig, ax = plt.subplots(1,2)

    mean_vr_qc.plot.contourf(ax=ax[0],levels=16,
                             vmin=-30, vmax=30,
                             add_labels=True,
                             cmap=cmap_ppi)

    plt_ops = dict(cmap='inferno',vmin=min_frac,vmax=vmax,
                   levels=9)

    frac_qc.plot.contourf(ax=ax[1], **plt_ops)

    ''' add radials and meridional/zonal lines '''
    locations = [(-x, -x) for x in range(10, 30, 5) + [35, 40]]
    for a in ax:
        rc = rho.plot.contour(ax=a, levels=range(0, 70, 10),
                              colors='k', alpha=0.6)
        plt.clabel(rc, fmt='%1.0f', manual=locations)

    ax[0].axhline(0, color='k', alpha=0.6)
    ax[0].axvline(0, color='k', alpha=0.6)

    ax[1].axhline(0, color='k', alpha=0.6)
    ax[1].axvline(0, color='k', alpha=0.6)

    # plt.tight_layout()
    # plt.subplots_adjust(left=0.01,top=0.95)


def rnd_rhi_ntta():

    ''' case 12 excluded because RHI scan is incomplete '''

    ds08 = make_xarray_rhi(xp08,regime='ntta')
    ds09 = make_xarray_rhi(xp09,regime='ntta')
    ds10 = make_xarray_rhi(xp10, regime='ntta')
    ds11 = make_xarray_rhi(xp11,regime='ntta')
    ds13 = make_xarray_rhi(xp13,regime='ntta')
    ds14 = make_xarray_rhi(xp14, regime='ntta')

    con = xr.concat([ds08, ds09], dim='time')
    con = xr.concat([con, ds10], dim='time')
    con = xr.concat([con, ds11], dim='time')
    con = xr.concat([con, ds13], dim='time')
    con = xr.concat([con, ds14], dim='time')

    rnd = np.random.randint(0, 632, 316)
    sample = con['VR'].isel(time=rnd)
    fig,ax = plt.subplots()
    sample.mean(dim='time').plot(ax=ax, levels=9,
                                 vmin=0,vmax=20)


def rnd_ppi_tta():

    ds08 = make_xarray_ppi(xp08, regime='tta')
    ds09 = make_xarray_ppi(xp09, regime='tta')
    ds11 = make_xarray_ppi(xp11, regime='tta')
    ds12 = make_xarray_ppi(xp12, regime='tta')
    ds13 = make_xarray_ppi(xp13, regime='tta')

    con = xr.concat([ds08, ds09], dim='time')
    con = xr.concat([con, ds11], dim='time')
    con = xr.concat([con, ds12], dim='time')
    con = xr.concat([con, ds13], dim='time')

    rnd = np.random.randint(0, 292, 146)
    sample = con['VR'].isel(time=rnd)
    fig,ax = plt.subplots()
    sample.mean(dim='time').plot(ax=ax, levels=13,
                                 )


def rnd_rhi(reps=None, regime=None):

    ''' case 12 excluded because RHI scan is incomplete '''

    if regime == 'tta':
        ds08 = make_xarray_rhi(xp08,regime='tta')
        ds09 = make_xarray_rhi(xp09,regime='tta')
        ds11 = make_xarray_rhi(xp11,regime='tta')
        ds13 = make_xarray_rhi(xp13,regime='tta')
        xpset = [ds08, ds09, ds11, ds13]

    elif regime == 'ntta':
        ds08 = make_xarray_rhi(xp08, regime='ntta')
        ds09 = make_xarray_rhi(xp09, regime='ntta')
        ds10 = make_xarray_rhi(xp10, regime='ntta')
        ds11 = make_xarray_rhi(xp11, regime='ntta')
        ds13 = make_xarray_rhi(xp13, regime='ntta')
        ds14 = make_xarray_rhi(xp14, regime='ntta')
        xpset = [ds08, ds09, ds10, ds11, ds13, ds14]

    fig, ax = plt.subplots(2, 1, sharex=True)

    if reps is None:
        rnd_VR, rnd_ZA = random_sample(xpset)
        rnd_VR.mean(dim='time').plot(ax=ax[0], levels=9,
                                     vmin=0,vmax=20)
        median = rnd_ZA.median()
        print 'median={}'.format(median.values)

        enh = rnd_ZA.where(rnd_ZA > median).count(
            dim='time').astype(float)
        all = rnd_ZA.count(dim='time').astype(float)
        frac = (enh / enh.max())*100

        rho = np.sqrt(frac.x ** 2 + frac.z ** 2)  # [km]
        max_range = rho<28
        min_frac = frac>20
        frac_qc = frac.where(max_range & min_frac)
        frac_qc.plot.contourf(ax=ax[1], cmap='inferno',
                                levels=9,
                                vmin=20, vmax=100)

    else:
        first = True
        nreps = reps
        while reps > 0:

            print reps

            ''' sample of VR and ZA '''
            rnd_VR, rnd_ZA = random_sample(xpset, verbose=True)

            ''' mean VR'''
            one_mean = rnd_VR.mean(dim='time')

            ''' precip freq '''
            median = rnd_ZA.median()
            print 'median={}'.format(median.values)
            exc_thres = rnd_ZA.where(rnd_ZA > median)
            enh = exc_thres.count(dim='time').astype(float)
            all = rnd_ZA.count(dim='time').astype(float)
            print 'all max={}'.format(all.max().values)
            frac = enh / enh.max()
            one_frac = frac.where(frac <= 1)

            if first:
                con_mean = one_mean.copy()
                con_frac = one_frac.copy()
                first = False
            else:
                con_mean = xr.concat([con_mean, one_mean])
                con_frac = xr.concat([con_frac, one_frac])

            reps -= 1

        mean = con_mean.mean(dim='concat_dims')
        frac = con_frac.mean(dim='concat_dims')*100

        ''' radial distance for masking '''
        rho = np.sqrt(frac.x ** 2 + frac.z ** 2)  # [km]
        theta = np.rad2deg(np.arctan(ds08.x / ds08.z))
        max_range = rho<28
        min_frac = frac>20
        max_theta = theta<73
        mean_qc = mean.where(max_range&max_theta)
        frac_qc = frac.where(max_range & min_frac&max_theta)

        m = mean_qc.plot.contourf(ax=ax[0],
                                levels=16, vmin=0, vmax=30,
                                add_labels=True,
                                cmap=cmap_rhi)

        f = frac_qc.plot.contourf(ax=ax[1],
                                cmap='inferno',
                                levels=9,
                                vmin=20, vmax=100)

        ax[0].text(-40, 5.3, 'repetitions: {}'.format(nreps),
                   ha='left')

        ax[0].set_xlim([-40,40])
        ax[1].set_xlim([-40,40])
        ax[0].set_ylim([0, 5])
        ax[1].set_ylim([0, 5])

        m.colorbar.set_label('VR')
        f.colorbar.set_label('FREQ[%]')

        return frac_qc


def rnd_ppi(reps=None, regime=None):

    ''' case 12 excluded because RHI scan is incomplete '''

    if regime == 'tta':

        ds08 = make_xarray_ppi(xp08,regime='tta')
        ds09 = make_xarray_ppi(xp09,regime='tta')
        ds11 = make_xarray_ppi(xp11,regime='tta')
        ds13 = make_xarray_ppi(xp13,regime='tta')
        xpset = [ds08, ds09, ds11, ds13]

    elif regime == 'ntta':

        ds08 = make_xarray_ppi(xp08, regime='ntta')
        ds09 = make_xarray_ppi(xp09, regime='ntta')
        ds10 = make_xarray_ppi(xp10, regime='ntta')
        ds11 = make_xarray_ppi(xp11, regime='ntta')
        ds13 = make_xarray_ppi(xp13, regime='ntta')
        ds14 = make_xarray_ppi(xp14, regime='ntta')
        xpset = [ds08, ds09, ds10, ds11, ds13, ds14]

    fig, ax = plt.subplots(2,1, figsize=[4.71,  7.12]
                                ,sharex=True)

    nreps=reps

    if reps is None:
        rnd_VR, rnd_ZA = random_sample(xpset)
        rnd_VR.mean(dim='time').plot(ax=ax[0], levels=13,
                                     vmin=-24, vmax=24,
                                     cmap='RdBu')

        median = rnd_ZA.median()
        print 'median={}'.format(median.values)

        enh = rnd_ZA.where(rnd_ZA > median).count(
            dim='time').astype(float)
        all = rnd_ZA.count(dim='time').astype(float)
        frac = enh / all

        frac = frac*100
        frac.name = 'FREQ[%]'

        frac.where(frac < 100).plot(ax=ax[1], vmin=20, vmax=80)
        ax[0].text(-50, 40, 'repetitions: {}'.format(1),ha='left')

    else:
        first = True
        while reps > 0:

            print reps
            rnd_VR, rnd_ZA = random_sample(xpset,verbose=False)
            one_mean = rnd_VR.mean(dim='time')
            median = rnd_ZA.median()
            print 'median={}'.format(median.values)
            enh = rnd_ZA.where(rnd_ZA > median).count(
                dim='time').astype(float)
            all = rnd_ZA.count(dim='time').astype(float)

            enh2 = enh.copy()
            enh2 = enh2.values
            enh2[[117, 117, 118, 118, 119],
                 [112, 113, 112, 113, 112]] = np.nan
            enh.values = enh2

            frac = enh / enh.max()
            one_frac = frac.where(frac <= 1)

            if first:
                con_mean = one_mean.copy()
                con_frac = one_frac.copy()
                first = False
            else:
                con_mean = xr.concat([con_mean, one_mean])
                con_frac = xr.concat([con_frac, one_frac])

            reps -= 1

        mean = con_mean.mean(dim='concat_dims')
        frac = con_frac.mean(dim='concat_dims')*100
        frac.name = 'FREQ[%]'

        ''' radial distance for masking '''
        rho = np.sqrt(frac.x ** 2 + frac.y ** 2) # [km]

        ''' azimuth for masking'''
        theta = rho.copy()  # copy dimensions
        y, x = np.meshgrid(ds08.y, ds08.x)
        th = np.rad2deg(np.arctan2(x,y))
        th = xr.DataArray(th)
        theta.values = th+180

        ''' qc '''
        max_rho = rho < 55
        min_frac = frac > 20
        th_qc = (theta < 120) | (theta > 320)

        mean_qc = mean.where(max_rho & th_qc)
        frac_qc = frac.where(max_rho & min_frac & th_qc)

        ''' check theta '''
        # fig2, ax2 = plt.subplots(2, 1)
        # theta.T.plot(ax=ax2[0])
        # th_qc.T.plot(ax=ax2[1])

        ''' make plots '''
        m=mean_qc.plot.contourf(ax=ax[0], levels=16,
                                           vmin=-30, vmax=30,
                                            cmap=cmap_ppi)
        f=frac_qc.plot.contourf(ax=ax[1], vmin=20,
                                           vmax=100,
                           cmap='inferno',
                                levels=9,)
        ax[0].text(-50, 35, 'repetitions: {}'.format(nreps),
                   ha='left')

        ''' add radials and meridional/zonal lines '''
        locations = [(-x, -x) for x in range(10,30,5)+[35, 40]]
        for a in ax:
            rc = rho.plot.contour(ax=a,levels=range(0, 70, 10),
                             colors='k', alpha=0.6)
            plt.clabel(rc,fmt='%1.0f',manual=locations)

        ax[0].axhline(0, color='k', alpha=0.6)
        ax[0].axvline(0, color='k', alpha=0.6)

        ax[1].axhline(0, color='k', alpha=0.6)
        ax[1].axvline(0, color='k', alpha=0.6)

        m.colorbar.set_label('VR')
        f.colorbar.set_label('FREQ[%]')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        ax[0].set_xlim([-58,30])
        ax[1].set_xlim([-58,30])
        ax[0].set_ylim([-55, 30])
        ax[1].set_ylim([-55, 30])

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')

        return frac_qc

if __name__ == '__main__':

    try:
        xp08
    except NameError:
        xp08 = xta.process(case=[8], params=params)

    try:
        xp09
    except NameError:
        xp09 = xta.process(case=[9], params=params)

    try:
        xp10
    except NameError:
        xp10 = xta.process(case=[10], params=params)

    try:
        xp11
    except NameError:
        xp11 = xta.process(case=[11], params=params)

    try:
        xp12
    except NameError:
        xp12 = xta.process(case=[12], params=params)

    try:
        xp13
    except NameError:
        xp13 = xta.process(case=[13], params=params)

    try:
        xp14
    except NameError:
        xp14 = xta.process(case=[14], params=params)


    # rnd_rhi(regime='tta')
    # rnd_ppi(regime='ntta')

    # rnd_rhi(reps=100, regime='tta')
    # plt.savefig('/Users/raulvalenzuela/random_rhi_tta.png')
    # rnd_rhi(reps=100, regime='ntta')
    # plt.savefig('/Users/raulvalenzuela/random_rhi_ntta.png')

    rnd_ppi(reps=100,regime='tta')
    plt.savefig('/Users/raulvalenzuela/random_ppi_tta.png')
    rnd_ppi(reps=100,regime='ntta')
    plt.savefig('/Users/raulvalenzuela/random_ppi_ntta.png')

    # composite_rhi(regime='tta')
    # plt.savefig('/Users/raulvalenzuela/composite_rhi_tta.png')
    # composite_rhi(regime='ntta')
    # plt.savefig('/Users/raulvalenzuela/composite_rhi_ntta.png')


    # composite_ppi(regime='tta')
    # plt.savefig('/Users/raulvalenzuela/composite_ppi_tta.png')
    # composite_ppi(regime='ntta')
    # plt.savefig('/Users/raulvalenzuela/composite_ppi_ntta.png')

    # regime_ppi(xp09,regime='tta',vmax=40)