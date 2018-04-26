

from figs_to_frames import figs_to_frames

from make_xarray import make_xarray_ppi
import xpol_tta_analysis as xta

params = dict(wdir_thres=150,
              rain_czd=0.25,
              nhours=2
              )

xp13 = xta.process(case=[13], params=params)

da = make_xarray_ppi(xp13)
da = da['ZA']

figs_to_frames(da)