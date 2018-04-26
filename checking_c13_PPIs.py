
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

elev = 'elev062'

files = glob('/Users/raulvalenzuela/Desktop/'
             'test_xpol/{}/*.cdf'.format(elev))
files.sort()

drop_this=[u'radar_or_data_origin',
 u'project_name',
 u'scientist_name',
 u'submitters_name',
 u'data_source',
 u'volume_header',
 u'start_date',
 u'end_date',
 u'start_time',
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
 u'x',
 u'y',
 u'z',
 u'el',
 u'z_spacing',
 u'field_names',
 u'TIME',
 u'VR',
 u'AZ',
 u'EL']

dss = [xr.open_dataset(ncfile, drop_variables=drop_this)
       for ncfile in files]

ds = xr.concat(dss, dim='time')

ZA = ds.isel(z=1)['ZA']/64.
z = 10**(ZA/10.)
zsum = z.sum(dim='time')

y = np.arange(-58,33.5,0.5)
x = np.arange(-58,45.5,0.5)

zsum.coords['x'] = x
zsum.coords['y'] = y

fig,ax = plt.subplots()
if elev == 'elev010':
    zsum.plot(vmax=0.5e6, ax=ax)
else:
    zsum.isel(x=slice(80,150),
              y=slice(80,150)).plot(vmax=1.5e5, ax=ax)
ax.plot([0, 0], [0, -50], color='r')