'''
	Processing of NOAA XPOL in cartesian coordinates

	Raul Valenzuela
	January, 2016
'''


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from glob import glob
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

basedir='/home/rvalenzuela/XPOL/netcdf/c09/PPI/elev005/'
def get_data():
	xfiles=glob(basedir+'*.cdf')
	xfiles.sort()

	za_arrays=[]
	vr_arrays=[]
	time_index=[]

	for x in xfiles:
		data=Dataset(x,'r')
		scale=64.
		VR=np.squeeze(data.variables['VR'][0,1,:,:])/scale
		ZA=np.squeeze(data.variables['ZA'][0,1,:,:])/scale
		date = ''.join(data.variables['start_date'][:]).replace('/01','/20')
		time = ''.join(data.variables['start_time'][:])
		raw_date=date+' '+time
		raw_fmt='%m/%d/%Y %H:%M:%S'
		dt = datetime.strptime(raw_date,raw_fmt)
		
		VR=np.ma.filled(VR,fill_value=np.nan)
		ZA=np.ma.filled(ZA,fill_value=np.nan)
		
		vr_arrays.append(VR)
		za_arrays.append(ZA)
		time_index.append(dt)
		data.close()
	
	df = pd.DataFrame(index=time_index,columns=['ZA','VR'])
	df['ZA']=za_arrays
	df['VR']=vr_arrays
	return df

def get_mean(arrays,minutes=60, name=None):

	g = pd.TimeGrouper(str(minutes)+'T')
	G = arrays.groupby(g)

	gindex = G.indices.items()
	gindex.sort()
	mean=[]
	dates=[]
	for gx in gindex:
		gr = arrays.ix[gx[1]].values
		a=gr[0]
		if name=='ZA':
			a=np.power(10,a/10.)
		for g in gr[1:]:
			a=np.dstack((a,g))			
		m = np.nanmean(a,axis=2)
		if name=='ZA':
			m=10*np.log10(m)
		mean.append(m)
		dates.append(gx[0])

	return dates, np.array(mean)

def plot(array,ax=None,show=True,name=None):

	if not ax:
		fig,ax=plt.subplots()

	if name=='ZA':
		vmin,vmax = [5,45]
		cmap='nipy_spectral'
	elif name=='VR':
		vmin,vmax = [-15,15]
		cmap='RdBu'

	im=ax.imshow(array,interpolation='none', origin='lower',
					vmin=vmin,vmax=vmax,cmap=cmap)
	add_colorbar(ax,im)
	plt.draw()
	if show:
		plt.show(block=False)

def plot_mean(means,dates,name):

	xpolmean=PdfPages('xpol_mean.pdf')
	ntimes,_,_=means.shape
	for n in range(ntimes):
		fig,ax = plt.subplots()
		plot(means[n,:,:],ax=ax, show=False,name=name)
		print dates[n]
		plt.suptitle(dates[n])
		xpolmean.savefig()
		plt.close('all')
	xpolmean.close()

def add_colorbar(ax,im):
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.05)
	cbar = plt.colorbar(im, cax=cax)