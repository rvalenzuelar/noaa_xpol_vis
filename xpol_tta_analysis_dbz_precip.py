import Windprof2 as wp 
import xpol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

reload(xpol)

sns.reset_orig()

setcase={	 8:[0.5, 180, 10],
			 9:[0.5, 180,  100],
		    10:[0.5, 180, 300],
		    11:[0.5, 180, 60],
		    12:[0.5,   6, 50],
		    13:[0.5, 180, 100],
		    14:[0.5, 180, 110]}

closeall=True

homedir=os.path.expanduser('~')

for case in range(8,15):

	elevation,azimuth,maxv =setcase[case]

	tta_times=wp.get_tta_times(case=str(case))
	print tta_times

	' PPIs'
	'***********************************************************************************'
	ppis=xpol.get_data(case,'PPI',elevation,homedir=homedir)

	tta_idxs=np.asarray([],dtype=int)
	for time in tta_times:
		idx=np.where((ppis.index.day == time.day) & (ppis.index.hour == time.hour))[0]
		if idx.size>0:
			tta_idxs=np.append(tta_idxs,idx)
	notta_idxs=np.delete(np.arange(len(ppis.index)), tta_idxs)

	fig,axes=plt.subplots(2,2,figsize=(11,10.5), sharex=True, sharey=True)
	ax=axes.flatten()

	ppis_tta=ppis.iloc[tta_idxs]
	
	if ppis_tta.size>0:
		print 'TTA'

		dbz_precip = xpol.get_dbz_precip_accum(ppis_tta['ZA'])
		xpol.plot(dbz_precip, ax=ax[2], name='rainrate',smode='ppi', colorbar=False, case=case, 
				   vmax=maxv)		
		
		ppi_tta_mean,good=xpol.get_mean(ppis_tta['VR'],name='VR')
		xpol.plot(ppi_tta_mean, ax=ax[0], name='VR',smode='ppi', colorbar=False, case=case)

		n=' (n='+str(ppis_tta.index.size)+', good='+str(good)+')'
	else:
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		ax[0].text(0.5,0.5, 'NO DATA', transform=ax[0].transAxes, ha='center', weight='bold')
		ax[2].text(0.5,0.5, 'NO DATA', transform=ax[2].transAxes, ha='center', weight='bold')
		n=''

	ax[0].set_title('TTA average'+n)

	ppis_notta=ppis.iloc[notta_idxs]
	
	if ppis_notta.size>0:
		print 'NO TTA'

		dbz_precip = xpol.get_dbz_precip_accum(ppis_notta['ZA'])
		xpol.plot(dbz_precip, ax=ax[3], name='rainrate',smode='ppi', colorbar=True, case=case, 
				    vmax=maxv)
		
		ppi_notta_mean,good = xpol.get_mean(ppis_notta['VR'],name='VR')
		xpol.plot(ppi_notta_mean,ax=ax[1],  name='VR',smode='ppi', colorbar=True, case=case)

		n = ' (n='+str(ppis_notta.index.size)+', good='+str(good)+')'

	ax[1].set_title('NO-TTA average'+n)

	t='XPOL Time Average - C{} {} - {} - Nscan={}\nBeg: {} - End: {} (UTC)'
	ym=ppis.index[0].strftime('%Y-%b')
	beg=ppis.index[0].strftime('%d %H:%M')
	end=ppis.index[-1].strftime('%d %H:%M')
	title=t.format(str(case).zfill(2), ym, ppis.index.name, ppis.index.size, beg, end)
	fig.suptitle(title, fontsize=14)
	plt.subplots_adjust(hspace=0.05, wspace=0.1,left=0.05, right=0.95,bottom=0.05)

	if closeall:
		o='c{}_xpol_tta_average_{}_{}.pdf'
		saveas=o.format(str(case).zfill(2), 'ppi', str(elevation*10).zfill(3))
		xpolpdf = PdfPages(saveas)
		xpolpdf.savefig()
		xpolpdf.close()


	' RHIs'
	'***********************************************************************************'
	rhis=xpol.get_data(case,'RHI',azimuth,homedir=homedir)
	elev=xpol.get_max(rhis['EL'])

	tta_idxs=np.asarray([],dtype=int)
	for time in tta_times:
		idx=np.where((rhis.index.day == time.day) & (rhis.index.hour == time.hour))[0]
		if idx.size>0:
			tta_idxs=np.append(tta_idxs,idx)
	notta_idxs=np.delete(np.arange(len(rhis.index)), tta_idxs)

	fig,axes=plt.subplots(2,2,figsize=(14,8), sharex=True, sharey=True)
	ax=axes.flatten()

	rhis_notta=rhis.iloc[notta_idxs]
	if rhis_notta.size>0:
		print 'NO TTA'

		dbz_precip = xpol.get_dbz_precip_accum(rhis_notta['ZA'])
		xpol.plot(dbz_precip, ax=ax[3], name='rainrate',smode='rhi', colorbar=True, case=case, 
				   vmax=maxv)

		rhi_notta_mean, good=xpol.get_mean(rhis_notta['VR'],name='VR')
		xpol.plot(rhi_notta_mean,ax=ax[1],  name='VR',smode='rhi', colorbar=True, case=case, elev=elev)
		n=' (n='+str(rhis_notta.index.size)+', good='+str(good)+')'
	ax[1].set_title('NO-TTA average'+n)

	rhis_tta=rhis.iloc[tta_idxs]
	if rhis_tta.size>0:
		print 'TTA'

		dbz_precip = xpol.get_dbz_precip_accum(rhis_tta['ZA'])
		xpol.plot(dbz_precip, ax=ax[2], name='rainrate',smode='rhi', colorbar=False, case=case, 
				   vmax=maxv)

		rhi_tta_mean, good=xpol.get_mean(rhis_tta['VR'],name='VR')
		xpol.plot(rhi_tta_mean, ax=ax[0], name='VR',smode='rhi', colorbar=False, case=case, 
					elev=elev, add_yticklabs=True)
		n=' (n='+str(rhis_tta.index.size)+', good='+str(good)+')'
	else:
		xticks=ax[1].get_xticks()
		yticks=ax[1].get_yticks()
		ax[0].set_xticks(xticks)
		ax[0].set_yticks(yticks)
		ax[0].set_xticklabels([str(int(x)) for x in xticks])
		ax[0].set_yticklabels([str(y) for y in yticks])
		ax[0].text(0.5,0.5, 'NO DATA', transform=ax[0].transAxes, ha='center', weight='bold')
		ax[2].text(0.5,0.5, 'NO DATA', transform=ax[2].transAxes, ha='center', weight='bold')
		n=''
	ax[0].set_title('TTA average'+n)
	ax[0].set_ylabel('Altitude MSL [km]', fontsize=14)
	ax[2].set_xlabel('Distance from the radar [km]', fontsize=14)

	t='XPOL Time Average - C{} {} - {} - Nscan={}\nBeg: {} - End: {} (UTC)'
	ym=rhis.index[0].strftime('%Y-%b')
	beg=rhis.index[0].strftime('%d %H:%M')
	end=rhis.index[-1].strftime('%d %H:%M')
	title=t.format(str(case).zfill(2), ym, rhis.index.name, rhis.index.size, beg, end)
	fig.suptitle(title, fontsize=14)
	plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.97)

	if closeall:
		o='c{}_xpol_tta_average_{}_{}.pdf'
		saveas=o.format(str(case).zfill(2), 'rhi', str(azimuth).zfill(3))
		xpolpdf = PdfPages(saveas)
		xpolpdf.savefig()
		xpolpdf.close()

	if closeall:
		plt.close('all')
	else:		
		plt.show(block=False)