
import xpol
import gapflow
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

'''	printing single and mean values
'''
# rhis=xpol.get_data(9,'RHI',225)
# minutes=120
# elevs=rhis['EL']

# # dates, rhis_means=xpol.get_mean(rhis['VR'],minutes=minutes)
# # _,elevs_means=xpol.get_mean(elevs,minutes=minutes)
# # xpol.plot_mean(rhis_means, dates, 'VR', 'rhi', elev=elevs_means)

# dates, rhis_means=xpol.get_mean(rhis['ZA'],name='ZA',minutes=minutes)
# xpol.plot_mean(rhis_means, dates, 'ZA', 'rhi',title='dBZe - Az: 225 SW(negative) - NE(positive)')

'''	gapflow analysis
'''
field='ZA'

case=10
gapf = gapflow.run(case)
gap_beg = gapf.index[0]
if case == 10:
	gap_end=datetime(2003,2,15,23,0,0)
elif case == 8:
	gap_end=datetime(2003,1,12,18,0,0)	
else:
	gap_end = gapf.index[-1]


azimuth=180
rhis=xpol.get_data(case,'RHI',azimuth)
rhi_beg = rhis.index[0]
rhi_end = rhis.index[-1]
elev=rhis['EL']
elev_angle=xpol.get_max(elev)

elevation=0.5
ppis=xpol.get_data(case,'PPI',elevation)
ppi_beg = ppis.index[0]
ppi_end = ppis.index[-1]

print '\nGap flow times:'
print gap_beg
print gap_end
print '\nRHI times:'
print rhi_beg
print rhi_end
print '\nPPI times:'
print ppi_beg
print ppi_end

fig =plt.figure(figsize=(12, 9))
ax1=plt.subplot2grid((3,4), (2,0), colspan=2, gid='ax1')
ax2=plt.subplot2grid((3,4), (2,2), colspan=2, gid='ax2')
ax3=plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2, gid='ax3')
ax4=plt.subplot2grid((3,4), (0,2), colspan=2, rowspan=2, gid='ax4')


' rhi all data '
'******************************'
# rhi_mean=xpol.get_mean(rhis['ZA'],name='ZA')
# xpol.plot(rhi_mean,name='ZA',smode='rhi', title= 'Mean dBZe - All c09')

# rhi_percent=xpol.get_percent(rhis['ZA'])
# xpol.plot(rhi_percent,name='percent',smode='rhi',title='Percentage of good gates - All c09')

' rhi gap flow '
'******************************'
rhi_gap = rhis[field].loc[gap_beg: gap_end]
rhi_nogap = rhis[field].loc[gap_end: rhi_end]
len_rhi_gap=len(rhi_gap)
len_rhi_nogap=len(rhi_nogap)

gap_title='Mean {0} (n={1}) - Gap flow c{2}'
nogap_title='Mean {0} (n={1}) - No gap flow c{2}'

if len_rhi_gap>0:
	rhi_gap_mean=xpol.get_mean(rhi_gap,name=field)
	rhi_gap_percent=xpol.get_percent(rhi_gap)
	xpol.plot(rhi_gap_mean,ax=ax1,name=field,smode='rhi', elev=elev_angle, 
				title= gap_title.format(field, len_rhi_gap, str(case).zfill(2)),colorbar=False)
	# xpol.plot(rhi_gap_percent,name='percent',smode='rhi', title= 'Percentage of good gates - Gap flow c09')

rhi_nogap_mean=xpol.get_mean(rhi_nogap,name=field)
rhi_nogap_percent=xpol.get_percent(rhi_nogap)
xpol.plot(rhi_nogap_mean,ax=ax2,name=field,smode='rhi', elev=elev_angle, 
			title= nogap_title.format(field, len_rhi_nogap, str(case).zfill(2)))
# xpol.plot(rhi_nogap_percent,name='percent',smode='rhi', title= 'Percentage of good gates - No gap flow c09')


' ppi gap flow'
'******************************'
ppi_gap = ppis[field].loc[gap_beg: gap_end]
ppi_nogap = ppis[field].loc[gap_end: ppi_end]
len_ppi_gap=len(ppi_gap)
len_ppi_nogap=len(ppi_nogap)

if len(ppi_gap)>0:
	ppi_gap_mean=xpol.get_mean(ppi_gap,name=field)
	ppi_gap_percent=xpol.get_percent(ppi_gap)
	xpol.plot(ppi_gap_mean,ax=ax3,name=field,smode='ppi', add_azline=azimuth, 
				title= gap_title.format(field, len_ppi_gap, str(case).zfill(2)),colorbar=False)
	# xpol.plot(ppi_gap_percent,name='percent',smode='ppi', title= 'Percentage of good gates - Gap flow c09')

ppi_nogap_mean=xpol.get_mean(ppi_nogap,name=field)
ppi_nogap_percent=xpol.get_percent(ppi_nogap)
xpol.plot(ppi_nogap_mean,ax=ax4,name=field,smode='ppi', add_azline=azimuth, 
			title= nogap_title.format(field, len_ppi_nogap, str(case).zfill(2)))
# xpol.plot(ppi_nogap_percent,name='percent',smode='ppi', title= 'Percentage of good gates - No gap flow c09')

' add dates '
'******************************'
box=dict(boxstyle='round', fc='white')
timetxt='Year: '+gap_beg.strftime('%Y')+'\n'
timetxt=timetxt+'Beg: '+gap_beg.strftime('%b-%d %H:%M')+'\n'
timetxt=timetxt+'End: '+gap_end.strftime('%b-%d %H:%M')
ax3.text(0.65,0.8,timetxt,bbox=box, transform=ax3.transAxes)
timetxt='Beg: '+gap_end.strftime('%b-%d %H:%M')+'\n'
timetxt=timetxt+'End: '+ppi_end.strftime('%b-%d %H:%M')
ax4.text(0.65,0.8,timetxt,bbox=box, transform=ax4.transAxes)