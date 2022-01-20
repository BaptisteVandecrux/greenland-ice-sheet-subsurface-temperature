# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
import progressbar
import matplotlib.pyplot as plt
import firn_temp_lib as ftl
import time
import xarray as xr
import time
import geopandas as gpd
from datetime import datetime as dt
import time
from rasterio.crs import CRS
from  scipy import stats, signal #Required for detrending data and computing regression
import statsmodels.api as sm

def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

    return date.year + fraction
land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

ds = xr.open_dataset("C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc")

ds_month = ds.resample(time = 'M').mean()



def calculate_trend(ds_month, year_start,year_end):
    ds_month = ds_month.loc[dict(time=slice(str(year_start)+"-01-01", str(year_end)+"-12-31"))]
    
    vals = ds_month['T10m'].values 
    time = np.array([toYearFraction(d) for d in pd.to_datetime(ds_month.time.values)])
    # Reshape to an array with as many rows as years and as many columns as there are pixels
    vals2 = vals.reshape(len(time), -1)
    # Do a first-degree polyfit
    regressions = np.nan * vals2[:2,:]
    ind_nan = np.all(np.isnan(vals2),axis=0)
    regressions[:,~ind_nan] = np.polyfit(time, vals2[:,~ind_nan], 1)
    # Get the coefficients back
    trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
    return (['rlat', 'rlon'],  trends)

year_ranges = np.array([[1990, 2021],[1960, 2020], [1996, 2013], [2014,2020]])
for i in range(4):
    ds_month["trend_"+str(i)] = calculate_trend(ds_month, year_ranges[i][0],year_ranges[i][1])

T10m_GrIS = (ds.T10m.mean(dim=('rlat','rlon'))-273.15).to_pandas()

# reprojecting
crs_racmo = CRS.from_proj4('-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +o_lon_p=0  +lon_0=-37.5')
ds_month = ds_month.rio.write_crs(crs_racmo)
ds_month = ds_month.rio.reproject("EPSG:3413")

#%%

import rioxarray # for the extension to load
import xarray
import rasterio
ABC = 'ABCDEF'
fig, ax = plt.subplots(2,2,figsize=(7,20))
plt.subplots_adjust(hspace = 0.1,wspace=0.1, right=0.8, left = 0, bottom=0.05, top = 0.8)
ax = ax.flatten()

for i in range(4):
    land.plot(ax=ax[i], zorder=0, color="black")

    if i in [0,1]:
        vmin = -0.1; vmax = 0.1
    else:
        vmin = -0.5; vmax = 0.5
        
    im = ds_month['trend_'+str(i)].plot(ax=ax[i],vmin = vmin, vmax = vmax,
                                        cmap='coolwarm',add_colorbar=False)
                
    ax[i].set_title(ABC[i+1]+'. '+str(year_ranges[i][0])+'-'+str(year_ranges[i][1]),fontweight = 'bold')
    if i in [1,3]:
        if i == 1:
            cbar_ax = fig.add_axes([0.85, 0.07, 0.03, 0.3])
        else:
            cbar_ax = fig.add_axes([0.85, 0.47, 0.03, 0.3])
        cb = plt.colorbar(im, ax = ax[i], cax=cbar_ax)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label('Trend in 10 m subsurface temperature ($^o$C yr $^{-1}$)', rotation=270)

    ax[i].set_xlim(-700000.0, 900000.0)
    ax[i].set_ylim(-3400000, -600000)
    ax[i].set_axis_off()
ax_bot = fig.add_axes([0.2, 0.85, 0.75, 0.12])
ax_bot.set_title(ABC[0]+'. Ice-sheet-wide average',fontweight = 'bold')
ax_bot.plot(ds.time, T10m_GrIS)
for r in range(4):
    tmp = T10m_GrIS.loc[str (year_ranges[r,0]):str (year_ranges[r,1])]
    X = np.array([toYearFraction(d) for d in tmp.index])
    y = tmp.values
    
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    x_plot = np.array([np.nanmin(X), np.nanmax(X)])
    if r ==1:
        linestyle = '--'
    else:
        linestyle = '-'
        
    ax_bot.plot([tmp.index.min(), tmp.index.max()], 
                est2.params[0] + est2.params[1] * x_plot,
                color = 'tab:red',
                linestyle=linestyle)
    print('slope (p-value): %0.3f (%0.3f)'% (est2.params[1], est2.pvalues[1]))
    
ax_bot.autoscale(enable=True, axis='x', tight=True)
ax_bot.set_ylabel('10 m subsurface \ntemperature ($^o$C yr $^{-1}$)')
fig.savefig( 'figures/RACMO_trend_map.png',dpi=300)

# %% 
fig, ax_bot = plt.subplots(1,1)
ax_bot.set_title(ABC[0]+'. Ice-sheet-wide average',fontweight = 'bold')
T10m_GrIS.plot()
for r in range(4):
    tmp = T10m_GrIS.loc[str (year_ranges[r,0]):str (year_ranges[r,1])]
    X = np.array([toYearFraction(d) for d in tmp.index])
    y = tmp.values
    
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    x_plot = np.array([np.nanmin(X), np.nanmax(X)])
    if r ==1:
        linestyle = '--'
    else:
        linestyle = '-'
        
    ax_bot.plot([tmp.index.min(), tmp.index.max()], 
                est2.params[0] + est2.params[1] * x_plot,
                color = 'tab:red',
                linestyle=linestyle)
    print('slope (p-value): %0.3f (%0.3f)'% (est2.params[1], est2.pvalues[1]))
    
    

# %% checking that the trends match with the ones calculated at the clusters
cluster_loc = [['CP1', [251, 136]],
    ['T1', [249, 128]],
    ['DYE-2', [182, 131]],
    ['Camp Century', [412,  99]],
    ['Crête', [270, 202]],
    ['SwissCamp', [247, 119]],
    ['Summit', [299, 195]],
    ['KAN-U', [194, 127]],
    ['SouthDome', [115, 134]],
    ['NASA-SE', [179, 161]],
    ['NASA-U', [331, 134]],
    ['Saddle', [171, 144]]]

for i in range(12):
    print(cluster_loc[i][0],
          '\n%0.3f \n%0.3f \n%0.3f \n%0.3f'% tuple([ds_month['trend_'+str(t)][dict(rlon=int(cluster_loc[i][1][1]),
                                                rlat=int(cluster_loc[i][1][0]))].values.tolist() for t in range(4)]))
    
for i in range(12):
    print(cluster_loc[i][0],
          ' %0.3f %0.3f'% tuple([ds_month[var][dict(time = 0,
                                              rlon=int(cluster_loc[i][1][1]),
                                              rlat=int(cluster_loc[i][1][0]))].values.tolist() for var in ['lat', 'lon']]))
    