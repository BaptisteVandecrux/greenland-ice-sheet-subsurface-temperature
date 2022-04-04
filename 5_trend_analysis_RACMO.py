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


df = pd.read_csv("10m_temperature_dataset_monthly.csv")

# ============ To fix ================

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

df_no_coord = df.loc[np.logical_or(df.latitude.isnull(), df.latitude.isnull()), :]
df = df.loc[~np.logical_or(df.latitude.isnull(), df.latitude.isnull()), :]

df_invalid_depth = df.loc[
    pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]
df = df.loc[
    ~pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]

df_no_elev = df.loc[df.elevation.isnull(), :]
df = df.loc[~df.elevation.isnull(), :]

df["year"] = pd.DatetimeIndex(df.date).year

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

dates = pd.DatetimeIndex(df.date)

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

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
ds_month_latlon = ds_month.set_coords(['lat','lon'])

ds_T10m = xr.open_dataset('predicted_T10m.nc')
T10m_GrIS_p = (ds_T10m.mean(dim=('latitude','longitude'))).to_pandas()


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
T10m_GrIS.plot(ax=ax_bot)
T10m_GrIS_p.plot(ax=ax_bot)

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
    
 # %% 

df = df.sort_values('date')
df['date'] = pd.to_datetime(df.date)

ds_month_latlon = ds_month.rio.reproject("EPSG:4326")
ds_month_latlon['T10m'] = xr.where(ds_month_latlon['T10m']>=0, ds_month_latlon['T10m'], 0)   

# %% checking that the trends match with the ones calculated at the clusters
cluster_list = ['CP1', 'DYE-2', 'Camp Century', 'SwissCamp',
               'Summit',  'KAN-U', 'SouthDome', 'NASA-SE',  'NASA-U', 'Saddle']

for site in cluster_list:
    tmp = df.loc[df.site==site]
    
    plt.figure()
    (ds_month_latlon.T10m.sel(x=tmp.iloc[0,:].longitude,
                              y=tmp.iloc[0,:].latitude, method = 'nearest').to_dataframe().T10m-273.15).plot()
    ds_T10m.sel(longitude = tmp.iloc[0,:].longitude,
                latitude = tmp.iloc[0,:].latitude, method = 'nearest').to_dataframe().T10m.plot()
    tmp.set_index('date').temperatureObserved.plot(linestyle='None',marker='o',markersize=5)
    plt.title(site)
