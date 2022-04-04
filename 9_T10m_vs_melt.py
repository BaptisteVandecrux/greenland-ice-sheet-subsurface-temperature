# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
# import GIS_lib as gis
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

#  find unique locations of measurements
df_all= df[['site','latitude','longitude']]
df_all.latitude = round(df_all.latitude,5)
df_all.longitude = round(df_all.longitude,5)
latlon =np.array([str(lat)+str(lon) for lat, lon in zip(df_all.latitude, df_all.longitude)])
uni, ind = np.unique(latlon, return_index = True)
print(latlon[:3])
print(latlon[np.sort(ind)][:3])
df_all = df_all.iloc[np.sort(ind)]
site_uni, ind = np.unique([str(s) for s in df_all.site], return_index = True)
df_all.iloc[np.sort(ind)].to_csv('coords.csv',index=False)

# %% Extracting closest cell in MAR
print("Extracting closest cell in MAR")


# % loading MAR
for yr in range(1980,2021):
    print(yr)
    # if yr == 1958:
    #     ds_mar = xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.9/MARv3.9-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK', 'SF', 'SH', 'TT', 'ME', 'RU','TI_10M_AVE']]
    # else:
    #     ds_mar = xr.concat((ds_mar,
    #                          xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.9/MARv3.9-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK', 'SF', 'SH', 'TT',  'ME', 'RU','TI_10M_AVE']]),dim='TIME')
        
    if yr == 1980:
        ds_mar = xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK',  'ME', 'RU','TI1']]
    else:
        ds_mar = xr.concat((ds_mar,
                             xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK',  'ME', 'RU','TI1']]),dim='TIME')


def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.TIME.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("TIME.year") / month_length.groupby("TIME.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("TIME.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(TIME="AS").sum(dim="TIME")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(TIME="AS").sum(dim="TIME")

    # Return the weighted average
    return obs_sum / ones_out


ds_mar['T10m'] = ds_mar.TI1.sel(OUTLAY = 10)
T10m_mar_yr = xr.where(ds_mar.isel(TIME=0)['MSK']>75, weighted_temporal_mean(ds_mar, 'T10m'), np.nan)
# T10m_mar_yr = xr.where(ds_mar.isel(TIME=0)['MSK']>75, weighted_temporal_mean(ds_mar, 'TI1'), np.nan)
melt_mar_yr = xr.where(ds_mar.isel(TIME=0)['MSK']>75, ds_mar['ME'].resample(TIME='AS').sum(),       np.nan)
melt_mar_yr = xr.where(ds_mar.isel(TIME=0)['MSK']>75, ds_mar['ME'].resample(TIME='AS').sum(),       np.nan)

T10m_mar_yr.isel(TIME=1).plot()
melt_mar_yr.isel(TIME=1).plot()

#%%
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
params = np.array([1,1])

def funcinv(x, a, b):
    return np.exp(a*x + b)

def residuals(params, x, data):
    # evaluates function given vector of params [a, b]
    # and return residuals: (observed_data - model_data)
    a, b = params
    func_eval = funcinv(x, a, b)
    err = (data - func_eval)
    # err[x<-15] = 0.5*(data[x<-15] - func_eval[x<-15])
    return err

from scipy.stats import kde

y = melt_mar_yr.values.ravel()
x = T10m_mar_yr.values.ravel()
x_sort = np.sort(x)
ind= np.argsort(x)
y_sort = y[ind]
ind_no_nan = ~np.isnan(x_sort+y_sort)
x = x_sort[ind_no_nan]
y = y_sort[ind_no_nan]

res = least_squares(residuals, params, args=(x, y))

nbins=300
ind_sub = np.random.choice(np.arange(0,len(x)), size = 1000)
k = kde.gaussian_kde([x[ind_sub],y[ind_sub]])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
plt.figure()
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto' ,cmap=plt.cm.nipy_spectral)
# plt.plot(x,y,'o',markersize=1, linestyle='None')
xs = np.linspace(-35, -0.001, 1000)
plt.plot(xs, funcinv(xs,res.x[0],res.x[1]), 'w', lw=3)
plt.xlim(-35,0)
plt.ylim(0,1000)
plt.colorbar(label = 'density')
plt.xlabel('10 m subsurface temperature')
plt.ylabel('Annual melt (mm w.e.)')