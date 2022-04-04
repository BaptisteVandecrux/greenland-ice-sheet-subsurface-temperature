# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from scipy.spatial.distance import cdist

# import GIS_lib as gis
import progressbar
import matplotlib


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

# Extracting ERA5 data

ds_era = xr.open_dataset('Data/ERA5/ERA5_monthly_temp_snowfall.nc')

     
# def weighted_temporal_mean(ds, var):
#     """
#     weight by days in each month
#     """
#     # Determine the month length
#     month_length = ds.time.dt.days_in_month

#     # Calculate the weights
#     wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

#     # Make sure the weights in each year add up to 1
#     np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

#     # Subset our dataset for our variable
#     obs = ds[var]

#     # Setup our masking for nan values
#     cond = obs.isnull()
#     ones = xr.where(cond, 0.0, 1.0)

#     # Calculate the numerator
#     obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

#     # Calculate the denominator
#     ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

#     # Return the weighted average
#     return obs_sum / ones_out

# ds_era_yr = weighted_temporal_mean(ds_era, 't2m').to_dataset(name='t2m')
# ds_era_yr['sf'] = ds_era['sf'].resample(time='AS').sum()

# fig,ax = plt.subplots(1,2)
# ds_era_yr.isel(time=1).t2m.plot(ax=ax[0])
# ds_era_yr.isel(time=1).sf.plot(ax=ax[1])

# %% 
time_era = ds_era.time

for i in range(6):
    df['t2m_'+str(i)] = np.nan
    df['sf_'+str(i)] = np.nan

for i in progressbar.progressbar(range(df.shape[0])):
    tmp = ds_era.sel(latitude = df.iloc[i,:].latitude, 
                  longitude = df.iloc[i,:].longitude, method= 'nearest')
    for k in range(6):
        time_end = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=0-k)
        time_start = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=-1-k)
        df.iloc[i,df.columns.get_loc('t2m_'+str(k))] = tmp.sel(time=slice(time_start, time_end)).t2m.mean().values
        df.iloc[i,df.columns.get_loc('sf_'+str(k))] =tmp.sel(time=slice(time_start, time_end)).sf.sum().values
        
df['t2m_avg'] = df[['t2m_'+str(i) for i in range(6)]].mean(axis = 1)
df['sf_avg'] = df[['sf_'+str(i) for i in range(6)]].mean(axis = 1)

#%% plot
fig,ax = plt.subplots(2,2)
ax = ax.flatten()
ax[0].plot(df.t2m_0, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[1].plot(df.sf_0, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[2].plot(df.t2m_avg, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[3].plot(df.sf_avg, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)


df['date'] = pd.to_datetime(df.date)
df['year'] = df['date'].dt.year
df['month'] = (df['date'].dt.month-1)/12

df = df.loc[df.t2m_avg.notnull(),:]
df = df.loc[df.sf_avg.notnull(),:]
for i in range(6):
    df = df.loc[df['sf_'+str(i)].notnull(),:]
    df = df.loc[df['t2m_'+str(i)].notnull(),:]
# %% MPM
from scipy.optimize import curve_fit

def func(X, a, b, c, d, e, f, g):
    return (a * X[0]**2 + b * X[0]) * (c * X[1]**2 + d * X[1]) * np.cos(e * X[2]*2*np.pi + f) + g

X = np.transpose(df[['t2m_avg','sf_avg','month']].values)
y =  np.transpose(df['temperatureObserved'].values)

popt, pcov = curve_fit(func, X, y)


df = df.sort_values('date')
def T10m_model(df):
    X = np.transpose(df[['t2m_avg','sf_avg','month']].values)
    Y = func(X, *popt)
    Predictions = pd.DataFrame(data =np.transpose(Y), index = df.index, columns=['temperaturePredicted']).temperaturePredicted
    Predictions.loc[Predictions>0] = 0
    return Predictions
    
plt.figure()
plt.plot(df.temperatureObserved,
         T10m_model(df),
         'o',linestyle= 'None')
ME = np.mean(df.temperatureObserved - T10m_model(df))
RMSE = np.sqrt(np.mean((df.temperatureObserved - T10m_model(df))**2))
plt.title('N = %i ME = %0.2f RMSE = %0.2f'%(len(df.temperatureObserved), ME, RMSE))

#%%
for site in ['DYE-2', 'Summit', 'KAN_U','EKT']:
    df_select = df.loc[df.site ==site,:]
    plt.figure()
    plt.plot(df_select.date,df_select.temperatureObserved, 'o', linestyle='None')
    plt.plot(df_select.date, T10m_model(df_select), 'o-')
    plt.title(site)
    
    
    

