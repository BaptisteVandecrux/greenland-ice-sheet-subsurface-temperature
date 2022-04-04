# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import pandas as pd
import numpy as np
import progressbar
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from datetime import datetime as dt


# loading temperature dataset
df = pd.read_csv("10m_temperature_dataset_monthly.csv")

df = df.loc[np.isin(df.reference_short, ['Steffen et al. (1996)', 'PROMICE', 'GC-Net']),:]
df = df.loc[np.isin(df.site, ['Summit', 
                              'CP1', 'NASA-U', 'TUNU-N', 'DYE-2', 'Saddle', 'SouthDome',
                              'NASA-E', 'NASA-SE']),:]

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

df_hans_tausen = df.loc[df.latitude > 82, :]
df = df.loc[~(df.latitude > 82), :]

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

df = (gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude)
                        ).set_crs(4326).to_crs(3413))
df['x_3413'] = df.geometry.x
df['y_3413'] = df.geometry.y


# % loading ERA T2m
for yr in range(1950,2020,6):
    print(yr)
    if yr == 1950:
        ds_era = xr.open_dataset('Data/ERA5/m_era5_t2m_'+str(yr)+'_'+str(yr+5)+'.nc')
    else:
        ds_era = xr.concat((ds_era,
                             xr.open_dataset('Data/ERA5/m_era5_t2m_'+str(yr)+'_'+str(yr+5)+'.nc')),'time')
   
# reprojecting
ds_era = ds_era.rio.write_crs("EPSG:4326")
ds_era = ds_era.rio.reproject("EPSG:3413")

# adding lat and lon as vatriable
ny, nx = len(ds_era['y']), len(ds_era['x'])
x, y = np.meshgrid(ds_era['x'], ds_era['y'])

# Rasterio works with 1D arrays
from rasterio.warp import transform
lon, lat = transform( {'init': 'EPSG:3413'}, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())
lon = np.asarray(lon).reshape((ny, nx))
lat = np.asarray(lat).reshape((ny, nx))
ds_era.coords['lon'] = (('y', 'x'), lon)
ds_era.coords['lat'] = (('y', 'x'), lat)

# % Extract ERA time series

df_era = pd.DataFrame()
time_ERA = ds_era["time"].values
df = df.reset_index(drop=True)

points = [str(x)+str(y) for x, y in zip(df.x_3413, df.y_3413)]
_, ind_list = np.unique(points, return_index =True) 
ind_list = np.sort(ind_list)
          
for ind in progressbar.progressbar(ind_list):
    df_point = (
        ds_era.t2m.sel(x = df.loc[ind].x_3413,
                       y = df.loc[ind].y_3413,
                       method = 'nearest',
                       ) - 273.15
    ).to_dataframe()
    df_point['ind'] = ind
    df_point=df_point.set_index(['ind','lat','lon'],append=True)
    df_point['lat_obs'] = df.loc[ind].latitude
    df_point['lon_obs'] = df.loc[ind].longitude
    df_era = df_era.append(df_point[['lat_obs','lon_obs','t2m']])
df_era = df_era.reset_index('lat').reset_index('lon')

# % Comparison of t2m and t10m: final comparison
firn_memory_time = 5
shift = 1
x=[]
y=[]
df['era_avg_t2m'] = np.nan
for ind_loc in progressbar.progressbar(df_era.index.get_level_values(1).unique()):
    lat = df_era.xs(ind_loc, level=1).lat_obs[0]
    lon = df_era.xs(ind_loc, level=1).lon_obs[0]
    df_loc = df.loc[(df.latitude == lat)&(df.longitude == lon),:].copy()
    for ind_meas in df_loc.index:
        date = pd.to_datetime(df.loc[ind].date)- pd.DateOffset(years=shift)
        date0 = date- pd.DateOffset(years=firn_memory_time)- pd.DateOffset(years=shift)
        
        x.append(df_era.xs(ind_loc, level=1).loc[date0:date,'t2m'].mean())
        df.loc[ind_meas,'era_avg_t2m'] = df_era.xs(ind_loc, level=1).loc[date0:date,'t2m'].mean()
        
        y.append(df.loc[ind_meas].temperatureObserved)

x = np.array(x)
y = np.array(y)
df_save = df


fig = plt.figure()
plt.plot(df.era_avg_t2m.values, df.temperatureObserved.values, 'o',markersize=1)
plt.plot([-30, 0], [-30, 0], color='black')
plt.title(str(shift)+' years shift. RMSE = %0.3f $^o$ C'%np.sqrt(np.nanmean((df.era_avg_t2m.values- df.temperatureObserved.values)**2)))
plt.xlabel('ERA5 average 2 m air temperature ($^o$ C)')
plt.ylabel('Observed 10 m firn temperature ($^o$ C)')
fig.savefig(str(shift)+'_years_shift_5_year_memory_air_firn_temp.png') 

df['era_dev'] = df.temperatureObserved - df.era_avg_t2m
df_save['era_dev'] = df_save.temperatureObserved - df_save.era_avg_t2m

# %% 
