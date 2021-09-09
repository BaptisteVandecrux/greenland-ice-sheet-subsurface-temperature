# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd 
# import GIS_lib as gis 

df = pd.read_csv('subsurface_temperature_summary.csv')

# ============ To fix ================

df_ambiguous_date = df.loc[pd.to_datetime(df.date,errors='coerce').isnull(),:]
df = df.loc[~pd.to_datetime(df.date,errors='coerce').isnull(),:]

df_bad_long = df.loc[df.longitude>0,:]
df['longitude'] = - df.longitude.abs().values

df_no_coord = df.loc[np.logical_or(df.latitude.isnull(), df.latitude.isnull()),:]
df = df.loc[~np.logical_or(df.latitude.isnull(), df.latitude.isnull()),:]

df_invalid_depth =  df.loc[pd.to_numeric(df.depthOfTemperatureObservation,errors='coerce').isnull(),:]
df = df.loc[~pd.to_numeric(df.depthOfTemperatureObservation,errors='coerce').isnull(),:]

df_no_elev =  df.loc[df.elevation.isnull(),:]
df = df.loc[~df.elevation.isnull(),:]

df['year'] = pd.DatetimeIndex(df.date).year

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude)).set_crs(4326).to_crs(3413)

land = gpd.GeoDataFrame.from_file('Data/misc/Ice-free_GEUS_GIMP.shp')
land = land.to_crs('EPSG:3413')

ice = gpd.GeoDataFrame.from_file('Data/misc/IcePolygon_3413.shp')
ice = ice.to_crs('EPSG:3413')

# %% Map
fig, ax = plt.subplots(1,1,figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0,top=1,bottom=0,left=0,right=1)
land.plot(ax = ax, zorder=0,color='black')
ice.plot(ax = ax, zorder=1,color='lightblue')
gdf.plot(ax = ax, column='year',  cmap = 'tab20c' , markersize=50, 
         edgecolor = 'gray', legend = True,
         legend_kwds={'label': 'Year of measurement', 
                      "orientation": "horizontal",
                      'shrink': 0.8})

plt.axis('off')
plt.savefig('figures/fig1_map.png')

# %% 
df_egig = df.loc[np.logical_and(df.latitude < 69,df.latitude < 90),:]
# df = df.loc[df.longitude <-36.5,:]

year = df_egig.latitude.values * np.nan
for i, x in enumerate(pd.to_datetime(df_egig.date)):
    try:
        year[i] = x.year
    except:
        year[i] = x
ind = np.logical_or(df.site == 'Eismitte',df.site == '200 km Randabstand')


plt.figure()
sc = plt.scatter(df_egig.elevation,
                 df_egig.temperatureObserved,
                 150, year,
                 cmap = 'tab20')
plt.scatter(df_egig.elevation.loc[ind],
                 df_egig.temperatureObserved.loc[ind],
                 200, year[ind],
                 cmap = 'tab20')
for i, txt in enumerate(df.reference):
    plt.annotate(#str(txt), #
                 txt.split(',')[0]+' '+str(np.floor(year[i])), 
                  (df_egig.elevation.iloc[i],
                  df_egig.temperatureObserved.iloc[i]))
plt.colorbar(sc, label='Year')

plt.ylabel('10 m firn temperature (deg C)')
plt.xlabel('Elevation (m a.s.l.)')
plt.savefig('figures/EGIG.png')

#%% Loading accumulation
print('Extracting data from rasters')

import GIS_lib as gis
   
    # Extracting SICE grain diameter and albedo

    
df['avg_snowfall_mm'] = gis.sample_raster_with_geopandas(gdf.to_crs('epsg:3413'),  
                        'Data/misc/Net_Snowfall_avg_1979-2014_MAR_3413.tif', 'c_avg')

# %% Trend analysis    
from sklearn import linear_model
X = df[['elevation','latitude']]
Y = df['temperatureObserved']
dates = pd.DatetimeIndex(df.date)
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

xx,yy = np.meshgrid(np.arange(0, 3400,100),np.arange(60,83,1)) #,np.arange(0,1600,10))
z = regr.intercept_ + regr.coef_[0]*xx + regr.coef_[1]*yy #+ regr.coef_[2]*zz

fig = plt.figure()
ax = fig.add_subplot(3,1, 1)
ax.scatter(df.elevation, df.temperatureObserved)
ax.set_xlabel('Elevation (m a.s.l.)')
ax.set_ylabel('10 m firn temperature')

ax = fig.add_subplot(3, 1, 2)
ax.scatter(df.latitude, df.temperatureObserved)
ax.set_xlabel('Latitude (deg N)')
ax.set_ylabel('10 m firn temperature')

ax = fig.add_subplot(3,1,3)
ax.scatter(df.avg_snowfall_mm, df.temperatureObserved)

ax.set_xlabel('Snow accumulation (mm)')
ax.set_ylabel('10 m firn temperature')
plt.show()
plt.savefig('figures/regression_lat_elev_accum.png')

fig = plt.figure()
ax = fig.add_subplot(1,1, 1)
ax.scatter(dates.month+dates.day/31, df.temperatureObserved)
ax.set_xlabel('Month')
ax.set_ylabel('10 m firn temperature')

#% Trend  fit
df['temperature_anomaly'] =  df.temperatureObserved - regr.intercept_ - regr.coef_[0]*df.elevation - regr.coef_[1]*df.latitude #- regr.coef_[2]*df.avg_snowfall_mm
y_data = df.temperature_anomaly.values[dates.year>1990]
x = dates[dates.year>1990].year + (dates[dates.year>1990].month +dates[dates.year>1990].day/31)/12
dec_year = dates.year + (dates.month +dates.day/31)/12
curve_fit = np.polyfit(x,y_data, 1)
y = curve_fit[1] + curve_fit[0]*x

plt.figure()
plt.plot(dec_year,df['temperature_anomaly'],'.')
plt.plot(x,y,'--',linewidth=3)
plt.xlabel('Year')
plt.ylabel('10 m firn temperature anomaly')
plt.savefig('figures/trend_time.png')

plt.figure()
plt.plot(df.avg_snowfall_mm,df['temperature_anomaly'],'.')
plt.xlabel('Average snowfall (mm)')
plt.ylabel('10 m firn temperature anomaly')
plt.savefig('figures/trend_accum.png')





