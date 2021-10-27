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
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd 
# import GIS_lib as gis 
matplotlib.rcParams.update({'font.size': 16})
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

print('Extracting data from rasters')
import GIS_lib as gis    
df['avg_snowfall_mm'] = gis.sample_raster_with_geopandas(gdf.to_crs('epsg:3413'),  
                        'Data/misc/Net_Snowfall_avg_1979-2014_MAR_3413.tif', 'c_avg')
df.date = pd.to_datetime(df.date)
df = df.set_index('date',drop=False)
df_save = df

# %% Map
fig, ax = plt.subplots(1,1,figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0,top=1,bottom=0,left=0,right=1)
land.plot(ax = ax, zorder=0,color='black')
ice.plot(ax = ax, zorder=1,color='lightblue')
gdf.plot(ax = ax, column='year',  cmap = 'tab20c' , markersize=30, 
         edgecolor = 'gray', legend = True,
         legend_kwds={'label': 'Year of measurement', 
                      "orientation": "horizontal",
                      'shrink': 0.8})

plt.axis('off')
plt.savefig('figures/fig1_map.png', dpi=300)

# %% 
df_select = df.loc[np.logical_and(df.latitude < 69,df.latitude < 90),:]
# df = df.loc[df.longitude <-36.5,:]

year = df_select.latitude.values * np.nan
for i, x in enumerate(pd.to_datetime(df_select.date)):
    try:
        year[i] = x.year
    except:
        year[i] = x
ind = np.logical_or(df_select.site == 'Eismitte',df_select.site == '200 km Randabstand')


plt.figure()
sc = plt.scatter(df_select.elevation,
                 df_select.temperatureObserved,
                 150, year,
                 cmap = 'tab20')
plt.scatter(df_select.elevation.loc[ind],
                 df_select.temperatureObserved.loc[ind],
                 200, year[ind],
                 cmap = 'tab20')
for i, txt in enumerate(df_select.site.unique()):
    print(txt,df_select.loc[df_select.site==txt,'elevation'].mean())
    plt.annotate(#str(txt), #
                 txt, 
                  (df_select.loc[df_select.site==txt,'elevation'].mean(),
                  df_select.loc[df_select.site==txt,'temperatureObserved'].mean()))
plt.colorbar(sc, label='Year')

plt.ylabel('10 m firn temperature (deg C)')
plt.xlabel('Elevation (m a.s.l.)')
plt.savefig('figures/EGIG.png')

# %% Model lat_elev_poly 
import itertools
from matplotlib import cm
df = df_save
site_remove = ['SouthDome','SDM','SE-Dome','H2', 'H3', 'FA_13', 'FA_15_1', 'FA_15_2']
df = df.loc[~np.isin(df.site,site_remove)]
elevation_bins = np.arange(0,3300,100)
latitude_bins = np.arange(60,85,5)

X = df[['elevation','latitude']]
Y = df['temperatureObserved']

def polyfit2d(x, y, z, order=2):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

# Fit a 3rd order, 2d polynomial
m = polyfit2d(df['elevation'],df['latitude'],df['temperatureObserved'], order = 5)

xx,yy = np.meshgrid(np.arange(0, 3400,100),np.arange(60,83,1)) #,np.arange(0,1600,10))
zz = polyval2d(xx.astype(float), yy.astype(float), m)
zz[zz>0]=0
zz[zz<-50]=-50

#% Trend  fit
df['temperature_anomaly'] =  df.temperatureObserved - polyval2d(df.elevation, df.latitude, m) 
df_save['temperature_anomaly'] =  df_save.temperatureObserved - polyval2d(df_save.elevation, df_save.latitude, m) 

dates = df.index
y_data = df.temperature_anomaly.values[dates.year>1990]
time_recent = dates[dates.year>1990]
curve_fit = np.polyfit(pd.to_numeric(time_recent),y_data, 1)
y = curve_fit[1] + curve_fit[0]*pd.to_numeric(time_recent)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(df['elevation'],df['latitude'], df['temperatureObserved'],color='black')
for site in site_remove:
    ax.scatter(df_save.loc[df_save.site==site,'elevation'],
                df_save.loc[df_save.site==site,'latitude'],
                df_save.loc[df_save.site==site,'temperatureObserved'])
    # ax.text(df_save.loc[df_save.site==site,'elevation'].mean(),
    #             df_save.loc[df_save.site==site,'latitude'].mean(),
    #             df_save.loc[df_save.site==site,'temperatureObserved'].mean(),
    #             site, size=10, zorder=1, color='k') 
ax.plot_surface(xx, yy, zz,cmap=cm.coolwarm, alpha=0.5)
ax.set_xlabel('Elevation (m)')
ax.set_ylabel('Latitude ($^o$)')
ax.set_zlabel('10 m firn temperature ($^o$C)')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(df['temperature_anomaly'],'.')
ax2.plot(time_recent,y,'--',linewidth=3)
for site in site_remove:
    ax2.plot(df_save.loc[df_save.site==site,'temperature_anomaly'],'o')
    ax2.annotate(site,
                 (df_save.loc[df_save.site==site,'temperature_anomaly'].index.mean(),
                df_save.loc[df_save.site==site,'temperature_anomaly'].mean()))
for txt, x, y in zip(df.loc[df.temperature_anomaly.abs()>3, 'site'],
                      df.loc[df.temperature_anomaly.abs()>3].index,
                      df.loc[df.temperature_anomaly.abs()>3, 'temperature_anomaly']):
    ax2.annotate(txt, (x,y))
ax2.set_xlabel('Year')
ax2.set_ylabel('10 m firn temperature anomaly')
ax2.set_title('Elevation - Latitude model\n RMSE = %0.3f'%np.sqrt(np.mean(df['temperature_anomaly']**2)))
# plt.savefig('figures/lat_elev_poly.png')

# %% Model lat_elev_grid
from matplotlib import cm
df = df_save
site_remove = ['SE-Dome','H2', 'H3', 'FA_13', 'FA_15_1', 'FA_15_2']
df = df.loc[~np.isin(df.site,site_remove)]
step_elev = 500
elevation_bins = np.arange(0,4000,step_elev)
step_lat=4
latitude_bins = np.arange(60,90,step_lat)
grid_temp = np.empty((len(latitude_bins),len(elevation_bins)))
grid_temp[:] = np.nan
for i in range(len(elevation_bins)-1):
    for j in range(len(latitude_bins)-1):
        conditions = np.array((df.elevation >= elevation_bins[i], 
                               df.elevation <  elevation_bins[i+1],
                               df.latitude  >= latitude_bins[j],
                               df.latitude  <  latitude_bins[j+1]))
        
        grid_temp[j,i] = df.loc[np.logical_and.reduce(conditions), 'temperatureObserved'].mean()

X = df[['elevation','latitude']]
Y = df['temperatureObserved']
# meshgrid on the centerpoints
x_center,y_center = np.meshgrid(elevation_bins+step_elev/2, latitude_bins+step_lat/2) 

# from scipy.interpolate import griddata
# grid_temp[np.isnan(grid_temp)] = griddata( (x_center[~np.isnan(grid_temp)], 
#                                              y_center[~np.isnan(grid_temp)]),
#                                              grid_temp[~np.isnan(grid_temp)],
#                                              (x_center[np.isnan(grid_temp)],
#                                               y_center[np.isnan(grid_temp)]),
#                                              fill_value=-15)
         
# Fit a 3rd order, 2d polynomial
# m = polyfit2d(x_center[~np.isnan(grid_temp)], 
#               y_center[~np.isnan(grid_temp)], 
#               grid_temp[~np.isnan(grid_temp)], order = 3)

from scipy.interpolate import Rbf
rbf3 = Rbf((x_center[~np.isnan(grid_temp)]-np.min(x_center))/np.max(x_center), 
              (y_center[~np.isnan(grid_temp)]-np.min(y_center))/np.max(y_center), 
              grid_temp[~np.isnan(grid_temp)],
              function="cubic")

real_x = np.arange(0, 3400,50)
real_y = np.arange(60,83,1)
xx,yy = np.meshgrid(real_x,real_y) #,np.arange(0,1600,10))
# zz = polyval2d(xx.astype(float), yy.astype(float), m)
zz = rbf3((xx-np.min(x_center))/np.max(x_center), 
          (yy-np.min(y_center))/np.max(y_center))

zz[zz>0]=0
zz[zz<-50]=-50
df['temperature_anomaly'] =  df.temperatureObserved - rbf3((df.elevation-np.min(x_center))/np.max(x_center), (df.latitude-np.min(y_center))/np.max(y_center)) 
# %% plot
import matplotlib as mpl

fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(2, 1, 1)

dx = (real_x[1]-real_x[0])/2.
dy = (real_y[1]-real_y[0])/2.
extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]
im = plt.imshow(zz, extent=extent , aspect='auto')

cax1 = fig.add_subplot(3, 11,10) # adding ax for first colorbar
ax.set_position([ax.get_position().x0 , 0.35,  
        ax.get_position().width - (1-cax1.get_position().x0)*.8, 
        0.55] ) # set a new position

cb1 =plt.colorbar(im, cax=cax1)
cb1.ax.set_ylabel('10 m firn \ntemperature ($^o$C)\n ', rotation=270, labelpad=40)

cmap=cm.seismic
bounds = [-5, -4, -3, -2, -1,   1,  2,  3,  4,  5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sct = ax.scatter(df['elevation'],df['latitude'],s=80, c= df['temperature_anomaly'], edgecolor='lightgray',cmap=cmap,norm=norm)

cax2 = fig.add_subplot(3, 11,21) # adding ax for second colorbar
cb2 =plt.colorbar(sct, cax=cax2)
cb2.ax.set_ylabel('Firn temperature \nanomaly ($^o$C)', rotation=270, labelpad=40)

ax.set_xticks(elevation_bins)
ax.set_yticks(latitude_bins)
ax.grid()
ax.set_xlim(0,3300)
ax.set_ylim(60,82)
ax.set_xlabel('Elevation (m)')
ax.set_ylabel('Latitude ($^o$N)')
ax.set_title('Elevation - Latitude model\n RMSE = %0.3f'%np.sqrt(np.mean(df['temperature_anomaly']**2)))

ax2 = fig.add_subplot(5,1, 5)

longitude_bins = np.arange(-70,-14,4)
bin_anomaly = df['temperature_anomaly'].groupby(pd.cut(df['longitude'].values,longitude_bins)).mean()

ax2.scatter(df['longitude'], df['temperature_anomaly'],color='black')
ax2.step(longitude_bins,
         np.append(bin_anomaly.values,bin_anomaly.values[-1]),
         where='post',linewidth=5)
ax2.set_ylabel('Firn temperature \nanomaly ($^o$C)')
ax2.set_xlabel('Longitude ($^o$E)')
ax2.autoscale(enable=True, axis='x', tight=True)
# longitude_center = longitude_bins[:-1] + np.gradient(longitude_bins[:-1])/2
# m, b = np.polyfit(longitude_center[bin_anomaly.notnull()],
#                   bin_anomaly.values[bin_anomaly.notnull()], 1)
# ax2.plot(df['longitude'].values, m*df['longitude'].values + b)
plt.savefig('figures/lat_elev_model.png')

# %% trend in each elevation bin
from scipy import stats

def lm(df):
    y_data = df.values
    x_data = pd.to_numeric(df.index.values)
    x_data_save=x_data
    ind = np.logical_and(~np.isnan(x_data) , ~np.isnan(y_data))
    y_data = y_data[ind]
    x_data = x_data[ind]
    res = stats.linregress(x_data, y_data)
    y = res.intercept + res.slope*x_data_save
    return y, res
 
plt.figure()
for i in range(len(elevation_bins)-1):
    ax = plt.subplot(7,1,i+1)
    tmp = df.loc[np.logical_and(df['elevation']>elevation_bins[i], 
                                df['elevation']<=elevation_bins[i+1]),
                 'temperature_anomaly']
    tmp.plot(ax=ax, marker='o', linestyle='none')
    tmp_recent = tmp.loc['1995':]
    y_lin,res = lm(tmp_recent)
    ax.plot(tmp_recent.index,y_lin)
    plt.title(str(elevation_bins[i])+' - '+str(elevation_bins[i+1]))
    ax.set_ylabel('Temperature anomlay (deg C)')
    


# %% Recent trend


dates = df.index
y_data = df.temperature_anomaly.values[dates.year>1995]
time_recent = dates[dates.year>1995]
curve_fit = np.polyfit(pd.to_numeric(time_recent),y_data, 1)
y = curve_fit[1] + curve_fit[0]*pd.to_numeric(time_recent)

fig = plt.figure()
ax2 = fig.add_subplot(1,1, 1)

ax2.plot(df['temperature_anomaly'],'.')
ax2.plot(time_recent,y,'--',linewidth=3)
for site in site_remove:
    ax2.plot(df_save.loc[df_save.site==site,'temperature_anomaly'],'o')
    ax2.annotate(site,
                  (df_save.loc[df_save.site==site,'temperature_anomaly'].index.mean(),
                df_save.loc[df_save.site==site,'temperature_anomaly'].mean()))
# for txt, x, y in zip(df.loc[df.temperature_anomaly.abs()>3, 'site'],
#                       df.loc[df.temperature_anomaly.abs()>3].index,
#                       df.loc[df.temperature_anomaly.abs()>3, 'temperature_anomaly']):
#     ax2.annotate(txt, (x,y))
ax2.autoscale(enable=True, axis='x', tight=True)

ax2.set_xlabel('Year')
ax2.set_ylabel('10 m firn temperature anomaly')
plt.savefig('figures/lat_elev_time.png')



# %% Trend analysis: elevation, latitude, avg_snowfall_mm 
from sklearn import linear_model
X = df[['elevation','latitude','avg_snowfall_mm']]
Y = df['temperatureObserved']
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

xx,yy,zz = np.meshgrid(np.arange(0, 3400,100),np.arange(60,83,1),np.arange(0,1600,10))
z = regr.intercept_ + regr.coef_[0]*xx + regr.coef_[1]*yy + regr.coef_[2]*zz

#% Trend  fit
df['temperature_anomaly'] =  df.temperatureObserved - regr.intercept_ - regr.coef_[0]*df.elevation - regr.coef_[1]*df.latitude - regr.coef_[2]*df.avg_snowfall_mm
y_data = df.temperature_anomaly.values[dates.year>1990]
time_recent = dates[dates.year>1990]
curve_fit = np.polyfit(pd.to_numeric(time_recent),y_data, 1)
temp_trend = curve_fit[1] + curve_fit[0]*pd.to_numeric(time_recent)

df = df.sort_index()
x = pd.to_numeric(df.index)
y = df.temperature_anomaly

coef = np.polyfit(x,y,4)
p = np.poly1d(coef)
y_pred= p(x)

fig, ax = plt.subplots(1,1,figsize=(15, 9))
fig.subplots_adjust(hspace=0.1, wspace=0.1,top=0.8,bottom=0.1,left=0.1,right=0.7)
for ref in df.reference_short.unique():
    ax.plot(df.loc[df.reference_short==ref, 'temperature_anomaly'],'o',markeredgecolor='gray', label=ref)
for txt, x, y in zip(df.loc[df.temperature_anomaly.abs()>5, 'site'],
                      df.loc[df.temperature_anomaly.abs()>5].index,
                      df.loc[df.temperature_anomaly.abs()>5, 'temperature_anomaly']):
    ax.annotate(txt, (x,y))
ax.plot(time_recent,temp_trend,'--',color='red', linewidth=3)
ax.plot(df.index,y_pred,'--',color='orange', linewidth=3)
ax.set_xlabel('Year')
ax.set_ylabel('10 m firn temperature anomaly')
ax.set_title('Elevation - Latitude - Snowfall model\n RMSE = %0.3f'%np.sqrt(np.mean(df['temperature_anomaly']**2)))
lgnd = plt.legend(bbox_to_anchor=(1.05, 1.25),  ncol=1, fontsize= 12)
for i in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[i]._legmarker.set_markersize(10)
fig.savefig('figures/ELS_trend_time.png')




