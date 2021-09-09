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

dates = pd.DatetimeIndex(df.date)

years  = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

# %% Extracting data from MAR

# First saving the closest coordinates to all temperature measurements
ds = xr.open_dataset('I:\Baptiste\Data\RCM\MAR\MARv3.12\MARv3.12-ERA5-15km-1950.nc',  decode_times = False)
points = [(x, y) for x,y in zip(ds['LAT'].values.flatten(),ds['LON'].values.flatten())]

df['MAR_distance'] = np.nan
df['MAR_lat'] = np.nan
df['MAR_lon'] = np.nan
df['MAR_i'] = np.nan
df['MAR_j'] = np.nan 
df['MAR_elevation'] = np.nan
lat_prev = -999
lon_prev = -999
for i, (lat,lon) in progressbar.progressbar(enumerate(zip(df.latitude,df.longitude))):
    if lat == lat_prev and lon == lon_prev:
        df.iloc[i,df.columns.get_loc('MAR_lat')] = df.iloc[i-1, df.columns.get_loc('MAR_lat')]
        df.iloc[i,df.columns.get_loc('MAR_lon')] = df.iloc[i-1, df.columns.get_loc('MAR_lon')]
        df.iloc[i,df.columns.get_loc('MAR_distance')] = df.iloc[i-1, df.columns.get_loc('MAR_distance')]
        df.iloc[i,df.columns.get_loc('MAR_i')] = df.iloc[i-1, df.columns.get_loc('MAR_i')]
        df.iloc[i,df.columns.get_loc('MAR_j')] = df.iloc[i-1, df.columns.get_loc('MAR_j')]
        df.iloc[i,df.columns.get_loc('MAR_elevation')] =  df.iloc[i-1, df.columns.get_loc('MAR_elevation')]
    else:
        closest = points[cdist([(lat,lon)], points).argmin()]
        df.iloc[i, df.columns.get_loc('MAR_lat')] = closest[0]
        df.iloc[i, df.columns.get_loc('MAR_lon')] = closest[1]
        df.iloc[i, df.columns.get_loc('MAR_distance')] = cdist([(lon,lat)], [closest]) [0][0]
        
        tmp = np.array(ds.SH)
        ind = np.argwhere(np.array(np.logical_and(ds['LAT']==closest[0], ds['LON']==closest[1])))[0]
        df.iloc[i, df.columns.get_loc('MAR_i')] = ind[0]
        df.iloc[i, df.columns.get_loc('MAR_j')] = ind[1]
     
        df.iloc[i, df.columns.get_loc('MAR_elevation')] =  np.array(ds.SH)[ind[0], ind[1]]
        lat_prev = lat
        lon_prev = lon
        

    # plt.figure()
    # plt.scatter(ds['LON'].values.flatten(), ds['LAT'].values.flatten())
    # plt.plot(lon,lat, marker='o',markersize=10,color='green')
    # plt.plot(np.array(ds.LON)[ind[0], ind[1]],np.array(ds.LAT)[ind[0], ind[1]], marker='o',markersize=10,color='red')
    # plt.plot(closest[1],closest[0], marker='o',markersize=5)    
    
    # date_in_days_since = (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).days + (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).seconds/60/60/24
# %% 
df['MAR_10m_temp'] = np.nan

for year in  years:
    print(year)
    if year < 1950:
        continue
    try:
        ds = xr.open_dataset('I:\Baptiste\Data\RCM\MAR\MARv3.12\MARv3.12-ERA5-15km-'+str(year)+'.nc', decode_times = False)
    except:
        print('Cannot read ',year)
    time_mar = pd.to_datetime([pd.to_datetime(str(year)+'-01-01') + pd.Timedelta(str(i)+' days') for i in range(len(ds.time))])
    
    # time_mar_org = np.array(ds.time)
    df_year = df.loc[df.year == year,:].copy()

    for ind in progressbar.progressbar(df_year.index):        
        df_year.loc[ind,'MAR_10m_temp'] = ds.TI1[dict(x=int(df_year.loc[ind].MAR_j),
                    y=int(df_year.loc[ind].MAR_i),
                    OUTLAY=15,
                    time=np.argmin(np.abs(time_mar - pd.to_datetime(df_year.loc[ind].date))) )].values
    df.loc[df.year == year,:] = df_year.values      
# %% 
df.to_csv('subsurface_temperature_summary_w_MAR.csv')

# %% Comparison
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'PROMICE',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'GC-Net',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'Hills et al. (2018)',:]
df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
df_10m  = df_10m.reset_index()
df_10m = df_10m.sort_values('year')
ref_list = df_10m['reference_short'].unique()

df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1, 1)
plt.subplots_adjust(left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None)

cmap = matplotlib.cm.get_cmap('tab20b')
for i, ref in enumerate(ref_list):
    ax.plot(df_10m.loc[df_10m['reference_short']==ref,:].MAR_10m_temp,
                   df_10m.loc[df_10m['reference_short']==ref,:].temperatureObserved,
                   marker='o',linestyle='none',markeredgecolor='gray',
                   markersize=10,
                   color=cmap(i/len(ref_list)), 
                   label = df_10m.loc[df_10m['reference_short']==ref,'reference_short'].values[0] )
plt.legend(title='Sources', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.plot([-40, 0], [-40, 0],c='black')
ax.set_xlabel('MAR simulated 10 m subsurface temperature ($^o$C)')
ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
fig.savefig('figures/MAR_comp_1.png')

# %% Comparison
dataset = ['PROMICE', 'GC-Net', 'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]
    df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
    df_10m  = df_10m.reset_index()
    df_10m = df_10m.sort_values('year')
    site_list = df_10m['site'].unique()
    
    df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1, 1)
    plt.subplots_adjust(left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None)
    
    cmap = matplotlib.cm.get_cmap('tab20b')
    for i, site in enumerate(site_list):
        ax.plot(df_10m.loc[df_10m['site']==site,:].MAR_10m_temp,
                       df_10m.loc[df_10m['site']==site,:].temperatureObserved,
                       marker='o',linestyle='none',markeredgecolor='gray',
                       markersize=10,
                       color=cmap(i/len(site_list)), 
                       label = df_10m.loc[df_10m['site']==site,'site'].values[0] )
    plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.plot([-40, 0], [-40, 0],c='black')
    ax.set_xlabel('MAR simulated 10 m subsurface temperature ($^o$C)')
    ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/MAR_comp_'+data_name+'.png')
    
# %% Comparison
dataset = ['PROMICE', 'GC-Net', 'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]
    df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
    df_10m  = df_10m.reset_index()
    df_10m = df_10m.sort_values('year')
    site_list = df_10m['site'].unique()
    
    df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1, 1)
    plt.subplots_adjust(left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None)
    
    cmap = matplotlib.cm.get_cmap('tab20b')
    for i, site in enumerate(site_list):
        tmp =df_10m.loc[df_10m['site']==site,:].set_index('date').sort_index()
        tmp.temperatureObserved.plot(ax =ax,
                       marker='o',linestyle='--',markeredgecolor='gray',
                       markersize=8,
                       color=cmap(i/len(site_list)), 
                       label = site + ' obs' )
        tmp.MAR_10m_temp.plot(ax=ax,
                       marker='^',linestyle=':',markeredgecolor='gray',
                       markersize=8,
                       color=cmap(i/len(site_list)), 
                       label = site + ' MAR' )
    plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel('10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/MAR_comp_'+data_name+'_2.png')


#%%
