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

# %% Extracting closest cell in MAR
print('Extracting closest cell in MAR')
ds = xr.open_dataset('I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-1950.nc',  decode_times = False)
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
# %% Extracting temperatures from MAR
print('Extracting temperatures from MAR')

df['MAR_10m_temp'] = np.nan

for year in  years:
    print(year)
    if year < 1950:
        continue
    try:
        ds = xr.open_dataset('I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-'+str(year)+'.nc')
    except:
        print('Cannot read ',year)
    time_mar = ds['TIME'].values
    df_year = df.loc[df.year == year,:].copy()

    for ind in progressbar.progressbar(df_year.index):        
        df_year.loc[ind,'MAR_10m_temp'] = ds.TI1[dict(x=int(df_year.loc[ind].MAR_j),
                    y=int(df_year.loc[ind].MAR_i),
                    OUTLAY=15,
                    TIME=np.argmin(np.abs(time_mar - np.datetime64(df_year.loc[ind].date))) )].values
    df.loc[df.year == year,:] = df_year.values      
# %% 
df.to_csv('subsurface_temperature_summary_w_MAR.csv')

# %% Comparison
df=pd.read_csv('subsurface_temperature_summary_w_MAR.csv')

df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'PROMICE',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'Hills et al. (2018)',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'Hills et al. (2017)',:]
df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
df_10m  = df_10m.reset_index()
df_10m = df_10m.sort_values('year')
ref_list = df_10m['reference_short'].unique()

df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]


fig = plt.figure(figsize=(15,9))
matplotlib.rcParams.update({'font.size': 16})
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
RMSE = np.sqrt(np.mean((df_10m.MAR_10m_temp- df_10m.temperatureObserved)**2))
ax.set_title('All except PROMICE and Hills (2017, 2018). RMSE = %0.2f'%RMSE)
plt.legend(title='Sources', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.plot([-40, 0], [-40, 0],c='black')
ax.set_xlabel('MAR simulated 10 m subsurface temperature ($^o$C)')
ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
fig.savefig('figures/MAR_comp_1.png')

# %% Comparison
dataset = ['PROMICE',  'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    if data_name == 'Hills et al. (2018)':
        df_10m = df_10m.loc[np.isin(df_10m['reference_short'],['Hills et al. (2017)','Hills et al. (2018)']),:]
    else:
        df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]

    df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
    df_10m  = df_10m.reset_index()
    df_10m = df_10m.sort_values('year')
    site_list = df_10m['site'].unique()
    
    df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
    
    fig = plt.figure(figsize=(15,9))
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
    RMSE = np.sqrt(np.mean((df_10m.MAR_10m_temp- df_10m.temperatureObserved)**2))
    ax.set_title(data_name+' RMSE = %0.2f'%RMSE)
    plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.plot([-40, 0], [-40, 0],c='black')
    ax.set_xlabel('MAR simulated 10 m subsurface temperature ($^o$C)')
    ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/MAR_comp_'+data_name+'.png')
    
# %% Comparison
dataset = ['PROMICE', 'Hills et al. (2018)']
# dataset = ['GC-Net', 'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]
    df_10m = df_10m.loc[df_10m.MAR_10m_temp>-200,:]
    df_10m  = df_10m.reset_index()
    df_10m = df_10m.sort_values('year')
    df_10m.date =pd.to_datetime(df_10m.date)
    site_list = df_10m['site'].unique()
    
    df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
    fig = plt.figure(figsize=(20,15))
    matplotlib.rcParams.update({'font.size': 12})
    
    cmap = matplotlib.cm.get_cmap('tab20b')
    for i, site in enumerate(site_list):
        ax = fig.add_subplot(int(np.floor(len(site_list)/3))+1,3, i+1)
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
        
        print(site,  tmp.latitude.mean() , tmp.longitude.mean(), tmp.temperatureObserved.std(),tmp.elevation.mean(),tmp.temperatureObserved.mean(),tmp.index.year.min(),tmp.index.year.max() )
        ax.set_title(site)
        ax.set_xlabel('')
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    # plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_ylabel('10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/MAR_comp_'+data_name+'_2.png')


#%%
# date	site	latitude	longitude	elevation	depthOfTemperatureObservation	temperatureObserved	reference	reference_short	note	year	geometry	MAR_distance	MAR_lat	MAR_lon	MAR_i	MAR_j	MAR_elevation	MAR_10m_temp
# 2016-08-31	T-16	67.18146999999999	-49.57025	987.0	10.0	-7.960000000000001	Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418	Hills et al. (2018)	interpolated at 10 m	2016	POINT (-49.57025 67.18146999999999)	165.10348372764497	67.16216278076172	-49.57770156860352	85.0	44.0	936.3746337890624	-15.278564453125

lat = 67.18146999999999
lon = -49.57025
from time import process_time
plt.figure() 
for year in  range(2016,2020):
    print(year)
    t1_start = process_time() 
    ds = xr.open_dataset('I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-'+str(year)+'.nc')
    points = [(x, y) for x,y in zip(ds['LAT'].values.flatten(),ds['LON'].values.flatten())]

    closest = points[cdist([(lat,lon)], points).argmin()]
    MAR_lat = closest[0]
    MAR_lon = closest[1]
    MAR_distance = cdist([(lon,lat)], [closest]) [0][0]
    
    tmp = np.array(ds.SH)
    ind = np.argwhere(np.array(np.logical_and(ds['LAT']==closest[0], ds['LON']==closest[1])))[0]
    MAR_i = ind[0]
    MAR_j = ind[1]
        
    ds_ice = ds.TI1[dict(x=MAR_j, y=MAR_i)].reset_coords() 
    ds_ice = ds_ice.load()
    t1_stop = process_time()
    print("Elapsed time:", t1_stop-t1_start) 
    for i in range(0,18):
        print(i)
        ds_ice[dict(OUTLAY=i)].plot(label=str(year))
    ds_ice[dict(OUTLAY=15)].plot(label=str(year),linewidth=5, color='black')
tmp =df_10m.loc[df_10m['site']=='T-16',:].set_index('date').sort_index()
tmp.temperatureObserved.plot(marker='o',linewidth=4, markersize=10)

plt.ylabel('Temperature at 10m')
plt.xlabel('time')
plt.title('MAR output at T-16: x=-200.0 (index 44), y=-2497.9277 (index 85)')


    