# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
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


df = pd.read_csv('subsurface_temperature_sumRACMOy.csv')

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


# %% Extracting closest cell in RACMO
print('Extracting closest cell in RACMO')
ds = xr.open_dataset('C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc')
points = [(x, y) for x,y in zip(ds['lat'].values.flatten(),ds['lon'].values.flatten())]

df['RACMO_distance'] = np.nan
df['RACMO_lat'] = np.nan
df['RACMO_lon'] = np.nan
df['RACMO_i'] = np.nan
df['RACMO_j'] = np.nan 
df['RACMO_elevation'] = np.nan
lat_prev = -999
lon_prev = -999
for i, (lat,lon) in progressbar.progressbar(enumerate(zip(df.latitude,df.longitude))):
    if lat == lat_prev and lon == lon_prev:
        df.iloc[i,df.columns.get_loc('RACMO_lat')] = df.iloc[i-1, df.columns.get_loc('RACMO_lat')]
        df.iloc[i,df.columns.get_loc('RACMO_lon')] = df.iloc[i-1, df.columns.get_loc('RACMO_lon')]
        df.iloc[i,df.columns.get_loc('RACMO_distance')] = df.iloc[i-1, df.columns.get_loc('RACMO_distance')]
        df.iloc[i,df.columns.get_loc('RACMO_i')] = df.iloc[i-1, df.columns.get_loc('RACMO_i')]
        df.iloc[i,df.columns.get_loc('RACMO_j')] = df.iloc[i-1, df.columns.get_loc('RACMO_j')]
        df.iloc[i,df.columns.get_loc('RACMO_elevation')] =  df.iloc[i-1, df.columns.get_loc('RACMO_elevation')]
    else:
        closest = points[cdist([(lat,lon)], points).argmin()]
        df.iloc[i, df.columns.get_loc('RACMO_lat')] = closest[0]
        df.iloc[i, df.columns.get_loc('RACMO_lon')] = closest[1]
        df.iloc[i, df.columns.get_loc('RACMO_distance')] = cdist([(lon,lat)], [closest]) [0][0]
        
        tmp = np.array(ds.lsm)
        ind = np.argwhere(np.array(np.logical_and(ds['lat']==closest[0], ds['lon']==closest[1])))[0]
        df.iloc[i, df.columns.get_loc('RACMO_i')] = ind[0]
        df.iloc[i, df.columns.get_loc('RACMO_j')] = ind[1]
     
        df.iloc[i, df.columns.get_loc('RACMO_elevation')] =  np.array(ds.lsm)[ind[0], ind[1]]
        lat_prev = lat
        lon_prev = lon
        

    # plt.figure()
    # plt.scatter(ds['lon'].values.flatten(), ds['lat'].values.flatten())
    # plt.plot(lon,lat, marker='o',markersize=10,color='green')
    # plt.plot(np.array(ds.lon)[ind[0], ind[1]],np.array(ds.lat)[ind[0], ind[1]], marker='o',markersize=10,color='red')
    # plt.plot(closest[1],closest[0], marker='o',markersize=5)    
    
    # date_in_days_since = (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).days + (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).seconds/60/60/24
# %% Extracting temperatures from RACMO
print('Extracting temperatures from RACMO')

df['RACMO_10m_temp'] = np.nan
time_RACMO = ds['time'].values
df = df.reset_index(drop=True)
for ind in progressbar.progressbar(df.index):   
    df.loc[ind,'RACMO_10m_temp'] = ds.T10m[dict(
                rlon=int(df.loc[ind].RACMO_j),
                rlat=int(df.loc[ind].RACMO_i),
                time=np.argmin(np.abs(time_RACMO - np.datetime64(df.loc[ind,'date'])))
                )].values-273.15
# %% 
df.to_csv('subsurface_temperature_sumRACMOy_w_RACMO.csv')

# %% Comparison
df=pd.read_csv('subsurface_temperature_sumRACMOy_w_RACMO.csv')

df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'PROMICE',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'Hills et al. (2018)',:]
df_10m = df_10m.loc[df_10m['reference_short'] != 'Hills et al. (2017)',:]
df_10m = df_10m.loc[df_10m.RACMO_10m_temp>-200,:]
df_10m  = df_10m.reset_index(drop = True)
df_10m = df_10m.sort_values('year')
ref_list = df_10m['reference_short'].unique()

df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]


fig = plt.figure(figsize=(15,9))
matplotlib.rcParams.update({'font.size': 16})
ax = fig.add_subplot(1,1, 1)
plt.subplots_adjust(left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None)

cmap = matplotlib.cm.get_cmap('tab20b')
for i, ref in enumerate(ref_list):
    ax.plot(df_10m.loc[df_10m['reference_short']==ref,:].RACMO_10m_temp,
                   df_10m.loc[df_10m['reference_short']==ref,:].temperatureObserved,
                   marker='o',linestyle='none',markeredgecolor='gray',
                   markersize=10,
                   color=cmap(i/len(ref_list)), 
                   label = df_10m.loc[df_10m['reference_short']==ref,'reference_short'].values[0] )
RMSE = np.sqrt(np.mean((df_10m.RACMO_10m_temp- df_10m.temperatureObserved)**2))
ax.set_title('All except PROMICE and Hills (2017, 2018). RMSE = %0.2f'%RMSE)
plt.legend(title='Sources', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.plot([-40, 0], [-40, 0],c='black')
ax.set_xlabel('RACMO simulated 10 m subsurface temperature ($^o$C)')
ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
fig.savefig('figures/RACMO_comp_1.png')

# %% Comparison
dataset = ['PROMICE',  'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    if data_name == 'Hills et al. (2018)':
        df_10m = df_10m.loc[np.isin(df_10m['reference_short'],['Hills et al. (2017)','Hills et al. (2018)']),:]
    else:
        df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]

    df_10m = df_10m.loc[df_10m.RACMO_10m_temp>-200,:]
    df_10m  = df_10m.reset_index(drop=True)
    df_10m = df_10m.sort_values('year')
    site_list = df_10m['site'].unique()
    
    df_10m['ref_id'] = [np.where(ref_list==x)[0] for x in df_10m['reference']]
    
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1, 1)
    plt.subplots_adjust(left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None)
    
    cmap = matplotlib.cm.get_cmap('tab20b')
    for i, site in enumerate(site_list):
        ax.plot(df_10m.loc[df_10m['site']==site,:].RACMO_10m_temp,
                       df_10m.loc[df_10m['site']==site,:].temperatureObserved,
                       marker='o',linestyle='none',markeredgecolor='gray',
                       markersize=10,
                       color=cmap(i/len(site_list)), 
                       label = df_10m.loc[df_10m['site']==site,'site'].values[0] )
    RMSE = np.sqrt(np.mean((df_10m.RACMO_10m_temp- df_10m.temperatureObserved)**2))
    ax.set_title(data_name+' RMSE = %0.2f'%RMSE)
    plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.plot([-40, 0], [-40, 0],c='black')
    ax.set_xlabel('RACMO simulated 10 m subsurface temperature ($^o$C)')
    ax.set_ylabel('Observed 10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/RACMO_comp_'+data_name+'.png')
    
# %% Comparison
dataset = ['PROMICE', 'Hills et al. (2018)']
# dataset = ['GC-Net', 'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float)==10,:]
    df_10m = df_10m.loc[df_10m['reference_short'] == data_name,:]
    df_10m = df_10m.loc[df_10m.RACMO_10m_temp>-200,:]
    df_10m  = df_10m.reset_index(drop=True)
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
        tmp.RACMO_10m_temp.plot(ax=ax,
                       marker='^',linestyle=':',markeredgecolor='gray',
                       markersize=8,
                       color=cmap(i/len(site_list)), 
                       label = site + ' RACMO' )
        ax.set_title(site)
        ax.set_xlabel('')
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    # plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_ylabel('10 m subsurface temperature ($^o$C)')
    fig.savefig('figures/RACMO_comp_'+data_name+'_2.png')


#%%
# date	site	latitude	longitude	elevation	depthOfTemperatureObservation	temperatureObserved	reference	reference_short	note	year	geometry	RACMO_distance	RACMO_lat	RACMO_lon	RACMO_i	RACMO_j	RACMO_elevation	RACMO_10m_temp
# 2016-08-31	T-16	67.18146999999999	-49.57025	987.0	10.0	-7.960000000000001	Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418	Hills et al. (2018)	interpolated at 10 m	2016	POINT (-49.57025 67.18146999999999)	165.10348372764497	67.16216278076172	-49.57770156860352	85.0	44.0	936.3746337890624	-15.278564453125

lat = 67.18146999999999
lon = -49.57025
from time import process_time
plt.figure() 
for year in  range(2017,2020):
    print(year)
    t1_start = process_time() 
    # ds = xr.open_dataset('I:\Baptiste\Data\RCM\RACMO\RACMOv3.12_fixed\RACMOv3.12-10km-daily-ERA5-'+str(year)+'.nc')
    ds = xr.open_dataset('I:/Baptiste/Data/RCM/RACMO/b92/ICE.'+str(year)+'.01-12.b92.nc')
    points = [(x, y) for x,y in zip(ds['LAT'].values.flatten(),ds['LON'].values.flatten())]

    closest = points[cdist([(lat,lon)], points).argmin()]
    RACMO_lat = closest[0]
    RACMO_lon = closest[1]
    RACMO_distance = cdist([(lon,lat)], [closest]) [0][0]
    
    tmp = np.array(ds.SH)
    ind = np.argwhere(np.array(np.logical_and(ds['LAT']==closest[0], ds['LON']==closest[1])))[0]
    RACMO_i = ind[0]
    RACMO_j = ind[1]
        
    ds_ice = ds.TI1[dict(X10_85=RACMO_j, Y20_155=RACMO_i)].reset_coords() 
    ds_ice = ds_ice.load()
    t1_stop = process_time()
    print("Elapsed time:", t1_stop-t1_start) 
    for i in range(0,18):
        print(i)
        plt.plot(pd.to_datetime(ds_ice.TIME.values), ds_ice[dict(OUTLAY=i)].TI1.values, label=str(year))
    plt.plot(pd.to_datetime(ds_ice.TIME.values), ds_ice[dict(OUTLAY=15)].TI1.values, label=str(year),linewidth=5, color='black')
tmp =df_10m.loc[df_10m['site']=='T-16',:].set_index('date').sort_index()
tmp.temperatureObserved.plot(marker='o',linewidth=4, markersize=10)

plt.ylabel('Temperature at 10m')
plt.xlabel('time')
plt.title('RACMO output at T-16: x=-200.0 (index 44), y=-2497.9277 (index 85)')


    