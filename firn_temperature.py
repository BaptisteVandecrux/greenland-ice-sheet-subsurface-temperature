# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:53:15 2020

@author: bav
"""
# This notebook opens the Greenland FirnCover data and puts it into pandas dataframes.
# The core data comes from core_data_df.pkl, which is created by running firncover_core_data_df.py
# (The dataframe is created in that script.)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime
import pandas as pd
import time
import xarray as xr
import firn_temp_lib as ftl
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
np.seterr(invalid='ignore')
# %matplotlib inline
# %matplotlib qt
import scipy.interpolate

#%% Loading data from firn cover
pathtodata = './data'
date1 = '2020_04_27' # Date on the hdf5 file
sites=['Summit','KAN-U','NASA-SE','Crawford','EKT','Saddle','EastGrip','DYE-2']

### Import the FirnCover data tables. 
filename='FirnCoverData_2.0_' + date1 + '.h5'
filepath=os.path.join(pathtodata,filename)

# Loading temperature data
statmeta_df, sonic_df, rtd_df, _, metdata_df = ftl.load_metadata(filepath,sites)
rtd_df=rtd_df.reset_index()
rtd_df.loc[rtd_df['sitename']=='Crawford','sitename'] = 'CP1'
rtd_df=rtd_df.set_index(['sitename','date'])
sonic_df=sonic_df.reset_index()
sonic_df.loc[sonic_df['sitename']=='Crawford','sitename'] = 'CP1'
sonic_df=sonic_df.set_index(['sitename','date'])

sites=['Summit','KAN-U','NASA-SE','CP1','EKT','Saddle','EastGrip','DYE-2']

    
#% Loading data from GC-Net
print('Loading GC-Net')
sites_2=['CP1','DYE-2','NASA-E', 'NASA-SE','NASA-U','Saddle', 'SouthDome','Summit','TUNU-N']

df_all = pd.DataFrame()
for site in sites_2:
    ds = xr.open_dataset('data/Vandecrux et al. 2020/'+site+'_T_firn_obs.nc')
    df = ds.to_dataframe()
    # df_avg = df.resample('D').mean()
    df.reset_index(inplace=True)  
    
    df2 = pd.DataFrame()
    df2['date'] = df.loc[df['level']==1,'time']
    for i in range(1,11):
        df2['rtd'+str(24-i)] = df.loc[df['level']==i,'T_firn'].values    
    for i in range(11,25):
        df2['rtd'+str(24-i)] = np.nan
    for i in range(1,11):
        df2['depth_'+str(24-i)] = df.loc[df['level']==i,'Depth'].values
    for i in range(11,25):
        df2['depth_'+str(24-i)] = np.nan
    df2.set_index(['date'],inplace=True)

    df_d = df2.resample('D').mean()
    df_d['sitename'] = site
    df_d.set_index(['sitename'],append=True,inplace=True)
    
    df_all = df_all.append(df_d)
df_all=df_all.reset_index()
df_all.set_index(['sitename','date'],inplace=True)

sites=['Summit','KAN-U','CP1','NASA-SE','EKT','Saddle','EastGrip','DYE-2']
sites_all = sorted(list(set(sites) | set(sites_2)))

df_all=df_all.combine_first(rtd_df)

#% Loading PROMICE KAN_U
print('Loading PROMICE at KAN-U')
ds = xr.open_dataset('data/T_firn_KANU_PROMICE.nc')
df = ds.to_dataframe()
# df_avg = df.resample('D').mean()
df.reset_index(inplace=True)
df2 = pd.DataFrame()
site='KAN-U'
df2['date'] = df.loc[df['level']==1,'time']
for i in range(1,9):
    df2['rtd'+str(i-1)] = df.loc[df['level']==i,'Firn temperature'].values    
for i in range(1,9):
    df2['depth_'+str(i-1)] = df.loc[df['level']==i,'depth'].values
df2.set_index(['date'],inplace=True)

df_d = df2.resample('D').mean()
df_d['sitename'] = site
df_d.set_index(['sitename'],append=True,inplace=True)
df_d=df_d.reset_index()
df_d.set_index(['sitename','date'],inplace=True)
df_all=df_all.combine_first(df_d)

#% Loading PROMICE KAN_U
print('Loading SPLAZ at KAN-U')
ds = xr.open_dataset('data/T_firn_KANU_SPLAZ_main.nc')
df = ds.to_dataframe()
# df_avg = df.resample('D').mean()
df.reset_index(inplace=True)
df2 = pd.DataFrame()
site='KAN-U'
df2['date'] = df.loc[df['level']==1,'time']
for i in range(1,33):
    df2['rtd'+str(i-1)] = df.loc[df['level']==i,'Firn temperature'].values    
for i in range(1,33):
    df2['depth_'+str(i-1)] = df.loc[df['level']==i,'depth'].values
df2.set_index(['date'],inplace=True)

df_d = df2.resample('D').mean()
df_d['sitename'] = site
df_d.set_index(['sitename'],append=True,inplace=True)
df_d=df_d.reset_index()
df_d.set_index(['sitename','date'],inplace=True)
df_all=df_all.combine_first(df_d)

#% Load Humphrey CP
print('loading Humphrey')
# The thermistors are placed with a 25cm spacing in the top 5.5m and...
# a 50cm spacing in the lower 4.5m.  The bottom of the string is ...
# nominally at 10m below the surface, with the top at the surface.
df = pd.read_csv('data/FirnTemperature_CP_Humphreyetal2012.txt', header=None, delim_whitespace=True)
df['date'] = np.nan
for i in range(df.shape[0]):
    df['date'][i] =  datetime.datetime(2007, 1, 1) + datetime.timedelta(days = df.iloc[i,0]-1)
df.columns
df.set_index(['date'],inplace=True)
df=df.resample('D').mean()
df=df.reset_index()
df['sitename'] = 'CP1'
df.set_index(['sitename','date'],inplace=True)
zz = np.array(['rtd'+str(x) for x in range(32)])
zz2 = np.array(['depth_'+str(x) for x in range(32)])
df.drop(0,axis=1,inplace=True)
df.columns=zz
df[zz2] = np.nan
ini_depth = np.hstack((np.array([0.01]), np.arange(0.25,5.75,0.25),np.arange(6, 10.5,0.5)))
for i, col in enumerate(zz2):
    df[col] = ini_depth[i]
    
df_all=df_all.combine_first(df)

#%% All firn temperature
# zz = ['rtd0', 'rtd1', 'rtd2', 'rtd3', 'rtd4', 'rtd5', 'rtd6', 'rtd7', 'rtd8', 'rtd9', 'rtd10', 'rtd11', 'rtd12', 'rtd13', 'rtd14', 'rtd15', 'rtd16', 'rtd17', 'rtd18', 'rtd19', 'rtd20', 'rtd21', 'rtd22', 'rtd23']
# zz2 = ['depth_0', 'depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5', 'depth_6', 'depth_7', 'depth_8', 'depth_9', 'depth_10', 'depth_11', 'depth_12', 'depth_13', 'depth_14', 'depth_15', 'depth_16', 'depth_17', 'depth_18', 'depth_19', 'depth_20', 'depth_21', 'depth_22', 'depth_23']
f1, ax = plt.subplots(4,3,figsize=(20, 15))
f1.subplots_adjust(hspace=0.2, wspace=0.1,
                   left = 0.08 , right = 0.85 ,
                   bottom = 0.2 , top = 0.9)
count = -1
for site in sites_all:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    
    sitetemp= df_all.loc[site,zz]
    sitedep = df_all.loc[site,zz2]
    n_grid = np.linspace(0,15,15)
    time=sitetemp.index.values
    temps = sitetemp.values

    time_sitedep = sitedep.index.get_level_values(0).values
    sitedep = sitedep.loc[np.isin(time_sitedep, time)]
    time_sitedep = sitedep.index.get_level_values(0).values
    depths = sitedep.values
    
    t_interp=np.zeros((depths.shape[0],len(n_grid)))
    for kk in range(depths.shape[0]):
            tif = sp.interpolate.interp1d(depths[kk,:], temps[kk,:], bounds_error=False)
            t_interp[kk,:]= tif(n_grid)
            
    cax1 = ax[i,j].contourf(time,n_grid,t_interp.T, 50, extend='both',
                            vmin=-50,
                            vmax=0)
    # cax1.cmap.set_over('cyan')
    # cax1.cmap.set_under('black')
    ax[i,j].set_title(site)    
    ax[i,j].set_xlim([datetime.date(1998, 5, 1), datetime.date(2019, 10, 1)])
    ax[i,j].set_ylim([15, 0])
    ax[i,j].xaxis.set_major_locator(years)
    ax[i,j].xaxis.set_major_formatter(years_fmt)
    ax[i,j].xaxis.set_minor_locator(months)
    ax[i,j].set_xlabel("")
    if count<len(sites_all)-3:
        ax[i,j].set_xticklabels("")
    else:
        for label in ax[i,j].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
cbar_ax = f1.add_axes([0.9, 0.15, 0.02, 0.7])
cb1 = f1.colorbar(cax1, cax=cbar_ax)
cb1.set_label('Temperature (C)')
f1.text(0.5, 0.1, 'Year', ha='center', size = 20)
f1.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 20)
f1.savefig('figures/RTD_temp.png')

# %% 10 m firn temp
T10m_df = pd.DataFrame()

count = -1
for site in sites_all:
    print(site)
    count = count+1
    
    sitetemp= df_all.loc[site,zz]
    sitedep = df_all.loc[site,zz2]
    n_grid = np.linspace(0,15,15)
    time=sitetemp.index.values
    temps = sitetemp.values

    time_sitedep = sitedep.index.get_level_values(0).values
    sitedep = sitedep.loc[np.isin(time_sitedep, time)]
    time_sitedep = sitedep.index.get_level_values(0).values
    depths = sitedep.values

    depths_from_surface = depths
    T10_temp = pd.DataFrame()
    T10_temp['T'] = np.nan*depths_from_surface[:,0]
    T10_temp['sitename'] = site
    T10_temp['date'] = sitedep.index.values
    for i in range(temps.shape[0]):
        ind_nonan = np.argwhere(np.logical_and(~np.isnan(temps[i,:]), ~np.isnan(depths_from_surface[i,:])))
        if (len(depths_from_surface[i,ind_nonan][:,0])>2 and \
                          len(temps[i,ind_nonan][:,0])>2) and np.max(depths_from_surface[i,ind_nonan][:,0])>8:
            T10_temp['T'][i] = sp.interpolate.interp1d(depths_from_surface[i,ind_nonan][:,0],
                                                       temps[i,ind_nonan][:,0],
                                                       fill_value="extrapolate", 
                                                       kind='linear', 
                                                       assume_sorted=False)(10)
    # applying gradient filter on KAN-U, Crawford, EwastGRIP, EKT, Saddle and Dye-2
    vals = T10_temp['T'].copy().values
    msk = np.where(np.abs(np.gradient(vals))>=0.5)[0]
    vals[msk] = np.nan
    vals[np.maximum(msk-1,0)] = np.nan
    vals[np.minimum(msk+1,len(vals)-1)] = np.nan
    vals[vals==-9999]=np.nan
    T10_temp['T']=vals

    T10m_df = T10m_df.append(T10_temp)
    # if np.sum(np.isnan(T_10m))>1:
    #     T_10m[np.argwhere(np.isnan(T_10m))]= \
    #         sp.interpolate.interp1d(np.argwhere(~np.isnan(T_10m))[:,0],\
    #                                 T_10m[np.argwhere(~np.isnan(T_10m))][:,0], \
    #                                     fill_value = 'extrapolate') (np.argwhere(np.isnan(T_10m)))

T10m_df.loc[T10m_df['T'].values>-8,'T'] = np.nan
T10m_df.loc[T10m_df['T'].values<-70,'T'] = np.nan

# ind=[]
# if site == 'Summit':
#     ind = np.where(np.logical_and(
#         time>np.datetime64('2017-05-15'),
#         time<np.datetime64('2017-09-10')))
# if site == 'Saddle':
#     ind = np.where( time>np.datetime64('2017-05-10') )
# if site == 'EKT':
#     ind = np.where(np.logical_and(
#         time>np.datetime64('2017-05-01'),
#         time<np.datetime64('2018-06-01')))
# if site == 'Crawford':
#     ind1 = np.logical_and(
#         time>np.datetime64('2017-03-01'),
#         time<np.datetime64('2017-07-01'))
#     ind2 = np.logical_and(
#         time>np.datetime64('2018-05-01'),
#         time<np.datetime64('2017-06-01'))
#     ind = np.where(np.logical_or(ind1,ind2))
# if site == 'KAN-U':
#     ind = np.where(np.logical_and(
#         time>np.datetime64('2017-11-01'),
#         time<np.datetime64('2018-05-01')))
# T_10m[ind] = np.nan
# print('Average T10: ' + str(np.nanmean(T_10m)))
T10m_df.set_index(['sitename','date'],inplace=True)

T10m_interp = T10m_df.copy()
for site in sites_all:
    T_10m = T10m_interp.loc[site,'T'].values
    ind1 = np.where(np.logical_not(np.isnan(T_10m)))[0][0]
    ind2 = np.where(np.logical_not(np.isnan(T_10m)))[0][-1]
    if np.sum(np.isnan(T_10m))>1:
        T_10m[np.argwhere(np.isnan(T_10m))]= sp.interpolate.interp1d(
            np.argwhere(~np.isnan(T_10m))[:,0],
            T_10m[np.argwhere(~np.isnan(T_10m))][:,0],
            fill_value = 'extrapolate') (np.argwhere(np.isnan(T_10m)))
    T_10m[T_10m>-8] = np.nan
    T_10m[T_10m<-70] = np.nan
    T_10m[ind2:] = np.nan
    T_10m[:ind1] = np.nan
    T10m_interp.loc[site,'T'] = T_10m
# %% 10 m firn temp
f1, ax = plt.subplots(1,1,figsize=(10, 10))
colors = [plt.cm.tab20(x) for x in np.linspace(0, 1, len(sites_all))]
for site in sites_all:
    time = T10m_df.loc[site].index.values
    ind = np.where(np.logical_and(
        time>np.datetime64('2015-02-01'),
        time<np.datetime64('2015-06-01')))
    tmp = T10m_df.loc[site,'T'].values
    tmp[ind,] = np.nan
    T10m_df.loc[site,'T']= tmp

count = -1
for i, site in enumerate(sites_all):
    ax.plot(T10m_df.loc[site,'T'].index.values,T10m_df.loc[site,'T'].values,
            color = colors[i], label=site,linewidth=2)
    ax.plot(T10m_interp.loc[site,'T'].index.values,T10m_interp.loc[site,'T'].values,
            color = colors[i], linestyle='--', linewidth=2)
ax.legend()
ax.set_xlim([datetime.date(1998, 5, 1), datetime.date(2019, 10, 1)])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
ax.set_xlabel("Year")
ax.set_ylabel('10 m firn temperature (C)')
f1.savefig('figures/T10.png')

#%% 
T10m_df_trimmed = T10m_df.dropna()
T10m_df_trimmed.to_csv('data/T10m_GC-Net_FirnCover_KANU_PROMICE_SPLAZ.csv')
