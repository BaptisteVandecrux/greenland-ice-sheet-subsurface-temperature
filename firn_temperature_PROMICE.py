# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:53:15 2020

@author: bav
"""


import errno
from pathlib import Path
import wget # conda install: https://anaconda.org/anaconda/pywget
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

#%% Adding PROMICE observations
# Array information of stations available at PROMICE official site: https://promice.org/WeatherStations.html
PROMICE_stations = [('EGP',(75.6247,-35.9748), 2660), 
                   ('KAN_B',(67.1252,-50.1832), 350), 
                   ('KAN_L',(67.0955,-35.9748), 670), 
                   ('KAN_M',(67.0670,-48.8355), 1270), 
                   ('KAN_U',(67.0003,-47.0253), 1840), 
                   ('KPC_L',(79.9108,-24.0828), 370),
                   ('KPC_U',(79.8347,-25.1662), 870), 
                   ('MIT',(65.6922,-37.8280), 440), 
                   ('NUK_K',(64.1623,-51.3587), 710), 
                   ('NUK_L',(64.4822,-49.5358), 530),
                   ('NUK_U',(64.5108,-49.2692), 1120),
                   ('QAS_L',(61.0308,-46.8493), 280),
                   ('QAS_M',(61.0998,-46.8330), 630), 
                   ('QAS_U',(61.1753,-46.8195), 900), 
                   ('SCO_L',(72.2230,-26.8182), 460),
                   ('SCO_U',(72.3933,-27.2333), 970),
                   ('TAS_A',(65.7790,-38.8995), 890),
                   ('TAS_L',(65.6402,-38.8987), 250),
                   ('THU_L',(76.3998,-68.2665), 570),
                   ('THU_U',(76.4197,-68.1463), 760),
                   ('UPE_L',(72.8932,-54.2955), 220), 
                   ('UPE_U',(72.8878,-53.5783), 940)]

path_to_PROMICE = "./data/PROMICE"  

# Function for making directories if they do not exists. 
def mkdir_p(path):
    try:
        os.makedirs(path)
        return 'Path created.'
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return 'Path already exists!'
        else:
            raise
            
mkdir_p(path_to_PROMICE)

# Goes through each station and fetch down data online. Necessary manipulations and sorting are made.
for ws in PROMICE_stations:
    if Path(f'{path_to_PROMICE}/{ws[0]}_day_v03.txt').is_file():
        print('\nPROMICE.csv file already exists.')
        pass
    else:
        print(ws)
        url = f'https://promice.org/PromiceDataPortal/api/download/f24019f7-d586-4465-8181-d4965421e6eb/v03/daily/{ws[0]}_day_v03.txt'
        filename = wget.download(url, out= path_to_PROMICE + f'/{ws[0]}_day_v03.txt')

if Path(f'{path_to_PROMICE}/PROMICE.csv').is_file():
    print('\nPROMICE.csv file already exists.')
    PROMICE = pd.read_csv (f'{path_to_PROMICE}/PROMICE.csv')
    pass
else: 
    # Create one big file "PROMICE.csv" with all the stations data. 
    # reading PROMICE file and joining them together
    for ws in PROMICE_stations:
        filepath = path_to_PROMICE + '/' + ws[0] + '_day_v03.txt'
    
        df = pd.read_csv (filepath, delim_whitespace=True)
        df = df[['Year','MonthOfYear','DayOfYear','AirTemperature(C)',
                 'AirTemperatureHygroClip(C)','SurfaceTemperature(C)','HeightSensorBoom(m)','HeightStakes(m)', 'IceTemperature1(C)', 'IceTemperature2(C)', 'IceTemperature3(C)', 'IceTemperature4(C)', 'IceTemperature5(C)', 'IceTemperature6(C)', 'IceTemperature7(C)', 'IceTemperature8(C)']]
        df['station_name'] = ws[0]
        df['latitude N'] = ws[1][0]
        df['longitude W'] = ws[1][1]
        df['elevation'] = float(ws[2])
        df.to_csv(path_to_PROMICE + '/' + ws[0] + '.csv', index=None)
    
    PROMICE = pd.DataFrame()
    filelist = [ f for f in os.listdir(path_to_PROMICE) if f.endswith(".csv") ]
    for f in filelist:
        print(f)
        PROMICE = PROMICE.append(pd.read_csv(f'{path_to_PROMICE}/{f}'))
    PROMICE.to_csv(f'{path_to_PROMICE}/PROMICE.csv', index=None)
    
PROMICE.rename(columns={'Year': 'year', 'DayOfYear': 'dayofyear', 
                            'IceTemperature1(C)': 'rtd0',
                            'IceTemperature2(C)': 'rtd1',
                            'IceTemperature3(C)': 'rtd2',
                            'IceTemperature4(C)': 'rtd3',
                            'IceTemperature5(C)': 'rtd4',
                            'IceTemperature6(C)': 'rtd5',
                            'IceTemperature7(C)': 'rtd6',
                            'IceTemperature8(C)': 'rtd7',
                            'station_name':'sitename'}, inplace=True)
PROMICE['date'] =   (np.asarray(PROMICE['year'], dtype='datetime64[Y]')-1970) + (np.asarray(PROMICE['dayofyear'], dtype='timedelta64[D]')-1)

PROMICE.drop( ['year', 'MonthOfYear', 'dayofyear', 'longitude W','latitude N','elevation'], axis=1,inplace=True)
PROMICE.set_index(['sitename', 'date'],inplace=True)
PROMICE.replace(to_replace=-999,value=np.nan,inplace=True)
sites_all =[item[0] for item in PROMICE_stations]
sites_all = np.delete(sites_all,1)
PROMICE = PROMICE.loc[sites_all,:]
PROMICE['surface_height'] = 2.6-PROMICE['HeightSensorBoom(m)']
PROMICE['surface_height2'] = 2.6-PROMICE['HeightStakes(m)']

gradthresh = 0.1

for site in ['EGP','KAN_U']: 
    vals = PROMICE.loc[site,'surface_height'].copy().values
    ind_maintenance = np.where(np.gradient(vals)<-0.2)[0]
    for i in range(len(ind_maintenance)):
        vals[(ind_maintenance[i]+1):] = vals[ind_maintenance[i]-4] - vals[ind_maintenance[i]+1] + vals[(ind_maintenance[i]+1):] 
    PROMICE.loc[site,'surface_height']=vals
    
for site in sites_all:
    tmp = PROMICE.loc[site,'surface_height']
    PROMICE.loc[site,'surface_height']= tmp.values - tmp[np.where(~np.isnan(tmp))[0][0]]
    tmp = PROMICE.loc[site,'surface_height2']
    PROMICE.loc[site,'surface_height2']= tmp.values - tmp[np.where(~np.isnan(tmp))[0][0]]
    # PROMICE.loc[site,'surface_height2'] = ftl.hampel(PROMICE.loc[site,'surface_height2']).values
    
    # applying gradient filter
    vals = PROMICE.loc[site,'surface_height'].copy().values
    vals[np.isnan(vals)]=-9999
    msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
    vals[msk] = np.nan
    vals[np.maximum(msk-1,0)] = np.nan
    vals[np.minimum(msk+1,len(vals)-1)] = np.nan
    vals[vals==-9999]=np.nan
    PROMICE.loc[site,'surface_height']=vals
    PROMICE.loc[site,'surface_height']= PROMICE.loc[site,'surface_height'].interpolate(method='linear').values
    PROMICE.loc[site,'surface_height']=ftl.smooth(PROMICE.loc[site,'surface_height'].values)

    vals = PROMICE.loc[site,'surface_height2'].copy().values
    vals[np.isnan(vals)]=-9999
    msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
    vals[msk] = np.nan
    vals[np.maximum(msk-1,0)] = np.nan
    vals[np.minimum(msk+1,len(vals)-1)] = np.nan
    vals[vals==-9999]=np.nan
    PROMICE.loc[site,'surface_height2']=vals
    PROMICE.loc[site,'surface_height2']= PROMICE.loc[site,'surface_height2'].interpolate(method='linear').values
    PROMICE.loc[site,'surface_height2']=ftl.smooth(PROMICE.loc[site,'surface_height2'].values)
    

#%%
f1, ax = plt.subplots(4,6,figsize=(20, 15))
f1.subplots_adjust(hspace=0.1, wspace=0.05,
                   left = 0.03 , right = 0.95 ,
                   bottom = 0.1 , top = 0.95)
count = -1
for site in sites_all:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    ax[i,j].plot(PROMICE.loc[site,'surface_height'].index.values,
                 PROMICE.loc[site,'surface_height'])
    ax[i,j].plot(PROMICE.loc[site,'surface_height2'].index.values,
                 PROMICE.loc[site,'surface_height2'])
    ax[i,j].set_title(site)    
    # ax[i,j].set_ylim([15, 0])
    ax[i,j].xaxis.set_major_locator(years)
    ax[i,j].xaxis.set_major_formatter(years_fmt)
    ax[i,j].xaxis.set_minor_locator(months)
    ax[i,j].set_xlim([datetime.date(2006, 5, 1), datetime.date(2019, 10, 1)])
    ax[i,j].set_xlabel("")
    if count<len(sites_all)-6:
        ax[i,j].set_xticklabels("")
    else:
        for label in ax[i,j].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
ax[-1, -1].axis('off')
ax[-1, -2].axis('off')
ax[-1, -3].axis('off')
f1.text(0.5, 0.1, 'Year', ha='center', size = 20)
f1.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 20)
f1.savefig('figures/PROMICE_surface_height.png')

#%% Correcting depth with surface height change
zz = ['rtd0', 'rtd1', 'rtd2', 'rtd3', 'rtd4', 'rtd5', 'rtd6', 'rtd7']
zz2 = ['depth_0', 'depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5', 'depth_6', 'depth_7']
ini_depth=[1,2,3,4,5,6,7,8]
PROMICE[zz2] = np.nan

for site in sites_all:
    for i,col in enumerate(zz2):
        PROMICE.loc[site,col]=ini_depth[i]

PROMICE[zz2] = PROMICE[zz2].add(PROMICE['surface_height'],axis='rows')
    
#%% Plotting thermistor depth
f1, ax = plt.subplots(4,6,figsize=(20, 15))
f1.subplots_adjust(hspace=0.1, wspace=0.05,
                   left = 0.03 , right = 0.95 ,
                   bottom = 0.1 , top = 0.95)
count = -1
for site in sites_all:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    ax[i,j].plot(PROMICE.loc[site,'surface_height'].index.values,
                 PROMICE.loc[site,'surface_height']*0)
    for k in range(0,9):
        ax[i,j].plot(PROMICE.loc[site,'surface_height'].index.values,
                     -PROMICE.loc[site,zz2[k]])
    ax[i,j].set_title(site)    
    # ax[i,j].set_ylim([15, 0])
    ax[i,j].xaxis.set_major_locator(years)
    ax[i,j].xaxis.set_major_formatter(years_fmt)
    ax[i,j].xaxis.set_minor_locator(months)
    ax[i,j].set_xlim([datetime.date(2006, 5, 1), datetime.date(2019, 10, 1)])
    ax[i,j].set_xlabel("")
    if count<len(sites_all)-6:
        ax[i,j].set_xticklabels("")
    else:
        for label in ax[i,j].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
ax[-1, -1].axis('off')
ax[-1, -2].axis('off')
ax[-1, -3].axis('off')
f1.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 20)
f1.savefig('figures/PROMICE_therm_depth.png')

#%% All firn temperature

f1, ax = plt.subplots(4,6,figsize=(20, 15))
f1.subplots_adjust(hspace=0.1, wspace=0.05,
                   left = 0.03 , right = 0.95 ,
                   bottom = 0.1 , top = 0.95)
count = -1
for site in sites_all:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    
    sitetemp= PROMICE.loc[site,zz]
    sitedep = PROMICE.loc[site,zz2]
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
f1.savefig('figures/PROMICE_temp.png')

# %% 10 m firn temp
T10m_df = pd.DataFrame()

count = -1
for site in sites_all:
    print(site)
    count = count+1
    
    sitetemp= PROMICE.loc[site,zz]
    sitedep = PROMICE.loc[site,zz2]
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
                          len(temps[i,ind_nonan][:,0])>2) and np.max(depths_from_surface[i,ind_nonan][:,0])>5:
            T10_temp['T'][i] = sp.interpolate.interp1d(depths_from_surface[i,ind_nonan][:,0],
                                                       temps[i,ind_nonan][:,0],
                                                       fill_value="extrapolate", 
                                                       kind='linear', 
                                                       assume_sorted=False)(10)
    vals = T10_temp['T'].copy().values
    vals[np.isnan(vals)]=-9999
    msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
    vals[msk] = np.nan
    vals[np.maximum(msk-1,0)] = np.nan
    vals[np.minimum(msk+1,len(vals)-1)] = np.nan
    vals[vals==-9999]=np.nan
    T10_temp['T']=vals
    # T10_temp['T']=ftl.smooth(T10_temp['T'].values)
    
    T10_temp.loc[T10_temp['T']>0, 'T'] = np.nan
    T10_temp.loc[T10_temp['T']<-70, 'T'] = np.nan
    T10m_df = T10m_df.append(T10_temp)
    # if np.sum(np.isnan(T_10m))>1:
    #     T_10m[np.argwhere(np.isnan(T_10m))]= \
    #         sp.interpolate.interp1d(np.argwhere(~np.isnan(T_10m))[:,0],\
    #                                 T_10m[np.argwhere(~np.isnan(T_10m))][:,0], \
    #                                     fill_value = 'extrapolate') (np.argwhere(np.isnan(T_10m)))

# T10m_df.loc[T10m_df['T'].values>-8,'T'] = np.nan
# T10m_df.loc[T10m_df['T'].values<-70,'T'] = np.nan

T10m_df.set_index(['sitename','date'],inplace=True)

T10m_interp = T10m_df.copy()
for site in sites_all:
    T_10m = T10m_interp.loc[site,'T'].values
    ind1 = np.where(np.logical_not(np.isnan(T_10m)))[0][0]
    ind2 = np.where(np.logical_not(np.isnan(T_10m)))[0][-1]
    if np.sum(np.isnan(T_10m))>1:
        T_10m[np.argwhere(np.isnan(T_10m))]= \
            sp.interpolate.interp1d(np.argwhere(~np.isnan(T_10m))[:,0],\
                                    T_10m[np.argwhere(~np.isnan(T_10m))][:,0], \
                                        fill_value = 'extrapolate') (np.argwhere(np.isnan(T_10m)))
    T_10m[T_10m>0] = np.nan
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
ax.set_xlim([datetime.date(2007, 5, 1), datetime.date(2019, 10, 1)])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
ax.set_xlabel("Year")
ax.set_ylabel('10 m firn temperature (C)')
f1.savefig('figures/PROMICE_T10.png')
    # plt.close(f1)   
    
#%% Plotting thermistor depth
f1, ax = plt.subplots(4,6,figsize=(20, 15))
f1.subplots_adjust(hspace=0.1, wspace=0.05,
                   left = 0.03 , right = 0.95 ,
                   bottom = 0.1 , top = 0.95)
count = -1
for site in sites_all:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    ax[i,j].plot(T10m_df.loc[site,'T'].index.values,T10m_df.loc[site,'T'].values,
            color = colors[i], label=site,linewidth=2)
    ax[i,j].plot(T10m_interp.loc[site,'T'].index.values,T10m_interp.loc[site,'T'].values,
            color = colors[i], linestyle='--', linewidth=2)
    ax[i,j].set_title(site)    
    # ax[i,j].set_ylim([15, 0])
    ax[i,j].xaxis.set_major_locator(years)
    ax[i,j].xaxis.set_major_formatter(years_fmt)
    ax[i,j].xaxis.set_minor_locator(months)
    ax[i,j].set_xlim([datetime.date(2006, 5, 1), datetime.date(2019, 10, 1)])
    ax[i,j].set_xlabel("")
    if count<len(sites_all)-6:
        ax[i,j].set_xticklabels("")
    else:
        for label in ax[i,j].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
ax[-1, -1].axis('off')
ax[-1, -2].axis('off')
ax[-1, -3].axis('off')
f1.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 20)
f1.savefig('figures/PROMICE_T10_2.png')