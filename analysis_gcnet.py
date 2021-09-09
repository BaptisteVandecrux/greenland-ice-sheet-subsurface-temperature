# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
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
#%%
print('Loading GC-Net')
time.sleep(0.2)
sites=['CP1','DYE-2','NASA-E', 'NASA-SE','NASA-U','Saddle',
       'SouthDome','Summit','TUNU-N']
lat = [69.8798, 66.4800, 66.4797, 65.9995, 63.1489, 75.0000, 73.8419,
       72.5797, 78.0168]
lon = [-46.9867, -46.2789, -42.5002, -44.5002, -44.8172, -29.9997, -49.4983,
       -38.5045, -33.9939]
elev = [2022.0, 2165.0, 2425.0, 2559.0, 2922.0, 2631.0, 2369.0, 3254.0, 2113.0]

df_gcnet = pd.DataFrame()
for ii, site in progressbar.progressbar(enumerate(sites)):
    ds = xr.open_dataset('Data/Vandecrux et al. 2020/'+site+'_T_firn_obs.nc')
    df = ds.to_dataframe()
    df = df.reset_index(0).groupby('level').resample('D').mean()
    df.reset_index(0,inplace=True, drop=True)  
    df.reset_index(inplace=True)  
    
    df_d = pd.DataFrame()
    df_d['date'] = df.loc[df['level']==1,'time']
    plt.figure()
    for i in range(1,11):
        plt.plot(df.loc[df['level']==i,'time'].values, 
                 df.loc[df['level']==i,'T_firn'].values)
                 # - df.loc[df['level']==10,'T_firn'].values
                 # + df.loc[df['level']==10,'T_firn'].mean())
        df_d['rtd'+str(i)] = df.loc[df['level']==i,'T_firn'].values  
    break
    for i in range(1,11):
        df_d['depth_'+str(i)] = df.loc[df['level']==i,'Depth'].values
    
    df_10 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,11)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,11)]].values, 10).set_index('date')
    df_5 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,11)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,11)]].values, 5).set_index('date')
    df_15 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,11)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,11)]].values, 15).set_index('date')
    df_20 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,11)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,11)]].values, 20).set_index('date')
    plt.figure()
    df_5.temperatureObserved.plot(label='5 m')
    df_10.temperatureObserved.plot(label='10 m')
    df_15.temperatureObserved.plot(label='15 m')
    df_20.temperatureObserved.plot(label='20 m')
    plt.legend()
    plt.title(site)