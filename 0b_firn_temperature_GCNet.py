# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import firn_temp_lib as ftl

plt.close('all')

#  all GC-Net stations
from nead import read
np.seterr(invalid="ignore")
df_info = pd.read_csv('Data/GC-Net/GC-Net_location.csv', skipinitialspace=True)
df_info = df_info.loc[df_info.Northing>0,:]
path_to_GCNet = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/GC-Net-Level-1-data-processing/L1/'

# df_info = df_info.loc[df_info.Name=='Swiss Camp']
plotting = False
if plotting:
    print('plotting')
    for ID, site in zip(df_info.ID, df_info.Name):   
        df_aws = read(path_to_GCNet+str(ID).zfill(2)+'-'+site.replace(' ','')+'.csv').to_dataframe()
        df_aws['time'] = pd.to_datetime(df_aws.timestamp, utc=True)
        df_aws = df_aws.set_index('time')
        
        depth_cols_name = [v for v in df_aws.columns if "DTS" in v]
        temp_cols_name = [v for v in df_aws.columns if "TS" in v and "DTS" not in v and "10m" not in v]
        print(site)
        if len(temp_cols_name)==0:
            print('No thermistor data')
            continue
    
        if df_aws[temp_cols_name].isnull().all().all():
            print('No thermistor data')
            continue
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        plt.subplots_adjust(left=0.08, right=0.98, wspace=0.25, top=0.9)
        df_aws["HS_combined"].plot(
            ax=ax[0], color="black", label="surface", linewidth=2
        )
        (df_aws["HS_combined"] - 10).plot(
            ax=ax[0],
            color="red",
            linestyle="-",
            linewidth=2,
            label="10 m depth",
        )
    
    
        for i, col in enumerate(depth_cols_name):
            (-df_aws[col] + df_aws["HS_combined"]).plot(
                ax=ax[0],
                label="_nolegend_",alpha=0.5,
            )
    
        ax[0].set_ylim(
            df_aws["HS_combined"].min() - 22,
            df_aws["HS_combined"].max() + 1,
        )
        # ax[1].set_ylim(-48, -11)
        
        for i in range(len(temp_cols_name)):
            df_aws[temp_cols_name[i]].interpolate(limit=14).plot(
                ax=ax[1], label="_nolegend_", alpha=0.5, linewidth=0.5
            )
    
        if len(df_aws["TS_10m"]) == 0:
            print("No 10m temp for ", site)
        else:
            df_aws["TS_10m"].resample(
                "W"
            ).mean().plot(ax=ax[1], color="red", linewidth=3, label="10 m temperature")
        for i in range(2):
            ax[i].plot(np.nan, np.nan, c='w', label=' ') 
            ax[i].plot(np.nan, np.nan, c='w', label='individual sensors') 
            ax[i].plot(np.nan, np.nan, c='w', label=' ') 
            ax[i].legend(loc='lower right')
        ax[0].set_ylabel("Height (m)")
        ax[1].set_ylabel("Subsurface temperature ($^o$C)")
        fig.suptitle(site)
        fig.savefig("figures/string processing/GC-Net_" + site + ".png", dpi=300)


# %% 10 m firn temp
df_gcn = pd.DataFrame()

for ID, site in zip(df_info.ID, df_info.Name):   
    df_aws = read(path_to_GCNet+str(ID).zfill(2)+'-'+site.replace(' ','')+'_daily.csv').to_dataframe()
    df_aws['time'] = pd.to_datetime(df_aws.timestamp, utc=True)
    df_aws = df_aws.set_index('time')
    print(site)
    if 'TS_10m' not in df_aws.columns:
        print('No 10m temperature data')
        continue        
    if df_aws['TS_10m'].isnull().all():
        print('No 10m temperature data')
        continue

    df_10 = df_aws[['TS_10m']].copy()
    df_10.columns = ['temperatureObserved']
    df_10.index = df_10.index.rename('date')
    
    df_10["latitude"] = df_info.loc[df_info.Name==site, "Northing"].values[0]
    df_10["longitude"] = df_info.loc[df_info.Name==site, "Easting"].values[0]
    df_10["elevation"] = df_info.loc[df_info.Name==site, "Elevationm"].values[0]
    df_10["site"] = site
    df_10["depthOfTemperatureObservation"] = 10
    df_10["note"] = ""

    # filtering
    df_10.loc[df_10["temperatureObserved"] > 0.1, "TS_10m"] = np.nan
    df_10.loc[df_10["temperatureObserved"] < -70, "TS_10m"] = np.nan
    df_save = df_10.copy()
    df_gcn = pd.concat((df_gcn, df_10.reset_index()))

df_gcn = df_gcn.loc[df_gcn.temperatureObserved.notnull(), :]

#%% Summit string 2007-2009
df = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2007-2009/t_hour.dat', delim_whitespace=True, header=None)
df.columns = 'id;year;day;hour_min;t_1;t_2;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14'.split(';')
df[df==-6999] = np.nan

df.loc[df.hour_min==2400,'day'] = df.loc[df.hour_min==2400,'day']+1
df.loc[df.hour_min==2400,'hour_min'] = 0
df.loc[df.day==367,'year'] = df.loc[df.day==367,'year']+1
df.loc[df.day==367,'day'] = 1

df['hour'] = np.trunc(df.hour_min/100)
df['minute'] = df.hour_min - df.hour*100
df['time'] = pd.to_datetime(df.year*100000+df.day*100+df.hour, format='%Y%j%H', utc=True, errors='coerce')
df = df.set_index('time').drop(columns=['year','hour','minute','id','day', 'hour_min'])


df3 = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/snow.dat', header = None, delim_whitespace=True)
df3.columns = 'year;day;hour_min;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14'.split(';')
df3[df3==-6999] = np.nan

df3.loc[df3.hour_min==2400,'day'] = df3.loc[df3.hour_min==2400,'day']+1
df3.loc[df3.hour_min==2400,'hour_min'] = 0
df3.loc[df3.day==367,'year'] = df3.loc[df3.day==367,'year']+1
df3.loc[df3.day==367,'day'] = 1

df3['hour'] = np.trunc(df3.hour_min/100)
df3['minute'] = df3.hour_min - df3.hour*100
df3['time'] = pd.to_datetime(df3.year*100000+df3.day*100+df3.hour, format='%Y%j%H', utc=True, errors='coerce')
df3 = df3.set_index('time').drop(columns=['year','hour','minute','day', 'hour_min'])
df3[['t_1','t_2']] = np.nan

df = pd.concat((df, df3))
df = df.resample('H').first()

col_temp = [v for v in df.columns if 't_' in v]
# df[col_temp].plot()

for i in range(len(col_temp)):
    df['depth_'+str(i)] = np.nan
col_depth = [v for v in df.columns if 'depth_' in v]

# adding the three dates
surface = [4.35, 5.00, 5.57]
depths =  np.array([[2.15, 2.65, 3.15, 3.65, 4.15, 4.65, 
                     4.65,  4.85, 5.35, 6.35, 9.35, 11.85, 14.35, 19.35],
           [2.75, 3.40, 3.90, 4.40, 4.90, 5.40, 5.40, 5.60, 
            6.40, 7.40, 10.4, 12.60, 15.4, 20.40],
           [np.nan, np.nan, 0.50, 2.00, 3.32, 4.97 ,5.97,  
            6.17, 6.97, 7.97, 10.97, 13.17, 16.17, 21.17]])

for i, date in enumerate(['2007-04-07', '2008-07-25', '2009-05-19']):
    tmp = df.iloc[0:1,:]*np.nan
    tmp.index = [pd.to_datetime(date, utc=True)]
    tmp['surface_height'] = surface[i]
    tmp[col_depth] = depths[i,:]
    df = pd.concat((tmp,df))
df = df.sort_index()
df.surface_height = df.surface_height.interpolate().values - df.surface_height[0]
tmp = df.surface_height.diff()
tmp.loc[tmp==0] = 0.62/365/24
df.surface_height = tmp.cumsum().values
df.loc[df[col_depth[-1]].notnull(), col_depth] = df.loc[df[col_depth[-1]].notnull(), col_depth] - np.repeat(np.expand_dims(df.loc[df[col_depth[-1]].notnull(), 'surface_height'].values,1), len(col_depth), axis=1)
df[col_depth] = df[col_depth].ffill().values

for col in col_depth:
    df[col] = df[col] + df['surface_height']
df_10 = ftl.interpolate_temperature(
    df.index, df[col_depth].values, df[col_temp].values, 
    surface_height=df.surface_height, kind="linear", title='Summit'
)


df_10 = df_10.reset_index()
site = 'Summit'
df_10["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df_10["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df_10["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df_10["reference"] = "GC-Net"
df_10["reference_short"] = "GC-Net"
df_10["site"] = site + ''
df_10["note"] = ''
df_10["depthOfTemperatureObservation"] = 10
df_gcn = pd.concat((df_gcn,
    df_10[
        [
            "date",
            "site",
            "latitude",
            "longitude",
            "elevation",
            "depthOfTemperatureObservation",
            "temperatureObserved",
            "reference",
            "reference_short",
            "note",
        ]
    ]
))
df_save = df.copy()

df = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2000-2002 thermistor string/2002_sun_00_01_raw.dat', header = None)
df.columns = 'id;day;hour_min;t_1;t_2;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14;t_15;t_16'.split(';')
df['year'] = 2000
df.loc[4781:,'year'] = 2001
df2 = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2000-2002 thermistor string/2002_sum_01_02_raw.dat', header = None)
df2.columns = 'id;day;hour_min;t_1;t_2;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14;t_15;t_16'.split(';')
df2['year'] = 2001
df2.loc[4901:,'year'] = 2002
df = pd.concat((df, df2))

df[df==-6999] = np.nan

df.loc[df.hour_min==2400,'day'] = df.loc[df.hour_min==2400,'day']+1
df.loc[df.hour_min==2400,'hour_min'] = 0
df.loc[df.day==367,'year'] = df.loc[df.day==367,'year']+1
df.loc[df.day==367,'day'] = 1

df['hour'] = np.trunc(df.hour_min/100)
df['minute'] = df.hour_min - df.hour*100
df['time'] = pd.to_datetime(df.year*100000+df.day*100+df.hour, format='%Y%j%H',utc=True, errors='coerce')
df = df.set_index('time')  #.drop(columns=['year','hour','minute','id','day', 'hour_min'])

col_temp = [v for v in df.columns if 't_' in v]

a=-9.0763671
b=0.704343
c=0.00919
d=0.000137
e=0.00000116676
f=0.00000000400674

# calibration coefficients from ice bath at Summit 2000
coef = [0.361 , 0.228, 0.036, 0.170, 0.170, 0.228, -0.022, 0.361, 0.036, 0.036, 0.228, 0.170, 0.323, 0.132, 0.323, 0.266]

#       convert UUB thermister reading to temperature      
df[col_temp]=a+b*df[col_temp]+c*(df[col_temp]**2)+d*(df[col_temp]**3)+e*(df[col_temp]**4)+f*(df[col_temp]**5)

# use the calibration coefficient from Summit 2000 ice bath

for i, col in enumerate(col_temp):
    df[col] = df[col] + coef[i]
    df.loc[df[col]>-5,col] = np.nan
    

# df_new = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2000-2002 thermistor string/2000_ice_ascii.dat')
# df_new['year'] = 2000
# df_new['day'] = np.trunc(df_new['jd.time'])
# df_new['hour'] = (df_new['jd.time'] - df_new['day'])*24
# df_new['time'] = pd.to_datetime(df_new.year*100000+df_new.day*100+df_new.hour, format='%Y%j%H', errors='coerce')
# df_new = df_new.set_index('time').drop(columns=['jd.time','year','day','hour'])
# plt.close('all')

# df_save[[v for v in df_save.columns if 't_' in v]].plot(ax=plt.gca())
# plt.figure()
depths = ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2.5m', '2m', '1.5m',
       '1m', '0.75m', '0.5m', '0.3m', '0.1m']
for col,col2 in zip(depths, col_temp):
    # df_new[[col]].plot(ax=plt.gca(), marker = 'o', linestyle = 'None')
    # df[[col2]].ds_ann.T10m.sel(time=slice('1980','2016')).mean(dim='time').weighted(weights).mean().valuest(ax=plt.gca())
    df['depth_'+col2.replace('t_','')]=np.nan
    df['depth_'+col2.replace('t_','')] = float(col.replace('m','')) +  np.arange(len(df.index)) * 0.62/365/24
df.loc['2001-06-11':'2001-12-15', 't_2'] = np.nan
col_depth = ['depth_'+str(i) for i in range(1,17)]
df_10 = ftl.interpolate_temperature(
    df.index, df[col_depth].values, df[col_temp].values,
    kind="linear", title='Summit'
)

df_10 = df_10.reset_index()
site = 'Summit'
df_10["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df_10["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df_10["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df_10["reference"] = "GC-Net"
df_10["reference_short"] = "GC-Net"
df_10["site"] = site + ''
df_10["note"] = ''
df_10["depthOfTemperatureObservation"] = 10
df_gcn = pd.concat((df_gcn,
    df_10[
        [
            "date",
            "site",
            "latitude",
            "longitude",
            "elevation",
            "depthOfTemperatureObservation",
            "temperatureObserved",
            "reference",
            "reference_short",
            "note",
        ]
    ]
))

# %% Swiss Camp TENT thermistor
import os
list_file = os.listdir('Data/GC-Net/Swiss Camp TENT thermistor')
list_file = [f for f in list_file if f.lower().endswith('.dat')]
# 2000_ICETEMP.DAT "l=channel3 +3.5 C"
df_swc = pd.DataFrame()
for f in list_file:
    print(f)
    df = pd.read_csv('Data/GC-Net/Swiss Camp TENT thermistor/'+f, 
                     header=None,
                     delim_whitespace=True)
    df = df.apply(pd.to_numeric)
    df.columns = ['doy'] + ['t_'+str(i) for i in df.columns[1:].values]
    year = float(f[:4])
    df['year'] = year
    
    if (any(df.doy.diff()<0))>0:
        for i, ind in enumerate(df.loc[df.doy.diff()<0,:].index.values):
            df.loc[slice(ind), 'year'] = year - (len(df.loc[df.doy.diff()<0,:].index.values)-i)
    df_swc = pd.concat((df_swc,df))
    
df_swc['time'] = pd.to_datetime(df_swc.year*1000+df_swc.doy, format='%Y%j', utc=True, errors='coerce')
df_swc = df_swc.set_index('time')  #.drop(columns=['year','hour','minute','id','day', 'hour_min'])

col_temp = [v for v in df.columns if 't_' in v]
for col in col_temp:
    df_swc.loc[df_swc[col]<-17,col] = np.nan
    df_swc.loc[df_swc[col]>10,col] = np.nan

df_swc = df_swc.resample('D').first()
df_swc[col_temp].plot()

df_10 = df_swc['t_10'].reset_index()
df_10 = df_10.rename(columns={'time':'date','t_10':'temperatureObserved'})
site = 'Swiss Camp'
df_10["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df_10["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df_10["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df_10["reference"] = "GC-Net"
df_10["reference_short"] = "GC-Net"
df_10["site"] = site + ''
df_10["note"] = ''
df_10["depthOfTemperatureObservation"] = 10
df_gcn = pd.concat((df_gcn,
    df_10[
        [
            "date",
            "site",
            "latitude",
            "longitude",
            "elevation",
            "depthOfTemperatureObservation",
            "temperatureObserved",
            "reference",
            "reference_short",
            "note",
        ]
    ]
))


# %% print to file

df_gcn.to_csv("Data/GC-Net/10m_firn_temperature.csv")
