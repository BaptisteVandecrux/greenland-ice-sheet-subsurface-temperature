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
needed_cols = ["date", "site", "latitude", "longitude", "elevation", 
               "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note"]
#  all GC-Net stations
from nead import read
np.seterr(invalid="ignore")
df_info = pd.read_csv('Data/GC-Net/GC-Net_location.csv', skipinitialspace=True)
df_info = df_info.loc[df_info.Northing>0,:]
path_to_GCNet = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/GC-Net-Level-1-data-processing/L1/'

# %% Plotting string data
plotting = False
if plotting:
    print('plotting')
    for ID, site in zip(df_info.ID, df_info.Name):   
        df_aws = read(path_to_GCNet+site.replace(' ','')+'.csv').to_dataframe()
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

# %% Compiling AWS subsurface temperatures

df_gcn = pd.DataFrame()
temp_var = ['TS1','TS2', 'TS3', 'TS4', 'TS5', 'TS6', 'TS7', 'TS8', 'TS9', 'TS10']
depth_var = ['DTS1', 'DTS2','DTS3', 'DTS4', 'DTS5', 'DTS6', 'DTS7', 'DTS8', 'DTS9', 'DTS10']
for ID, site in zip(df_info.ID, df_info.Name):   
    df_aws = read(path_to_GCNet+'/daily/'+site.replace(' ','')+'_daily.csv').to_dataframe()
    df_aws['time'] = pd.to_datetime(df_aws.timestamp, utc=True)
    df_aws = df_aws.set_index('time')
    print(site)
    if 'DTS1' not in df_aws.columns:
        print('No temperature data')
        continue        
    if df_aws[temp_var].isnull().all().all():
        print('No temperature data')
        continue

    df_aws.index = df_aws.index.rename('date')

    df_aws = df_aws[temp_var+depth_var+['TS_10m']]
    df_aws["latitude"] = df_info.loc[df_info.Name==site, "Northing"].values[0]
    df_aws["longitude"] = df_info.loc[df_info.Name==site, "Easting"].values[0]
    df_aws["elevation"] = df_info.loc[df_info.Name==site, "Elevationm"].values[0]
    df_aws["site"] = site
    df_aws["note"] = ""

    # filtering
    for v in temp_var+['TS_10m']:
        df_aws.loc[df_aws[v] > 0.1, v] = np.nan
        df_aws.loc[df_aws[v] < -70, v] = np.nan
    df_gcn = pd.concat((df_gcn, df_aws.reset_index()))

df_gcn = df_gcn.loc[df_gcn.TS_10m.notnull(), :]

df_gcn[
    "reference"
] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103. and Steffen, K. and J. Box: Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001 and Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023. and Vandecrux, B., Box, J.E., Ahlstrøm, A.P., Andersen, S.B., Bayou, N., Colgan, W.T., Cullen, N.J., Fausto, R.S., Haas-Artho, D., Heilig, A., Houtz, D.A., How, P., Iosifescu Enescu , I., Karlsson, N.B., Kurup Buchholz, R., Mankoff, K.D., McGrath, D., Molotch, N.P., Perren, B., Revheim, M.K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P.J., Zwally, J., Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented Level 1 dataset, Submitted to ESSD, 2023"
df_gcn["reference_short"] = "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)"
df_gcn= df_gcn.reset_index().drop(columns='index')
#%%
# saving full-res data
ds_gcn = ftl.df_to_xarray(df_gcn, temp_var, depth_var)

# saving monthly T10m
num_col = ['date', 'TS_10m', 'latitude', 'longitude', 'elevation', 'site']
non_num_col = ['date',  'site', 'note', 'reference', 'reference_short']
df_gcn_m = df_gcn[num_col].set_index('date').groupby('site').resample('M').mean()
df_gcn_m[non_num_col[2:]] = df_gcn[non_num_col].set_index('date').groupby('site').resample('M').first()[non_num_col[2:]] 
df_gcn_m['depthOfTemperatureObservation'] = 10
df_all = df_gcn_m.reset_index().rename(columns={'TS_10m':'temperatureObserved',
                                'time': 'date'})[needed_cols].copy()

#%% Summit string 2005-2009
print('Summit string 2005-2009')
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

for i in range(1, len(col_temp)+1):
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

df.loc[:'2008',col_depth]  = df.loc[:'2008',col_depth].bfill() 
for col in col_depth:
    df[col] = df[col] + df['surface_height']

df['TS_10m'] = ftl.interpolate_temperature(
    df.index, df[col_depth].values, df[col_temp].values, 
    surface_height=df.surface_height, kind="linear", title='Summit'
).set_index('date')

df = df[col_temp+col_depth+['TS_10m']].reset_index().rename(columns={'index':'date'})

site = 'Summit'
df["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df["reference"] = "GC-Net unpublished"
df["reference_short"] = "GC-Net unpublished"
df["site"] = site
df["note"] = ''
df = df.rename(columns=dict(zip(col_depth, ['DTS'+str(i) for i in range(1,len(col_depth)+1)])))
df = df.rename(columns=dict(zip(col_temp, ['TS'+str(i) for i in range(1,len(col_depth)+1)])))

# saving full-res
ds_sum_1 = ftl.df_to_xarray(df, 
                        ['TS'+str(i) for i in range(1,len(col_depth)+1)],
                        ['DTS'+str(i) for i in range(1,len(col_depth)+1)])
# saving T10m
df['depthOfTemperatureObservation'] = 10
df_all = pd.concat((df_all, df.rename(columns={'TS_10m':'temperatureObserved',
                                'time': 'date'})[needed_cols]),
                   ignore_index=True)
# %% Summit string 2000-2002
print('Summit string 2000-2002')
df = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2000-2002 thermistor string/2002_sun_00_01_raw.dat', header = None)
df.columns = 'id;day;hour_min;TS1;TS2;TS3;TS4;TS5;TS6;TS7;TS8;TS9;TS10;TS11;TS12;TS13;TS14;TS15;TS16'.split(';')
df['year'] = 2000
df.loc[4781:,'year'] = 2001
df2 = pd.read_csv('Data/GC-Net/Summit Snow Thermistors/2000-2002 thermistor string/2002_sum_01_02_raw.dat', header = None)
df2.columns = 'id;day;hour_min;TS1;TS2;TS3;TS4;TS5;TS6;TS7;TS8;TS9;TS10;TS11;TS12;TS13;TS14;TS15;TS16'.split(';')
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
df['date'] = pd.to_datetime(df.year*100000+df.day*100+df.hour, format='%Y%j%H',utc=True, errors='coerce')
df = df.set_index('date')  #.drop(columns=['year','hour','minute','id','day', 'hour_min'])

col_temp = [v for v in df.columns if 'TS' in v]

a=-9.0763671; b=0.704343; c=0.00919; d=0.000137; e=0.00000116676; f=0.00000000400674

# calibration coefficients from ice bath at Summit 2000
coef = [0.361 , 0.228, 0.036, 0.170, 0.170, 0.228, -0.022, 0.361, 0.036, 0.036, 0.228, 0.170, 0.323, 0.132, 0.323, 0.266]

#       convert UUB thermister reading to temperature      
df[col_temp]=a+b*df[col_temp]+c*(df[col_temp]**2)+d*(df[col_temp]**3)+e*(df[col_temp]**4)+f*(df[col_temp]**5)

# use the calibration coefficient from Summit 2000 ice bath

for i, col in enumerate(col_temp):
    df[col] = df[col] + coef[i]
    df.loc[df[col]>-5,col] = np.nan

depths = ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2.5m', '2m', '1.5m',
       '1m', '0.75m', '0.5m', '0.3m', '0.1m']
for col, col2 in zip(depths, col_temp):   
    df['DTS'+col2.replace('TS','')]=np.nan
    df['DTS'+col2.replace('TS','')] = float(col.replace('m','')) +  np.arange(len(df.index)) * 0.62/365/24
df.loc['2001-06-11':'2001-12-15', 'DTS2'] = np.nan
col_depth = ['DTS'+str(i) for i in range(1,17)]
df['TS_10m'] = ftl.interpolate_temperature(
    df.index, df[col_depth].values, df[col_temp].values,
    kind="linear", title='Summit'
).set_index('date')

df = df[col_temp+col_depth+['TS_10m']]
df = df.reset_index()
site = 'Summit'
df["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df["reference"] = "GC-Net unpublished"
df["reference_short"] = "GC-Net unpublished"
df["site"] = site
df["note"] = ''

# saving full-res
ds_sum_2 = ftl.df_to_xarray(df, 
                        ['TS'+str(i) for i in range(1,len(col_depth)+1)],
                        ['DTS'+str(i) for i in range(1,len(col_depth)+1)])
# saving T10m
df['depthOfTemperatureObservation'] = 10
df_all = pd.concat((df_all, df.rename(columns={'TS_10m':'temperatureObserved',
                                'time': 'date'})[needed_cols]),
                   ignore_index=True)

# %% Swiss Camp TENT thermistor
print('Swiss Camp TENT thermistor')
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
    df.columns = ['doy'] + ['TS'+str(i) for i in df.columns[1:].values]
    year = float(f[:4])
    df['year'] = year
    
    if (any(df.doy.diff()<0))>0:
        for i, ind in enumerate(df.loc[df.doy.diff()<0,:].index.values):
            df.loc[slice(ind), 'year'] = year - (len(df.loc[df.doy.diff()<0,:].index.values)-i)
    df_swc = pd.concat((df_swc,df))
    
df_swc['date'] = pd.to_datetime(df_swc.year*1000+df_swc.doy, format='%Y%j', utc=True, errors='coerce')
df_swc = df_swc.set_index('date') 

col_temp = [v for v in df_swc.columns if 'TS' in v]
col_depth = ['DTS'+str(i) for i in range(1,len(col_temp)+1)]
for col in col_temp:
    df_swc.loc[df_swc[col]<-17,col] = np.nan
    df_swc.loc[df_swc[col]>10,col] = np.nan

df_swc[col_temp].plot()

df_swc['TS_10m'] = df_swc['TS10']
# Installation depth
depth_ini = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0.75, 0.5]
for i,d in enumerate(depth_ini):
    df_swc[col_depth[i]] = d

df_swc = df_swc.loc[df_swc[col_temp].notnull().any(axis=1),:]

df_swc = df_swc.drop(columns=['doy','year']).reset_index().rename(columns={'time':'date'})
site = 'Swiss Camp'
df_swc["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df_swc["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df_swc["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df_swc["reference"] = "GC-Net unpublished"
df_swc["reference_short"] = "GC-Net unpublished"
df_swc["site"] = site
df_swc["note"] = ''
for v in df_swc.columns:
    if v not in df_all.columns:
        df_all[v] = np.nan

# saving full res
ds_swc = ftl.df_to_xarray(df_swc, col_temp,col_depth)

# saving T10m
df_swc['depthOfTemperatureObservation'] = 10
df_all = pd.concat((df_all, df_swc.rename(columns={'TS_10m':'temperatureObserved',
                                'time': 'date'})[needed_cols]),
                   ignore_index=True)
# %% exporting all subsurface temperatures to netcdf
export_nc = False
import xarray as xr
if export_nc:
    print('exporting to nc')
    ds_sum = ftl.merge_two_xr(ds_sum_1, ds_sum_2)
    ds_extra = ftl.merge_two_xr(ds_sum, ds_swc)
    ds_gcn_merged = ftl.merge_two_xr(ds_gcn, ds_extra)
    ds_gcn_merged['reference'] = ds_gcn_merged.reference.astype(str)
    ftl.write_netcdf(ds_gcn_merged, 
                     'Data/netcdf/Historical_GC-Net_subsurface_temperatures.nc')

# %% print T10m to file
df_all.to_csv("Data/GC-Net/10m_firn_temperature.csv")


