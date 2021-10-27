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
import os
import xarray as xr
import time
def interp_pandas(s, kind ="quadratic"):
    # A mask indicating where `s` is not null
    m = s.notna().values
    s_save=s.copy()
    # Construct an interpolator from the non-null values
    # NB 'kind' instead of 'method'!
    kw = dict(kind=kind, fill_value="extrapolate")
    f = interp1d(s[m].index, s.loc[m].values.reshape(1,-1)[0], **kw) 
    
    # Apply this to the indices of the nulls; reconstruct a series
    s[~m] = f(s[~m].index)[0]
    
    # plt.figure()
    # s.plot(marker='o',linestyle='none')
    # s_save.plot(marker='o',linestyle='none')
    # plt.xlim(0, 60)
    return s


# %% Mock and Weeks  
print('Loading Mock and Weeks')
df_all = pd.DataFrame(columns=['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note'])

df_MW = pd.read_excel('Data/MockandWeeks/CRREL RR- 170 digitized.xlsx')
df_MW.loc[df_MW.Month.isnull(), 'Day']=1
df_MW.loc[df_MW.Month.isnull(), 'Month']=1
df_MW['date'] = pd.to_datetime(df_MW[['Year', 'Month', 'Day']])
df_MW['note'] = 'as reported in Mock and Weeks 1965'
df_MW['depthOfTemperatureObservation'] = 10
df_MW=df_MW.rename(columns={'Refstationnumber':'site',
                            'Lat(dec_deg)':'latitude',
                            'Lon(dec_deg)':'longitude',
                            'Elevation':'elevation',
                            '10msnowtemperature(degC)':'temperatureObserved',
                            'Reference':'reference'})
df_MW = df_MW.loc[df_MW['temperatureObserved'].notnull()]
df_all = df_all.append(df_MW[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

# %% Benson (not reported in Mock and Weeks)  
print('Loading Benson 1962')
df_benson = pd.read_excel('Data/Benson 1962/Benson_1962.xlsx')
df_benson.loc[df_benson.Month.isnull(), 'Day']=1
df_benson.loc[df_benson.Month.isnull(), 'Month']=1
df_benson['date'] = pd.to_datetime(df_MW[['Year', 'Month', 'Day']])
df_benson['note'] = ''
df_benson['depthOfTemperatureObservation'] = 10
df_benson=df_benson.rename(columns={'Refstationnumber':'site',
                            'Lat(dec_deg)':'latitude',
                            'Lon(dec_deg)':'longitude',
                            'Elevation':'elevation',
                            '10msnowtemperature(degC)':'temperatureObserved',
                            'Reference':'reference'})
df_all = df_all.append(df_benson[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

# %% Polashenski  
print('Loading Polashenski')
df_Pol = pd.read_csv('Data/Polashenski/2013_10m_Temperatures.csv')
df_Pol.columns = df_Pol.columns.str.replace(' ', '')
df_Pol.date = pd.to_datetime(df_Pol.date,format='%m/%d/%y')
df_Pol['reference'] = 'Polashenski, C., Z. Courville, C. Benson, A. Wagner, J. Chen, G. Wong, R. Hawley, and D. Hall (2014), Observations of pronounced Greenland ice sheet firn warming and implications for runoff production, Geophys. Res. Lett., 41, 4238–4246, doi:10.1002/2014GL059806.'
df_Pol['reference_short'] = 'Polashenski et al. (2014)'
df_Pol['note'] = ''
df_Pol['longitude'] = -df_Pol['longitude']
df_Pol['depthOfTemperatureObservation'] = df_Pol['depthOfTemperatureObservation'].str.replace('m','').astype(float)
df_all = df_all.append(df_Pol[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

# %% Ken's dataset  
print('Loading Kens dataset')
df_Ken = pd.read_excel('Data/greenland_ice_borehole_temperature_profiles-main/data_filtered.xlsx')

df_all = df_all.append(df_Ken[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)
# %% Sumup
# df_sumup = pd.read_csv('Data/Sumup/sumup_10m_borehole_temperature.csv')
# df_sumup = df_sumup.loc[df_sumup.Latitude>0]
# df_sumup['date'] = pd.to_datetime([str(t) for t in df_sumup.Date_Taken],format= '%Y%m%d')
# df_sumup['reference'] = 'Miller, O.L., Solomon, D.K.,,Miege, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R.M., Legchenko, A.., Voss, C.I., Montgomery, L., McConnell, J.R. (in preparation) Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data'
# df_sumup['reference_short'] = 'Miller et al. (2018) as in sumup'
# df_sumup['site']='FA'
# df_sumup['note']='as reported in Sumup'

# df_sumup = df_sumup.rename(columns={'Latitude':'latitude','Longitude':'longitude','Elevation':'elevation','Depth_Taken':'depthOfTemperatureObservation','Temperature':'temperatureObserved'})
# df_all = df_all.append(df_sumup[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']],ignore_index=True)

# ====> Sumup not used. Redundant with Miege and containing positive temperatures

# %% Miege aquifer  
print('Loading firn aquifer data')
time.sleep(0.2)
metadata = np.array([['FA-13', 66.181, 39.0435, 1563],
 ['FA-15-1', 66.3622, 39.3119, 1664],
 ['FA-15-2', 66.3548, 39.1788, 1543]])

# mean_accumulation = 1 # m w.e. from Miege et al. 2014
# thickness_accum = 2.7 # thickness of the top 1 m w.e. in the FA13 core
thickness_accum = 1.4 # Burial of the sensor between their installation in Aug 2015 and revisit in Aug 2016
df_miege = pd.DataFrame()

for k, site in enumerate(['FA_13','FA_15_1','FA_15_2']):
    depth = pd.read_csv('Data/Miege firn aquifer/'+site+'_Firn_Temperatures_Depths.csv').transpose()
    if k == 0:
        depth = depth.iloc[5:].transpose()
    else:
        depth = depth.iloc[5:,0]
    temp = pd.read_csv('Data/Miege firn aquifer/'+site+'_Firn_Temperatures.csv')
    dates = pd.to_datetime((temp.Year*1000000+temp.Month*10000+temp.Day*100+temp['Hours (UTC)']).apply(str),format='%Y%m%d%H')
    temp = temp.iloc[:,4:]
    
    ellapsed_hours = (dates-dates[0]).astype('timedelta64[h]')
    accum_depth =  ellapsed_hours.values * thickness_accum/365/24
    depth_cor = pd.DataFrame()
    depth_cor  = depth.values.reshape((1, -1)).repeat(len(dates),axis=0) + accum_depth.reshape((-1, 1)).repeat(len(depth.values),axis=1)
    
    df_10 = ftl.interpolate_temperature(dates, depth_cor, temp.values,
                                 title=site)
    df_10.loc[np.greater(df_10['temperatureObserved'],0), 'temperatureObserved']=0
    df_10 = df_10.set_index('date',drop=False).resample('M').first()
    df_10['site'] = site
    df_10['latitude'] = float(metadata[k,1])
    df_10['longitude'] = - float(metadata[k,2])
    df_10['elevation'] =  float(metadata[k,3])
    df_10['depthOfTemperatureObservation'] = 10
    df_10['reference'] = 'Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W'
    df_10['reference_short'] = 'Miller et al. (2020)'
    df_10['note'] = 'interpolated to 10 m, monthly snapshot'
    # plt.figure()
    # df_10.temperatureObserved.plot()
    df_miege =df_miege.append(df_10)
    

df_all = df_all.append(df_miege[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% McGrath  
print('Loading McGrath')
df_mcgrath = pd. read_excel('Data/McGrath/McGrath et al. 2013 GL055369_Supp_Material.xlsx')
df_mcgrath = df_mcgrath.loc[df_mcgrath['Data Type']!= 'Met Station']
df_mcgrath['depthOfTemperatureObservation'] = np.array(df_mcgrath['Data Type'].str.split('m').to_list())[:,0].astype(int)

df_mcgrath= df_mcgrath.rename(columns = {'Observed\nTemperature (°C)':'temperatureObserved',  'Latitude\n(°N)':'latitude', 'Longitude (°E)':'longitude','Elevation\n(m)':'elevation',  'Reference':'reference','Location':'site'})
df_mcgrath['note'] = 'as reported in McGrath et al. (2013)'

df_mcgrath['date'] = pd.to_datetime((df_mcgrath.Year*10000+101).apply(str),format='%Y%m%d')
df_mcgrath['site'] = df_mcgrath['site'].str.replace('B4','4')  
df_mcgrath['site'] = df_mcgrath['site'].str.replace('B5','5')  

df_all = df_all.append(df_mcgrath[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# adding real date to Benson's measurement
df_fausto = pd.read_excel('Data/misc/Data_Sheet_1_ASnowDensityDatasetforImprovingSurfaceBoundaryConditionsinGreenlandIceSheetFirnModeling.XLSX', skiprows=[0])

df_fausto.Name=df_fausto.Name.str.replace('station ','')       
for site in df_fausto.Name:
    if any(df_all.site==site):
        if np.sum(df_all.site==site)>1:
            print(df_all.loc[df_all.site==site,['temperatureObserved','depthOfTemperatureObservation']])
            print('duplicate, removing from McGrath')
            df_all =  df_all.loc[~np.logical_and(df_all.site==site, df_all['note'] == 'as reported in McGrath et al. (2013)')]

        print(site, df_all.loc[df_all.site==site].date.values,
              df_fausto.loc[df_fausto.Name==site, ['year','Month','Day']].values)
        if df_all.loc[df_all.site==site].date.iloc[0].year ==                df_fausto.loc[df_fausto.Name==site, ['year']].iloc[0].values:
            df_all.loc[df_all.site==site,'date'] =                pd.to_datetime(df_fausto.loc[df_fausto.Name==site, ['year','Month','Day']]).astype('datetime64[ns]').iloc[0]
            print('Updating date')
        else:
            print('Different years')

# %% Hawley GrIT  
print('Loading Hawley GrIT')
df_hawley = pd.read_excel('Data/Hawley GrIT/GrIT2011_9m-borehole_calc-temps.xlsx')
df_hawley = df_hawley.rename(columns={'Pit name (tabs)':'site', 'Date':'date', 'Lat (dec.degr)':'latitude', 'Long (dec.degr)':'longitude','Elevation':'elevation', '9-m temp':'temperatureObserved'})
df_hawley['depthOfTemperatureObservation']=9
df_hawley['note']=''
df_hawley['reference']='Bob Hawley. 2014. Traverse physical, chemical, and weather observations. arcitcdata.io, doi:10.18739/A2W232. '
df_hawley['reference_short']='Hawley (2014) GrIT'

df_hawley=df_hawley.loc[[isinstance(x, float) for x in df_hawley.temperatureObserved]]
df_hawley=df_hawley.loc[df_hawley.temperatureObserved.notnull()]

df_all = df_all.append(df_hawley[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% PROMICE  
print('Loading PROMICE')
df_promice = pd.read_csv('Data/PROMICE/PROMICE_10m_firn_temperature.csv',sep=';')
df_promice = df_promice.loc[df_promice.temperatureObserved.notnull()]
df_promice['reference_short'] = 'PROMICE'
df_all = df_all.append(df_promice[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% GC-Net  
print('Loading GC-Net')
df_GCN = pd.read_csv('Data/GC-Net/10m_firn_temperature.csv')
df_GCN = df_GCN.loc[df_GCN.temperatureObserved.notnull()]
df_GCN.reference = 'Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103.'
df_all = df_all.append(df_GCN[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% GCNv2  
print('Loading GCNv2')
df_GCNv2 = pd.read_csv('Data/GCNv2/10m_firn_temperature.csv')
df_GCNv2 = df_GCNv2.loc[df_GCNv2.temperatureObserved.notnull()]
df_all = df_all.append(df_GCNv2[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% Harper ice temperature  
print('Loading Harper ice temperature')
df_harper = pd.read_csv('Data/Harper ice temperature/harper_iceTemperature_2015-2016.csv')
num_row = df_harper.shape[0]
df_harper['date'] = np.nan
df_harper['temperatureObserved'] = np.nan
df_harper['note'] = ''
df_harper = df_harper.append(df_harper)
df_harper['borehole'].iloc[:num_row] = df_harper['borehole'].iloc[:num_row]+'_2015'
df_harper['borehole'].iloc[num_row:] = df_harper['borehole'].iloc[num_row:]+'_2016'
df_harper['date'].iloc[:num_row] = pd.to_datetime('2015-01-01')
df_harper['date'].iloc[num_row:] = pd.to_datetime('2016-01-01')
df_harper['temperatureObserved'].iloc[:num_row] = df_harper['temperature_2015_celsius'].iloc[:num_row]
df_harper['temperatureObserved'].iloc[num_row:] = df_harper['temperature_2016_celsius'].iloc[num_row:]
df_harper['depth'] = df_harper.depth_m - df_harper.height_m
df_harper=df_harper.loc[df_harper.depth<100]
df_harper=df_harper.drop(columns=['height_m', 'temperature_2015_celsius', 'temperature_2016_celsius','yearDrilled', 'dateDrilled', 'depth_m'])
df_harper=df_harper.loc[df_harper.temperatureObserved.notnull()]

for borehole in df_harper['borehole'].unique():
    print(borehole,df_harper.loc[df_harper['borehole'] == borehole,'depth'].min())
    if df_harper.loc[df_harper['borehole'] == borehole,'depth'].min() > 20:
        df_harper = df_harper.loc[df_harper['borehole'] != borehole]
        continue        
    
    new_row = df_harper.loc[df_harper['borehole'] == borehole].iloc[0,:].copy()
    new_row['depth']=10
    new_row['temperatureObserved']=np.nan
    df_harper = df_harper.append(new_row)
    
df_harper = df_harper.set_index(['depth']).sort_index()

# plt.figure()
for borehole in df_harper['borehole'].unique():
    s = df_harper.loc[df_harper['borehole']==borehole,'temperatureObserved']
    s.iloc[0] =         interp1d(s.iloc[1:].index, s.iloc[1:], kind='linear',fill_value='extrapolate')(10)
    # s.plot(marker='o',label='_no_legend_')
    # df_harper.loc[df_harper['borehole']==borehole,'temperatureObserved'].plot(marker='o',label=borehole)
    # plt.legend()
    df_harper.loc[df_harper['borehole']==borehole,'temperatureObserved'] = s.values
    df_harper.loc[df_harper['borehole']==borehole,'note'] = 'interpolated from ' +str(s.iloc[1:].index.values)+' m depth'

df_harper = df_harper.reset_index()
df_harper = df_harper.loc[df_harper.depth==10]

df_harper['reference'] = 'Hills, B. H., Harper, J. T., Humphrey, N. F., & Meierbachtol, T. W. (2017). Measured horizontal temperature gradients constrain heat transfer mechanisms in Greenland ice. Geophysical Research Letters, 44. https://doi.org/10.1002/2017GL074917;  https://doi.org/10.18739/A24746S04'

df_harper['reference_short'] = 'Hills et al. (2017)'

df_harper = df_harper.rename(columns={'borehole': 'site', 'latitude_WGS84':'latitude', 'longitude_WGS84':'longitude', 'Elevation_m':'elevation','depth':'depthOfTemperatureObservation'})

df_all = df_all.append(df_harper[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)
   
# %%  GC-Net  

print('Loading GC-Net')
time.sleep(0.2)

sites=['CP1', 'NASA-U', 'Summit', 'TUNU-N', 'DYE-2', 'Saddle', 'SouthDome', 'NASA-E', 'NASA-SE']
lat = [69.87975, 73.84189, 72.57972, 78.01677, 66.48001, 65.99947, 63.14889, 75, 66.4797]
lon = [-46.98667, -49.49831, -38.50454, -33.99387, -46.27889, -44.50016, -44.81717, -29.99972, -42.5002]
elev = [2022, 2369, 3254, 2113, 2165, 2559, 2922, 2631, 2425]

df_gcnet = pd.DataFrame()
for ii, site in progressbar.progressbar(enumerate(sites)):
    ds = xr.open_dataset('Data/Vandecrux et al. 2020/'+site+'_T_firn_obs.nc')
    df = ds.to_dataframe()
    df = df.reset_index(0).groupby('level').resample('D').mean()
    df.reset_index(0,inplace=True, drop=True)  
    df.reset_index(inplace=True)  
    
    df_d = pd.DataFrame()
    df_d['date'] = df.loc[df['level']==1,'time']
    for i in range(1,11):
        df_d['rtd'+str(i)] = df.loc[df['level']==i,'T_firn'].values    
    for i in range(1,11):
        df_d['depth_'+str(i)] = df.loc[df['level']==i,'Depth'].values
    
    df_10 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,11)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,11)]].values,
                                 title=site)
    df_10['site'] = site
    df_10['latitude'] = lat[ii]
    df_10['longitude'] = lon[ii]
    df_10['elevation'] = elev[ii]
    df_10 = df_10.set_index('date').resample('M').first().reset_index()
    
    df_gcnet = df_gcnet.append(df_10)

df_gcnet['reference'] = 'Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103.'

df_gcnet['reference_short'] = 'GC-Net'

df_gcnet['note'] = 'as processed in Vandecrux, B., Fausto, R.S., Van As, D., Colgan, W., Langen, P.L., Haubner, K., Ingeman-Nielsen, T., Heilig, A., Stevens, C.M., Macferrin, M. and Niwano, M., 2020. Firn cold content evolution at nine sites on the Greenland ice sheet between 1998 and 2017. Journal of Glaciology, 66(258), pp.591-602.'
df_gcnet['depthOfTemperatureObservation'] = 10

df_all = df_all.append(df_gcnet[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %%  FirnCover 
print('Loading FirnCover')
time.sleep(0.2)

filepath=os.path.join('Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5')
sites=['Summit','KAN-U','NASA-SE','Crawford','EKT','Saddle','EastGrip','DYE-2']
statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath,sites)
statmeta_df['elevation'] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

rtd_df=rtd_df.reset_index()
rtd_df=rtd_df.set_index(['sitename','date'])

df_firncover = pd.DataFrame()
for site in sites:
    df_d = rtd_df.xs(site,level='sitename').reset_index()
    df_10 = ftl.interpolate_temperature(df_d['date'],
                                 df_d[['depth_'+str(i) for i in range(1,24)]].values,
                                 df_d[['rtd'+str(i) for i in range(1,24)]].values,
                                 title=site)
    df_10['site'] = site
    if site == 'Crawford':
        df_10['site'] = 'CP1'
    df_10['latitude'] = statmeta_df.loc[site,'latitude']
    df_10['longitude'] = statmeta_df.loc[site,'longitude']
    df_10['elevation'] = statmeta_df.loc[site,'elevation']
    df_10 = df_10.set_index('date').resample('M').first().reset_index()
    
    df_firncover = df_firncover.append(df_10)
df_firncover['reference'] = 'MacFerrin, M., Stevens, C.M., Vandecrux, B., Waddington, E., Abdalati, W.: The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) Dataset, 2013-2019, submitted to ESSD'
df_firncover['reference_short'] = 'FirnCover'
df_firncover['note'] = ''
df_firncover['depthOfTemperatureObservation'] = 10

df_all = df_all.append(df_firncover[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% SPLAZ KAN_U  
print('Loading SPLAZ at KAN-U')
site= 'KAN_U'
num_therm = [32, 12, 12]
df_splaz = pd.DataFrame()
for k, note in enumerate(['SPLAZ_main','SPLAZ_2','SPLAZ_3']):
    
    ds = xr.open_dataset('Data/SPLAZ/T_firn_KANU_'+note+'.nc')
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df2 = pd.DataFrame()
    df2['date'] = df.loc[df['level']==1,'time']
    for i in range(1,num_therm[k]+1):
        df2['rtd'+str(i-1)] = df.loc[df['level']==i,'Firn temperature'].values    
    for i in range(1,num_therm[k]+1):
        df2['depth_'+str(i-1)] = df.loc[df['level']==i,'depth'].values
    df2[df2==-999]=np.nan
    df2=df2.set_index(['date']).resample('D').mean()
    
    df_10 = ftl.interpolate_temperature(df2.index,
                    df2[['depth_'+str(i) for i in range(num_therm[k])]].values,
                    df2[['rtd'+str(i) for i in range(num_therm[k])]].values,
                    min_diff_to_depth = 1.5, kind = 'linear',
                                 title=note)
    # for i in range(10):
    #     plt.figure()
    #     plt.plot(df2.iloc[i*20,0:12].values,-df2.iloc[i*20,12:].values)
    #     plt.plot(df_10.iloc[i*20,1],-10,marker='o')
    #     plt.title(df2.index[i*20])
    #     plt.xlim(-15,0)
        
    df_10['note'] = note
    df_10['latitude'] = 67.000252 
    df_10['longitude'] = -47.022999
    df_10['elevation'] = 1840
    df_10 = df_10.set_index('date').resample('M').first().reset_index()
    df_splaz = df_splaz.append(df_10)
df_splaz['reference'] = 'Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10.'
df_splaz['reference_short'] = 'SPLAZ'
df_splaz['site'] = site
df_splaz['depthOfTemperatureObservation'] = 10

df_all = df_all.append(df_splaz[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% Load Humphrey data
print('loading Humphrey')
df = pd.read_csv("Data/Humphrey string/location.txt",  delim_whitespace=True)
df_humphrey = pd.DataFrame(columns=['site','latitude','longitude',
                               'elevation','date','T10m'])
for site in df.site:
    try:
        df_site = pd.read_csv('Data/Humphrey string/'+site+".txt", header = None, delim_whitespace=True, names = ['doy']+['IceTemperature'+str(i)+'(C)' for i in range(1,33)])
    except: # Exception as e:
        # print(e)
        continue
    
    print(site)
    temp_label = df_site.columns[1:]
    # the first column is a time stamp and is the decimal days after the first second of January 1, 2007. 
    df_site['time'] = [datetime(2007,1,1) + timedelta(days=d) for d in df_site.iloc[:,0]]
    if site == 'T1old':
        df_site['time'] = [datetime(2006,1,1) + timedelta(days=d) for d in df_site.iloc[:,0]]
    df_site = df_site.loc[df_site['time']<=df_site['time'].values[-1],:]
    df_site = df_site.set_index('time')
    df_site = df_site.resample('H').mean()
  
    depth = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0]
    
    if site != 'H5':
        df_site = df_site.iloc[24*30:,:]
    if site == 'T4':
        df_site = df_site.loc[:'2007-12-05']    
    if site == 'H2':
        depth = np.array(depth)-1
    if site == 'H4':
        depth = np.array(depth)-0.75
    if site in ['H3', 'G165', 'T1new']:
        depth = np.array(depth)-0.50
    
    df_hs = pd.read_csv('Data/Humphrey string/'+site+"_surface_height.csv")
    df_hs.time = pd.to_datetime(df_hs.time)
    df_hs = df_hs.set_index('time')
    df_hs = df_hs.resample('H').mean()
    df_site['surface_height'] = np.nan
    
    df_site['surface_height'] = df_hs.iloc[[df_hs.index.get_loc(index, method='nearest') for index in df_site.index]].values 

    depth_label = ['depth_'+str(i) for i in range(1,len(temp_label)+1)]
    for i in range(len(temp_label)):
        df_site[depth_label[i]] = depth[i] + df_site['surface_height'].values - df_site['surface_height'].iloc[0]
        if site != 'H5':
            df_site[temp_label[i]] = df_site[temp_label[i]].rolling(24*3, center=True).mean().values
    
    df_10 = ftl.interpolate_temperature(df_site.index,
                                     df_site[depth_label].values,
                                     df_site[temp_label].values,
                                     title=site)
    df_10 = df_10.set_index('date').resample('M').mean().reset_index()
        
    df_10 ['site'] = site
    df_10['latitude'] = df.loc[df.site == site,'latitude'].values[0]
    df_10 ['longitude']=df.loc[df.site == site,'longitude'].values[0]
    df_10 ['elevation']=df.loc[df.site == site,'elevation'].values[0]
    
    df_humphrey = df_humphrey.append(df_10)
df_humphrey = df_humphrey.reset_index(drop=True)
df_humphrey=df_humphrey.loc[df_humphrey.temperatureObserved.notnull()]
df_humphrey['depthOfTemperatureObservation']=10
df_humphrey['reference'] = 'Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/'
df_humphrey['reference_short'] = 'Humphrey et al. (2012)'
df_humphrey['note'] = 'no surface height measurements, using interpolating surface height using CP1 and SwissCamp stations'

df_all = df_all.append(df_humphrey[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% loading Hills  
print('Loading Hills')
df_meta = pd.read_csv('Data/Hills/metadata.txt',sep=' ')
df_meta.date_start=pd.to_datetime(df_meta.date_start, format='%m/%d/%y')
df_meta.date_end=pd.to_datetime(df_meta.date_end, format='%m/%d/%y')

df_hills = pd.DataFrame()
for site in df_meta.site:
    if site == 'Meteorological_station':
        continue
    df = pd.read_csv('Data/Hills/Hills_'+site+'_IceTemp.txt',sep='\t')
    
    depth = df.columns[1:].str.replace('Depth_','').values.astype(np.float)

    df['date'] = df_meta.loc[df_meta.site==site].date_start.iloc[0] + timedelta(days=1)*(df.Time-df.Time[0])
    
    if 10 not in depth:
        if 10 < depth.max():
            depth_1 = depth[depth < 10].max()
            depth_2 = depth[depth > 10].min()
        else:
            depth_1 = np.partition(depth, -2)[-2]
            depth_2 = np.partition(depth, -2)[-1]
            
        label_1 = 'Depth_%0.3f'%depth_1
        label_2 = 'Depth_%0.3f'%depth_2 
        df['Depth_10.000'] = ((df[label_1] - df[label_2])/(depth_1 - depth_2) * 10 
        + df[label_2] - (df[label_1] - df[label_2])/(depth_1 - depth_2) * depth_2).values

    df = df[['date','Depth_10.000']]
    df['site']= site
    df['latitude']= df_meta.latitude[df_meta.site == site].iloc[0]
    df['longitude']= df_meta.longitude[df_meta.site == site].iloc[0]
    df['elevation']= df_meta.elevation[df_meta.site == site].iloc[0]
    df['depthOfTemperatureObservation']= 10
    df['temperatureObserved']= df['Depth_10.000'].values
    df = df.set_index('date').resample('M').first().reset_index()
    if site in ['T-11a', 'T-11b', 'T-14']:
        df = df.iloc[1:,:]

    df_hills=df_hills.append(df)

df_hills['note'] = 'interpolated at 10 m'
df_hills['reference'] = 'Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418'

df_hills['reference_short'] = 'Hills et al. (2018)'

df_all = df_all.append(df_hills[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% Achim Dye-2

print('Loading Achim Dye-2')
time.sleep(0.2)

ds = xr.open_dataset('Data/Achim/T_firn_DYE-2_16.nc')
df = ds.to_dataframe()
df = df.reset_index(0).groupby('level').resample('D').mean()
df.reset_index(0,inplace=True, drop=True)  
df.reset_index(inplace=True)  

df_d = pd.DataFrame()
df_d['date'] = df.loc[df['level']==1,'time']
for i in range(1,9):
    df_d['rtd'+str(i)] = df.loc[df['level']==i,'Firn temperature'].values    
for i in range(1,9):
    df_d['depth_'+str(i)] = df.loc[df['level']==i,'depth'].values

df_achim = ftl.interpolate_temperature(df_d['date'],
                             df_d[['depth_'+str(i) for i in range(1,9)]].values,
                             df_d[['rtd'+str(i) for i in range(1,9)]].values,
                                 title='Dye-2 Achim')
df_achim['site'] = 'DYE-2'
df_achim['latitude'] = 66.4800
df_achim['longitude'] = -46.2789
df_achim['elevation'] = 2165.0
df_achim['depthOfTemperatureObservation'] = 2165.0
df_achim['note'] = 'interpolated at 10 m, using surface height from FirnCover station'
df_achim['reference'] = 'Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018. '
df_achim['reference_short'] = 'Heilig et al. (2018)'

df_achim = df_achim.set_index('date').resample('M').first().reset_index()
    
df_all = df_all.append(df_achim[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)

# %% Camp Century Climate
df = pd.read_csv('Data/Camp Century Climate/data_long.txt',sep=',',header=None)
df = df.rename(columns={0:'date'})
df['date'] = pd.to_datetime(df.date)
df[df==-999] = np.nan
df = df.set_index('date').resample('D').first()
df = df.iloc[:,:-2]

df_promice = pd.read_csv('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3/EGP_hour_v03_L3.txt',sep='\t')
df_promice[df_promice==-999] = np.nan
df_promice = df_promice.rename(columns={'time':'date'})
df_promice['date'] = pd.to_datetime(df_promice.date)
df_promice=df_promice.set_index('date').resample('D').mean()
df_promice = df_promice['SurfaceHeight_summary(m)']
temp_label = ['T_'+str(i+1) for i in range(len(df.columns))]
depth_label = ['depth_'+str(i+1) for i in range(len(df.columns))]
df['surface_height'] = df_promice[np.array([x for x in df_promice.index if x in df.index])].values

depth = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 53, 58, 63, 68, 73]

for i in range(len(temp_label)):
    df = df.rename(columns={i+1: temp_label[i]})
    df[depth_label[i]] = depth[i] + df['surface_height'].values - df['surface_height'].iloc[0]

df_10 = ftl.interpolate_temperature(df.index,
                                 df[depth_label].values,
                                 df[temp_label].values,
                                 title='Camp Century Climate')
df_10.loc[np.greater(df_10['temperatureObserved'],-15), 'temperatureObserved']=np.nan
df_10 = df_10.set_index('date',drop=False).resample('M').first()
df_10['site'] = 'CEN'
df_10['latitude'] = 77.1333
df_10['longitude'] = -61.0333
df_10['elevation'] = 1880
df_10['depthOfTemperatureObservation'] = 10
df_10['note'] = 'THM_long, interpolated at 10 m'
df_10['reference'] = 'Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F'
df_10['reference_short'] = 'Camp Century Climate (long string)'

df_all = df_all.append(df_10[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

df = pd.read_csv('Data/Camp Century Climate/data_short.txt',sep=',',header=None)
df = df.rename(columns={0:'date'})
df['date'] = pd.to_datetime(df.date)
df[df==-999] = np.nan
df = df.set_index('date').resample('D').first()

df_promice = pd.read_csv('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3/EGP_hour_v03_L3.txt',sep='\t')
df_promice[df_promice==-999] = np.nan
df_promice = df_promice.rename(columns={'time':'date'})
df_promice['date'] = pd.to_datetime(df_promice.date)
df_promice=df_promice.set_index('date').resample('D').mean()
df_promice = df_promice['SurfaceHeight_summary(m)']
temp_label = ['T_'+str(i+1) for i in range(len(df.columns))]
depth_label = ['depth_'+str(i+1) for i in range(len(df.columns))]
df['surface_height'] = df_promice[np.array([x for x in df_promice.index if x in df.index])].values

# plt.figure()
# df_10.temperatureObserved.plot()
depth = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 22, 25, 28, 31, 34, 38, 42, 46, 50, 54]

for i in range(len(temp_label)):
    df = df.rename(columns={i+1: temp_label[i]})
    df[depth_label[i]] = depth[i] + df['surface_height'].values - df['surface_height'].iloc[0]

df_10 = ftl.interpolate_temperature(df.index,
                                 df[depth_label].values,
                                 df[temp_label].values,
                                 title='Camp Century Climate')
df_10.loc[np.greater(df_10['temperatureObserved'],-15), 'temperatureObserved']=np.nan
df_10 = df_10.set_index('date',drop=False).resample('M').first()
df_10['site'] = 'CEN'
df_10['latitude'] = 77.1333
df_10['longitude'] = -61.0333
df_10['elevation'] = 1880
df_10['depthOfTemperatureObservation'] = 10
df_10['note'] = 'THM_short, interpolated at 10 m'
df_10['reference'] = 'Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F'
# df_10.temperatureObserved.plot()
df_10['reference_short'] = 'Camp Century Climate (short string)'

df_all = df_all.append(df_10[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% Camp Century historical 
# % In 1977, firn density was measured to a depth of 100.0 m, and firn temperature was measured at 10 m depth (Clausen and Hammer, 1988). In 1986, firn density was measured to a depth of 12.0 m, and the 10 m firn temperature was again measured (Gundestrup et al., 1987).

df_cc_hist = pd.DataFrame()
df_cc_hist['date'] = pd.to_datetime(['1977-07-01'])
df_cc_hist['site'] = 'Camp Century'
df_cc_hist['temperatureObserved'] = -24.29
df_cc_hist['depthOfTemperatureObservation'] = 10
df_cc_hist['note'] = 'original data cannot be found'
df_cc_hist['reference'] = 'Clausen, H., and Hammer, C. (1988). The laki and tambora eruptions as revealed in Greenland ice cores from 11 locations. J. Glaciology 10, 16–22. doi:10.1017/s0260305500004092'
df_cc_hist['reference_short'] = 'Clausen and Hammer (1988)'

df_cc_hist['latitude'] = 77.1333
df_cc_hist['longitude'] = -61.0333
df_cc_hist['elevation'] = 1880

df_all = df_all.append(df_cc_hist[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% Davies dataset
df_davies = pd.read_excel('Data/Davies South Dome/table_3_digitized.xlsx')
df_davies['depthOfTemperatureObservation'] = 10
df_davies['note'] = ''
df_davies['reference'] = 'Davies, T.C., Structures in the upper snow layers of the southern Dome Greenland ice sheet, CRREL research report 115, 1954'
df_davies['reference_short'] = 'Davies (1954)'

df_all = df_all.append(df_10[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %%  Echelmeyer Jakobshavn isbræ
df_echel = pd.read_excel('Data/Echelmeyer jakobshavn/Fig5_points.xlsx')
df_echel.date = pd.to_datetime(df_echel.date)
df_echel['reference'] = 'Echelmeyer Κ, Harrison WD, Clarke TS and Benson C (1992) Surficial Glaciology of Jakobshavns Isbræ, West Greenland: Part II. Ablation, accumulation and temperature. Journal of Glaciology 38(128), 169–181 (doi:10.3189/S0022143000009709)'
df_echel['reference_short'] = 'Echelmeyer et al. (1992)'
df_echel = df_echel.rename(columns = {'12 m temperature':'temperatureObserved', 'Name':'site'})
df_echel['depthOfTemperatureObservation'] = 12

df_profiles = pd.read_excel('Data/Echelmeyer jakobshavn/Fig3_profiles.xlsx')
df_profiles = df_profiles[pd.DatetimeIndex(df_profiles.date).month>3]
df_profiles['date'] = pd.to_datetime(df_profiles.date)
df_profiles= df_profiles.set_index(['site','date'],drop=False)
for site in ['L20', 'L23']:
    dates = df_profiles.loc[site,'date'].unique()
    for date in dates:
        df_profiles.loc[(site,date),'depth'] = -df_profiles.loc[(site,date),'depth'].values + df_profiles.loc[(site,date),'depth'].max()
        tmp = df_echel.iloc[0,:]
        tmp.Name =  df_profiles.loc[(site,date),'site'].iloc[0]
        tmp.longitude =  df_profiles.loc[(site,date),'longitude'].iloc[0]
        tmp.latitude =  df_profiles.loc[(site,date),'latitude'].iloc[0]
        tmp.elevation =  df_profiles.loc[(site,date),'elevation'].iloc[0]
        f= interp1d(df_profiles.loc[(site,date),'depth'].values,
                    df_profiles.loc[(site,date),'temperature'].values,
                    fill_value = 'extrapolate')
        tmp.temperatureObserved =  f(10)
        tmp.depthOfTemperatureObservation =  10
        tmp.note = 'digitized, interpolated at 10 m'
        df_echel = df_echel.append(tmp)
df_all = df_all.append(df_echel[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% Fischer EGIG 1990 1992
meta = pd.read_excel('Data/Fischer EGIG/metadata.xlsx',header=None)
meta.columns=['site','latitude','longitude','elevation']

df_15 = pd.read_csv('Data/Fischer EGIG/T15.txt')
df_15[['latitude','longitude','elevation']]=np.nan
df_15['date'] = pd.to_datetime(df_15.year.astype(str)+'-01-01')
for site in df_15.site:
    df_15.loc[df_15.site == site, 'latitude'] = meta.loc[meta.site == site,'latitude'].iloc[0]
    df_15.loc[df_15.site == site, 'longitude'] = meta.loc[meta.site == site,'longitude'].iloc[0]
    df_15.loc[df_15.site == site, 'elevation'] = meta.loc[meta.site == site,'elevation'].iloc[0]
df_15['depthOfTemperatureObservation']=15
df_15['temperatureObserved']=df_15.T15.values
df_15['note']='Western measurement subject to higher uncertainty'
df_15['reference']='Fischer, H., Wagenbach, D., Laternser, M. & Haeberli, W., 1995. Glacio-meteorological and isotopic studies along the EGIG line, central Greenland. Journal of Glaciology, 41(139), pp. 515-527.'
df_15['reference_short']='Fischer et al. (1995)'

  
df_dequervain = pd.read_csv('Data/Fischer EGIG/DeQuervain.txt', index_col=False)
df_dequervain.date= pd.to_datetime(df_dequervain.date.str[:-2]+'19'+df_dequervain.date.str[-2:])
df_dequervain=df_dequervain.rename(columns={'depth': 'depthOfTemperatureObservation', 'temp':'temperatureObserved'})
df_dequervain['note']='as reported in Fischer et al. (1995)'
df_dequervain['reference']='de Quervain, M, 1969. Schneekundliche Arbeiten der Internationalen Glaziologischen Grönlandexpedition (Nivologie). Medd. Grønl. 177(4)'
df_dequervain['reference_short']='de Quervain (1969)'

df_all = df_all.append(df_15[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)
df_all = df_all.append(df_dequervain[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% Wegener 1929-1930
df1 = pd.read_csv('Data/Wegener 1930/200mRandabst_firtemperature_wegener.csv',sep=';')
df3 = pd.read_csv('Data/Wegener 1930/ReadMe.txt',sep=';')

df1['depth']=df1.depth/100
df1 = df1.append({'Firntemp':np.nan, 'depth': 10},ignore_index=True)
df1 = interp_pandas(df1.set_index('depth'))

df_wegener = pd.DataFrame.from_dict({'date': [df3.date.iloc[0]],
                                    'temperatureObserved': df1.loc[10].values[0],
                                    'depthOfTemperatureObservation': 10,
                                    'latitude': [df3.latitude.iloc[0]],
                                    'longitude': [df3.longitude.iloc[0]],
                                    'elevation': [df3.elevation.iloc[0]],
                                    'reference': [df3.reference.iloc[0]],
                                    'site': [df3.name.iloc[0]]})
df2 = pd.read_csv('Data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv',sep=';')
date = '1930-'+df2.month.astype(str).apply(lambda x: x.zfill(2)) + '-15'
df2 = -df2.iloc[:,1:].transpose().reset_index(drop=True)
df2.iloc[:,0] =  interp_pandas(df2.iloc[:,0])
df2 = df2.iloc[10,:].transpose()
df_new = pd.DataFrame()
df_new['temperatureObserved'] = df2.values
df_new['date'] = date.values
df_new['depthOfTemperatureObservation'] = 10
df_new['latitude'] = df3.latitude.iloc[1]
df_new['longitude'] = df3.longitude.iloc[1]
df_new['elevation'] = df3.elevation.iloc[1]
df_new['site'] = df3.name.iloc[1]
df_new['reference'] = df3.reference.iloc[1]
df_wegener=df_wegener.append(df_new)
df_wegener['reference_short'] = 'Wegener (1930)'
df_wegener['note'] = ''
df_all = df_all.append(df_wegener[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference', 'reference_short', 'note']],ignore_index=True)

# %% Japanese stations
df = pd.read_excel('Data/Japan/Qanaaq.xlsx')
df.date = pd.to_datetime(df.date)
df['note'] = 'interpolated to 10 m'
df_all = df_all.append(df[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

df = pd.read_excel('Data/Japan/Sigma.xlsx')
df['note'] = 'interpolated to 10 m'
df_all = df_all.append(df[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

# %% Ambach

meta = pd.read_csv('Data/Ambach1979b/metadata.txt',sep='\t', header = None,names = ['site', 'file', 'date', 'latitude','longitude','elevation'])
meta.date=pd.to_datetime(meta.date)
for file in meta.file:
    df = pd.read_csv('Data/Ambach1979b/'+file+'.txt',header=None, names=['temperature','depth'])
    if df.depth.max()<7.5:
        meta.loc[meta.file== file,'temperatureObserved'] = df.temperature.iloc[-1]
        meta.loc[meta.file== file,'depthOfTemperatureObservation'] = df.index.values[-1]
    else:
        df.loc[df.shape[0]] = [np.nan, 10]  # adding a row
        df = df.set_index('depth')
        df = interp_pandas(df)
        plt.figure()
        df.plot()
        plt.title(file)
        meta.loc[meta.file== file,'temperatureObserved'] = df.temperature.iloc[-1]
        meta.loc[meta.file== file,'depthOfTemperatureObservation'] = df.index.values[-1]
                 
meta['reference'] = 'Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979'
meta['reference_short'] = 'Ambach EGIG 1959'
meta['note'] = ''
df_all = df_all.append(meta[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference','reference_short', 'note']], ignore_index=True)

# %% 
# 'Clausen HB and Stauffer B (1988) Analyses of Two Ice Cores Drilled at the Ice-Sheet Margin in West Greenland. Annals of Glaciology 10, 23–27 (doi:10.3189/S0260305500004109) and Stauffer, B, Oeschger, H 1979 Temperaturprofile in Bohrlöchern am Rande des grönländischen Inlandeises. Mitteilungen der Versuchsanstalt für Wasserbau, Hydrologie und Glaziologie an der Eidgenössischen Technischen Hochschule (Zürich) 41: 301—313'

# %% Checking values
# df_ambiguous_date = df_all.loc[pd.to_datetime(df_all.date,errors='coerce').isnull(),:]
# df_bad_long = df_all.loc[df_all.longitude.astype(float)>0,:]
# df_no_coord = df_all.loc[np.logical_or(df_all.latitude.isnull(), df_all.latitude.isnull()),:]
# df_invalid_depth =  df_all.loc[pd.to_numeric(df_all.depthOfTemperatureObservation,errors='coerce').isnull(),:]
# df_no_elev =  df_all.loc[df_all.elevation.isnull(),:]
# df_no_temp =  df_all.loc[df_all.temperatureObserved.isnull(),:]

# %% Removing nan and saving
tmp = df_all.loc[np.isnan(df_all.temperatureObserved.astype(float).values)]
df_all = df_all.loc[~np.isnan(df_all.temperatureObserved.astype(float).values)]

df_all.to_csv('subsurface_temperature_summary.csv')