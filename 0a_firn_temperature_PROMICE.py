# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""


import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
np.seterr(invalid='ignore')


#%% Adding PROMICE observations
# Array information of stations available at PROMICE official site: https://promice.org/WeatherStations.html

PROMICE_stations = [('EGP',(75.6247,-35.9748), 2660), 
                    ('CEN',(77.1333,-61.0333), 1880), 
                   ('KAN_L',(67.0955,-35.9748), 670), 
                   ('KAN_M',(67.0670,-48.8355), 1270), 
                   ('KAN_U',(67.0003,-47.0253), 1840), 
                   ('KPC_L',(79.9108,-24.0828), 370),
                   ('KPC_U',(79.8347,-25.1662), 870), 
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

path_to_PROMICE = "C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3"  
PROMICE = pd.DataFrame()

for ws in PROMICE_stations:
    print(ws)
    filepath = path_to_PROMICE + '/' + ws[0] + '_hour_v03_L3.txt'

    df = pd.read_csv (filepath,  sep='\t', index_col=0, parse_dates=True,na_values=-999).resample('D').mean()
    df = df[['Year','MonthOfYear','DayOfYear','AirTemperature(C)',
             'AirTemperatureHygroClip(C)','SurfaceTemperature(C)','HeightSensorBoom(m)','HeightStakes(m)', 'SurfaceHeight_summary(m)', 'IceTemperature1(C)', 'IceTemperature2(C)', 'IceTemperature3(C)', 'IceTemperature4(C)', 'IceTemperature5(C)', 'IceTemperature6(C)', 'IceTemperature7(C)', 'IceTemperature8(C)']]
    df['station_name'] = ws[0]
    df['latitude N'] = ws[1][0]
    df['longitude W'] = ws[1][1]
    df['elevation'] = float(ws[2])
    PROMICE = PROMICE.append(df)

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

PROMICE.drop( ['year', 'MonthOfYear', 'dayofyear'], axis=1,inplace=True)
PROMICE.set_index(['sitename', 'date'],inplace=True)
PROMICE.replace(to_replace=-999,value=np.nan,inplace=True)
sites_all =[item[0] for item in PROMICE_stations]
PROMICE = PROMICE.loc[sites_all,:]
PROMICE['surface_height_summary'] = PROMICE['SurfaceHeight_summary(m)']
PROMICE_save = PROMICE.copy()

# %% Correcting depth with surface height change and maintenance and filtering
     
PROMICE = PROMICE_save.copy()

try :
    url='https://docs.google.com/spreadsheets/d/19WT3BOspQAI3p-r5LPUxIr3AY8jWlEuaf3M_jIT1Z8E/export?format=csv&gid=0'
    pd.read_csv(url).to_csv('Data/PROMICE/fieldwork_summary_PROMICE.csv')
except:
    pass

maintenance_string = pd.read_csv('Data/PROMICE/fieldwork_summary_PROMICE.csv')
maintenance_string=maintenance_string.replace('OUT',np.nan)
maintenance_string['Length of thermistor string on surface from surface marking'] = maintenance_string['Length of thermistor string on surface from surface marking'].astype(float)
maintenance_string['date'] = pd.to_datetime(maintenance_string.Year.astype(str) + '-' + maintenance_string['Date visit'].astype(str))

temp_cols_name = ['rtd0', 'rtd1', 'rtd2', 'rtd3', 'rtd4', 'rtd5', 'rtd6', 'rtd7']
depth_cols_name = ['depth_0', 'depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5', 'depth_6', 'depth_7']
ini_depth=[1,2,3,4,5,6,7,10]
PROMICE[depth_cols_name] = np.nan

for site in sites_all:
    print(site)
    # filtering the surface height
    tmp = PROMICE.loc[site,'surface_height_summary'].copy()
    ind_filter = tmp.rolling(window=14,center=True).var()>0.1
    if any(ind_filter):
        tmp[ind_filter] = np.nan
    PROMICE.loc[site,'surface_height_summary'] = tmp.values
    PROMICE.loc[site,'surface_height_summary']=PROMICE.loc[site,'surface_height_summary'].interpolate().values
    
    maintenance = maintenance_string.loc[maintenance_string.Station==site]

    # first initialization of the depths            
    for i,col in enumerate(depth_cols_name):
        PROMICE.loc[site,col] = ini_depth[i] + PROMICE.loc[site,'surface_height_summary'].values - PROMICE.loc[site,'surface_height_summary'][PROMICE.loc[site,'surface_height_summary'].first_valid_index()]
        
    #reseting depth at maintenance
    if len(maintenance.date)==0:
        print('No maintenance at ',site)
    
    for date in maintenance.date: 
        if date >  PROMICE.loc[site,'surface_height_summary'].last_valid_index():
            continue
        new_depth = maintenance.loc[maintenance.date==date].depth_new_thermistor_m.values[0]               
        if isinstance(new_depth, str):
            new_depth = [float(x) for x in new_depth.split(',')]
            for i,col in enumerate(depth_cols_name):
                tmp = PROMICE.loc[site,col].copy()
                tmp.loc[date:] = new_depth[i]  + PROMICE.loc[site,'surface_height_summary'][date:].values  - PROMICE.loc[site,'surface_height_summary'][date:][PROMICE.loc[site,'surface_height_summary'][date:].first_valid_index()]
                PROMICE.loc[site,col] = tmp.values

# % Filtering thermistor data
    
    temp_cols_name = ['rtd0', 'rtd1', 'rtd2', 'rtd3', 'rtd4', 'rtd5', 'rtd6', 'rtd7']
    for i in range(8):       
        tmp = PROMICE.loc[site,'rtd'+str(i)].copy()
        # variance filter
        ind_filter  = PROMICE.loc[site,'rtd'+str(i)].interpolate(limit=14).rolling(window=7).var()>0.5
        month = PROMICE.loc[site,'rtd'+str(i)].interpolate(limit=14).index.month.values
        ind_filter.loc[np.isin(month, [5,6,7])] = False
        if any(ind_filter):
            tmp.loc[ind_filter] = np.nan
            
        # before and after maintenance adaptation filter
        maintenance = maintenance_string.loc[maintenance_string.Station == site]
        if len(maintenance.date)>0:
            for date in maintenance.date:
                if isinstance(maintenance.loc[maintenance.date==date].depth_new_thermistor_m.values[0], str):
                    ind_adapt =  np.abs(tmp.interpolate(limit=14).index.values - pd.to_datetime(date).to_datetime64()) < np.timedelta64(7, 'D')
                    if any(ind_adapt):
                        tmp.loc[ind_adapt] = np.nan
            
        # surfaced thermistor
        ind_pos =  PROMICE.loc[site,'depth_'+str(i)]<0.1
        if any(ind_pos):
            tmp.loc[ind_pos] = np.nan
            
        # porting the filtered values to the original table
        PROMICE.loc[site,'rtd'+str(i)] =tmp.values
        # PROMICE.loc[site,'rtd'+str(i)] = PROMICE.loc[site,'rtd'+str(i)].interpolate(limit=14).values

# %% 10 m firn temp
df_PROMICE = pd.DataFrame()
import firn_temp_lib as ftl

# ['EGP', 'CEN', 'KAN_L', 'KAN_M', 'KAN_U', 'KPC_L', 'KPC_U', 'NUK_L', 'NUK_U', 'QAS_L',
#  'QAS_M', 'QAS_U', 'SCO_L', 'SCO_U', 'TAS_A', 'TAS_L', 'THU_L', 'THU_U', 'UPE_L', 'UPE_U']

for site in sites_all:
    df_10 = ftl.interpolate_temperature(PROMICE.loc[site].index,
                                    PROMICE.loc[site,depth_cols_name].values.astype(float),
                            PROMICE.loc[site,temp_cols_name].values.astype(float),
                            kind = 'linear',title=site, plot=False)

    df_10['latitude'] = PROMICE.loc[site,'latitude N'].values
    df_10['longitude'] = PROMICE.loc[site,'longitude W'].values
    df_10['elevation'] = PROMICE.loc[site,'elevation'].values
    df_10['site'] = site
    df_10['depthOfTemperatureObservation'] = 10
    df_10['note'] = ''
    df_10=df_10.set_index('date')
    
    # filtering
    ind_pos = df_10['temperatureObserved'] > 0.1
    ind_low = df_10['temperatureObserved'] < -70
    df_10.loc[ind_pos,'temperatureObserved'] = np.nan
    df_10.loc[ind_low,'temperatureObserved'] = np.nan
    df_PROMICE = df_PROMICE.append (df_10.reset_index())

df_PROMICE = df_PROMICE.loc[df_PROMICE.temperatureObserved.notnull(),:]
df_PROMICE['reference'] = 'Fausto, R.S. and van As, D., (2019). Programme for monitoring of the Greenland ice sheet (PROMICE): Automatic weather station data. Version: v03, Dataset published via Geological Survey of Denmark and Greenland. DOI: https://doi.org/10.22008/promice/data/aws'


df_PROMICE=df_PROMICE.set_index('date')
df_PROMICE_month = df_PROMICE.groupby('site').resample('M').first()
df_PROMICE_month.to_csv('Data/PROMICE/PROMICE_10m_firn_temperature.csv',sep=';')

# %% Plotting
for site in sites_all:
    print(site)
    
    fig, ax = plt.subplots(1,2,figsize=(25,15))
    PROMICE.loc[site,'surface_height_summary'].plot(ax=ax[0])
    (PROMICE.loc[site,'surface_height_summary']-10).plot(ax=ax[0],color='red',linestyle='--')
    maintenance = maintenance_string.loc[maintenance_string.Station==site]

    for i,col in enumerate(depth_cols_name):
        (-PROMICE.loc[site,col]+PROMICE.loc[site,'surface_height_summary']).plot(ax=ax[0])
        
    if len(maintenance.date)>0:    
        for date in maintenance.date:
            index=PROMICE.loc[site,'surface_height_summary'].index
            date2 = index[index.get_loc(date, method='nearest')]
            if np.abs(date-date2) <= pd.Timedelta('7 days'):
                depth_top_therm_found = (maintenance.loc[maintenance.date==date,'Length of thermistor string on surface from surface marking']/100-1+PROMICE.loc[site,'surface_height_summary'][date2])
                ax[0].plot(date,depth_top_therm_found, markersize= 10, marker='o', linestyle='None')
                if isinstance(maintenance.loc[maintenance.date==date].depth_new_thermistor_m.values[0], str):        
                    ax[0].axvline(date)
    ax[0].set_ylim(PROMICE.loc[site,'surface_height_summary'].min()-11, 
    PROMICE.loc[site,'surface_height_summary'].max()+1)

    temp_cols_name = ['rtd0', 'rtd1', 'rtd2', 'rtd3', 'rtd4', 'rtd5', 'rtd6', 'rtd7']
    for i in range(8):
        PROMICE_save.loc[site,'rtd'+str(i)].interpolate(limit=14).plot(ax=ax[1],label='_nolegend_')
        
        tmp = PROMICE_save.loc[site,'rtd'+str(i)].copy()
        # variance filter
        ind_filter  = PROMICE_save.loc[site,'rtd'+str(i)].interpolate(limit=14).rolling(window=7).var()>0.5
        month = PROMICE_save.loc[site,'rtd'+str(i)].interpolate(limit=14).index.month.values
        ind_filter.loc[np.isin(month, [5,6,7])] = False
        if any(ind_filter):
            tmp.loc[ind_filter].plot(ax=ax[1], marker='o', linestyle='none',color='pink', label='_nolegend_')
            
        # before and after maintenance adaptation filter
        maintenance = maintenance_string.loc[maintenance_string.Station == site]
        if len(maintenance.date)>0:
            for date in maintenance.date:
                if isinstance(maintenance.loc[maintenance.date==date].depth_new_thermistor_m.values[0], str):
                    ind_adapt =  np.abs(tmp.interpolate(limit=14).index.values - pd.to_datetime(date).to_datetime64()) < np.timedelta64(7, 'D')
                    if any(ind_adapt):
                        tmp.loc[ind_adapt].plot(ax=ax[1], marker='o', linestyle='none',color='purple', label='_nolegend_')
            
        # surfaced thermistor
        ind_pos =  PROMICE.loc[site,'depth_'+str(i)]<0.1
        if any(ind_pos):
            tmp.loc[ind_pos].plot(ax=ax[1], marker='o', linestyle='none',color='black', label='_nolegend_')
    if len(df_PROMICE.loc[df_PROMICE.site == site,'temperatureObserved']) ==0:
        print('No 10m temp for ',site)
    else:
        df_PROMICE.loc[df_PROMICE.site == site,'temperatureObserved'].resample('W').mean().plot(ax = ax[1], linewidth=3, label='10 m temperature')  
    ax[1].plot(np.nan,np.nan, marker='o', linestyle='none',color='black', label='surfaced')
    ax[1].plot(np.nan,np.nan, marker='o', linestyle='none',color='purple', label='maintenance')
    ax[1].plot(np.nan,np.nan, marker='o', linestyle='none',color='pink', label='var filter')
    ax[1].legend()
    fig.suptitle(site) 
    fig.savefig('figures/PROMICE'+site+'.png',dpi=90, bbox_inches='tight')
