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
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib as mpl
from scipy import stats
import GIS_lib as gis
import itertools
from matplotlib import cm
from scipy.interpolate import Rbf

df = pd.read_csv("subsurface_temperature_summary.csv")

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

df_hans_tausen = df.loc[df.latitude > 82, :]
df = df.loc[~(df.latitude > 82), :]

df_no_coord = df.loc[np.logical_or(df.latitude.isnull(), df.latitude.isnull()), :]
df = df.loc[~np.logical_or(df.latitude.isnull(), df.latitude.isnull()), :]

df_invalid_depth = df.loc[
    pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]
df = df.loc[
    ~pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]

df_no_elev = df.loc[df.elevation.isnull(), :]
df = df.loc[~df.elevation.isnull(), :]

df = df.loc[df.depthOfTemperatureObservation==10, :]
df = df.loc[df.temperatureObserved.notnull(), :]

df['date'] = pd.to_datetime(df.date)
df_m = pd.DataFrame()
# generating the monthly file


for ref in df.reference_short.unique():
    for site in df.loc[df.reference_short == ref,'site'].unique():
        df_loc = df.loc[(df.reference_short == ref)&(df.site == site),:]
        
        if np.sum((df_loc.date.diff() < '2 days') & (df_loc.date.diff() > '0 days')) > 5:
            print(ref, site, '... averaging to monthly')
            df_loc_first = df_loc.set_index('date').resample('M').first()
            df_loc = df_loc.set_index('date').resample('M').mean()
            df_loc[['Unnamed: 0', 'site', 'reference',
       'reference_short', 'note']] = df_loc_first[['Unnamed: 0', 'site', 'reference',
       'reference_short', 'note']]
            if any(df_loc.depthOfTemperatureObservation.unique() != 10):
                print('Some non-10 m depth')
                print(df_loc.depthOfTemperatureObservation.unique())
                print(df_loc.loc[df_loc.depthOfTemperatureObservation!=10].head())
                df_loc = df_loc.loc[df_loc.depthOfTemperatureObservation!=10]
        
            df_m = df_m.append(df_loc.reset_index()[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)
        else:
            df_m = df_m.append(df_loc[['date','site', 'latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved', 'reference',  'reference_short', 'note']],ignore_index=True)
                
df_m = df_m.drop(columns = ['Unnamed: 0'])
df_m.to_csv('10m_temperature_dataset_monthly.csv',index=False)
