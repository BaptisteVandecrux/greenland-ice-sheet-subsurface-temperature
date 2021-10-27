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
# import GIS_lib as gis 
matplotlib.rcParams.update({'font.size': 16})
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

df = df.loc[~df.temperatureObserved.isnull(),:]

df.date = pd.to_datetime(df.date)
df = df.set_index('date',drop=False)
print('Number of observations:',len(df.depthOfTemperatureObservation))
df = df.loc[df.depthOfTemperatureObservation == 10,:]
print('Number of observations at 10 m:',len(df.depthOfTemperatureObservation))

#%% Summary table
df = df.sort_values('year')
for ref in df.reference.unique():
    tmp = df.loc[df.reference == ref,:]
    print(tmp.reference_short.values[0],'\t',
          len(tmp.depthOfTemperatureObservation),'\t',
          str(tmp.year.min())+'-'+str(tmp.year.max()),'\t',
          tmp.reference.values[0],
          tmp.note.unique()[0])