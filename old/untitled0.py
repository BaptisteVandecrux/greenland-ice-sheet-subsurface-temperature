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


df = df.reset_index('date')

df = df.drop(columns = ['Unnamed: 0', 'depthOfTemperatureObservation',])
df.to_csv('10m_temperature_dataset.csv',index=False)
