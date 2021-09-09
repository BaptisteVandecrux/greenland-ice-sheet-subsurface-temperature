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

    
# %% analystical solution to heat diffusion
# kappa = 0.00114 # m2h-1
# om = 2*np.pi/24
# t = np.arange(24)
# A =3
# T = A*np.exp(-z * np.sqrt(om/2/kappa) * np.cos(om*t - z*np.sqrt(om/2/kappa) + eps)

# %% Studying clusters
df = pd.read_csv('subsurface_temperature_summary.csv')
import geopandas as gpd 

# ============ To fix ================

df_ambiguous_date = df.loc[pd.to_datetime(df.date,errors='coerce').isnull(),:]
df = df.loc[~pd.to_datetime(df.date,errors='coerce').isnull(),:]

df_bad_long = df.loc[df.longitude>0,:]
df['longitude'] = - df.longitude.abs().values

df_big_lon = df.loc[df.longitude.abs()>100,:]
df.loc[df.longitude.abs()>100,'longitude'] = df.loc[df.longitude.abs()>100,'longitude'].values/10

df_big_lat = df.loc[df.latitude>100,:]
df.loc[df.latitude>100,'latitude'] = df.loc[df.latitude>100,'latitude'].values/10

df_no_coord = df.loc[np.logical_or(df.latitude.isnull(), df.longitude.isnull()),:]
df = df.loc[~np.logical_or(df.latitude.isnull(), df.longitude.isnull()),:]

df_invalid_depth =  df.loc[pd.to_numeric(df.depthOfTemperatureObservation,errors='coerce').isnull(),:]
df = df.loc[~pd.to_numeric(df.depthOfTemperatureObservation,errors='coerce').isnull(),:]

# df_no_elev =  df.loc[df.elevation.isnull(),:]
# df = df.loc[~df.elevation.isnull(),:]

df_no_temp =  df.loc[df.temperatureObserved.isnull(),:]
df = df.loc[~df.temperatureObserved.isnull(),:]

df['year'] = pd.DatetimeIndex(df.date).year

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude)).set_crs(4326).to_crs(3413)

land = gpd.GeoDataFrame.from_file('Data/misc/Ice-free_GEUS_GIMP.shp')
land = land.to_crs('EPSG:3413')

ice = gpd.GeoDataFrame.from_file('Data/misc/IcePolygon_3413.shp')
ice = ice.to_crs('EPSG:3413')

fig, ax = plt.subplots(1,1,figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0,top=1,bottom=0,left=0,right=1)
land.plot(ax = ax, zorder=0,color='black')
ice.plot(ax = ax, zorder=1,color='lightblue')
gdf.plot(ax = ax, column='year',  cmap = 'tab20c' , markersize=50, 
         edgecolor = 'gray', legend = True,
         legend_kwds={'label': 'Year of measurement', 
                      "orientation": "horizontal",
                      'shrink': 0.8})


plt.axis('off')
plt.savefig('figures/fig1_map.png')

from sklearn.cluster import DBSCAN

epsilon = 5000
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y

coords = gdf[['x', 'y']].values
db = (DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric = 'euclidean').fit(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#  Black removed and is used for noise instead.
fig, ax = plt.subplots(1,1,figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0,top=1,bottom=0,left=0,right=1)
land.plot(ax = ax, zorder=0,color='black')
ice.plot(ax = ax, zorder=1,color='lightblue')
unique_labels = set(labels)
colors = [plt.cm.tab20c(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = coords[class_member_mask & ~core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
gdf.plot(ax = ax, color='black')
gdf.loc[gdf.reference=='GC-Net_v2 by GEUS'].plot(ax = ax, color='red',marker='^')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

gdf['clusters'] = labels
gdf.date = gdf.date.astype("datetime64[ns]")
gdf = gdf.set_index(['clusters','date'])
# %% 

for cluster_id in unique_labels:
    if cluster_id == -1:
        continue
    tmp = gdf.loc[cluster_id]
    ref_list = tmp.reference_short.unique()
    if len(ref_list)>1:
        fig, ax = plt.subplots(1,2,figsize=(15, 9))
        fig.subplots_adjust(hspace=0.1, wspace=0.1,top=0.8,bottom=0.1,left=0.1,right=0.9)
        land.plot(ax = ax[0], zorder=0,color='black')
        ice.plot(ax = ax[0], zorder=1,color='lightblue')
        gdf.plot(ax = ax[0], color='black')
        tmp.plot(ax = ax[0], color='red',markersize=50)
        xlimit = tmp.index.min()-pd.Timedelta("360 days"),  tmp.index.max()+pd.Timedelta("360 days")
        for ref in ref_list:
            tmp.loc[tmp.reference_short==ref].temperatureObserved.plot(ax = ax[1],
                                                                       marker='o',
                                                                       markersize=8,
                                                                       linestyle='none',
                                                                       label=ref)
        ax[1].legend()
        ax[1].set_xlim(xlimit)
        ax[1].set_title(str(cluster_id) + ' ' + str(tmp.site.unique()))
        ax[1].set_ylabel('10 m firn temperature ($^o$C)')

