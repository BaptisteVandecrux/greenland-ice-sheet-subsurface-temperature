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
matplotlib.rcParams.update({"font.size": 16})
df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")

# ============ To fix ================

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

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

df["year"] = pd.DatetimeIndex(df.date).year

df = df.loc[~df.temperatureObserved.isnull(), :]

df.date = pd.to_datetime(df.date)
df = df.set_index("date", drop=False)
print("Number of observations:", len(df.depthOfTemperatureObservation))
df = df.loc[df.depthOfTemperatureObservation == 10, :]
print("Number of observations at 10 m:", len(df.depthOfTemperatureObservation))

#%% Summary table
df = df.sort_values("year")
for ref in df.reference.unique():
    tmp = df.loc[df.reference == ref, :]
    print(
        tmp.reference_short.values[0],
        "\t",
        len(tmp.depthOfTemperatureObservation),
        "\t",
        len(tmp.site.unique()),
        "\t",
        np.array_repr(tmp.site.unique()).replace('\n', ''),
        "\t",
        str(tmp.year.min()) + "-" + str(tmp.year.max()),
        "\t",
        tmp.reference.values[0],
        "\t",
        tmp.note.unique()[0],
    )

# %%  Looking for duplicates
duplicate = df[['date','temperatureObserved']].duplicated(keep=False)
tmp = df.loc[duplicate,:]
# %% 
word = 'Steffen'
tmp = df.loc[np.array([word in ref for ref in df.reference]), :]
print(tmp)  

for site in tmp.site.unique():
    print(site)
    plt.figure()
    tmp2 = tmp.loc[tmp.site==site] 
    for note in tmp2.reference.unique():
        plt.plot(tmp2.loc[tmp2.reference == note,'date'], 
                 tmp2.loc[tmp2.reference == note,'temperatureObserved'],
                 marker='o',
                 linestyle='None',
                 label=note)
    plt.title(site)
    plt.legend()
# %% plotting Observation uncertainty
fig, ax = plt.subplots(7, 1, figsize=(13, 15))
# ax = ax.flatten()
fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.97, bottom=0.1, left=0.1, right=0.9)

import matplotlib.cm as cm
cmap = cm.get_cmap('tab20', len(ref_all))
handles = list()
labels = list()
import matplotlib.dates as mdates

names = ['T-15 T-16','KAN_U','CP1', 'Dye-2', 'EKT', 'Camp Century','Summit']
clusters = [42,19, 0, 2, 43, 4,  15]
for k, cluster_id in enumerate(clusters):

    tmp = gdf.loc[cluster_id]
    ref_list = tmp.reference_short.unique()
    
    # plotting observation
    for ref in ref_list:
        i_col = np.where(ref_all == ref)
        if len(i_col[0]) == 0:
            ref_all = np.append(ref_all,ref)
            i_col = np.where(ref_all == ref)
            # cmap.append('b')
        tmp.loc[tmp.reference_short == ref].temperatureObserved.plot(
            ax=ax[k], marker="o", color=cmap(i_col[0]), 
            markersize=8, linestyle="none", label=ref,
            markeredgecolor='w'
        )
           

    ax[k].set_xlabel('')
    ax[k].text(0.03, 0.95, names[k], transform=ax[k].transAxes, 
                 fontsize=11, fontweight='bold',
                 verticalalignment='top')
    ax[k].set_ylabel("")

    h, l = ax[k].get_legend_handles_labels()
    handles = handles + h
    labels = labels + l
    ylim = ax[k].get_ylim()
    if ylim[1]-ylim[0] < 2:
        ax[k].set_ylim(ylim[0]-2, ylim[1]+2)
    
fig.text(0.05, 0.7, "10 m subsurface temperature ($^o$C)", ha='center', va='center', rotation='vertical',fontsize=12)
fig.savefig('figures/clusters/clusters_w_RCM.png')

df_1 = df.loc[(df.site=='Summit') & (df.reference_short == 'GC-Net'),['date', 'temperatureObserved'] ].set_index('date')
df_2 = df.loc[(df.site=='Summit') & (df.reference_short == 'Giese and Hawley (2015)'),['date', 'temperatureObserved'] ].set_index('date')
df_merge = df_1-df_2
print('Summit', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='GC-Net')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='Giese and Hawley (2015)')
ax.set_title('Summit')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='CEN')&(df.reference_short == 'PROMICE') ,['date', 'temperatureObserved'] ].set_index('date')
df_2 = df.loc[(df.reference_short == 'Camp Century Climate'),['date', 'temperatureObserved'] ].set_index('date').resample('M').first()
df_merge = df_1-df_2
print('Camp Century', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='CEN')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='Camp Century Climate')
ax.set_title('Camp Century')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='CP1') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site == 'CP2'),['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()

df_merge = df_1-df_2
print('Crawford Point', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='CP1')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='CP2')
ax.set_title('Crawford Point')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()


df_1 = df.loc[(df.site=='DYE-2') & (df.reference_short == 'FirnCover') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='DYE-2') & (df.reference_short == 'Covi et al.') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()

df_merge = df_1-df_2
print('Dye-2', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='FirnCover')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='Covi et al.')
ax.set_title('EKT')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='EKT') & (df.reference_short == 'FirnCover') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='EKT') & (df.reference_short == 'Covi et al.') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()

df_merge = df_1-df_2
print('EKT', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))
fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='FirnCover')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='Covi et al.')
ax.set_title('EKT')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='KAN_U') & (df.reference_short == 'PROMICE') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='KAN-U') & (df.reference_short == 'FirnCover') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()

df_merge = df_1-df_2
print('KAN_U', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_2.temperatureObserved.plot(ax=ax, marker='o', label='FirnCover')
df_1.temperatureObserved.plot(ax=ax, marker='o', label='PROMICE')
ax.set_title('KAN_U')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='T-15a') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='T-14') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()
df_merge = df_1-df_2
print('T-15a T-14', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))

fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='T-15a')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='T-14')
ax.set_title('Hills et al. (2018)')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='T-15b') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='T-14') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()
df_merge = df_1-df_2
print('T-15b T-14', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))
fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='T-15b')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='T-14')
ax.set_title('Hills et al. (2018)')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()


df_1 = df.loc[(df.site=='T-15c') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='T-14') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()
df_merge = df_1-df_2
print('T-15c T-14', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))
fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='T-15c')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='T-14')
ax.set_title('Hills et al. (2018)')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()

df_1 = df.loc[(df.site=='T-14') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_1.date = pd.to_datetime(df_1.date)
df_1 = df_1.set_index('date').resample('D').mean()
df_2 = df.loc[(df.site=='T-16') & (df.reference_short == 'Hills et al. (2018)') ,['date', 'temperatureObserved'] ]
df_2.date = pd.to_datetime(df_2.date)
df_2 =df_2.set_index('date').resample('D').mean()
df_merge = df_1-df_2
print('T-14 T-16', (df_1-df_2).mean().values, np.sqrt(((df_1-df_2)**2).mean().values))
fig, ax = plt.subplots(1,1)
df_1.temperatureObserved.plot(ax=ax, marker='o', label='T-14')
df_2.temperatureObserved.plot(ax=ax, marker='o', label='T-16')
ax.set_title('Hills et al. (2018)')
ax.set_xlim([(df_merge.loc[df_merge.temperatureObserved.notnull(), :].index.values[i]) for i in [0, -1]])
plt.legend()


# %% 

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
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from scipy.spatial.distance import cdist
from progressbar import progressbar
import time
# import GIS_lib as gis
import matplotlib
from rasterio.crs import CRS
from rasterio.warp import transform
ABC = 'ABCDEFGHIJKL'

print('loading dataset')
df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

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

df["year"] = pd.DatetimeIndex(df.date).year
dates = pd.DatetimeIndex(df.date)

df[['latitude','longitude','elevation','date','depthOfTemperatureObservation','temperatureObserved']].to_csv('add_to_sumup_v1.csv',index=False)
