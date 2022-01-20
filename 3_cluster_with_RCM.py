# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import statsmodels.api as sm
from datetime import datetime as dt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

    return date.year + fraction


df = pd.read_csv("subsurface_temperature_summary.csv")
df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

df_big_lon = df.loc[df.longitude.abs() > 100, :]
df.loc[df.longitude.abs() > 100, "longitude"] = (
    df.loc[df.longitude.abs() > 100, "longitude"].values / 10
)

df_big_lat = df.loc[df.latitude > 100, :]
df.loc[df.latitude > 100, "latitude"] = (
    df.loc[df.latitude > 100, "latitude"].values / 10
)

df_no_coord = df.loc[np.logical_or(df.latitude.isnull(), df.longitude.isnull()), :]
df = df.loc[~np.logical_or(df.latitude.isnull(), df.longitude.isnull()), :]

df_invalid_depth = df.loc[
    pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]
df = df.loc[
    ~pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :
]

# df_no_elev =  df.loc[df.elevation.isnull(),:]
# df = df.loc[~df.elevation.isnull(),:]

df_no_temp = df.loc[df.temperatureObserved.isnull(), :]
df = df.loc[~df.temperatureObserved.isnull(), :]

df["year"] = pd.DatetimeIndex(df.date).year

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

epsilon = 5000
gdf["x"] = gdf.geometry.x
gdf["y"] = gdf.geometry.y

coords = gdf[["x", "y"]].values
db = DBSCAN(eps=epsilon, min_samples=10, algorithm="ball_tree", metric="euclidean").fit(
    coords
)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
gdf["clusters"] = labels
gdf.date = gdf.date.astype("datetime64[ns]")
gdf = gdf.set_index(["clusters", "date"])

cluster_list = np.array([
    [0, 'CP1', '1955', '2021', 69.91666666666667, -46.93333333333333, 2012.0],
    [1, 'T1', '1955', '2009', 69.73166666666667, -48.05, 1746.0],
    [2, 'DYE-2', '1964', '2019', 66.46166666666667, -46.23333333333333, 2100.0],
    # [3, 'H2', '1950', '2009', 69.7, -48.266667, 1598.0],
    [4, 'Camp Century', '1954', '2019', 77.21833333333333, -61.025, 1834.0],
    [11, 'Crête', '1959', '1992', 71.1204, -37.3164, 3224.0],
    [6, 'SwissCamp', '1990', '2005', 69.57, -49.3, 1174.0],
    [15, 'Summit', '1997', '2018', 72.5667, -38.5, 3248.0],
    [19, 'KAN-U', '2011', '2019', 67.0003, -47.0253, 1840.0],
    [36, 'SouthDome', '1998', '2021', 63.148985, -44.81738, 2894.0],
    [37, 'NASA-SE', '1998', '2021', 66.47776999999999, -42.493634, 2385.0],
    [38, 'NASA-U', '2003', '2021', 73.840558, -49.531539, 2340.0],
    [41, 'Saddle', '1998', '2017', 65.99947, -44.50016, 2559.0]
    ])

ref_all = np.array([])
for k, cluster_id in enumerate(cluster_list[:, 0].astype(int)):

    tmp = gdf.loc[cluster_id]
    ref_all = np.append(ref_all, tmp.reference_short.unique())

ref_all = np.unique(ref_all)

df_meta = pd.DataFrame(cluster_list, columns = ['cluster_id','site','year_min','year_max','latitude','longitude','elevation'])

# %% extracting RACMO 10m temperature for these clusters
ds = xr.open_dataset("C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc")
points = [
    (x, y) for x, y in zip(ds["lat"].values.flatten(), ds["lon"].values.flatten())
]

df_RACMO = pd.DataFrame()
df_RACMO['time'] = ds["time"].values


for i, (lat, lon) in enumerate(zip(df_meta.latitude.astype(float), df_meta.longitude.astype(float))):
    closest = points[cdist([(lat, lon)], points).argmin()]

    ind = np.argwhere(
        np.array(np.logical_and(ds["lat"] == closest[0], ds["lon"] == closest[1]))
    )[0]
    
    print(df_meta.iloc[i,1], ind)
    df_RACMO[df_meta.iloc[i, 1]] = ds.T10m[ dict(rlon=int(ind[1]), rlat=int(ind[0])) ].values - 273.15
    

df_RACMO = df_RACMO.set_index('time')

# %% extracting HIRHAM 10m temperature for these clusters
ds = xr.open_dataset( "C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m_1980.nc" )

points = [
    (x, y) for x, y in zip(ds["lat"].values.flatten(), ds["lon"].values.flatten())
]

df_HIRHAM = pd.DataFrame()

for year in range(1980,2017):
    ds = xr.open_dataset( "C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m_"+str(year)+".nc" )
    df_year = pd.DataFrame()
    df_year['time'] = ds["time"].values
    df_year.time = pd.to_datetime(
        [int(s) for s in np.floor(df_year.time.values)], format="%Y%m%d"
    )
    for i, (lat, lon) in enumerate(zip(df_meta.latitude.astype(float), df_meta.longitude.astype(float))):
        closest = points[cdist([(lat, lon)], points).argmin()]
    
        ind = np.argwhere(
            np.array(np.logical_and(ds["lat"] == closest[0], ds["lon"] == closest[1]))
        )[0]
        
        df_year[df_meta.iloc[i, 1]] = ds.T10m[ dict(x=int(ind[1]), y=int(ind[0])) ].values - 273.15
        
    
    df_year = df_year.set_index('time')
    df_HIRHAM = df_HIRHAM.append(df_year)

#%% Extracting MAR
ds2 = xr.open_dataset(
    "I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-1950.nc",
    decode_times=False,
)
points = [
    (x, y) for x, y in zip(ds2["LAT"].values.flatten(), ds2["LON"].values.flatten())
]
ds = xr.open_dataset('MAR_all_year.nc')

df_MAR = pd.DataFrame()
df_MAR['time'] = ds["TIME"].values
    
for i, (lat, lon) in enumerate(zip(df_meta.latitude.astype(float), df_meta.longitude.astype(float))):   
    print(i)    
    closest = points[cdist([(lat, lon)], points).argmin()]
    ind = np.argwhere(np.array(np.logical_and(ds2["LAT"] == closest[0], ds2["LON"] == closest[1])))[0]
    
    df_MAR[df_meta.iloc[i, 1]] = ds.TI1[ dict(x = int(ind[1]), y = int(ind[0]))].values
df_MAR = df_MAR.set_index('time')

#%% Extracting MAR
# ds = xr.open_dataset(
#     "I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-1990.nc",
#     decode_times=False,
# )
# points = [
#     (x, y) for x, y in zip(ds["LAT"].values.flatten(), ds["LON"].values.flatten())
# ]

# df_MAR = pd.DataFrame()

# for year in range(1990,2021):
#     print(year)
#     try:
#         ds = xr.open_dataset(
#             "I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-"
#             + str(year)
#             + ".nc"
#         )
#     except:
#         print("Cannot read ", year)


#     df_year = pd.DataFrame()
#     df_year['time'] = ds["TIME"].values
    
#     tmp = ds.TI1[dict(OUTLAY=15)].load()
#     for i, (lat, lon) in enumerate(zip(df_meta.latitude.astype(float), df_meta.longitude.astype(float))):   
#         print(i)    
#         closest = points[cdist([(lat, lon)], points).argmin()]
#         ind = np.argwhere(np.array(np.logical_and(ds["LAT"] == closest[0], ds["LON"] == closest[1])))[0]
        
#         df_year[df_meta.iloc[i, 1]] = tmp[ dict(x = int(ind[1]), y = int(ind[0]))].values
    
#     df_year = df_year.set_index('time')
#     df_MAR = df_MAR.append(df_year)   
    
# df_MAR.to_csv('cluster_MAR.csv')

# %% plotting
fig, ax = plt.subplots(7, 2, figsize=(13, 15))
# ax = ax.flatten()
fig.subplots_adjust(hspace=0.1, wspace=0.2, top=0.97, bottom=0.1, left=0.1, right=0.9)
i = -1
j = -1
import matplotlib.cm as cm
cmap = cm.get_cmap('tab20', len(ref_all))
handles = list()
labels = list()
import matplotlib.dates as mdates


for k, cluster_id in enumerate(cluster_list[:, 0].astype(int)):

    tmp = gdf.loc[cluster_id]
    ref_list = tmp.reference_short.unique()

    if k < 5:
        i = k
        j = 0
    else:
        i = k - 5
        j = 1

    # plotting RACMO 10 m temperature for that location
    df_MAR[cluster_list[k, 1]].plot(ax = ax[i,j],label='MAR',color='lightgreen')
    df_RACMO[cluster_list[k, 1]].plot(ax = ax[i,j],label='RACMO',color='tab:red')
    df_HIRHAM[cluster_list[k, 1]].plot(ax = ax[i,j],label='HIRHAM',color='tab:blue')
    
    # % linear trend analysis    
    print(cluster_list[k, 1])
    year_ranges = np.array([[1960, 1990],[1960, 2020], [1990, 2013], [2013,2020]])
    for r in range(4):
        RACMO = df_RACMO[cluster_list[k, 1]].loc[str (year_ranges[r,0]):str (year_ranges[r,1])]
        X = np.array([toYearFraction(d) for d in RACMO.index])
        y = RACMO.values
        
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        x_plot = np.array([np.nanmin(X), np.nanmax(X)])
        # ax[i,j].plot([RACMO.index.min(), RACMO.index.max()], est2.params[0] + est2.params[1] * x_plot)
        print('slope (p-value): %0.3f (%0.3f)'% (est2.params[1], est2.pvalues[1]))


    # %
    df_RACMO[cluster_list[k, 1]].plot(ax = ax[i,j],label='RACMO',color='tab:red')

    # plotting observation
    for ref in ref_list:
        i_col = np.where(ref_all == ref)
        if len(i_col[0]) == 0:
            ref_all = np.append(ref_all,ref)
            i_col = np.where(ref_all == ref)
            # cmap.append('b')
        if tmp.loc[tmp.reference_short == ref].year.min() <1990:
            tmp.loc[tmp.reference_short == ref].temperatureObserved.plot(
                ax=ax[i,j], marker="o", color=cmap(i_col[0]), 
                markersize=8, linestyle="none", label=ref,
                markeredgecolor='w'
            )
        else:
            tmp.loc[tmp.reference_short == ref].temperatureObserved.plot(
                ax=ax[i,j], marker="o", color=cmap(i_col[0]), 
                markersize=3, linestyle="none", label=ref
            )            
    if k < 5:
        ax[i,j].set_xlim('1950-01-01','2022-01-01')
    else:
        ax[i,j].set_xlim('1989-01-01','2022-01-01')
    ax[i,j].set_xlabel('')
    ax[i,j].text(0.03, 0.95, cluster_list[k, 1], transform=ax[i,j].transAxes, 
                 fontsize=11, fontweight='bold',
                 verticalalignment='top')
    ax[i,j].set_ylabel("")

    h, l = ax[i,j].get_legend_handles_labels()
    handles = handles + h
    labels = labels + l
    ylim = ax[i,j].get_ylim()
    if ylim[1]-ylim[0] < 2:
        ax[i,j].set_ylim(ylim[0]-2, ylim[1]+2)

    if not (((i,j) == (4,0)) | ((i,j) == (6,1))) :
        ax[i,j].set_xticklabels('')
tmp , ind_uni = np.unique(labels,return_index = True)
ind_uni = np.append(ind_uni[np.where(tmp!='RACMO')[0]],
                    ind_uni[np.where(tmp=='RACMO')[0]])
tmp = np.append(tmp[np.where(tmp!='RACMO')[0]],
                    tmp[np.where(tmp=='RACMO')[0]])
ind_uni = np.append(ind_uni[np.where(tmp!='HIRHAM')[0]],
                    ind_uni[np.where(tmp=='HIRHAM')[0]])
tmp = np.append(tmp[np.where(tmp!='HIRHAM')[0]],
                    tmp[np.where(tmp=='HIRHAM')[0]])
ind_uni = np.append(ind_uni[np.where(tmp!='MAR')[0]],
                    ind_uni[np.where(tmp=='MAR')[0]])
tmp = np.append(tmp[np.where(tmp!='MAR')[0]],
                    tmp[np.where(tmp=='MAR')[0]])
lgnd = ax[-2,0].legend(handles=[handles[ind] for ind in ind_uni], loc='center', ncol = 2,bbox_to_anchor=(0.45, -0.5))
for i in range(len(lgnd.legendHandles)):
    if lgnd.legendHandles[i]._legmarker.get_markersize() == 3:
        lgnd.legendHandles[i]._legmarker.set_markersize(6)
    else:
        lgnd.legendHandles[i]._legmarker.set_markersize(8)
    
ax[-1,0].set_axis_off()
ax[-2,0].set_axis_off()
fig.text(0.05, 0.7, "10 m subsurface temperature ($^o$C)", ha='center', va='center', rotation='vertical',fontsize=12)
fig.savefig('figures/clusters/clusters_w_RCM.png')


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

# %% 
df['date'] = pd.to_datetime(df.date)

#%% Uncertainty analysis
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
