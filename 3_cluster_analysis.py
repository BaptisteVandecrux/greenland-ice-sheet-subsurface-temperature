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


#loading data
df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")
import geopandas as gpd

# ============ To fix ================

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
df['date'] = pd.to_datetime(df.date,utc=True, errors='coerce')
df["year"] = pd.DatetimeIndex(df.date).year

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

df['x_3413'] = gdf.geometry.x
df['y_3413'] = gdf.geometry.y

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
DSA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/DSA_MAR_4326.shp")
LAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/LAPA_MAR_4326.shp")
HAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/HAPA_MAR_4326.shp")
firn = gpd.GeoDataFrame.from_file(
    "Data/misc/firn areas/FirnLayer2000-2017_final_4326.shp"
)
PA = pd.concat([LAPA, HAPA])

df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()
df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

import matplotlib
matplotlib.rcParams.update({"font.size": 14})
from matplotlib import cm
from matplotlib import patches as mpatches

# % spatial distribution plot
from matplotlib import gridspec
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         width_ratios=[1], wspace=0.1,
                         hspace=0.05 , height_ratios=[3, 1])

fig = plt.figure(figsize=(9, 13))
ax1 = fig.add_subplot(spec[0])
box = ax1.get_position()
box.x0 = box.x0 - 0.2
box.x1 = box.x1 - 0.2
ax1.set_position(box)
land.to_crs("EPSG:3413").plot(ax=ax1, color="lightgray")
ice.to_crs("EPSG:3413").plot(ax=ax1, color="gray")
DSA.to_crs("EPSG:3413").plot(ax=ax1, color="tab:blue")
# LAPA.to_crs("EPSG:3413").plot(ax=ax1, color="m")
PA.to_crs("EPSG:3413").plot(ax=ax1, color="tab:red")

ax1.axis("off")
h = [np.nan, np.nan, np.nan, np.nan, np.nan]
h[0] = mpatches.Patch(facecolor="lightgray", label="Land")
h[1] = mpatches.Patch(facecolor="gray", label="Bare ice area")
h[2] = mpatches.Patch(facecolor="tab:blue", label="Dry snow area")
# h[3] = mpatches.Patch(facecolor="m", label="Low accumulation\npercolation area")
h[3] = mpatches.Patch(facecolor="tab:red", label="Percolation area")
h[4] = plt.plot(np.nan, np.nan,
    marker="h", color="lightgray", markersize=8, markerfacecolor="k",
    linestyle="None", label="Observation sites")[0]

ax1.legend(handles=h, bbox_to_anchor=(1.1, 0.5), loc="lower left", 
           fontsize=14, frameon=False)
ax1.text(0.02, 0.95, "(a)" ,
        transform=ax1.transAxes, fontsize=14, 
        verticalalignment="top",fontweight='bold')
hb = ax1.hexbin(df.x_3413, df.y_3413,
    bins="log", gridsize=(20, 26), mincnt=1,
    linewidth=0.5, edgecolors="k", cmap="gnuplot2")
cbar_ax = fig.add_axes([0.62, 0.58, 0.2, 0.01])
cb = plt.colorbar(hb, ax=ax1, cax=cbar_ax, orientation="horizontal")
cb.ax.get_yaxis().fontsize = 14
cb.set_label("Number of \n$T_{10m}$ observations", fontsize=12, rotation=0)

ax2 = fig.add_subplot(spec[1])
ax2.set_title("(b)",loc='left',fontweight='bold')
ax2.hist(df.date.dt.year.values + df.date.dt.month.values/12, density=False, bins=np.arange(1910,2024,2.5), alpha=0.7, edgecolor="white")
ax2.set_yscale('log')
ax2.set_ylabel('Number of observations')
ax2.set_xlabel('Year')
ax2.grid()  

fig.savefig('figures/figure1.tif', dpi=300, bbox_inches='tight')
# fig.savefig('figures/figure1.png', dpi=120)

# %% Studying clusters
fig, ax = plt.subplots(1, 1, figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0, top=1, bottom=0, left=0, right=1)
land.plot(ax=ax, zorder=0, color="black")
ice.plot(ax=ax, zorder=1, color="lightblue")
gdf.plot(ax=ax, column="year", cmap="tab20c", markersize=50,
    edgecolor="gray", legend=True,
    legend_kwds={"label": "Year of measurement",
        "orientation": "horizontal",
        "shrink": 0.8},
)


plt.axis("off")
plt.savefig("figures/cluster_map.png")

from sklearn.cluster import DBSCAN

epsilon = 4500
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

#  Black removed and is used for noise instead.
plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0, top=1, bottom=0, left=0, right=1)
land.plot(ax=ax, zorder=0, color="black")
ice.plot(ax=ax, zorder=1, color="lightblue")
unique_labels = set(labels)
colors = [plt.cm.tab20c(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = coords[class_member_mask & core_samples_mask]
    ax.plot(
        xy[:, 0], xy[:, 1], "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = coords[class_member_mask & ~core_samples_mask]
    ax.plot(
        xy[:, 0], xy[:, 1], "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
gdf.plot(ax=ax, color="black")
gdf.loc[gdf.reference == "GC-Net_v2 by GEUS"].plot(ax=ax, color="red", marker="^")
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

gdf["clusters"] = labels
# gdf.date = gdf.date.astype("datetime64[ns]")
gdf = gdf.set_index(["clusters", "date"])
from rasterio.crs import CRS

print("loading ANN")
ds_ann = xr.open_dataset("output/T10m_prediction.nc")
ds_ann["time"] = pd.to_datetime(ds_ann["time"])
crs_ann = CRS.from_string("EPSG:4326")
ds_ann = ds_ann.rio.write_crs(crs_ann)
cluster_coord = gdf.reset_index()[['latitude','longitude','clusters']].groupby('clusters').mean()
cluster_coord['site'] = gdf.reset_index()[['site','clusters']].groupby('clusters').site.apply(lambda x: list(np.unique(x)))

# %% Plotting all clusters

i = -1
import matplotlib.cm as cm
plt.close('all')
handles = list()
labels = list()
import matplotlib.dates as mdates
sym= 'o d ^ v > < s x *'.split()
for cluster_id in unique_labels:
    if cluster_id == -1:
        continue

    tmp = gdf.loc[cluster_id]
    ref_list = tmp.reference_short.unique()
    print(cluster_id, tmp.site.unique(), np.nanmin(tmp.year), np.nanmax(tmp.year),',', tmp.shape[0])
    if len(ref_list) > 1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax = [ax]
        i = 0
        df_ANN = ds_ann.T10m.sel(longitude=cluster_coord.loc[cluster_id,'longitude'],
            latitude=cluster_coord.loc[cluster_id,'latitude'],
            method="nearest").to_dataframe()
        df_ANN.T10m.plot(ax=ax[i],label='ANN',color='lightgray')
        df_ANN.T10m.resample("Y").mean().plot(ax=ax[i],color='k',
                                              label='ANN annual average')
        for ref in ref_list:
            if ref == "PROMICE":
                tmp2 = tmp.loc[tmp.reference_short == ref,:]
                for k, site in enumerate(tmp2.site.unique()):
                    tmp2.loc[tmp2.site == site, :].temperatureObserved.plot(
                    ax=ax[i],
                    marker=sym[k],
                    markersize=6,
                    linestyle="none",
                    label=ref+' '+site,
                    color="purple",
                )
            else:
                tmp.loc[tmp.reference_short == ref].temperatureObserved.plot(
                    ax=ax[i], marker="o", markersize=6, linestyle="none", label=ref
                )
            
        ax[i].set_ylabel("Temperature 10 m below the surface ($^oC$)")
        ax[i].set_xlabel("")
        ax[i].set_title(str(np.unique(tmp.site)))
        ax[i].legend(loc='lower center',bbox_to_anchor=(0.5,1.1))
        ax[i].set_ylim(-33, 2)
        ax[i].grid()

        fig.savefig("figures/clusters/"+tmp.site.unique()[-1]+".png")


# %% Plotting at specific clusters
from sklearn.linear_model import LinearRegression
def trend(series_with_nan):
    start_year = series_with_nan.first_valid_index().year
    series_dropped_nan = series_with_nan.dropna()
    
    # Convert the datetime index to a numeric format (years since start) for the modified series
    years_since_start_dropped_nan = series_dropped_nan.index.year - start_year
    
    # Prepare data for linear regression with NaN values dropped
    X_dropped_nan = years_since_start_dropped_nan.values.reshape(-1, 1)
    y_dropped_nan = series_dropped_nan.values
    
    # Perform linear regression on the modified data
    model_dropped_nan = LinearRegression()
    model_dropped_nan.fit(X_dropped_nan, y_dropped_nan)
    
    # The slope of the regression line indicates the trend
    trend_slope_dropped_nan = model_dropped_nan.coef_[0]
    trend_slope_dropped_nan
    return trend_slope_dropped_nan 

abc = 'abcdefghijkl'

fig, ax = plt.subplots(2,1, figsize=(10, 10), sharex=True, sharey=False)
ax = ax.flatten()
# fig.subplots_adjust(
#     hspace=0.3, wspace=0.05, top=0.88, bottom=-0.5, left=0.1, right=0.85
# )

handles = list()
labels = list()

sym= 'o d ^ v > < s x *'.split()
site_list = ['QAS_U','QAS_L'] #['NASA-SE','NASA-E','Summit_THM', 'Tunu-N', 'South Dome', 'Saddle', 'Humboldt','Crawford Point 1','DYE-2','Swiss Camp']

for i, site in enumerate(site_list):
    cluster_id = cluster_coord.index[cluster_coord.site.apply(lambda x: site in x)][0]

    tmp = gdf.loc[cluster_id].sort_index()[['temperatureObserved']].resample('M').mean()
    

    df_ANN = ds_ann.T10m.bfill(dim='latitude').sel(longitude=cluster_coord.loc[cluster_id,'longitude'],
        latitude=cluster_coord.loc[cluster_id,'latitude'],
        method="nearest").to_dataframe()
    df_ANN.T10m.plot(ax=ax[i],label='ANN annual values',
                     color='k',drawstyle='steps')
    
    df_ANN['T10m_pred'] =  (
               trend(df_ANN.T10m)*(df_ANN.index.year + df_ANN.index.month/12)
                   -trend(df_ANN.T10m)*1950.0833333333333 
                       + df_ANN.T10m.iloc[0]
                       )
    df_ANN.T10m_pred.plot(ax=ax[i],label='__nolegend__',
                     color='k')
    df_ANN_site = df_ANN.T10m.copy()
    tmp.temperatureObserved.plot(
        ax=ax[i], marker="o",c='tab:red',
        markersize=5, linestyle="none", label='observation'
    )
    msk = np.isin(df_ANN_site.index.year, tmp.index.year)
    df_ANN_site.loc[~msk] = np.nan
    # df_ANN_site.plot(ax=ax[i],color='k', 
    #  label='years when observations are available',
    #                  drawstyle='steps', alpha=0.7, lw=3)  
    try:
        print('%s, %0.1f, %0.1f, %0.1f, %i, %0.1f, %0.1f, %0.1f, %i'%(site.replace('_THM',''),
              10*trend(df_ANN_site.loc['1998':'2010']),
              10*trend(tmp.temperatureObserved.loc['1998':'2010']),
              10*trend(df_ANN_site.loc['1998':'2010']) - 10*trend(tmp.temperatureObserved.loc['1998':'2010']),
              tmp.temperatureObserved.loc['1998':'2010'].notnull().sum(),
                    10*trend(df_ANN_site.loc['1998':'2022']),
                    10*trend(tmp.temperatureObserved.loc['1998':'2022']),
                    10*trend(df_ANN_site.loc['1998':'2022']) - 10*trend(tmp.temperatureObserved.loc['1998':'2022']),
                    tmp.temperatureObserved.loc['1998':'2022'].notnull().sum()))
    except:
        print('%s, nan, nan, %0.1f, %0.1f, %0.1f, %i'%(site.replace('_THM',''),
                    10*trend(df_ANN_site.loc['1998':'2022']),
                    10*trend(tmp.temperatureObserved.loc['1998':'2022']),
                    10*trend(df_ANN_site.loc['1998':'2022']) - 10*trend(tmp.temperatureObserved.loc['1998':'2022']),
                    tmp.temperatureObserved.loc['1998':'2022'].notnull().sum()))
        pass
    
        
    ax[i].set_ylabel("$T_{10m}$ ($^oC$)")
    ax[i].set_xlabel("")
    ax[i].set_title('('+abc[i]+') '+site.replace('_THM',''))
    if i == 0: ax[i].legend(loc='lower center')##,bbox_to_anchor=(1.1,1.2))
    # ax[i].set_yticks(np.arange(-30,0,5))
    # ax[i].set_ylim(df_ANN.T10m.resample("Y").mean().mean()-5,
    #                df_ANN.T10m.resample("Y").mean().mean()+5)
    # if i%2==1:
    #     ax[i].yaxis.set_label_position("right")
    #     ax[i].yaxis.tick_right()
    ax[i].set_xlim('1950', '2023')
    ax[i].grid()

fig.savefig("figures/clusters/selected.png")
