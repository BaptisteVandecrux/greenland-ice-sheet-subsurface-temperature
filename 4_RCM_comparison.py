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

# import GIS_lib as gis
import progressbar
import matplotlib
from rasterio.crs import CRS
from rasterio.warp import transform

print('loading dataset')
df = pd.read_csv("subsurface_temperature_summary.csv")

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

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

dates = pd.DatetimeIndex(df.date)

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

# %% Loading RCM in native CRS
print('loading RACMO')
ds_racmo = xr.open_dataset("C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc").set_coords('lat').set_coords('lon')
ds_racmo = ds_racmo.rename({'rlat': 'y', 'rlon': 'x'})

print('loading MAR')
ds_mar = xr.open_dataset("C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-T10m_2.nc")
ds_mar = ds_mar.rename({'X10_85': 'x', 'Y20_155': 'y'})
ds_mar = ds_mar.drop_vars('OUTLAY')

print('loading HIRHAM')
ds_hh = xr.open_dataset("C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m.nc")
ds_hh['time'] = pd.to_datetime([int(s) for s in np.floor(ds_hh["time"].values)], format="%Y%m%d")


# Reprojection of RCM into same CRS (RACMO's rotated pole)
print('reprojecting RACMO')
crs_racmo_proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +o_lon_p=0'
crs_racmo = CRS.from_string(crs_racmo_proj)
ds_racmo = ds_racmo.rio.write_crs(crs_racmo)
x,y = np.meshgrid(ds_racmo.x.values, ds_racmo.y.values)
lon, lat = transform( crs_racmo,{'init': 'EPSG:4326'}, x.flatten(),y.flatten())
lat = np.reshape(lat,x.shape)
lon = np.reshape(lon,x.shape)
ds_racmo['lat'] = (('y','x'), lat)
ds_racmo['lon'] = (('y','x'), lon)

print('reprojecting MAR')
crs_mar_proj = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs'
crs_mar = CRS.from_string(crs_mar_proj)
ds_mar = ds_mar.rio.write_crs(crs_mar)
ds_mar = ds_mar.rio.reproject(crs_racmo)
x,y = np.meshgrid(ds_mar.x.values, ds_mar.y.values)
lon, lat = transform( crs_racmo,{'init': 'EPSG:4326'}, x.flatten(),y.flatten())
lat = np.reshape(lat,x.shape)
lon = np.reshape(lon,x.shape)
ds_mar['lat'] = (('y','x'), lat)
ds_mar['lon'] = (('y','x'), lon)

print('reprojecting HIRHAM')
x, y = transform( {'init': 'EPSG:4326'}, crs_racmo,
                     ds_hh.lon.values.flatten(), ds_hh.lat.values.flatten())
x = np.reshape(x,ds_hh.lon.values.shape)
x = np.round(x[0,:],2)
y = np.reshape(y,ds_hh.lat.values.shape)
y= np.round(y[:,0],2)
ds_hh['x'] = x
ds_hh['y'] = y
ds_hh = ds_hh.rio.write_crs(crs_racmo)

# finding observation coordinate in RACMO's CRS
df.date = pd.to_datetime(df.date)
df['rlon'], df['rlat'] = transform( {'init': 'EPSG:4326'}, crs_racmo,
                     df.longitude, df.latitude)
df['T10m_RACMO'] = np.nan
df['T10m_MAR'] = np.nan
df['T10m_HIRHAM'] = np.nan
from progressbar import progressbar
print('extracting T10m from RCMs')

for i in progressbar(range(df.shape[0])):
    if (df.date.iloc[i] >= ds_racmo.time.min()) & (df.date.iloc[i] <= ds_racmo.time.max()):
        df.iloc[i,-3] = ds_racmo.T10m.sel(y = df.rlat.values[i], 
                                             x = df.rlon.values[i], 
                                             time = df.date.values[i],
                                             method = 'nearest').values-273.15
    if (df.date.iloc[i] >= ds_mar.time.min()) & (df.date.iloc[i] <= ds_mar.time.max()):
        df.iloc[i,-2] = ds_mar.T10m.sel(y = df.rlat.values[i], 
                                             x = df.rlon.values[i], 
                                             time = df.date.values[i],
                                             method = 'nearest').values
    if (df.date.iloc[i] >= ds_hh.time.min()) & (df.date.iloc[i] <= ds_hh.time.max()):
        df.iloc[i,-1] = ds_hh.T10m.sel(y = df.rlat.values[i], 
                                             x = df.rlon.values[i], 
                                             time = df.date.values[i],
                                             method = 'nearest').values-273.15
df_save = df.copy()  
    
# %% Plotting RCM performance
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]


matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2,figsize=(10, 9))
ax = ax.flatten()
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)
cmap = matplotlib.cm.get_cmap("tab20b")
version =['v3.12','','5','2.3p2']
sym = ['o','^','v','d']
rcm_list = ['MAR','RACMO','HIRHAM']
for i,rcm in enumerate(rcm_list):
    sym_i = 0
    if i == 1:
        i = 3
    for j, ref in enumerate(ref_list):
        ax[i].plot(
            df_10m.loc[df_10m["reference_short"] == ref, 'T10m_'+rcm],
            df_10m.loc[df_10m["reference_short"] == ref, 'temperatureObserved'],
            marker=sym[sym_i % 4],
            linestyle="none",
            markeredgecolor="lightgray",
            markeredgewidth=0.5,
            markersize=8,
            color=cmap(j / len(ref_list)),
            label=df_10m.loc[df_10m["reference_short"] == ref, "reference_short"].values[0],
        )
        sym_i=sym_i+1
    RMSE = np.sqrt(np.mean((df_10m['T10m_'+rcm] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m['T10m_'+rcm] - df_10m.temperatureObserved)
    RMSE2 = np.sqrt(np.mean((df_10m['T10m_'+rcm].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()]) ** 2))
    ME2 = np.mean(df_10m['T10m_'+rcm].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()])
    if rcm == 'HIRHAM':
        textstr = '\n'.join((
            r'MD = %.2f $^o$C' % (ME,),
            r'RMSD=%.2f $^o$C' % (RMSE,),
            r'N=%.0f' % (np.sum(~np.isnan(df_10m['T10m_'+rcm])),)))
    else:
        textstr = '\n'.join((
            r'MD = %.2f  (%.2f) $^o$C' % (ME, ME2),
            r'RMSD=%.2f (%.2f) $^o$C' % (RMSE, RMSE2),
            r'N=%.0f  (%.0f)' % (np.sum(~np.isnan(df_10m['T10m_'+rcm])),
                                  np.sum(df_10m.T10m_HIRHAM.notnull()))))
    ax[i].text(0.5, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')

    ax[i].set_title(rcm+version[i])
    ax[i].plot([-35, 2], [-35, 2], c="black")
    ax[i].set_xlim(-35, 2)
    ax[i].set_ylim(-35, 2)
    
    if i == 0:
        lgnd = ax[i].legend(title="Sources",ncol=2, bbox_to_anchor=(2.33, -0.2), loc="lower right")
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._legmarker.set_markersize(10)

fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_all.png")

    
# Comparison for ablation datasets
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.loc[(df_10m["reference_short"].astype(str) == "PROMICE")
                    | (df_10m["reference_short"].astype(str) == "Hills et al. (2018)") 
                    | (df_10m["reference_short"].astype(str) == "Hills et al. (2017)")
                    | (df_10m["site"].astype(str) == "SwissCamp"), :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["site"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2,figsize=(10, 9))
ax = ax.flatten()
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3
)
cmap = matplotlib.cm.get_cmap("tab20b")
for i,rcm in enumerate(rcm_list):
    if i == 1:
        i = 3
    for j, ref in enumerate(ref_list):
        ax[i].plot(
            df_10m.loc[df_10m["site"] == ref, 'T10m_'+rcm],
            df_10m.loc[df_10m["site"] == ref, 'temperatureObserved'],
            marker="o",
            linestyle="none",
            markeredgecolor="lightgray",
            markeredgewidth=0.5,
            markersize=6,
            color=cmap(j / len(ref_list)),
            label=df_10m.loc[df_10m["site"] == ref, "site"].values[0],
        )
    RMSE = np.sqrt(np.mean((df_10m['T10m_'+rcm] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m['T10m_'+rcm] - df_10m.temperatureObserved)

    textstr = '\n'.join((
        r'$MD=%.2f ^o$C ' % (ME, ),
        r'$RMSD=%.2f ^o$C' % (RMSE, ),
        r'$N=%.0f$' % (np.sum(~np.isnan(df_10m['T10m_'+rcm])), )))

    ax[i].text(0.63, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')
    ax[i].set_title(rcm+version[i])
    ax[i].plot([-18, 2], [-18, 2], c="black")
    ax[i].set_xlim(-18, 2)
    ax[i].set_ylim(-18, 2)
    if i == 0:
        ax[i].legend(title="Sources",ncol=3, bbox_to_anchor=(2.1, 0.1), loc="lower right")
    
fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_ablation.png")
    
# Comparison for ablation datasets
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.loc[(df_10m["reference_short"].astype(str) != "PROMICE")
                    & (df_10m["reference_short"].astype(str) != "Hills et al. (2018)") 
                    & (df_10m["reference_short"].astype(str) != "Hills et al. (2017)"), :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]


matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2,figsize=(10, 9))
ax = ax.flatten()
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)
cmap = matplotlib.cm.get_cmap("tab20b")
version =['v3.12','','5','2.3p2']
sym = ['o','^','v','d']
rcm_list = ['MAR','RACMO','HIRHAM']
for i,rcm in enumerate(rcm_list):
    sym_i = 0
    if i == 1:
        i = 3
    for j, ref in enumerate(ref_list):
        ax[i].plot(
            df_10m.loc[df_10m["reference_short"] == ref, 'T10m_'+rcm],
            df_10m.loc[df_10m["reference_short"] == ref, 'temperatureObserved'],
            marker=sym[sym_i % 4],
            linestyle="none",
            markeredgecolor="lightgray",
            markeredgewidth=0.5,
            markersize=8,
            color=cmap(j / len(ref_list)),
            label=df_10m.loc[df_10m["reference_short"] == ref, "reference_short"].values[0],
        )
        sym_i=sym_i+1
    RMSE = np.sqrt(np.mean((df_10m['T10m_'+rcm] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m['T10m_'+rcm] - df_10m.temperatureObserved)
    RMSE2 = np.sqrt(np.mean((df_10m['T10m_'+rcm].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()]) ** 2))
    ME2 = np.mean(df_10m['T10m_'+rcm].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()])
    if rcm == 'HIRHAM':
        textstr = '\n'.join((
            r'MD = %.2f $^o$C' % (ME,),
            r'RMSD=%.2f $^o$C' % (RMSE,),
            r'N=%.0f' % (np.sum(~np.isnan(df_10m['T10m_'+rcm])),)))
    else:
        textstr = '\n'.join((
            r'MD = %.2f  (%.2f) $^o$C' % (ME, ME2),
            r'RMSD=%.2f (%.2f) $^o$C' % (RMSE, RMSE2),
            r'N=%.0f  (%.0f)' % (np.sum(~np.isnan(df_10m['T10m_'+rcm])),
                                  np.sum(df_10m.T10m_HIRHAM.notnull()))))
    ax[i].text(0.5, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')

    ax[i].set_title(rcm+version[i])
    ax[i].plot([-35, 2], [-35, 2], c="black")
    ax[i].set_xlim(-35, 2)
    ax[i].set_ylim(-35, 2)
    
    if i == 0:
        lgnd = ax[i].legend(title="Sources",ncol=2, bbox_to_anchor=(2.33, -0.2), loc="lower right")
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._legmarker.set_markersize(10)

fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_non_ablation.png")

# %% Clusters
from sklearn.cluster import DBSCAN
df = df_save

df["year"] = pd.DatetimeIndex(df.date).year

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(crs_racmo)
)

epsilon = 0.03
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
# gdf = gdf.set_index(["clusters", "date"],drop=False)

gdf['ind'] = gdf.index.values
df_meta = pd.DataFrame()
df_meta['cluster_id'] = gdf.groupby('clusters').date.min().index.values
df_meta['year_min'] = gdf.groupby('clusters').date.min().dt.year.values
df_meta['year_max'] = gdf.groupby('clusters').date.max().dt.year.values
df_meta['rlat'] = gdf.groupby('clusters').rlat.mean().values
df_meta['rlon'] = gdf.groupby('clusters').rlon.mean().values
df_meta['latitude'] = gdf.groupby('clusters').latitude.mean().values
df_meta['latitude_std'] = gdf.groupby('clusters').latitude.std().values
df_meta['longitude'] = gdf.groupby('clusters').longitude.mean().values
df_meta['longitude_std'] = gdf.groupby('clusters').longitude.std().values
df_meta['elevation'] = gdf.groupby('clusters').elevation.mean().values
df_meta['site_all'] = gdf.groupby('clusters').site.transform(lambda x: ', '.join(x.astype(str)))[gdf.groupby('clusters').ind.first().values].values
df_meta['site_all'] = gdf.groupby('clusters').site.transform(lambda x: ', '.join(x.astype(str)))[gdf.groupby('clusters').ind.first().values].values
df_meta['site'] = gdf.groupby('clusters').site.nth(3)

df_meta = df_meta.loc[(df_meta.year_max-df_meta.year_min)>3,:]
df_meta = df_meta.iloc[1:,:]

ref_all = np.array([])
for k, cluster_id in enumerate(df_meta.cluster_id[df_meta.cluster_id>0]):
    ref_all = np.append(ref_all,
                        gdf.loc[gdf.clusters==cluster_id].reference_short.unique())
ref_all = np.unique(ref_all)

# plotting clusters

gdf_3413 = gdf.to_crs(3413)
fig, ax = plt.subplots(1, 1, figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0, top=1, bottom=0, left=0, right=1)
land.plot(ax=ax, zorder=0, color="black")
ice.plot(ax=ax, zorder=1, color="lightblue")
unique_labels = np.unique(labels)
gdf_3413.plot(ax=ax, color="black")
for k in unique_labels[unique_labels>0]:
    gdf_3413.loc[gdf_3413.clusters == k].plot(ax=ax, markersize=50,label='cluster '+str(k))
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


# %% extracting RCM  10m temperature for these clusters
# RACMO
points = [
    (x, y) for x, y in zip(ds_racmo["lat"].values.flatten(), ds_racmo["lon"].values.flatten())
]

df_RACMO = pd.DataFrame()
df_RACMO['time'] = ds_racmo["time"].values

for(site, rlat, rlon) in (zip(df_meta.site, df_meta.rlat, df_meta.rlon)):
    df_RACMO[site] = ds_racmo.T10m.sel(
        x=rlon, y=rlat, method='nearest') - 273.15
    
df_RACMO = df_RACMO.set_index('time')

# HIRHAM
points = [
    (x, y) for x, y in zip(ds_hh["lat"].values.flatten(), ds_hh["lon"].values.flatten())
]

df_HIRHAM = pd.DataFrame()
df_HIRHAM['time'] = ds_hh["time"].values
for(site, rlat, rlon) in (zip(df_meta.site, df_meta.rlat, df_meta.rlon)):
    df_HIRHAM[site] = ds_hh.T10m.sel(
        x=rlon, y=rlat, method='nearest') - 273.15
    
df_HIRHAM = df_HIRHAM.set_index('time')
    

# MAR
points = [
    (x, y) for x, y in zip(ds_mar["lat"].values.flatten(), ds_mar["lon"].values.flatten())
]

df_MAR = pd.DataFrame()
df_MAR['time'] = ds_mar["time"].values
for(site, rlat, rlon) in (zip(df_meta.site, df_meta.rlat, df_meta.rlon)):
    df_MAR[site] = ds_mar.T10m.sel(
        x=rlon, y=rlat, method='nearest')
    
df_MAR = df_MAR.set_index('time')

# %% plotting
def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

    return date.year + fraction
from datetime import datetime as dt
import statsmodels.api as sm

fig, ax = plt.subplots(11, 2, figsize=(13, 15))
ax = ax.flatten()
fig.subplots_adjust(hspace=0.1, wspace=0.2, top=0.97, bottom=0.1, left=0.1, right=0.9)

import matplotlib.cm as cm
cmap = cm.get_cmap('tab20', len(ref_all))
handles = list()
labels = list()
import matplotlib.dates as mdates


for k, cluster_id in enumerate(df_meta.cluster_id.astype(int)):
    # plotting RACMO 10 m temperature for that location
    df_MAR[df_meta.site.iloc[k]].plot(ax = ax[k],label='MAR',color='tab:orange')
    df_RACMO[df_meta.site.iloc[k]].plot(ax = ax[k],label='RACMO',color='tab:red')
    df_HIRHAM[df_meta.site.iloc[k]].plot(ax = ax[k],label='HIRHAM',color='tab:blue')
    gdf.loc[gdf.clusters == cluster_id].set_index('date').temperatureObserved.plot(
        marker='o',linestyle='None', ax=ax[k], color='k', markersize=4)
    
    ax[k].set_xlim('1950-01-01','2022-01-01')
    ax[k].set_xlabel('')
    ax[k].text(0.03, 0.95, gdf.loc[gdf.clusters == cluster_id].site.iloc[0], 
                 transform=ax[k].transAxes, 
                 fontsize=11, fontweight='bold',
                 verticalalignment='top')
    ax[k].set_ylabel("")
    ax[k].plot(np.nan,np.nan,'ok',label='Observations')
    h, l = ax[k].get_legend_handles_labels()
    handles = handles + h
    labels = labels + l
    ylim = ax[k].get_ylim()
    if ylim[1]-ylim[0] < 2:
        ax[k].set_ylim(ylim[0]-2, ylim[1]+2)


lgnd = ax[-2,0].legend(loc='center', ncol = 2,bbox_to_anchor=(0.45, -1))

ax[-1,0].set_axis_off()
# ax[-2,0].set_axis_off()
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
