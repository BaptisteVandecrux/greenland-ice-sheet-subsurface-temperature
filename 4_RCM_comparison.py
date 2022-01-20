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


df = pd.read_csv("subsurface_temperature_summary.csv")

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

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

dates = pd.DatetimeIndex(df.date)

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)


# %% Adding RCM data to dataframe
file_list = ["subsurface_temperature_summary_w_MAR.csv",
             "subsurface_temperature_summary_w_RACMO.csv",
             "subsurface_temperature_summary_w_HIRHAM.csv"]
rcm_list = ['MAR', 'RACMO', 'HIRHAM']

for file, rcm in zip(file_list,rcm_list):
    df_rcm = pd.read_csv(file)
    df[rcm+'_10m_temp'] = np.nan
    df[rcm+'_distance'] = np.nan
    df[rcm+'_elevation'] = np.nan
    missing = list()
    for i, (lat, lon, date) in progressbar.progressbar(enumerate(zip(df.latitude, df.longitude, df.date))):
        tmp = df_rcm.loc[((df_rcm['latitude'] == lat)
                             & (df_rcm['longitude'] == lon)
                             & (df_rcm['date'] == date))]
        if len(tmp) == 0:
            ref = df.site.iloc[i] +' '+ df.reference_short.iloc[i]
            if ref not in missing:
                missing.append(ref)
        else:
            df.iloc[i,  df.columns.get_loc(rcm+'_10m_temp')] = tmp[rcm+'_10m_temp'].iloc[0]
            df.iloc[i,  df.columns.get_loc(rcm+'_distance')] = tmp[rcm+'_distance'].iloc[0]
            df.iloc[i,  df.columns.get_loc(rcm+'_elevation')] = tmp[rcm+'_elevation'].iloc[0]
    print(' ')
    print('Missing measurements in '+rcm+' file:')
    for line in missing:
        print(line)
        
# %% Plotting RCM performance
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "PROMICE", :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "Hills et al. (2018)", :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "Hills et al. (2017)", :]
# df_10m = df_10m.loc[df_10m.MAR_10m_temp > -200, :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]


matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2,figsize=(10, 9))
ax = ax.flatten()
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3
)
cmap = matplotlib.cm.get_cmap("tab20b")
version =['v3.12','','5','2.3p2']
sym = ['o','^','v','d']
for i,rcm in enumerate(rcm_list):
    sym_i = 0
    if i == 1:
        i = 3
    for j, ref in enumerate(ref_list):
        ax[i].plot(
            df_10m.loc[df_10m["reference_short"] == ref, rcm+'_10m_temp'],
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
    RMSE = np.sqrt(np.mean((df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved)
    RMSE2 = np.sqrt(np.mean((df_10m[rcm+'_10m_temp'].loc[df_10m.HIRHAM_10m_temp.notnull()] - df_10m.temperatureObserved.loc[df_10m.HIRHAM_10m_temp.notnull()]) ** 2))
    ME2 = np.mean(df_10m[rcm+'_10m_temp'].loc[df_10m.HIRHAM_10m_temp.notnull()] - df_10m.temperatureObserved.loc[df_10m.HIRHAM_10m_temp.notnull()])
    if rcm == 'HIRHAM':
        textstr = '\n'.join((
            r'MD = %.2f $^o$C' % (ME,),
            r'RMSD=%.2f $^o$C' % (RMSE,),
            r'N=%.0f' % (np.sum(~np.isnan(df_10m[rcm+'_10m_temp'])),)))
    else:
        textstr = '\n'.join((
            r'MD = %.2f  (%.2f) $^o$C' % (ME, ME2),
            r'RMSD=%.2f (%.2f) $^o$C' % (RMSE, RMSE2),
            r'N=%.0f  (%.0f)' % (np.sum(~np.isnan(df_10m[rcm+'_10m_temp'])),
                                  np.sum(df_10m.HIRHAM_10m_temp.notnull()))))
    ax[i].text(0.5, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')

    ax[i].set_title(rcm+version[i])
    ax[i].plot([-46, 5], [-46, 5], c="black")
    ax[i].set_xlim(-46, 2)
    ax[i].set_ylim(-46, 2)
    
    if i == 0:
        lgnd = ax[i].legend(title="Sources",ncol=2, bbox_to_anchor=(2.33, -0.2), loc="lower right")
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._legmarker.set_markersize(10)

fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_all.png")
    
    # %% Comparison for ablation datasets
    
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
            df_10m.loc[df_10m["site"] == ref, rcm+'_10m_temp'],
            df_10m.loc[df_10m["site"] == ref, 'temperatureObserved'],
            marker="o",
            linestyle="none",
            markeredgecolor="lightgray",
            markeredgewidth=0.5,
            markersize=6,
            color=cmap(j / len(ref_list)),
            label=df_10m.loc[df_10m["site"] == ref, "site"].values[0],
        )
    RMSE = np.sqrt(np.mean((df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved)

    textstr = '\n'.join((
        r'$MD=%.2f ^o$C ' % (ME, ),
        r'$RMSD=%.2f ^o$C' % (RMSE, ),
        r'$N=%.0f$' % (np.sum(~np.isnan(df_10m[rcm+'_10m_temp'])), )))

    ax[i].text(0.63, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')
    ax[i].set_title(rcm+version[i])
    ax[i].plot([-46, 5], [-46, 5], c="black")
    ax[i].set_xlim(-46, 2)
    ax[i].set_ylim(-46, 2)
    if i == 0:
        ax[i].legend(title="Sources",ncol=3, bbox_to_anchor=(2.1, 0.1), loc="lower right")
    
fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_ablation.png")
    
    # %% Comparison for ablation datasets
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
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3
)
cmap = matplotlib.cm.get_cmap("tab20b")
for i,rcm in enumerate(rcm_list):
    if i == 1:
        i = 3
    for j, ref in enumerate(ref_list):
        ax[i].plot(
            df_10m.loc[df_10m["reference_short"] == ref, rcm+'_10m_temp'],
            df_10m.loc[df_10m["reference_short"] == ref, 'temperatureObserved'],
            marker="o",
            linestyle="none",
            markeredgecolor="gray",
            markersize=10,
            color=cmap(j / len(ref_list)),
            label=df_10m.loc[df_10m["reference_short"] == ref, "reference_short"].values[0],
        )
    RMSE = np.sqrt(np.mean((df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m[rcm+'_10m_temp'] - df_10m.temperatureObserved)

    textstr = '\n'.join((
        r'$MD=%.2f ^o$C ' % (ME, ),
        r'$RMSD=%.2f ^o$C' % (RMSE, ),
        r'$N=%.0f$' % (np.sum(~np.isnan(df_10m[rcm+'_10m_temp'])), )))

    ax[i].text(0.63, 0.25, textstr, transform=ax[i].transAxes, fontsize=12,
            verticalalignment='top')
    ax[i].set_title(rcm+version[i])
    ax[i].plot([-46, 5], [-46, 5], c="black")
    ax[i].set_xlim(-46, 2)
    ax[i].set_ylim(-46, 2)
    if i == 0:
        ax[i].legend(title="Sources",ncol=2, bbox_to_anchor=(2.33, -0.2), loc="lower right")

fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
ax[1].set_axis_off()
fig.savefig("figures/RCM_comp_non_ablation.png")
