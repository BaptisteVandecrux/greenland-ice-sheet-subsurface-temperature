# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
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


# %% Extracting closest cell in HIRHAM
print("Extracting closest cell in HIRHAM")
ds = xr.open_dataset( "C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m_1980.nc" )
points = [
    (x, y) for x, y in zip(ds["lat"].values.flatten(), ds["lon"].values.flatten())
]

df["HIRHAM_distance"] = np.nan
df["HIRHAM_lat"] = np.nan
df["HIRHAM_lon"] = np.nan
df["HIRHAM_i"] = np.nan
df["HIRHAM_j"] = np.nan
df["HIRHAM_elevation"] = np.nan
lat_prev = -999
lon_prev = -999
for i, (lat, lon) in progressbar.progressbar(enumerate(zip(df.latitude, df.longitude))):
    if lat == lat_prev and lon == lon_prev:
        df.iloc[i, df.columns.get_loc("HIRHAM_lat")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_lat")
        ]
        df.iloc[i, df.columns.get_loc("HIRHAM_lon")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_lon")
        ]
        df.iloc[i, df.columns.get_loc("HIRHAM_distance")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_distance")
        ]
        df.iloc[i, df.columns.get_loc("HIRHAM_i")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_i")
        ]
        df.iloc[i, df.columns.get_loc("HIRHAM_j")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_j")
        ]
        df.iloc[i, df.columns.get_loc("HIRHAM_elevation")] = df.iloc[
            i - 1, df.columns.get_loc("HIRHAM_elevation")
        ]
    else:
        closest = points[cdist([(lat, lon)], points).argmin()]
        df.iloc[i, df.columns.get_loc("HIRHAM_lat")] = closest[0]
        df.iloc[i, df.columns.get_loc("HIRHAM_lon")] = closest[1]
        df.iloc[i, df.columns.get_loc("HIRHAM_distance")] = cdist(
            [(lon, lat)], [closest]
        )[0][0]

        ind = np.argwhere(
            np.array(np.logical_and(ds["lat"] == closest[0], ds["lon"] == closest[1]))
        )[0]
        df.iloc[i, df.columns.get_loc("HIRHAM_i")] = ind[0]
        df.iloc[i, df.columns.get_loc("HIRHAM_j")] = ind[1]

        df.iloc[i, df.columns.get_loc("HIRHAM_elevation")] = np.nan
        lat_prev = lat
        lon_prev = lon

    # plt.figure()
    # plt.scatter(ds['lon'].values.flatten(), ds['lat'].values.flatten())
    # plt.plot(lon,lat, marker='o',markersize=10,color='green')
    # plt.plot(np.array(ds.lon)[ind[0], ind[1]],np.array(ds.lat)[ind[0], ind[1]], marker='o',markersize=10,color='red')
    # plt.plot(closest[1],closest[0], marker='o',markersize=5)

    # date_in_days_since = (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).days + (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).seconds/60/60/24
# %% Extracting temperatures from HIRHAM
print("Extracting temperatures from HIRHAM")

df["HIRHAM_10m_temp"] = np.nan
for year in range(1980,2017):
    print(year)
    ds = xr.open_dataset( "C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m_"+str(year)+".nc" )

    time_HIRHAM = pd.to_datetime(
        [int(s) for s in np.floor(ds["time"].values)], format="%Y%m%d"
    )
    df = df.reset_index(drop=True)
    for ind in df.loc[df.year==year,:].index:
        
        df.loc[ind, "HIRHAM_10m_temp"] = (
            ds.T10m[
                dict(
                    x=int(df.loc[ind].HIRHAM_j),
                    y=int(df.loc[ind].HIRHAM_i),
                    time=np.argmin(
                        np.abs(time_HIRHAM - np.datetime64(df.loc[ind, "date"]))
                    ),
                )
            ].values
            - 273.15
        )
# %%
df.to_csv("subsurface_temperature_summary_w_HIRHAM.csv")

# %% Comparison
df = pd.read_csv("subsurface_temperature_summary_w_HIRHAM.csv")

df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "PROMICE", :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "Hills et al. (2018)", :]
# df_10m = df_10m.loc[df_10m["reference_short"] != "Hills et al. (2017)", :]
df_10m = df_10m.loc[df_10m.HIRHAM_10m_temp > -200, :]
df_10m = df_10m.reset_index(drop=True)
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]


fig = plt.figure(figsize=(15, 9))
matplotlib.rcParams.update({"font.size": 16})
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(
    left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None
)

cmap = matplotlib.cm.get_cmap("tab20b")
for i, ref in enumerate(ref_list):
    ax.plot(
        df_10m.loc[df_10m["reference_short"] == ref, :].HIRHAM_10m_temp,
        df_10m.loc[df_10m["reference_short"] == ref, :].temperatureObserved,
        marker="o",
        linestyle="none",
        markeredgecolor="gray",
        markersize=10,
        color=cmap(i / len(ref_list)),
        label=df_10m.loc[df_10m["reference_short"] == ref, "reference_short"].values[0],
    )
RMSE = np.sqrt(np.mean((df_10m.HIRHAM_10m_temp - df_10m.temperatureObserved) ** 2))
ax.set_title("All. RMSE = %0.2f" % RMSE)
plt.legend(title="Sources", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.plot([-40, 0], [-40, 0], c="black")
ax.set_xlabel("HIRHAM simulated 10 m subsurface temperature ($^o$C)")
ax.set_ylabel("Observed 10 m subsurface temperature ($^o$C)")
fig.savefig("figures/HIRHAM_comp_1.png")

# %% Comparison
dataset = ["PROMICE", "Hills et al. (2018)"]
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
    if data_name == "Hills et al. (2018)":
        df_10m = df_10m.loc[
            np.isin(
                df_10m["reference_short"],
                ["Hills et al. (2017)", "Hills et al. (2018)"],
            ),
            :,
        ]
    else:
        df_10m = df_10m.loc[df_10m["reference_short"] == data_name, :]

    df_10m = df_10m.loc[df_10m.HIRHAM_10m_temp > -200, :]
    df_10m = df_10m.reset_index(drop=True)
    df_10m = df_10m.sort_values("year")
    site_list = df_10m["site"].unique()

    df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(
        left=None, bottom=None, right=0.5, top=None, wspace=None, hspace=None
    )

    cmap = matplotlib.cm.get_cmap("tab20b")
    for i, site in enumerate(site_list):
        ax.plot(
            df_10m.loc[df_10m["site"] == site, :].HIRHAM_10m_temp,
            df_10m.loc[df_10m["site"] == site, :].temperatureObserved,
            marker="o",
            linestyle="none",
            markeredgecolor="gray",
            markersize=10,
            color=cmap(i / len(site_list)),
            label=df_10m.loc[df_10m["site"] == site, "site"].values[0],
        )
    RMSE = np.sqrt(np.mean((df_10m.HIRHAM_10m_temp - df_10m.temperatureObserved) ** 2))
    ax.set_title(data_name + " RMSE = %0.2f" % RMSE)
    plt.legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.plot([-40, 0], [-40, 0], c="black")
    ax.set_xlabel("HIRHAM simulated 10 m subsurface temperature ($^o$C)")
    ax.set_ylabel("Observed 10 m subsurface temperature ($^o$C)")
    fig.savefig("figures/HIRHAM_comp_" + data_name + ".png")

# %% Comparison
dataset = ["PROMICE", "Hills et al. (2018)"]
# dataset = ['GC-Net', 'Hills et al. (2018)']
for data_name in dataset:
    df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
    df_10m = df_10m.loc[df_10m["reference_short"] == data_name, :]
    df_10m = df_10m.loc[df_10m.HIRHAM_10m_temp > -200, :]
    df_10m = df_10m.reset_index(drop=True)
    df_10m = df_10m.sort_values("year")
    df_10m.date = pd.to_datetime(df_10m.date)
    site_list = df_10m["site"].unique()

    df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]
    fig = plt.figure(figsize=(20, 15))
    matplotlib.rcParams.update({"font.size": 12})

    cmap = matplotlib.cm.get_cmap("tab20b")
    for i, site in enumerate(site_list):
        ax = fig.add_subplot(int(np.floor(len(site_list) / 3)) + 1, 3, i + 1)
        tmp = df_10m.loc[df_10m["site"] == site, :].set_index("date").sort_index()
        tmp.temperatureObserved.plot(
            ax=ax,
            marker="o",
            linestyle="--",
            markeredgecolor="gray",
            markersize=8,
            color=cmap(i / len(site_list)),
            label=site + " obs",
        )
        tmp.HIRHAM_10m_temp.plot(
            ax=ax,
            marker="^",
            linestyle=":",
            markeredgecolor="gray",
            markersize=8,
            color=cmap(i / len(site_list)),
            label=site + " HIRHAM",
        )
        ax.set_title(site)
        ax.set_xlabel("")
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    # plt.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_ylabel('10 m subsurface temperature ($^o$C)')
    fig.savefig("figures/HIRHAM_comp_" + data_name + "_2.png")
