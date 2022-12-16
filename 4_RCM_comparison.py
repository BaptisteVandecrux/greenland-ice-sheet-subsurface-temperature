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

ABC = "ABCDEFGHIJKL"

print("loading dataset")
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

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

# Loading RCM and reprojecting them
crs_racmo_proj = (
    "-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +o_lon_p=0"
)
crs_racmo = CRS.from_string(crs_racmo_proj)
# target_crs = crs_racmo
target_crs = CRS.from_string("EPSG:3413")

print("loading RACMO")
ds_racmo = (
    xr.open_dataset("Data/RCM/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc")
    .set_coords("lat")
    .set_coords("lon")
)
ds_racmo = ds_racmo.rename({"rlat": "y", "rlon": "x"})
# adding crs info and lat lon grids
ds_racmo = ds_racmo.rio.write_crs(crs_racmo).drop_vars(["lat", "lon"])
# ds_racmo = ds_racmo.rio.reproject(target_crs)
# x, y = np.meshgrid(ds_racmo.x.values, ds_racmo.y.values)
# lon, lat = transform({'init': 'EPSG:3413'}, {'init': 'EPSG:4326'}, x.flatten(), y.flatten())
# lat = np.reshape(lat, x.shape)
# lon = np.reshape(lon, x.shape)
# ds_racmo['lat'] = (('y', 'x'), lat)
# ds_racmo['lon'] = (('y', 'x'), lon)

print("loading MAR")
ds_mar = xr.open_dataset( "Data/RCM/MARv3.12.1_2022_T10m_ME_1980-2021.nc" )
ds_mar = ds_mar.rename({"X10_85": "x", "Y20_155": "y", 'TIME':'time'}).drop_vars(["LAT", "LON"])
crs_mar_proj = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs"
crs_mar = CRS.from_string(crs_mar_proj)
ds_mar = ds_mar.rio.write_crs(crs_mar)
# ds_mar = ds_mar.rio.reproject(target_crs)

print("loading HIRHAM")
# HIRHAM has no projection, just a lat-lon grid
ds_hh = xr.open_dataset(
    "Data/RCM/RetMIP_2Doutput_Daily_DMIHH_T10m.nc"
)
ds_hh["time"] = pd.to_datetime(
    [int(s) for s in np.floor(ds_hh["time"].values)], format="%Y%m%d"
)
x, y = transform(
    {"init": "EPSG:4326"},
    crs_racmo,
    ds_hh.lon.values.flatten(),
    ds_hh.lat.values.flatten(),
)
x = np.reshape(x, ds_hh.lon.values.shape)
x = np.round(x[0, :], 2)
y = np.reshape(y, ds_hh.lat.values.shape)
y = np.round(y[:, 0], 2)
ds_hh["x"] = x
ds_hh["y"] = y
ds_hh = ds_hh.rio.write_crs(crs_racmo).drop_vars(["lat", "lon"])
# ds_hh = ds_hh.drop_vars(['lat', 'lon']).rio.reproject(target_crs)

print("loading ANN")
ds_ann = xr.open_dataset("output/predicted_T10m.nc")
ds_ann["time"] = pd.to_datetime(ds_ann["time"])
crs_ann = CRS.from_string("EPSG:4326")
ds_ann = ds_ann.rio.write_crs(crs_ann)
# ds_ann = ds_ann.rio.reproject(target_crs)

# finding observation coordinate in RACMO's CRS
df.date = pd.to_datetime(df.date)
df["x"], df["y"] = transform(
    {"init": "EPSG:4326"}, crs_racmo, df.longitude.values, df.latitude.values
)
df["x_mar"], df["y_mar"] = transform(
    {"init": "EPSG:4326"}, crs_mar, df.longitude.values, df.latitude.values
)
df["x_3413"], df["y_3413"] = transform(
    {"init": "EPSG:4326"},
    {"init": "EPSG:3413"},
    df.longitude.values,
    df.latitude.values,
)
df["T10m_RACMO"] = np.nan
df["T10m_MAR"] = np.nan
df["T10m_HIRHAM"] = np.nan
df["T10m_ANN"] = np.nan

# interpolating HIRHAM output at the observation locations


def extract_T10m_values(ds, df, dim1="x", dim2="y", name_out="out"):
    coords_uni = np.unique(df[[dim1, dim2]].values, axis=0)
    x = xr.DataArray(coords_uni[:, 0], dims="points")
    y = xr.DataArray(coords_uni[:, 1], dims="points")
    try:
        ds_interp = ds.interp(x=x, y=y, method="linear")
    except:
        ds_interp = ds.interp(longitude=x, latitude=y, method="linear")
    try:
        ds_interp_2 = ds.interp(x=x, y=y, method="nearest")
    except:
        ds_interp_2 = ds.interp(longitude=x, latitude=y, method="nearest")

    for i in progressbar(range(df.shape[0])):
        query_point = (df[dim1].values[i], df[dim2].values[i])
        index_point = np.where((coords_uni == query_point).all(axis=1))[0][0]
        tmp = ds_interp.T10m.isel(points=index_point).sel(
            time=df.date.values[i], method="nearest"
        )
        if tmp.isnull().all():
            tmp = ds_interp_2.T10m.isel(points=index_point).sel(
                time=df.date.values[i], method="nearest"
            )
        if (
            tmp[dim1.replace("_mar", "")].values,
            tmp[dim2.replace("_mar", "")].values,
        ) != query_point:
            print(wtf)
        if np.size(tmp) > 1:
            print(wtf)
        df.iloc[i, df.columns.get_loc(name_out)] = tmp.values
    return df


print("extracting from RACMO")
df = extract_T10m_values(ds_racmo, df, dim1="x", dim2="y", name_out="T10m_RACMO")
df.T10m_RACMO = df.T10m_RACMO - 273.15
print("extracting from MAR")
df = extract_T10m_values(ds_mar, df, dim1="x_mar", dim2="y_mar", name_out="T10m_MAR")
print("extracting from HIRHAM")
df = extract_T10m_values(ds_hh, df, dim1="x", dim2="y", name_out="T10m_HIRHAM")
df.T10m_HIRHAM = df.T10m_HIRHAM - 273.15
print("extracting from ANN")
df = extract_T10m_values(
    ds_ann, df, dim1="longitude", dim2="latitude", name_out="T10m_ANN"
)

df_save = df.copy()

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
DSA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/DSA_MAR_4326.shp")
LAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/LAPA_MAR_4326.shp")
HAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/HAPA_MAR_4326.shp")
firn = gpd.GeoDataFrame.from_file(
    "Data/misc/firn areas/FirnLayer2000-2017_final_4326.shp"
)
# %% Plotting RCM performance
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()
df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

matplotlib.rcParams.update({"font.size": 9})
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import matplotlib.patheffects as pe
from matplotlib import patches as mpatches

fig, ax = plt.subplots(2, 2, figsize=(9, 15))
ax = ax.flatten()
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.97, top=0.6, wspace=0.2, hspace=0.2)
cmap = matplotlib.cm.get_cmap("tab20b")
version = ["", "v3.12", "2.3p2", "5"]
sym = ["o", "^", "v", "d"]

ax_map = fig.add_axes([0.01, 0.6, 0.8, 0.4])
land.to_crs("EPSG:3413").plot(ax=ax_map, color="k")
ice.to_crs("EPSG:3413").plot(ax=ax_map, color="gray")
DSA.to_crs("EPSG:3413").plot(ax=ax_map, color="tab:blue")
LAPA.to_crs("EPSG:3413").plot(ax=ax_map, color="m")
HAPA.to_crs("EPSG:3413").plot(ax=ax_map, color="tab:red")
plt.annotate("A.", (0.01, 0.9), fontsize=12, xycoords="axes fraction")

ax_map.axis("off")
h = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
h[0] = mpatches.Patch(facecolor="k", label="Land")
h[1] = mpatches.Patch(facecolor="gray", label="Bare ice area")
h[2] = mpatches.Patch(facecolor="tab:blue", label="Dry snow area")
h[3] = mpatches.Patch(facecolor="m", label="Low accumulation\npercolation area")
h[4] = mpatches.Patch(facecolor="tab:red", label="High accumulation\npercolation area")
h[5] = plt.plot(
    np.nan,
    np.nan,
    marker="h",
    color="lightgray",
    markersize=8,
    markerfacecolor="k",
    linestyle="None",
    label="Observation sites",
)[0]

ax_map.legend(
    handles=h, bbox_to_anchor=(1.1, 0.5), loc="lower left", fontsize=11, frameon=False
)
ax_map.set_title("A. Ice sheet areas")

hb = ax_map.hexbin(
    df.x_3413,
    df.y_3413,
    bins="log",
    gridsize=(20, 26),
    mincnt=1,
    linewidth=0.5,
    edgecolors="white",
    cmap="magma",
)
cbar_ax = fig.add_axes([0.64, 0.79, 0.1, 0.01])
cb = plt.colorbar(hb, ax=ax_map, cax=cbar_ax, orientation="horizontal")
cb.ax.get_yaxis().fontsize = 12
cb.set_label("Number of monthly \n$T_{10m}$ observations", fontsize=12, rotation=0)

model_list = ["ANN", "RACMO", "HIRHAM", "MAR"]
for i, model in enumerate(model_list):
    sym_i = 0
    ax[i].plot(
        df_10m["T10m_" + model],
        df_10m["temperatureObserved"],
        marker="+",
        linestyle="none",
        # markeredgecolor="lightgray",
        markeredgewidth=0.5,
        markersize=5,
        color="k",
    )
    RMSE = np.sqrt(np.mean((df_10m["T10m_" + model] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m["T10m_" + model] - df_10m.temperatureObserved)
    RMSE2 = np.sqrt(
        np.mean(
            (
                df_10m["T10m_" + model].loc[df_10m.T10m_HIRHAM.notnull()]
                - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()]
            )
            ** 2
        )
    )
    ME2 = np.mean(
        df_10m["T10m_" + model].loc[df_10m.T10m_HIRHAM.notnull()]
        - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()]
    )
    if model == "HIRHAM":
        textstr = "\n".join(
            (
                r"MD = %.2f $^o$C" % (ME,),
                r"RMSD=%.2f $^o$C" % (RMSE,),
                r"N=%.0f" % (np.sum(~np.isnan(df_10m["T10m_" + model])),),
            )
        )
    else:
        textstr = "\n".join(
            (
                r"MD = %.2f  (%.2f) $^o$C" % (ME, ME2),
                r"RMSD=%.2f (%.2f) $^o$C" % (RMSE, RMSE2),
                r"N=%.0f  (%.0f)"
                % (
                    np.sum(~np.isnan(df_10m["T10m_" + model])),
                    np.sum(df_10m.T10m_HIRHAM.notnull()),
                ),
            )
        )
    ax[i].text(
        0.02,
        0.95,
        "All sites\n" + textstr,
        transform=ax[i].transAxes,
        fontsize=11,
        verticalalignment="top",
    )

    ax[i].set_title(ABC[i + 1] + ". " + model, loc="left")
    ax[i].plot([-35, 2], [-35, 2], c="black")
    ax[i].set_xlim(-35, 2)
    ax[i].set_ylim(-35, 2)
    ax[i].grid()

# Comparison for ablation datasets
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.loc[
    (df_10m["reference_short"].astype(str) == "PROMICE")
    | (df_10m["reference_short"].astype(str) == "Hills et al. (2018)")
    | (df_10m["reference_short"].astype(str) == "Hills et al. (2017)")
    | (df_10m["site"].astype(str) == "SwissCamp"),
    :,
]
df_10m = df_10m.loc[
    (df_10m["site"].astype(str) != "CEN")
    & (df_10m["site"].astype(str) != "EGP")
    & (df_10m["site"].astype(str) != "KAN_U"),
    :,
]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["site"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

for i, model in enumerate(model_list):
    ax[i].plot(
        df_10m["T10m_" + model],
        df_10m["temperatureObserved"],
        marker="+",
        linestyle="none",
        # markeredgecolor="lightgray",
        markeredgewidth=0.5,
        markersize=5,
        color="gray",
    )
    RMSE = np.sqrt(np.mean((df_10m["T10m_" + model] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m["T10m_" + model] - df_10m.temperatureObserved)

    textstr = "\n".join(
        (
            r"$MD=%.2f ^o$C " % (ME,),
            r"$RMSD=%.2f ^o$C" % (RMSE,),
            r"$N=%.0f$" % (np.sum(~np.isnan(df_10m["T10m_" + model])),),
        )
    )

    ax[i].text(
        0.64,
        0.3,
        "Ablation sites\n" + textstr,
        transform=ax[i].transAxes,
        fontsize=11,
        verticalalignment="top",
        color="gray",
    )

fig.text(
    0.5,
    0.02,
    "Simulated 10 m subsurface temperature ($^o$C)",
    ha="center",
    va="center",
    fontsize=12,
)
fig.text(
    0.03,
    0.35,
    "Observed 10 m subsurface temperature ($^o$C)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=12,
)
fig.savefig("figures/model_comp_all_no_legend.png")

# %% Plotting at selected sites
from math import sin, cos, sqrt, atan2, radians


def get_distance(point1, point2):
    R = 6370
    lat1 = radians(point1[0])  # insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


all_points = df[["latitude", "longitude"]].values

site_list = pd.DataFrame(
    np.array(
        [
            ["CP1", "1955", "2021", 69.91666666666667, -46.93333333333333, 2012.0],
            ["DYE-2", "1964", "2019", 66.46166666666667, -46.23333333333333, 2100.0],
            ["Camp Century", "1954", "2019", 77.21833333333333, -61.025, 1834.0],
            ["SwissCamp", "1990", "2005", 69.57, -49.3, 1174.0],
            ["Summit", "1997", "2018", 72.5667, -38.5, 3248.0],
            ["KAN-U", "2011", "2019", 67.0003, -47.0253, 1840.0],
            ["NASA-SE", "1998", "2021", 66.47776999999999, -42.493634, 2385.0],
            ["NASA-U", "2003", "2021", 73.840558, -49.531539, 2340.0],
            ["Saddle", "1998", "2017", 65.99947, -44.50016, 2559.0],
            ["THU_U", "1954", "2021", 76.4197, -68.1460, 770],
            ["French Camp VI", "1950", "2021", 69.706167, -48.344967, 1555.0],
            ["FA_13", "1950", "2021", 66.181, -39.0435, 1563.0],
        ]
    )
)
site_list.columns = ["site", "date start", "date end", "lat", "lon", "elev"]
site_list = site_list.set_index("site")
site_list.lon = site_list.lon.astype(float)
site_list.lat = site_list.lat.astype(float)
site_list["x"], site_list["y"] = transform(
    {"init": "EPSG:4326"},
    crs_racmo,
    site_list.lon.astype(float).values,
    site_list.lat.astype(float).values,
)
site_list["x_mar"], site_list["y_mar"] = transform(
    {"init": "EPSG:4326"},
    crs_mar,
    site_list.lon.astype(float).values,
    site_list.lat.astype(float).values,
)

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

fig, ax = plt.subplots(6, 2, figsize=(9, 17))
fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.03, hspace=0.25)
ax = ax.flatten()

for i, site in enumerate(site_list.index):
    # plotting observations
    coords = np.expand_dims(site_list.loc[site, ["lat", "lon"]].values.astype(float), 0)
    dm = cdist(all_points, coords, get_distance)
    df_select = df.loc[dm < 5, :]
    ax[i].plot(
        df_select.date,
        df_select.temperatureObserved,
        ".",
        markersize=9,
        color="darkgray",
        markeredgecolor="lightgray",
        linestyle="None",
        label="observations",
    )

    # plotting ANN
    df_ANN = ds_ann.T10m.interp(
        longitude=site_list.loc[site, "lon"],
        latitude=site_list.loc[site, "lat"],
        method="linear",
    ).to_dataframe()
    df_ANN_interp = np.interp(df_select.date, df_ANN.index, df_ANN.T10m)
    MD = np.sqrt(np.mean((df_ANN_interp - df_select.temperatureObserved) ** 2))
    RMSD = np.mean((df_ANN_interp - df_select.temperatureObserved))
    N = df_select.temperatureObserved.notnull().sum()
    print("%s, %i, %0.2f, %0.2f" % (site, N, RMSD, MD))
    # ax[i].fill_between(df_era.index, bs_out[0]-bs_out[1], bs_out[0]+bs_out[1], color = 'lightgray')
    # df_ANN.T10m.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_ANN.T10m.resample("Y").mean().plot(
        ax=ax[i],
        drawstyle="steps-post",
        color=CB_color_cycle[0],
        linewidth=2,
        label="ANN",
    )

    # plotting RACMO
    df_racmo = (
        ds_racmo.T10m.interp(
            x=site_list.loc[site, "x"], y=site_list.loc[site, "y"], method="linear"
        )
        .to_dataframe()
        .T10m
        - 273.15
    )
    if df_racmo.isnull().all():
        df_racmo = (
            ds_racmo.T10m.interp(
                x=site_list.loc[site, "x"], y=site_list.loc[site, "y"], method="nearest"
            )
            .to_dataframe()
            .T10m
            - 273.15
        )
    # df_racmo.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_racmo.resample("Y").mean().plot(
        ax=ax[i],
        drawstyle="steps-post",
        color=CB_color_cycle[1],
        linewidth=2,
        label="RACMO",
    )

    # plotting HIRHAM
    df_hh = (
        ds_hh.T10m.interp(
            x=site_list.loc[site, "x"], y=site_list.loc[site, "y"], method="linear"
        )
        .to_dataframe()
        .T10m
        - 273.15
    )
    if df_hh.isnull().all():
        df_hh = (
            ds_hh.T10m.interp(
                x=site_list.loc[site, "x"], y=site_list.loc[site, "y"], method="nearest"
            )
            .to_dataframe()
            .T10m
        )
    # df_hh.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_hh.resample("Y").mean().plot(
        ax=ax[i],
        drawstyle="steps-post",
        color=CB_color_cycle[2],
        linewidth=2,
        label="HIRHAM",
    )
    
    # plotting MAR
    df_mar = (
        ds_mar.T10m.interp(
            x=site_list.loc[site, "x_mar"],
            y=site_list.loc[site, "y_mar"],
            method="linear",
        )
        .to_dataframe()
        .T10m
    )
    if df_mar.isnull().all():
        df_mar = (
            ds_mar.T10m.interp(
                x=site_list.loc[site, "x_mar"],
                y=site_list.loc[site, "y_mar"],
                method="nearest",
            )
            .to_dataframe()
            .T10m
        )
    # df_mar.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_mar.resample("Y").mean().plot(
        ax=ax[i],
        drawstyle="steps-post",
        color=CB_color_cycle[3],
        linewidth=2,
        label="MAR",
    )
    # ax[i].set_ylim(np.nanmean(bs_out[0])-4, np.nanmean(bs_out[0])+4)
    ax[i].set_title(ABC[i] + ". " + site, loc="left")
    ax[i].grid()
    ax[i].set_xlim(pd.to_datetime("1950-01-01"), pd.to_datetime("2022-01-01"))
for i, site in enumerate(site_list.index):
    if i < 10:
        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].set_xlabel("")
    else:
        ax[i].set_xlabel("Year")
ax[0].legend(ncol=5, loc="lower right", bbox_to_anchor=(2, 1.13), fontsize=13)
fig.text(
    0.02,
    0.5,
    "10 m subsurface temperature ($^o$C)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=12,
)
fig.savefig("figures/model_comp_selected_sites.png")

# %% Preparing input for trend analysis
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")
ice_4326 = ice.to_crs("EPSG:4326")


# RACMO
# if 'ds_racmo_3413' not in locals():
#     ds_racmo_3413 = ds_racmo.rio.reproject("EPSG:3413")
ds_racmo = ds_racmo.rio.clip(
    ice.to_crs(ds_hh.rio.crs).geometry.values, ds_racmo.rio.crs
)
ds_racmo_GrIS = (ds_racmo.mean(dim=("x", "y"))).to_pandas().T10m - 273.15
# if 'ds_hh_3413' not in locals():
#     ds_hh_3413 = ds_hh.rio.reproject("EPSG:3413")
ds_hh = ds_hh.rio.clip(ice.to_crs(ds_hh.rio.crs).geometry.values, ds_hh.rio.crs)
# ds_hh_3413['T10m'] = ds_hh_3413.T10m.where(ds_hh_3413.T10m!=3.4028234663852886e+38)
ds_hh_GrIS = (ds_hh.mean(dim=("x", "y"))).to_pandas().T10m - 273.15
# ds_hh.to_netcdf('HIRHAM_T10m.nc')
# if 'ds_mar_3413' not in locals():
#     ds_mar_3413 = ds_mar.rio.reproject("EPSG:3413")
ds_mar = ds_mar.rio.clip(ice.to_crs(ds_mar.rio.crs).geometry.values, ds_mar.rio.crs)
ds_mar["T10m"] = ds_mar.T10m.where(ds_mar.T10m > -200)
ds_mar_GrIS = (ds_mar.mean(dim=("x", "y"))).to_pandas().T10m
if "ds_ann_3413" not in locals():
    ds_ann_3413 = ds_ann.rio.reproject("EPSG:3413")

ds_ann = ds_ann.rio.clip(ice_4326.geometry.values, ice_4326.crs)
ds_ann_3413 = ds_ann_3413.rio.clip(ice.geometry.values, ice.crs)
# ds_ann_3413_2 = ds_ann_3413.rio.clip(ice.geometry.values, ice.crs, all_touched=True)

ds_ann_3413["T10m"] = ds_ann_3413.T10m.where(
    ds_ann_3413["T10m"] != 3.4028234663852886e38
)
weights = np.cos(np.deg2rad(ds_ann.latitude))
weights.name = "weights"
ds_ann_weighted = ds_ann.weighted(weights)
ds_ann_GrIS = (ds_ann_weighted.mean(("latitude", "longitude"))).to_pandas().T10m

from datetime import datetime as dt

np.seterr(invalid="ignore")


def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def linregress_3D(x, y):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time.
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon).
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression
    between the two datasets along their aligned time dimension.
    Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount.
    """
    # 1. Ensure that the data are properly alinged to each other.
    x, y = xr.align(x, y)

    # 3. Compute data length, mean and standard deviation along time axis for further use:
    n = y.notnull().sum(dim="time")
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd = x.std(axis=0)
    ystd = y.std(axis=0)

    # 4. Compute covariance along time axis
    cov = np.sum((x - xmean) * (y - ymean), axis=0) / (n)

    # 5. Compute correlation along time axis
    cor = cov / (xstd * ystd)

    # 6. Compute regression slope and intercept:
    slope = cov / (xstd ** 2)
    intercept = ymean - xmean * slope

    # 7. Compute P-value and standard error
    # Compute t-statistics
    tstats = cor * np.sqrt(n - 2) / np.sqrt(1 - cor ** 2)
    stderr = slope / tstats

    from scipy.stats import t

    pval = t.sf(abs(tstats), n - 2) * 2
    pval = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cov, cor, slope, intercept, pval, stderr


# finding ice-sheet-wide average and stdev values for 1954, 1985 and 2021
# ds_ann_y = ds_ann.resample(time='Y').mean()
# ds_ann_y = ds_ann_y.weighted(weights)
# print((ds_ann_y.mean(('latitude','longitude'))).to_pandas().T10m.loc[['1954-12-31', '1985-12-31','2021-12-31']])
# print((ds_ann_y.std(('latitude','longitude'))).to_pandas().T10m.loc[['1954-12-31', '1985-12-31','2021-12-31']])

# calculating breakpoint
# x = ds_ann_GrIS.loc['1954':].index.values.astype(float)
# y = ds_ann_GrIS.loc['1954':].values
# RMSD = x*np.nan
# for i, x0 in enumerate(x):
#     def piecewise_linear(x, y0, k1, k2):
#         return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
#     p , e = optimize.curve_fit(piecewise_linear, x, y)
#     y_pred = piecewise_linear(x, *p)

#     RMSD[i] = np.sqrt(np.mean((y_pred - y)**2))


# x0 = x[RMSD == RMSD.min()]
# ds_ann_GrIS.loc['1954':].index[RMSD == RMSD.min()]

# def piecewise_linear(x, y0, k1, k2):
#     return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
# p , e = optimize.curve_fit(piecewise_linear, x, y)
# y_pred = piecewise_linear(x, *p)
# plt.figure()
# plt.plot(x, y, "o")
# plt.plot(x, RMSD*10-20)
# plt.plot(x, y_pred)
# Answer: 1985-04-01

year_ranges = np.array([[1954, 1985], [1985, 2022], [1954, 2022]])

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
# %% plotting
import statsmodels.api as sm
import rioxarray  # for the extension to load
import scipy.odr
import scipy.stats
from scipy import optimize

ds_T10m = ds_ann_3413
T10m_GrIS = ds_ann_GrIS
model = "ANN"
land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
land = land.to_crs(ds_T10m.rio.crs)

fig, ax = plt.subplots(2, 3, figsize=(9, 12))
plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.8, left=0.01, bottom=0.01, top=0.8)
ax = ax.flatten()
ax_bot = fig.add_axes([0.12, 0.84, 0.87, 0.11])
ax_bot.set_title(ABC[0] + ". Ice-sheet-wide average", fontweight="bold", loc="left")
# ax_bot.plot(T10m_GrIS.index, T10m_GrIS,color='lightgray')
ax_bot.step(
    T10m_GrIS.resample("Y").mean().index,
    T10m_GrIS.resample("Y").mean(),
    color=CB_color_cycle[0],
    label="ANN",
)
ds_T10m_dy = ds_T10m.copy()
ds_T10m_dy["time"] = [toYearFraction(d) for d in pd.to_datetime(ds_T10m_dy.time.values)]
ds_T10m_dy = ds_T10m_dy.T10m.transpose("time", "y", "x")
for i in range(3):
    # tmp = T10m_GrIS.resample('Y').mean().loc[str(year_ranges[i,0]):str (year_ranges[i,1])]

    tmp = ds_T10m_dy.sel(time=slice(year_ranges[i][0], year_ranges[i][1]))
    _, _, slope, _, pval, _ = linregress_3D(tmp.time, tmp)

    land.plot(ax=ax[i], zorder=0, color="black")
    vmin = -1.5
    vmax = 1.5

    im = (slope * 10).plot(
        ax=ax[i], vmin=vmin, vmax=vmax, cmap="coolwarm", add_colorbar=False
    )
    significant = slope.where(pval > 0.1)
    X, Y = np.meshgrid(slope.x, slope.y)
    ax[i].hexbin(
        X.reshape(-1),
        Y.reshape(-1),
        significant.data[:, :].reshape(-1),
        gridsize=(50, 50),
        hatch="..",
        alpha=0,
    )

    year_start = max(year_ranges[i, 0], int(tmp.time.values.min()))
    year_end = min(year_ranges[i, 1] - 1, int(tmp.time.values.max()))
    if i in [1, 3]:
        if i == 1:
            cbar_ax = fig.add_axes([0.85, 0.07, 0.03, 0.6])
        cb = plt.colorbar(im, ax=ax[i], cax=cbar_ax)
        cb.ax.get_yaxis().labelpad = 18
        cb.set_label(
            "Trend in 10 m subsurface temperature ($^o$C decade $^{-1}$)",
            fontsize=12,
            rotation=270,
        )
    ax[i].set_axis_off()
    ax[i].set_title(
        ABC[i + 1] + ". ANN, " + str(year_start) + " to " + str(year_end),
        fontweight="bold",
    )

    ax[i].set_xlim(land.bounds.minx.min(), land.bounds.maxx.max())
    ax[i].set_ylim(land.bounds.miny.min(), land.bounds.maxy.max())

# calculating trend on entire period
X = np.array([toYearFraction(d) for d in T10m_GrIS.loc[T10m_GrIS.notnull()].index])
y = T10m_GrIS.loc[T10m_GrIS.notnull()].values

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(
    "%s, %i-%i, %0.3f, %0.3f"
    % (model, X[0], X[-1], est2.params[1] * 10, est2.pvalues[1])
)

# calculating piecewise function, trends and pvalues of the trends
def piecewise_linear(x, y0, k1, k2):
    x0 = 1985.33
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


p, e = optimize.curve_fit(piecewise_linear, X, y)


def f_wrapper_for_odr(beta, x):  # parameter order for odr
    return piecewise_linear(x, *beta)


func = scipy.odr.odrpack.Model(f_wrapper_for_odr)
data = scipy.odr.odrpack.Data(X, y)
myodr = scipy.odr.odrpack.ODR(data, func, beta0=p, maxit=0)
myodr.set_job(fit_type=2)
ptatistics = myodr.run()
df_e = len(X) - len(p)  # degrees of freedom, error
cov_beta = ptatistics.cov_beta  # parameter covariance matrix from ODR
sd_beta = ptatistics.sd_beta * ptatistics.sd_beta
ci = []
t_df = scipy.stats.t.ppf(0.975, df_e)
ci = []
for i in range(len(p)):
    ci.append(
        [p[i] - t_df * ptatistics.sd_beta[i], p[i] + t_df * ptatistics.sd_beta[i]]
    )

tstat_beta = p / ptatistics.sd_beta  # coeff t-statistics
pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0  # coef. p-values
print("%s, %i-%i, %0.3f, %0.3f" % (model, X[0], 1985, p[1] * 10, pstat_beta[1]))
print("%s, %i-%i, %0.3f, %0.3f" % (model, 1985, X[-1], p[2] * 10, pstat_beta[2]))

y_pred = piecewise_linear(X, *p)
if model == "ANN":
    ax_bot.plot(
        T10m_GrIS.loc[T10m_GrIS.notnull()].index,
        y_pred,
        color=CB_color_cycle[0],
        linestyle="--",
    )
ax_bot.autoscale(enable=True, axis="x", tight=True)
ax_bot.set_ylabel("10 m subsurface \ntemperature ($^o$C)", fontsize=12)

# trend over common period
X = np.array([toYearFraction(d) for d in T10m_GrIS.loc["1980":"2016"].index])
y = T10m_GrIS.loc["1980":"2016"].values
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(
    "%s, %i-%i, %0.3f, %0.3f"
    % (model, X[0], X[-1], est2.params[1] * 10, est2.pvalues[1])
)

ds_T10m_l = (ds_racmo, ds_hh, ds_mar)
T10m_GrIS_l = (ds_racmo_GrIS, ds_hh_GrIS, ds_mar_GrIS)
model_l = ("RACMO", "HIRHAM", "MAR")

for k in range(len(ds_T10m_l)):
    ds_T10m = ds_T10m_l[k]
    T10m_GrIS = T10m_GrIS_l[k]
    model = model_l[k]

    land = land.to_crs(ds_T10m.rio.crs)
    ax_bot.step(
        T10m_GrIS.resample("Y").mean().index,
        T10m_GrIS.resample("Y").mean(),
        label=model,
        color=CB_color_cycle[1 + k],
        alpha=0.8,
    )
    ds_T10m_dy = ds_T10m.copy()
    ds_T10m_dy["time"] = [
        toYearFraction(d) for d in pd.to_datetime(ds_T10m_dy.time.values)
    ]
    ds_T10m_dy = ds_T10m_dy.T10m.transpose("time", "y", "x")

    tmp = ds_T10m_dy.sel(time=slice(1980, 2016))
    _, _, slope, _, pval, _ = linregress_3D(tmp.time, tmp)

    land.plot(ax=ax[k + 3], zorder=0, color="black")
    im = (slope * 10).plot(
        ax=ax[k + 3], vmin=vmin, vmax=vmax, cmap="coolwarm", add_colorbar=False
    )
    X, Y = np.meshgrid(slope.x, slope.y)
    ax[k + 3].hexbin(
        X.reshape(-1),
        Y.reshape(-1),
        slope.where(pval > 0.1).data[:, :].reshape(-1),
        gridsize=(50, 50),
        hatch="..",
        alpha=0,
    )

    ax[k + 3].set_axis_off()
    ax[k + 3].set_xlim(land.bounds.minx.min(), land.bounds.maxx.max())
    ax[k + 3].set_ylim(land.bounds.miny.min(), land.bounds.maxy.max())
    ax[k + 3].set_title(
        ABC[k + 4] + ". " + model + ", 1980 to 2016", fontsize=12, fontweight="bold"
    )
    # calculating trend on entire period
    X = np.array([toYearFraction(d) for d in T10m_GrIS.loc[T10m_GrIS.notnull()].index])
    y = T10m_GrIS.loc[T10m_GrIS.notnull()].values

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(
        "%s, %i-%i, %0.3f, %0.3f"
        % (model, X[0], X[-1], est2.params[1] * 10, est2.pvalues[1])
    )

    # trend over common period
    X = np.array([toYearFraction(d) for d in T10m_GrIS.loc["1980":"2016"].index])
    y = T10m_GrIS.loc["1980":"2016"].values
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(
        "%s, %i-%i, %0.3f, %0.3f"
        % (model, X[0], X[-1], est2.params[1] * 10, est2.pvalues[1])
    )
ax_bot.legend(ncol=4, loc="upper right", fontsize=12, bbox_to_anchor=(1, 1.35))
fig.savefig("figures/all_RCMs_trend_map.png", dpi=300)

# %% Firn area averages analysis


def selected_area(ds, shape, mask=0):
    ds_select = ds.rio.clip(
        shape.to_crs(ds.rio.crs).geometry.values, ds.rio.crs, all_touched=True
    )

    if mask:
        ds_select = ds.where(ds_select.T10m.isel(time=100).isnull())

    try:
        tmp = (ds_select.mean(dim=("x", "y"))).to_pandas().T10m
    except:
        weights = np.cos(np.deg2rad(ds_select.latitude))
        weights.name = "weights"
        tmp_w = ds_select.weighted(weights)
        tmp = (tmp_w.mean(dim=("latitude", "longitude"))).to_pandas().T10m
    return tmp.resample("Y").mean()


ds_ann_DSA = selected_area(ds_ann, DSA)
ds_ann_LAPA = selected_area(ds_ann, LAPA)
ds_ann_HAPA = selected_area(ds_ann, HAPA)
ds_ann_BIA = selected_area(ds_ann, firn, mask=1)

ds_racmo_DSA = selected_area(ds_racmo, DSA) - 273.15
ds_racmo_LAPA = selected_area(ds_racmo, LAPA) - 273.15
ds_racmo_HAPA = selected_area(ds_racmo, HAPA) - 273.15
ds_racmo_BIA = selected_area(ds_racmo, firn, mask=1) - 273.15

ds_mar_DSA = selected_area(ds_mar, DSA)
ds_mar_LAPA = selected_area(ds_mar, LAPA)
ds_mar_HAPA = selected_area(ds_mar, HAPA)
ds_mar_BIA = selected_area(ds_mar, firn, mask=1)

ds_hh_DSA = selected_area(ds_hh, DSA) - 273.15
ds_hh_LAPA = selected_area(ds_hh, LAPA) - 273.15
ds_hh_HAPA = selected_area(ds_hh, HAPA) - 273.15
ds_hh_BIA = selected_area(ds_hh, firn, mask=1) - 273.15


# # %% 
# plt.close('all')
# fig, ax = plt.subplots(3,4)
# ax = ax.flatten()
# for i in range(1,13):
#     ds_mar_1990 = ds_mar.sel(time='1990-'+str(i).zfill(2)).T10m.rio.reproject("EPSG:3413")
#     ds_hh_1990 = (ds_hh.sel(time='1990-'+str(i).zfill(2)).T10m.rio.reproject("EPSG:3413").interp_like(ds_mar_1990)-273.15)
    
#     (ds_mar_1990-ds_hh_1990).plot(ax=ax[i-1], cbar_kwargs=dict(label='T10m MAR - T10m HIRHAM'))
#     ax[i-1].set_title('1990-'+str(i).zfill(2))
#     ax[i-1].axes.get_xaxis().set_visible(False)
#     ax[i-1].axes.get_yaxis().set_visible(False)
#%% Plotting for different firn areas
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def plot_selected_ds(tmp, ax, label, mask=0):
    if label == "ANN":
        col = CB_color_cycle[0]
    elif label == "RACMO":
        col = CB_color_cycle[1]
    elif label == "HIRHAM":
        col = CB_color_cycle[2]
    elif label == "MAR":
        col = CB_color_cycle[3]
    else:
        col = CB_color_cycle[4]

    # tmp.plot(ax=ax, color='black', label='_no_legend_',alpha=0.3)
    tmp = tmp.resample("Y").mean()
    tmp.plot(
        ax=ax, label=label, drawstyle="steps-post", linewidth=3, color=col, alpha=0.8
    )

    # print(tmp.loc['1980-01-01':'1990-01-01'].mean())
    # print(tmp.loc['2010-01-01':'2020-01-01'].mean())
    # print(tmp.loc['1998-01-01':'2009-01-01'].mean())
    # print(tmp.loc['2010-01-01':'2018-01-01'].mean())

    tmp = tmp.loc["1985":]
    X = np.array([toYearFraction(d) for d in tmp.index])
    y = tmp.values

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()

    print(
        "%s, %i-%i, %0.3f, %0.3f"
        % (label, X[0], X[-1], est2.params[1] * 10, est2.pvalues[1])
    )


fig, ax = plt.subplots(4, 1, figsize=(8, 10))
fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.07, hspace=0.25)
ax = ax.flatten()
print("bare ice")
plot_selected_ds(ds_ann_BIA, ax[0], "ANN")
plot_selected_ds(ds_racmo_BIA, ax[0], "RACMO")
plot_selected_ds(ds_hh_BIA, ax[0], "HIRHAM")
plot_selected_ds(ds_mar_BIA, ax[0], "MAR")
# plot_selected_ds(ds_era, firn,ax[0], 'ERA5 $T_{2m}$')
ax[0].set_title("A. Bare ice area", loc="left")
ax[0].legend(ncol=4, bbox_to_anchor=(1, 1.05), loc="lower right", fontsize=11)
print("DSA")
plot_selected_ds(ds_ann_DSA, ax[1], "ANN")
plot_selected_ds(ds_racmo_DSA, ax[1], "RACMO")
plot_selected_ds(ds_hh_DSA, ax[1], "HIRHAM")
plot_selected_ds(ds_mar_DSA, ax[1], "MAR")
# plot_selected_ds(ds_era_DSA, ax[1], 'ERA5 $T_{2m}$')
ax[1].set_title("B. Dry snow area", loc="left")
print("LAPA")
plot_selected_ds(ds_ann_LAPA, ax[2], "ANN")
plot_selected_ds(ds_racmo_LAPA, ax[2], "RACMO")
plot_selected_ds(ds_hh_LAPA, ax[2], "HIRHAM")
plot_selected_ds(ds_mar_LAPA, ax[2], "MAR")
# plot_selected_ds(ds_era_LAPA, ax[2], 'ERA5 $T_{2m}$')
ax[2].set_title("C. Low accumulation percolation area", loc="left")
print("HAPA")
plot_selected_ds(ds_ann_HAPA, ax[3], "ANN")
plot_selected_ds(ds_racmo_HAPA, ax[3], "RACMO")
plot_selected_ds(ds_hh_HAPA, ax[3], "HIRHAM")
plot_selected_ds(ds_mar_HAPA, ax[3], "MAR")
# plot_selected_ds(ds_era, HAPA, ax[3], 'ERA5 $T_{2m}$')
ax[3].set_title("D. High accumulation percolation area", loc="left")

ax[0].set_ylim(-16, -16 + 10)
ax[1].set_ylim(-29, -29 + 10)
ax[2].set_ylim(-18, -18 + 10)
ax[3].set_ylim(-15, -15 + 10)
for i in range(4):
    ax[i].set_xlim(pd.to_datetime("1954"), pd.to_datetime("2021"))
    ax[i].tick_params(axis="y", labelsize=10.5)
    ax[i].grid()

    if i < 3:
        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].set_xlabel("")
    else:
        ax[i].set_xlabel("Year", fontsize=11)
fig.text(
    0.01,
    0.5,
    "Average 10 m subsurface temperature ($^o$C)",
    fontsize=12,
    va="center",
    rotation="vertical",
)
fig.savefig("figures/model_comp_ice_sheet_areas.png", dpi=300)


# %% comparison with ERA5 temperature: trend difference
ds_era_m = xr.open_dataset("Data/ERA5/ERA5_monthly_temp_snowfall.nc")
ds_era = ds_era_m.resample(time="Y").mean()
ds_era["sf"] = ds_era_m.sf.resample(time="Y").sum()
ice = ice.to_crs("EPSG:4326")
ds_era = (
    ds_era.rio.write_crs("EPSG:4326")
    .rio.clip(ice.geometry.values, ice.crs)
    .rio.reproject("EPSG:3413")
)
ds_era = ds_era.sel(time=slice("1985-01-01", "2021-12-31"))
ds_era = ds_era.interp(x=ds_ann_3413["x"], y=ds_ann_3413["y"])

ds_era_dy = ds_era.copy()
ds_era_dy["time"] = [toYearFraction(d) for d in pd.to_datetime(ds_era_dy.time.values)]
_, _, slope_era, _, pval_era, _ = linregress_3D(ds_era_dy.time, ds_era_dy.t2m)
_, _, slope_sf, _, pval_sf, _ = linregress_3D(ds_era_dy.time, ds_era_dy.sf)

ds_ann_dy = (
    ds_ann_3413.resample(time="Y")
    .mean()
    .sel(time=slice("1985-01-01", "2021-12-31"))
    .copy()
)
ds_ann_dy["time"] = [toYearFraction(d) for d in pd.to_datetime(ds_ann_dy.time.values)]

_, _, slope_ann, _, pval_ann, _ = linregress_3D(ds_ann_dy.time, ds_ann_dy.T10m)

labels = [
    "A. 1985-2021 trend in 2m \nair temperature",
    "B. 1985-2021 trend in snowfall",
    "C. 1985-2021 trend in 10 m \nsubsurface temperature",
    "D. 1985-2021 trend difference\n bewteen T10m and Ta2m",
]
units = [
    "$^o$C decade$^{-1}$",
    "mm decade$^{-1}$",
    "$^o$C decade$^{-1}$",
    "$^o$C decade$^{-1}$",
]
# %%
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 12))
ax = ax.flatten()
vmin = -1.5
vmax = 1.5
(slope_era * 10).plot(
    ax=ax[0], vmin=vmin, vmax=vmax, cmap="coolwarm", cbar_kwargs={"label": units[0]}
)
X, Y = np.meshgrid(slope_era.x, slope_era.y)
ax[0].hexbin(
    X.reshape(-1),
    Y.reshape(-1),
    slope_era.where(pval_era > 0.05).data[:, :].reshape(-1),
    gridsize=(50, 50),
    hatch="..",
    alpha=0,
)

(slope_ann * 10).plot(
    ax=ax[2], vmin=vmin, vmax=vmax, cmap="coolwarm", cbar_kwargs={"label": units[1]}
)
X, Y = np.meshgrid(slope_ann.x, slope_ann.y)
ax[2].hexbin(
    X.reshape(-1),
    Y.reshape(-1),
    slope_ann.where(pval_ann > 0.05).data[:, :].reshape(-1),
    gridsize=(50, 50),
    hatch="..",
    alpha=0,
)

(slope_sf * 1000 * 10).plot(
    ax=ax[1], vmin=-10, vmax=10, cmap="coolwarm", cbar_kwargs={"label": units[2]}
)
X, Y = np.meshgrid(slope_sf.x, slope_sf.y)
ax[1].hexbin(
    X.reshape(-1),
    Y.reshape(-1),
    slope_sf.where(pval_sf > 0.05).data[:, :].reshape(-1),
    gridsize=(50, 50),
    hatch="..",
    alpha=0,
)

(slope_ann * 10 - slope_era * 10).plot(
    ax=ax[3], vmin=vmin, vmax=vmax, cmap="seismic", cbar_kwargs={"label": units[3]}
)

for i in range(4):
    ax[i].set_title(labels[i])
    ax[i].axes.xaxis.set_ticklabels([])
    ax[i].axes.yaxis.set_ticklabels([])
    land.plot(ax=ax[i], zorder=0, color="black")
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")

# %% T2m vs T10m difference
x = ds_era_dy.t2m.values.flatten()
y = ds_ann_dy.T10m.values.flatten()

ind = ~np.isnan(x + y)
x = x[ind]
y = y[ind]
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

nbins = 300
k = gaussian_kde([x, y])
xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto")
plt.show()

# Change color palette
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap=plt.cm.Greens_r)
plt.colorbar()
plt.show()
