# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

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
import rioxarray

# import GIS_lib as gis
matplotlib.rcParams.update({"font.size": 16})
df = pd.read_csv("subsurface_temperature_summary.csv")

# ============ To fix ================

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

df["year"] = pd.DatetimeIndex(df.date).year

gdf = (
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    .set_crs(4326)
    .to_crs(3413)
)
gdf =gdf.set_index('Unnamed: 0')

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

print("Extracting data from rasters")

df["avg_snowfall_mm"] = gis.sample_raster_with_geopandas(
    gdf.to_crs("epsg:3413"),
    "Data/misc/Net_Snowfall_avg_1979-2014_MAR_3413.tif",
    "c_avg",
)
df.date = pd.to_datetime(df.date)
df = df.set_index("date", drop=False)
df_save = df

# %% Map
fig, ax = plt.subplots(1, 1, figsize=(6, 9))
fig.subplots_adjust(hspace=0.0, wspace=0.0, top=1, bottom=0, left=0, right=1)
land.plot(ax=ax, zorder=0, color="black")
ice.plot(ax=ax, zorder=1, color="lightblue")
gdf.plot(
    ax=ax,
    column="year",
    cmap="tab20c",
    markersize=30,
    edgecolor="gray",
    legend=True,
    legend_kwds={
        "label": "Year of measurement",
        "orientation": "horizontal",
        "shrink": 0.8,
    },
)

plt.axis("off")
plt.savefig("figures/fig1_map.png", dpi=300)
#%% histograms
fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
unit = [' (deg N)', ' (deg E)', ' (m a.s.l.)', '']
for i, col in  enumerate(['latitude', 'longitude', 'elevation', 'year']):
    df.hist(column = col, bins = 15, ax=ax[i], figsize=(20, 18))
    ax[i].set_xlabel(col+unit[i])
    ax[i].set_title('')


# %% Model lat_elev_grid all in

df = df_save

# df = df.loc[(df.date>'1990') & (df.date<'2010'),:]
# df = df.loc[(df.date>'2010') & (df.date<'2014'),:]
# site_remove = ["SE-Dome", "H2", "H3", "FA_13", "FA_15_1", "FA_15_2"]
# df = df.loc[~np.isin(df.site, site_remove)]
step_elev = 500
elevation_bins = np.arange(0, 4000, step_elev)
step_lat = 4
latitude_bins = np.arange(60, 90, step_lat)
grid_temp = np.empty((len(latitude_bins), len(elevation_bins)))
grid_temp[:] = np.nan
for i in range(len(elevation_bins) - 1):
    for j in range(len(latitude_bins) - 1):
        conditions = np.array(
            (
                df.elevation >= elevation_bins[i],
                df.elevation < elevation_bins[i + 1],
                df.latitude >= latitude_bins[j],
                df.latitude < latitude_bins[j + 1],
            )
        )

        grid_temp[j, i] = df.loc[
            np.logical_and.reduce(conditions), "temperatureObserved"
        ].mean()

X = df[["elevation", "latitude"]]
Y = df["temperatureObserved"]
# meshgrid on the centerpoints
x_center, y_center = np.meshgrid(
    elevation_bins + step_elev / 2, latitude_bins + step_lat / 2
)


rbf3 = Rbf(
    (x_center[~np.isnan(grid_temp)] - np.min(x_center)) / np.max(x_center),
    (y_center[~np.isnan(grid_temp)] - np.min(y_center)) / np.max(y_center),
    grid_temp[~np.isnan(grid_temp)],
    function="cubic",
)

real_x = np.arange(0, 3400, 50)
real_y = np.arange(60, 83, 1)
xx, yy = np.meshgrid(real_x, real_y)  # ,np.arange(0,1600,10))
# zz = polyval2d(xx.astype(float), yy.astype(float), m)
zz = rbf3(
    (xx - np.min(x_center)) / np.max(x_center),
    (yy - np.min(y_center)) / np.max(y_center),
)

zz[zz > 0] = 0
zz[zz < -50] = -50
df["temperature_anomaly"] = df.temperatureObserved - rbf3(
    (df.elevation - np.min(x_center)) / np.max(x_center),
    (df.latitude - np.min(y_center)) / np.max(y_center),
)

# applying to Greenland map
gdf['temperature_anomaly'] = np.nan
gdf.loc[df['Unnamed: 0'].values, 'temperature_anomaly'] = df.temperature_anomaly.values

import rasterio as rio
import rioxarray
elevation = rioxarray.open_rasterio('Data/misc/Ice.tiff').squeeze()
latitude = rioxarray.open_rasterio('Data/misc/lat.tif').squeeze()
latitude = latitude.where(elevation>1)
elevation = elevation.where(elevation>1)
T10_mod = elevation.copy()
T10_mod.values = rbf3(
    (elevation - np.min(x_center)) / np.max(x_center),
    (latitude - np.min(y_center)) / np.max(y_center)
    )
T10_mod = T10_mod.where(T10_mod<0)
T10_mod = T10_mod.where(T10_mod>-60)
# % plot
fig = plt.figure(figsize=(10, 12))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.3)

# subsurface temperature regression latitude elevation
ax = fig.add_subplot(2, 2, 1)
dx = (real_x[1] - real_x[0]) / 2.0
dy = (real_y[1] - real_y[0]) / 2.0
extent = [real_x[0] - dx, real_x[-1] + dx, real_y[-1] + dy, real_y[0] - dy]
im = plt.imshow(zz, extent=extent, aspect="auto", cmap='coolwarm',vmin=-35,vmax=0)

# scatter residual
cmap = cm.seismic
bounds = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sct = ax.scatter(
    df["elevation"],
    df["latitude"],
    s=80,
    c=df["temperature_anomaly"],
    edgecolor="lightgray",
    cmap=cmap,
    norm=norm,
)

# colorbar subsurface temperature 
cax1 = fig.add_subplot(14, 4, 3)  
cb1 = plt.colorbar(im, cax=cax1,orientation = 'horizontal')
cb1.ax.set_xlabel("10 m subsurface \ntemperature ($^o$C)")
cb1.ax.xaxis.tick_top()
cb1.ax.xaxis.set_label_position('top') 

# colorbar residual
cax2 = fig.add_subplot(14, 4, 4)  # adding ax for second colorbar
cb2 = plt.colorbar(sct, cax=cax2, orientation = 'horizontal')
cb2.ax.set_xlabel("Residual ($^o$C)")
cb2.ax.xaxis.tick_top()
cb2.ax.xaxis.set_label_position('top') 

ax.set_xticks(elevation_bins)
ax.set_yticks(latitude_bins)
ax.grid()
ax.set_xlim(0, 3300)
ax.set_ylim(60, 82)
ax.set_xlabel("Elevation (m a.s.l.)")
ax.set_ylabel("Latitude ($^o$N)")
ax.set_title(
    "Elevation - Latitude model\n RMSE = %0.3f"
    % np.sqrt(np.mean(df["temperature_anomaly"] ** 2))
)
ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

# temperature map
ax2 = fig.add_subplot(1, 2, 2)
land.plot(ax=ax2, zorder=0, color="black")
# ice.plot(ax=ax, zorder=1, color="lightblue")
T10_mod.plot(cmap='coolwarm', ax = ax2, add_colorbar = False, vmin = -35, vmax = 0,
             )
ax2.set_xlim(-700000.0, 900000.0)
ax2.set_ylim(-3400000, -400000)
ax2.set_title('')
ax2.set_axis_off()

gdf.plot(
    ax=ax2,
    column="temperature_anomaly",
    cmap=cmap,
    norm=norm,
    markersize=30,
    edgecolor="gray",
    legend=False,
)
ax2.text(0.1, 0.9, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

ax3 = fig.add_subplot(2, 2, 3)
#  Recent trend
dates = df.index
y_data = df.temperature_anomaly.values[dates.year > 1995]
time_recent = dates[dates.year > 1995]
curve_fit = np.polyfit(pd.to_numeric(time_recent), y_data, 1)
y = curve_fit[1] + curve_fit[0] * pd.to_numeric(np.sort(time_recent))

ax3.plot(df["temperature_anomaly"], ".")
ax3.plot(np.sort(time_recent), y, "--", linewidth=3)

ax3.autoscale(enable=True, axis="x", tight=True)
ax3.set_xlabel("Year")
ax3.set_ylabel("10 m firn \ntemperature anomaly")
ax3.text(0.05, 0.95, 'C', transform=ax3.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

plt.savefig("figures/fig2_lat_elev_model.png")

T10_mod.mean()

# %% Model lat_elev_grid two decades

df = df_save

df_1 = df.loc[(df.date>'1990') & (df.date<'2010'),:].copy()
# df = df.loc[(df.date>'2010') & (df.date<'2014'),:]
# site_remove = ["SE-Dome", "H2", "H3", "FA_13", "FA_15_1", "FA_15_2"]
# df_1 = df_1.loc[~np.isin(df_1.site, site_remove)]

# defining latitude-elevation grid:
step_elev = 500
elevation_bins = np.arange(0, 4000, step_elev)
step_lat = 4
latitude_bins = np.arange(60, 90, step_lat)
x_center, y_center = np.meshgrid(
    elevation_bins + step_elev / 2, latitude_bins + step_lat / 2
)

# binning and averaging measurements within each grid cell
grid_temp = np.empty((len(latitude_bins), len(elevation_bins)))
grid_temp[:] = np.nan
for i in range(len(elevation_bins) - 1):
    for j in range(len(latitude_bins) - 1):
        conditions = np.array(
            (
                df_1.elevation >= elevation_bins[i],
                df_1.elevation < elevation_bins[i + 1],
                df_1.latitude >= latitude_bins[j],
                df_1.latitude < latitude_bins[j + 1],
            )
        )

        grid_temp[j, i] = df_1.loc[
            np.logical_and.reduce(conditions), "temperatureObserved"
        ].mean()

# # fitting a surface:
# fitted_surface = Rbf(
#     (x_center[~np.isnan(grid_temp)] - np.min(x_center)) / np.max(x_center),
#     (y_center[~np.isnan(grid_temp)] - np.min(y_center)) / np.max(y_center),
#     grid_temp[~np.isnan(grid_temp)],
#     function="cubic",
# )
fitted_surface = Rbf(x_center[~np.isnan(grid_temp)],
    y_center[~np.isnan(grid_temp)],
    grid_temp[~np.isnan(grid_temp)],
    function="cubic",
)

# function that either applies the surface or gives NaN if no data was in the bin
def rbf3(x,y, fitted_surface):
    out = fitted_surface(x,y)
    # if len(x.shape)>1:
    #     shape = x.shape
    #     x = x.flatten()
    #     y = y.flatten()
    # else:
    #     shape = []
    for i in range(len(elevation_bins) - 1):
        for j in range(len(latitude_bins) - 1):
            if np.isnan(grid_temp[j, i]):
                conditions = np.array(
                    (
                        x >= elevation_bins[i],
                        x < elevation_bins[i + 1],
                        y >= latitude_bins[j],
                        y < latitude_bins[j + 1],
                    )
                )
    
                out[np.logical_and.reduce(conditions)] = np.nan

    # point_list_out = [(x, y) for (x, y) in zip(x,y)]
    # for k, (x0, y0) in enumerate(point_list_out):
    #     ind_x_org_closest = (np.abs(elevation_bins - x0)).argmin()
    #     ind_y_org_closest = (np.abs(latitude_bins - y0)).argmin()
    #     if np.isnan(grid_temp[ind_y_org_closest, ind_x_org_closest]):
    #         if shape:
    #             i, j =  np.unravel_index(k, shape)
    #             out[i, j] = np.nan
    #         else:
    #             out[k] = np.nan
    out[out>0] = 0
    return out

# finer latitude-elevation for illustration
real_x = elevation_bins #np.arange(0, 3400, 50)
real_y = latitude_bins #np.arange(60, 83, 1)
# xx, yy = np.meshgrid(real_x, real_y)  # ,np.arange(0,1600,10))

dx = (real_x[1] - real_x[0]) / 2.0
dy = (real_y[1] - real_y[0]) / 2.0

xx, yy =  np.meshgrid(
    real_x + step_elev / 2, real_y + step_lat / 2
)

extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]

# output of the fitted surface in latitude-elevation space
# zz_1 = rbf3(
#     (xx - np.min(x_center)) / np.max(x_center),
#     (yy - np.min(y_center)) / np.max(y_center),
#     fitted_surface
# )
zz_1 = rbf3(xx, yy, fitted_surface)

zz_1[zz_1 > 0] = 0
zz_1[zz_1 < -50] = -50

# output of the fitted surface for each measurement
# df_1["temperature_anomaly"] = df_1.temperatureObserved - rbf3(
#     (df_1.elevation - np.min(x_center)) / np.max(x_center),
#     (df_1.latitude - np.min(y_center)) / np.max(y_center),
#     fitted_surface
# )
df_1["temperature_anomaly"] = df_1.temperatureObserved - rbf3(df_1.elevation, 
                                                              df_1.latitude, fitted_surface)
gdf['temperature_anomaly'] = np.nan
gdf.loc[df_1['Unnamed: 0'].values, 'temperature_anomaly'] = df_1.temperature_anomaly.values

# applying the fitted surface for the entire ice sheet
coarsening = 10
elevation = rioxarray.open_rasterio('Data/misc/Ice.tiff').squeeze().coarsen(x=coarsening, boundary='trim').mean().coarsen(y=coarsening, boundary='trim').mean()

latitude = rioxarray.open_rasterio('Data/misc/lat.tif').squeeze().coarsen(x=coarsening, boundary='trim').mean().coarsen(y=coarsening, boundary='trim').mean()

latitude = latitude.where(elevation>1)
elevation = elevation.where(elevation>1)

T10_mod_1 = elevation.copy()
# T10_mod_1.values = rbf3(
#     (elevation.values - np.min(x_center)) / np.max(x_center),
#     (latitude.values - np.min(y_center)) / np.max(y_center),
#     fitted_surface
#     )
T10_mod_1.values = rbf3(elevation.values,
                        latitude.values,
                        fitted_surface)
T10_mod_1 = T10_mod_1.where(T10_mod_1<0)
T10_mod_1 = T10_mod_1.where(T10_mod_1>-60)

df = df_save
# df = df.loc[(df.date>'1990') & (df.date<'2010'),:]
df_2 = df.loc[(df.date>='2010') & (df.date<'2022'),:].copy()
# site_remove = ["SE-Dome", "H2", "H3", "FA_13", "FA_15_1", "FA_15_2"]
# df_2 = df_2.loc[~np.isin(df_2.site, site_remove)]

grid_temp = np.empty((len(latitude_bins), len(elevation_bins)))
grid_temp[:] = np.nan
for i in range(len(elevation_bins) - 1):
    for j in range(len(latitude_bins) - 1):
        conditions = np.array(
            (
                df_2.elevation >= elevation_bins[i],
                df_2.elevation < elevation_bins[i + 1],
                df_2.latitude >= latitude_bins[j],
                df_2.latitude < latitude_bins[j + 1],
            )
        )

        grid_temp[j, i] = df_2.loc[
            np.logical_and.reduce(conditions), "temperatureObserved"
        ].mean()
# fitted_surface = Rbf(
#     (x_center[~np.isnan(grid_temp)] - np.min(x_center)) / np.max(x_center),
#     (y_center[~np.isnan(grid_temp)] - np.min(y_center)) / np.max(y_center),
#     grid_temp[~np.isnan(grid_temp)],
#     function="cubic",
# )

fitted_surface = Rbf(x_center[~np.isnan(grid_temp)],
    y_center[~np.isnan(grid_temp)],
    grid_temp[~np.isnan(grid_temp)],
    function="cubic",
)

# zz_2 = rbf3(
#     (xx - np.min(x_center)) / np.max(x_center),
#     (yy - np.min(y_center)) / np.max(y_center),
#     fitted_surface
# )
zz_2 = rbf3(xx, yy, fitted_surface)


zz_2[zz_2 > 0] = 0
zz_2[zz_2 < -50] = -50
df_2["temperature_anomaly"] = df_2.temperatureObserved - rbf3(
    (df_2.elevation - np.min(x_center)) / np.max(x_center),
    (df_2.latitude - np.min(y_center)) / np.max(y_center),
    fitted_surface
)

# applying to Greenland map
gdf['temperature_anomaly'] = np.nan
gdf.loc[df_2['Unnamed: 0'].values, 'temperature_anomaly'] = df_2.temperature_anomaly.values
T10_mod_2 = elevation.copy()
# T10_mod_2.values = rbf3(
#     (elevation.values - np.min(x_center)) / np.max(x_center),
#     (latitude.values - np.min(y_center)) / np.max(y_center),
#     fitted_surface
#     )
T10_mod_2.values = rbf3(elevation.values,
                        latitude.values,
                        fitted_surface)
T10_mod_2 = T10_mod_2.where(T10_mod_2<0)
T10_mod_2 = T10_mod_2.where(T10_mod_2>-60)

# % plotting lat_elev_grid

def plot_lat_elev_space(ax, zz, extent, df, vmin = -35, vmax = 0, cmap = 'coolwarm'):
    im = plt.imshow(zz, extent=extent, aspect="auto", cmap=cmap, vmin=vmin,vmax=vmax)

    if len(df)>0:
        sct = ax.scatter(
            df["elevation"],
            df["latitude"],
            s=80,
            c=df["temperatureObserved"],
            edgecolor="lightgray",
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax
        )
    ax.set_xticks(elevation_bins)
    ax.set_yticks(latitude_bins)
    ax.grid()
    # ax.set_xlim(0, 3300)
    # ax.set_ylim(60, 82)
    ax.set_xlabel("Elevation (m a.s.l.)")
    ax.set_ylabel("Latitude ($^o$N)")
    if len(df)>0:
        ax.set_title(
            "Elevation - Latitude model\n RMSE = %0.3f"
            % np.sqrt(np.mean(df["temperature_anomaly"] ** 2))
        )
    return im, ax


def plot_T10m_map(ax, T10_mod, gdf=gdf, cmap = 'coolwarm',vmin = -35, vmax = 0):
    land.plot(ax=ax, zorder=0, color="black")
    # ice.plot(ax=ax, zorder=1, color="lightblue")
    T10_mod.plot(cmap=cmap, ax = ax2, add_colorbar = False, vmin = vmin, vmax = vmax,
                 )
    ax2.set_xlim(-700000.0, 900000.0)
    ax2.set_ylim(-3400000, -400000)
    ax2.set_title('')
    ax2.set_axis_off()
    
    if len(gdf)>0:
        gdf.plot(
            ax=ax,
            column="temperatureObserved",
            cmap='coolwarm',
            vmin=-35,
            vmax=0,
            markersize=30,
            edgecolor="gray",
            legend=False,
        )
    return ax


fig = plt.figure(figsize=(10, 12))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.3)

ax = fig.add_subplot(2, 3, 1)
im, ax = plot_lat_elev_space(ax, zz_1, extent, df_1)

ax.text(0.05, 0.95, 'A. 1990-2009', transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

# colorbar subsurface temperature 
cax1 = fig.add_subplot(4, 6, 5)  
cb1 = plt.colorbar(im, cax=cax1,orientation = 'vertical')
cb1.ax.set_ylabel("10 m subsurface \ntemperature ($^o$C)")
# colorbar residual
# cax2 = fig.add_subplot(4, 6, 11)  # adding ax for second colorbar
# cb2 = plt.colorbar(sct, cax=cax2, orientation = 'vertical')
# cb2.ax.set_ylabel("Residual ($^o$C)")

# temperature map
ax2 = fig.add_subplot(2, 3, 4)
ax2 = plot_T10m_map(ax2,T10_mod_1)
ax2.text(0.1, 1, 'B. 1990-2009', transform=ax2.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

# subsurface temperature regression latitude elevation
ax = fig.add_subplot(2, 3, 2)
im, ax = plot_lat_elev_space(ax, zz_2, extent, df_2)
ax.text(0.05, 0.95, 'C. 2010-2020', transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

# temperature map
ax2 = fig.add_subplot(2, 3, 5)
ax2 = plot_T10m_map(ax2,T10_mod_2)
ax2.text(0.1, 1, 'D. 2010-2020', transform=ax2.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

ax = fig.add_subplot(2, 3, 3)
im, ax = plot_lat_elev_space(ax, zz_2-zz_1, extent, [], vmin=-5, vmax=5)
ax.text(0.05, 0.95, 'C. 2010-2020', transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')


ax2 = fig.add_subplot(2, 3, 6)
land.plot(ax=ax2, zorder=0, color="black")
# ice.plot(ax=ax, zorder=1, color="lightblue")
(T10_mod_1-T10_mod_2).plot(cmap='seismic', ax = ax2, add_colorbar = False, vmin = -5, vmax = 5)
   
ax2.set_xlim(-700000.0, 900000.0)
ax2.set_ylim(-3400000, -400000)
ax2.set_title('')
ax2.set_axis_off()
ax2.text(-0.2, 1, 'E. 2010-2020 minus 1990-2009', transform=ax2.transAxes, fontsize=14, fontweight='bold', verticalalignment='top')

plt.savefig("figures/fig2_lat_elev_model_2.png")

# %% residual dependence to longitude
fig = plt.figure(figsize=(10, 4))
ax2 = fig.add_subplot(1, 1, 1)
longitude_bins = np.arange(-70, -14, 4)
bin_anomaly = (
    df["temperature_anomaly"]
    .groupby(pd.cut(df["longitude"].values, longitude_bins))
    .mean()
)

ax2.scatter(df["longitude"], df["temperature_anomaly"], color="black")
ax2.step(
    longitude_bins,
    np.append(bin_anomaly.values, bin_anomaly.values[-1]),
    where="post",
    linewidth=5,
)
ax2.set_ylabel("Firn temperature \nanomaly ($^o$C)")
ax2.set_xlabel("Longitude ($^o$E)")
ax2.autoscale(enable=True, axis="x", tight=True)
# longitude_center = longitude_bins[:-1] + np.gradient(longitude_bins[:-1])/2
# m, b = np.polyfit(longitude_center[bin_anomaly.notnull()],
#                   bin_anomaly.values[bin_anomaly.notnull()], 1)
# ax2.plot(df['longitude'].values, m*df['longitude'].values + b)
plt.savefig("figures/figS1_residuals_vs_longitude.png")

# %% trend in each elevation bin
ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def lm(df):
    y_data = df.values
    x_data = pd.to_numeric(df.index.values)
    x_data_save = x_data
    ind = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
    y_data = y_data[ind]
    x_data = x_data[ind]
    res = stats.linregress(x_data, y_data)
    y = res.intercept + res.slope * x_data_save
    return y, res


fig,ax_all = plt.subplots(7,1,figsize=(12,12),sharex=True)
ax_all=ax_all.flatten()
plt.subplots_adjust(left=0.07, bottom=0.01, right=0.95, top=0.99, wspace=0.1, hspace=0.02)

for i in range(len(elevation_bins) - 1):
    ax = ax_all[i]
    tmp = df.loc[
        np.logical_and(
            df["elevation"] > elevation_bins[i],
            df["elevation"] <= elevation_bins[i + 1],
        ),
        "temperature_anomaly",
    ]
    tmp.plot(ax=ax, marker="o", linestyle="none")
    tmp_recent = tmp.loc["1995":]
    y_lin, res = lm(tmp_recent)
    ax.plot(tmp_recent.index, y_lin)
    ax.grid()
    ax.set_xlabel('')
    ax.text(0.01, 0.9, ABC[i]+'. '+str(elevation_bins[i]) + " - " + str(elevation_bins[i + 1])+' m a.s.l.', transform=ax.transAxes, fontsize=10, fontweight='bold', verticalalignment='top')

fig.text(0.02, 0.5, "Firn temperature anomaly ($^o$C)", ha='center', va='center', rotation='vertical',fontsize=12)
plt.savefig("figures/figS2_trends_elevation_bins.png")

#%% latitude bin and elevation exponential:
df = df_save
df = df.loc[df.temperatureObserved<1]
df = df.loc[df.temperatureObserved>-100]
# lats = np.array([61.5, 63.5, 65.5, 67.5, 69.5, 71.5, 75, 78])
lats = np.arange(60,80,2)

plt.figure()

cmap = plt.get_cmap("tab20")
norm = plt.Normalize(1930, 2021)

for i in range(len(lats)-1):
    ax = plt.subplot(3,3,i+1)
    msk = (df.latitude>=lats[i]) & (df.latitude<lats[i+1])
    sc = ax.scatter(df.loc[msk].elevation,
                    df.loc[msk].temperatureObserved, 
                    5, 
                    df.loc[msk].year,
                    cmap = cmap,
                    norm = norm)
    ax.set_title(str(lats[i])+' - '+str(lats[i+1]))
plt.colorbar(sc)