# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
# import GIS_lib as gis
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
import geopandas as gpd

df = pd.read_csv("10m_temperature_dataset_monthly.csv")

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

# land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
# land = land.to_crs("EPSG:3413")

# ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
# ice = ice.to_crs("EPSG:3413")

dates = pd.DatetimeIndex(df.date)

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

#  find unique locations of measurements
df_all = df[["site", "latitude", "longitude"]]
df_all.latitude = round(df_all.latitude, 5)
df_all.longitude = round(df_all.longitude, 5)
latlon = np.array(
    [str(lat) + str(lon) for lat, lon in zip(df_all.latitude, df_all.longitude)]
)
uni, ind = np.unique(latlon, return_index=True)
print(latlon[:3])
print(latlon[np.sort(ind)][:3])
df_all = df_all.iloc[np.sort(ind)]
site_uni, ind = np.unique([str(s) for s in df_all.site], return_index=True)
df_all.iloc[np.sort(ind)].to_csv("coords.csv", index=False)

# %% Extracting closest cell in MAR
print("Extracting closest cell in MAR")


# % loading MAR
for yr in range(1980, 2021):
    print(yr)
    # if yr == 1958:
    #     ds_mar = xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.9/MARv3.9-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK', 'SF', 'SH', 'TT', 'ME', 'RU','TI_10M_AVE']]
    # else:
    #     ds_mar = xr.concat((ds_mar,
    #                          xr.open_dataset('C:/Data_save/RCM/MAR/MARv3.9/MARv3.9-'+str(yr)+'.nc')[['LON', 'LAT', 'MSK', 'SF', 'SH', 'TT',  'ME', 'RU','TI_10M_AVE']]),dim='TIME')

    if yr == 1980:
        ds_mar = xr.open_dataset(
            "C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-"
            + str(yr)
            + ".nc"
        )[["LON", "LAT", "MSK", "ME", "RU", "TI1"]]
    else:
        ds_mar = xr.concat(
            (
                ds_mar,
                xr.open_dataset(
                    "C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-"
                    + str(yr)
                    + ".nc"
                )[["LON", "LAT", "MSK", "ME", "RU", "TI1"]],
            ),
            dim="TIME",
        )


def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.TIME.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("TIME.year") / month_length.groupby("TIME.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("TIME.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(TIME="AS").sum(dim="TIME")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(TIME="AS").sum(dim="TIME")

    # Return the weighted average
    return obs_sum / ones_out


ds_mar["T10m"] = ds_mar.TI1.sel(OUTLAY=10)
T10m_mar_yr = xr.where(
    ds_mar.isel(TIME=0)["MSK"] > 75, weighted_temporal_mean(ds_mar, "T10m"), np.nan
)
# T10m_mar_yr = xr.where(ds_mar.isel(TIME=0)['MSK']>75, weighted_temporal_mean(ds_mar, 'TI1'), np.nan)
melt_mar_yr = xr.where(
    ds_mar.isel(TIME=0)["MSK"] > 75, ds_mar["ME"].resample(TIME="AS").sum(), np.nan
)
melt_mar_yr = xr.where(
    ds_mar.isel(TIME=0)["MSK"] > 75, ds_mar["ME"].resample(TIME="AS").sum(), np.nan
)

T10m_mar_yr.isel(TIME=1).plot()
melt_mar_yr.isel(TIME=1).plot()

# %% Extracting closest cell in MAR
from scipy.spatial.distance import cdist
import progressbar

print("Extracting closest cell in MAR")
ds = xr.open_dataset(
    "C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-"
    + str(1980)
    + ".nc"
)

points = [
    (x, y) for x, y in zip(ds["LAT"].values.flatten(), ds["LON"].values.flatten())
]

df["MAR_distance"] = np.nan
df["MAR_lat"] = np.nan
df["MAR_lon"] = np.nan
df["MAR_i"] = np.nan
df["MAR_j"] = np.nan
df["MAR_elevation"] = np.nan
lat_prev = -999
lon_prev = -999
for i, (lat, lon) in progressbar.progressbar(enumerate(zip(df.latitude, df.longitude))):
    if lat == lat_prev and lon == lon_prev:
        df.iloc[i, df.columns.get_loc("MAR_lat")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_lat")
        ]
        df.iloc[i, df.columns.get_loc("MAR_lon")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_lon")
        ]
        df.iloc[i, df.columns.get_loc("MAR_distance")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_distance")
        ]
        df.iloc[i, df.columns.get_loc("MAR_i")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_i")
        ]
        df.iloc[i, df.columns.get_loc("MAR_j")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_j")
        ]
        df.iloc[i, df.columns.get_loc("MAR_elevation")] = df.iloc[
            i - 1, df.columns.get_loc("MAR_elevation")
        ]
    else:
        closest = points[cdist([(lat, lon)], points).argmin()]
        df.iloc[i, df.columns.get_loc("MAR_lat")] = closest[0]
        df.iloc[i, df.columns.get_loc("MAR_lon")] = closest[1]
        df.iloc[i, df.columns.get_loc("MAR_distance")] = cdist([(lon, lat)], [closest])[
            0
        ][0]

        tmp = np.array(ds.SH)
        ind = np.argwhere(
            np.array(np.logical_and(ds["LAT"] == closest[0], ds["LON"] == closest[1]))
        )[0]
        df.iloc[i, df.columns.get_loc("MAR_i")] = ind[0]
        df.iloc[i, df.columns.get_loc("MAR_j")] = ind[1]

        df.iloc[i, df.columns.get_loc("MAR_elevation")] = np.array(ds.SH)[
            ind[0], ind[1]
        ]
        lat_prev = lat
        lon_prev = lon

    # plt.figure()
    # plt.scatter(ds['LON'].values.flatten(), ds['LAT'].values.flatten())
    # plt.plot(lon,lat, marker='o',markersize=10,color='green')
    # plt.plot(np.array(ds.LON)[ind[0], ind[1]],np.array(ds.LAT)[ind[0], ind[1]], marker='o',markersize=10,color='red')
    # plt.plot(closest[1],closest[0], marker='o',markersize=5)
    # break

    # date_in_days_since = (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).days + (pd.to_datetime(date) - pd.to_datetime('01-SEP-1947 00:00:00')).seconds/60/60/24

# %% Extracting temperatures from MAR
print("Extracting melt from MAR")
df["MAR_ME"] = np.nan
ds_ME = xr.open_dataset(
    "C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-ME-1980-2021.nc"
)

for ind in progressbar.progressbar(df.index):
    if df.loc[ind, "date"].year < 1985:
        continue

    tmp = ds_ME[dict(X10_85=int(df.loc[ind].MAR_j), Y20_155=int(df.loc[ind].MAR_i))]
    time_start = tmp.sel(
        time=pd.to_datetime(df.loc[ind, "date"]) + pd.DateOffset(years=-5),
        method="ffill",
    ).time.values
    time_end = tmp.sel(
        time=pd.to_datetime(df.iloc[i, :].date) + pd.DateOffset(days=1), method="ffill"
    ).time.values

    df.loc[ind, "MAR_ME"] = tmp.sel(time=slice(time_start, time_end)).ME.mean().values

# %%
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

params = np.array([1, 1])


def funcinv(x, a, b):
    return np.exp(a * x + b)


def residuals(params, x, data):
    # evaluates function given vector of params [a, b]
    # and return residuals: (observed_data - model_data)
    a, b = params
    func_eval = funcinv(x, a, b)
    err = data - func_eval
    # err[x<-15] = 0.5*(data[x<-15] - func_eval[x<-15])
    return err


from scipy.stats import kde

y = melt_mar_yr.values.ravel()
x = T10m_mar_yr.values.ravel()
x_sort = np.sort(x)
ind = np.argsort(x)
y_sort = y[ind]
ind_no_nan = ~np.isnan(x_sort + y_sort)
x = x_sort[ind_no_nan]
y = y_sort[ind_no_nan]

res = least_squares(residuals, params, args=(x, y))
x_obs = df.temperatureObserved[df.temperatureObserved.notnull() & df.MAR_ME.notnull()]
y_obs = df.MAR_ME[df.temperatureObserved.notnull() & df.MAR_ME.notnull()]

res_obs = least_squares(residuals, params, args=(x_obs, y_obs))

nbins = 50
ind_sub = np.random.choice(np.arange(0, len(x)), size=175316)
k = kde.gaussian_kde([x[ind_sub], y[ind_sub]])
xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, 0 : 8000 : nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# %%
plt.figure()
plt.pcolormesh(
    xi, yi, zi.reshape(xi.shape), shading="auto", cmap="magma_r", vmin=0, vmax=0.000005
)
# plt.plot(x,y,'o',markersize=1, linestyle='None')
xs = np.linspace(-35, -0.001, 1000)


plt.plot(
    df.temperatureObserved,
    df.MAR_ME,
    c="tab:red",
    marker="+",
    linestyle="None",
    label="observed $T_{10m}$ vs. MAR melt",
)

plt.plot(
    xs,
    funcinv(xs, res_obs.x[0], res_obs.x[1]),
    "lightcoral",
    lw=2,
    label="exponential fit, melt in MAR vs. observed $T_{10m}$",
)
plt.plot(
    xs,
    funcinv(xs, res.x[0], res.x[1]),
    "lightgray",
    lw=2,
    label="exponential fit, melt in MAR vs. $T_{10m}$ in MAR",
)
plt.xlim(-35, 0)
plt.ylim(0, 8000)
cbar = plt.colorbar(label="Density of MAR $T_{10m}$ vs. MAR melt points (-)")
cbar.ax.set_yticks([0, 0.00001])
cbar.ax.set_yticklabels(["low", "", "", "", "", "high"])
plt.xlabel("$T_{10m} (^oC)$")
plt.ylabel("Average annual melt (mm w.e.)")
plt.legend(loc="upper left")
