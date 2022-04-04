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
import geopandas as gpd
from datetime import datetime as dt
import time
from rasterio.crs import CRS
from  scipy import stats, signal
from scipy.interpolate import Rbf
import matplotlib as mpl
from matplotlib import cm
import rasterio as rio
import rioxarray
import matplotlib

# loading Greenland elevation and latitude maps
coarsening = 5
elevation = rioxarray.open_rasterio('Data/misc/Ice.tiff').squeeze().coarsen(x=coarsening, boundary='trim').mean().coarsen(y=coarsening, boundary='trim').mean()

latitude = rioxarray.open_rasterio('Data/misc/lat.tif').squeeze().coarsen(x=coarsening, boundary='trim').mean().coarsen(y=coarsening, boundary='trim').mean()
latitude = latitude.where(elevation>1)
elevation = elevation.where(elevation>1)

def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

    return date.year + fraction
def calculate_trend(ds_month, year_start,year_end, var, dim1, dim2):
    ds_month = ds_month.loc[dict(time=slice(str(year_start)+"-01-01", str(year_end)+"-12-31"))]
    
    vals = ds_month[var].values 
    time = np.array([toYearFraction(d) for d in pd.to_datetime(ds_month.time.values)])
    # Reshape to an array with as many rows as years and as many columns as there are pixels
    vals2 = vals.reshape(len(time), -1)
    # Do a first-degree polyfit
    regressions = np.nan * vals2[:2,:]
    ind_nan = np.all(np.isnan(vals2),axis=0)
    regressions[:,~ind_nan] = np.polyfit(time, vals2[:,~ind_nan], 1)
    # Get the coefficients back
    trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
    return ([dim1, dim2],  trends)

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

elev_contours = gpd.GeoDataFrame.from_file("Data/misc/Contours1000.shp").to_crs("EPSG:3413")

# loading temperature dataset
df = pd.read_csv("10m_temperature_dataset_monthly.csv")
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

msk1 = (df.site == 'QAS_A') & (df.temperatureObserved<-35)
msk2 = (df.site == 'CEN') & (df.temperatureObserved<-35)
df = df.loc[ ~msk1,:]
df = df.loc[ ~msk2,:]

df["year"] = pd.DatetimeIndex(df.date).year

df = (gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude)
                        ).set_crs(4326).to_crs(3413))
df['x_3413'] = df.geometry.x
df['y_3413'] = df.geometry.y


# % Loading RACMO T10m
# ds = xr.open_dataset("C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc")
# ds_month = ds.resample(time = 'M').mean()

# year_ranges = np.array([[1960, 1995],[1960, 2020], [1996, 2013], [2014,2020]])
# for i in range(4):
#     ds_month["trend_"+str(i)] = calculate_trend(ds_month, year_ranges[i][0],year_ranges[i][1],
#                                                 'T10m','rlat','rlon')

# T10m_GrIS = (ds.T10m.mean(dim=('rlat','rlon'))-273.15).to_pandas()

# # reprojecting
# crs_racmo = CRS.from_proj4('-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +o_lon_p=0  +lon_0=-37.5')
# ds_month = ds_month.rio.write_crs(crs_racmo)
# ds_month = ds_month.rio.reproject("EPSG:3413")

# Plotting RACMO firn temp vs ERA air temp
# plt.figure()
# T10m_GrIS.resample('Y').mean().plot(label='10 m firn temperature in RACMO')
# # t2m_land.t2m.plot()
# t2m_GrIS.resample('Y').mean().rolling(5).mean().t2m.plot(drawstyle='steps',label='2 m air temperature in ERA5')
# plt.ylabel('GrIS-wide average temperature ($^o$ C)')
# plt.legend()


# % loading ERA T2m
for yr in range(1950,2020,6):
    print(yr)
    if yr == 1950:
        ds_era = xr.open_dataset('Data/ERA5/m_era5_t2m_'+str(yr)+'_'+str(yr+5)+'.nc')
    else:
        ds_era = xr.concat((ds_era,
                             xr.open_dataset('Data/ERA5/m_era5_t2m_'+str(yr)+'_'+str(yr+5)+'.nc')),'time')
   
# reprojecting
ds_era = ds_era.rio.write_crs("EPSG:4326")
ds_era = ds_era.rio.reproject("EPSG:3413")

# adding lat and lon as vatriable
ny, nx = len(ds_era['y']), len(ds_era['x'])
x, y = np.meshgrid(ds_era['x'], ds_era['y'])

# Rasterio works with 1D arrays
from rasterio.warp import transform
lon, lat = transform( {'init': 'EPSG:3413'}, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())
lon = np.asarray(lon).reshape((ny, nx))
lat = np.asarray(lat).reshape((ny, nx))
ds_era.coords['lon'] = (('y', 'x'), lon)
ds_era.coords['lat'] = (('y', 'x'), lat)

# % Extract ERA time series

df_era = pd.DataFrame()
time_ERA = ds_era["time"].values
df = df.reset_index(drop=True)

points = [str(x)+str(y) for x, y in zip(df.x_3413, df.y_3413)]
_, ind_list = np.unique(points, return_index =True) 
ind_list = np.sort(ind_list)
          
for ind in progressbar.progressbar(ind_list):
    df_point = (
        ds_era.t2m.sel(x = df.loc[ind].x_3413,
                       y = df.loc[ind].y_3413,
                       method = 'nearest',
                       ) - 273.15
    ).to_dataframe()
    df_point['ind'] = ind
    df_point=df_point.set_index(['ind','lat','lon'],append=True)
    df_point['lat_obs'] = df.loc[ind].latitude
    df_point['lon_obs'] = df.loc[ind].longitude
    df_era = df_era.append(df_point[['lat_obs','lon_obs','t2m']])
df_era = df_era.reset_index('lat').reset_index('lon')

# % Comparison of t2m and t10m: final comparison
firn_memory_time = 5
shift = 1
x=[]
y=[]
df['era_avg_t2m'] = np.nan
for ind_loc in progressbar.progressbar(df_era.index.get_level_values(1).unique()):
    lat = df_era.xs(ind_loc, level=1).lat_obs[0]
    lon = df_era.xs(ind_loc, level=1).lon_obs[0]
    df_loc = df.loc[(df.latitude == lat)&(df.longitude == lon),:].copy()
    for ind_meas in df_loc.index:
        date = pd.to_datetime(df.loc[ind].date)- pd.DateOffset(years=shift)
        date0 = date- pd.DateOffset(years=firn_memory_time)- pd.DateOffset(years=shift)
        
        x.append(df_era.xs(ind_loc, level=1).loc[date0:date,'t2m'].mean())
        df.loc[ind_meas,'era_avg_t2m'] = df_era.xs(ind_loc, level=1).loc[date0:date,'t2m'].mean()
        
        y.append(df.loc[ind_meas].temperatureObserved)

x = np.array(x)
y = np.array(y)
df_save = df

df['era_dev'] = df.temperatureObserved - df.era_avg_t2m
df_save['era_dev'] = df_save.temperatureObserved - df_save.era_avg_t2m

# %% Comparison of t2m and t10m: sensitivity to the memory of the subsurface
# for firn_memory_time in range(1,21):
#     x=[]
#     y=[]
#     for ind in progressbar.progressbar(df_era.index.get_level_values(1).unique()):
#         date = pd.to_datetime(df.loc[ind].date)
#         date0 = date- pd.DateOffset(years=firn_memory_time)
#         x.append(df_era.xs(ind, level=1).loc[date0:date,'t2m'].mean())
#         y.append(df.loc[ind].temperatureObserved)
#     x = np.array(x)
#     y = np.array(y)
    
#     fig = plt.figure()
#     plt.plot(x,y, 'o',markersize=1)
#     plt.plot([np.nanmin(y), np.nanmax(y)], [np.nanmin(y), np.nanmax(y)], color='black')
#     plt.title(str(firn_memory_time)+' years memory. RMSE = %0.3f $^o$ C'%np.sqrt(np.nanmean((x-y)**2)))
#     plt.xlabel('ERA5 average 2 m air temperature ($^o$ C)')
#     plt.ylabel('Observed 10 m firn temperature ($^o$ C)')
#     fig.savefig(str(firn_memory_time)+'_years_memory_air_firn_temp.png')

# for ind in progressbar.progressbar(df_era.index.get_level_values(1).unique()):
#     date = pd.to_datetime(df.loc[ind].date)
#     date0 = date- pd.DateOffset(years=3)
#     if date < time_ERA[0]:
#         continue

#     fig, ax = plt.subplots(1,1)
#     ax.set_title(df.loc[ind].site)
#     ax.plot([date0, date], 
#             np.array([1, 1])*df.loc[ind].temperatureObserved,
#             label='firn temp',linewidth=2)
#     for firn_memory_time in range(1,11):
#         date = pd.to_datetime(df.loc[ind].date)
#         date0 = date- pd.DateOffset(years=firn_memory_time)
#         ax.plot(pd.to_datetime([date0, date]), 
#             np.array([1, 1])*df_era.xs(ind, level=1).loc[date0:date,'t2m'].mean(),
#                     label=str(firn_memory_time)+' years average')
#         ax.legend()
#     df_era.xs(ind, level=1).t2m.resample('Y').mean().plot(ax=ax, label='air temp')

# BEST averaging period: 15 years

# % Comparison of t2m and t10m: sensitivity to the lag time
# firn_memory_time = 5
# for shift in range(1,20):
#     x=[]
#     y=[]
#     for ind in progressbar.progressbar(df_era.index.get_level_values(1).unique()):
#         date = pd.to_datetime(df.loc[ind].date)- pd.DateOffset(years=shift)
#         date0 = date- pd.DateOffset(years=firn_memory_time)- pd.DateOffset(years=shift)
#         x.append(df_era.xs(ind, level=1).loc[date0:date,'t2m'].mean())
#         y.append(df.loc[ind].temperatureObserved)
#     x = np.array(x)
#     y = np.array(y)
    
#     fig = plt.figure()
#     plt.plot(x,y, 'o',markersize=1)
#     plt.plot([np.nanmin(y), np.nanmax(y)], [np.nanmin(y), np.nanmax(y)], color='black')
#     plt.title(str(shift)+' years shift. RMSE = %0.3f $^o$ C'%np.sqrt(np.nanmean((x-y)**2)))
#     plt.xlabel('ERA5 average 2 m air temperature ($^o$ C)')
#     plt.ylabel('Observed 10 m firn temperature ($^o$ C)')
#     fig.savefig(str(shift)+'_years_shift_5_year_memory_air_firn_temp.png') 




#%% Investigating how much the meassurements depend on the month of collection
# site_list = df.index.get_level_values(1).unique()
# temp_month_all=[]
# temp_year_all=[]
# month_all=[]
# temp_avg_all = []
# for site in site_list:
#     if (df.xs(site,level='site').shape[0] < 12) | (df.xs(site,level='site').index.unique().shape[0] < 12):
#         continue
#     else:
#         print(site)
#         df_site = df.xs(site,level='site')
#         df_site.loc[:,'date'] = [pd.to_datetime(d) for d in df_site.reset_index().date]
#         df_site = df_site.set_index('date')
#         df_site = df_site.resample('M').mean()
#         df_rolling = df_site.rolling(12, min_periods = 12).mean()
        
#         temp_month = df_site.loc[df_rolling.temperatureObserved.notnull(), 'temperatureObserved']
#         temp_year = df_rolling.loc[df_rolling.temperatureObserved.notnull(), 'temperatureObserved']
#         month = df_site.loc[df_rolling.temperatureObserved.notnull(), 'temperatureObserved'].index.month
#         temp_avg = df_site.temperatureObserved.mean()
        
#         temp_month_all.extend(temp_month.tolist())
#         temp_year_all.extend(temp_year.tolist())
#         month_all.extend(month.tolist())
#         temp_avg_all.extend((temp_avg+temp_year*0).tolist())

#         # df_site.temperatureObserved.plot()
#         # df_rolling.temperatureObserved.plot()
#         # plt.plot(month, temp_month - temp_year, 'o')
# temp_month_all=np.array(temp_month_all)
# temp_year_all=np.array(temp_year_all)
# month_all=np.array(month_all)
# temp_avg_all = np.array(temp_avg_all)


# plt.figure()
# plt.scatter(month_all, temp_month_all-temp_year_all, 5, temp_avg_all)

# plt.figure()
# plt.scatter(month_all, temp_avg_all, 20, temp_month_all-temp_year_all)


# diff_month_year = temp_month_all-temp_year_all

# data = [diff_month_year[month_all==i] for i in range(1,13)]
 
# fig = plt.figure(figsize =(10, 7))
# plt.boxplot(data)
# plt.plot([0, 13],[0, 0 ],'--',color='black')


# %% Model lat_elev_grid to map deviation between Ta and T10m
import itertools
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from scipy.spatial import cKDTree

step_lat = 5.
latitude_bins = np.arange(60., 90., step_lat)

step_elev = 500.
elevation_bins = np.arange(0., 4000., step_elev)

x_center, y_center = np.meshgrid(latitude_bins + step_lat / 2,
    elevation_bins + step_elev / 2, 
)
x_org = np.unique((x_center - np.min(x_center)) / np.max(x_center).flatten())
y_org = np.unique((y_center - np.min(y_center)) / np.max(y_center).flatten())

real_x = np.arange(60., 86., 0.1) # latitude_bins 
real_y = np.arange(0., 3400., 10.) # elevation_bins
xx, yy = np.meshgrid(real_x, real_y)  # ,np.arange(0,1600,10))
dx = (real_x[1] - real_x[0]) / 2.0
dy = (real_y[1] - real_y[0]) / 2.0
extent = [real_x[0], real_x[-1]+dx*2, real_y[-1]+dy*2, real_y[0]]

def fitting_surface(df, target_var = "temperatureObserved", order = 3):
    grid_temp = np.empty((len(elevation_bins), len(latitude_bins)))
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
            msk = np.logical_and.reduce(conditions)
            grid_temp[i, j] = df.loc[msk, target_var].mean()
            if df.loc[msk, "temperatureObserved"].count()==0:
                df.loc[msk, "weight"] = 0
            else:
                df.loc[msk, "weight"] = 1/df.loc[msk, "temperatureObserved"].count()

    # method #1
    def polyfit2d(x, y, z, W = [], order=3):
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        if len(W)==0:
            W = np.diag(z*0+1)            
        else:
            W = np.sqrt(np.diag(W))
        Gw = np.dot(W,G)
        zw = np.dot(z,W)
        m, _, _, _ = np.linalg.lstsq(Gw, zw, rcond=None)
        return m
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`
    
        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)
    
        return hull.find_simplex(p)>=0
    
    points = np.array([[x,y] for x,y in zip(df.latitude.values, df.elevation.values)])
    hull = ConvexHull(points)
    
    def polyval2d(x, y, m):
        order = int(np.sqrt(len(m))) - 1
        ij = itertools.product(range(order+1), range(order+1))
        z = np.zeros_like(x)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
        
        # # removing output for cells that are further than a certain 
        # # distance from the df training set
        s1 = np.array([[x, y/200] for x,y in zip(x.flatten(),y.flatten())])
        s2 = np.array([[x, y/200] for x,y in zip(df.latitude.values, df.elevation.values)])
        
        min_dists, min_dist_idx = cKDTree(s2).query(s1, 1)
        z = z.flatten()

        points_query = np.array([[x,y] for x,y in zip(x.flatten(), y.flatten())])
        ind_in_hull = in_hull(points_query, points)
        msk = (~ind_in_hull) & (min_dists>1)
        z[msk] = np.nan
        z = np.reshape(z,x.shape)
        return z


    m = polyfit2d(df.loc[df[target_var].notnull(), 'latitude'],
                  df.loc[df[target_var].notnull(), 'elevation'],
                  df.loc[df[target_var].notnull(), target_var],
                  W = df.loc[df[target_var].notnull(), 'weight'],
                  order = order)
    zz = polyval2d(xx+dx, yy+dy, m)
    
    res =  df[target_var] - polyval2d(df.latitude.values, df.elevation.values, m)
    T10_mod = elevation.copy()
    T10_mod.values = polyval2d(latitude.values, 
                               elevation.values,
                               m)

    return zz, T10_mod, res


def plot_latitude_elevation_space(ax, zz, df = [], 
                                  target_var = "temperatureObserved",
                                  vmin = -35, vmax = 0,
                                  contour_levels = [],
                                  cmap ='coolwarm'):
    ax.set_facecolor("black")
    im = ax.imshow(zz, extent=extent,
                    aspect="auto", cmap=cmap,vmin=vmin, vmax=vmax)
    if len(contour_levels)>0:
        CS = ax.contour(zz, contour_levels, colors='k', origin='upper', extent=extent)
        ax.clabel(CS, CS.levels, inline=True, fontsize=10)


    # scatter residual
    if len(df)>0:
        sct = ax.scatter(
            df["latitude"],
            df["elevation"],
            s=80,
            c=df[target_var],
            edgecolor="gray",
            cmap = cmap,
            vmin=vmin,vmax=vmax,
            zorder=10
        )
    ax.set_yticks(elevation_bins)
    ax.set_xticks(latitude_bins)
    ax.grid()
    ax.set_ylim(0, 3500)
    ax.set_xlim(60, 82)
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_xlabel("Latitude ($^o$N)")
    return im

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
def plot_greenland_map(ax, T10_mod, df, 
                       target_var = "temperatureObserved",
                       vmin = -5, vmax = 5,
                       colorbar_label = '',
                       colorbar = True,
                       cmap = 'coolwarm'):
    if colorbar:
        cbar_kwargs={'label': colorbar_label, 'orientation':'vertical', 'location':'left'}
    else:
        cbar_kwargs={}
    land.plot(ax=ax, zorder=0, color="black", transform = ccrs.epsg(3413))
    
    T10_mod.plot(ax =ax, cmap=cmap, add_colorbar = colorbar, 
                 cbar_kwargs=cbar_kwargs,
                 vmin = vmin, vmax = vmax, transform = ccrs.epsg(3413)
                 )
    elev_contours.plot(ax=ax, color="gray", transform = ccrs.epsg(3413))

    ax.set_extent([-57, -30, 59, 84], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator([-60, -40, -20])
    gl.ylocator = mticker.FixedLocator([80, 70, 60])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = dict(rotation= 0, ha='center', fontsize = 10)
    # gl.ylabel_style = dict(rotation= 0, ha='center', fontsize = 10)
    gl.top_labels=False
    gl.right_labels=False
    ax.set_title('')
    
    if len(df)>0:
        df.plot(
        ax=ax,
        column=target_var,
        cmap=cmap,
        vmin = vmin,
        vmax = vmax,
        markersize=30,
        edgecolor="gray",
        legend=False, 
        transform = ccrs.epsg(3413)
        )   
    return ax

target_var = 'temperatureObserved'

if target_var == "era_dev":
    colorbar_label = "Deviation between $T_{10m}$ \nand $Ta_{ERA5, y}$ ($^o$C)"
    vmin = -5
    vmax = 5
    contour_levels = np.arange(-5,5)
    order=4

elif target_var == "temperatureObserved":
    colorbar_label = "$T_{10m}$ ($^o$C)"
    contour_levels = np.arange(-50,0,5)
    vmin = -35
    vmax = 0
    order=3  

    

# fig = plt.figure(figsize=(10, 25))
# plt.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.87, wspace=0.05, hspace=0.3)
 
# ax = fig.add_subplot(1,1,1, projection = ccrs.NorthPolarStereo(-45))
# print(ax.get_position())
# plot_greenland_map(ax, T10_mod, df, target_var, 
#                         vmin = vmin, vmax = vmax, 
#                         colorbar_label=colorbar_label,
#                         colorbar = False, cmap = cmap)   
# %% Fitting surfaces and caluclating maps
order_temp=3
order_delta_t=3
start = [1950, 1990, 2010]
end = [1990, 2010, 2022]

# initializing output variables
zz = [xx, xx, xx]
zz_delta_t = [xx, xx, xx]
T10_mod = [elevation, elevation, elevation]
delta_t = [elevation, elevation, elevation]
df_save['T10m_res'] = np.nan
for i in range(len(start)):
    s=start[i]
    e=end[i]
    df = df_save.reset_index()
    df = df.loc[(df.date>=str(s)) & (df.date<str(e)),:]
    zz[i], T10_mod[i], res = fitting_surface(df, "temperatureObserved", order = order_temp)
    df_save.loc[(df_save.date>=str(s)) & (df_save.date<str(e)),'T10m_res'] = res.values
    zz_delta_t[i], delta_t[i], _ = fitting_surface(df, "era_dev", order = order_delta_t)

zz_diff_1 = zz[1]- zz[0]
zz_diff_2 = zz[2]- zz[1]
T10_mod_diff_1 = T10_mod[1]- T10_mod[0]
T10_mod_diff_2 = T10_mod[2]- T10_mod[1]

#%% printing tables
print('period, % within 0.5 deg, % < 1 deg, % > 1 deg')
for i in range(len(start)):
    s=start[i]
    e=end[i]
    print(str(s)+'-'+str(e)+',', 
          '%0.1f, %0.1f, %0.1f'% 
          (np.sum(np.abs(delta_t[i]).values<0.5)/np.sum(elevation.notnull().values)*100,
           np.sum(delta_t[i].values<1)/np.sum(elevation.notnull().values)*100,
           np.sum(delta_t[i].values>1)/np.sum(elevation.notnull().values)*100))
    
# over common mask
msk =  (delta_t[0].notnull()) & (delta_t[1].notnull()) & (delta_t[2].notnull())
print('over common mask')
for i in range(len(start)):
    s=start[i]
    e=end[i]
    print(str(s)+'-'+str(e)+',', 
          '%0.1f, %0.1f, %0.1f'% 
          (np.sum(np.abs(delta_t[i].where(msk)).values<0.5)/np.sum(elevation.notnull().values)*100,
           np.sum(delta_t[i].where(msk).values<1)/np.sum(elevation.notnull().values)*100,
           np.sum(delta_t[i].where(msk).values>1)/np.sum(elevation.notnull().values)*100))
    
# measurements in the dataset
print('---')
for i in range(len(start)):
    s=start[i]
    e=end[i]
    df = df_save.reset_index()
    df = df.loc[(df.date>=str(s)) & (df.date<str(e)),:]
    print('%0.0f-%0.0f, %0.0f / %0.0f, %0.1f %% '%  
          (s, e,  df.shape[0], df_save.shape[0], 
           np.sum(T10_mod[i].notnull().values)/np.sum(elevation.notnull().values)*100)) 
    
# stats on the T10m maps
print('---')
for i in range(len(start)):
    s=start[i]
    e=end[i]
    print('%0.0f-%0.0f, %0.1f degC, %0.1f %%, %0.1f degC, %0.1f %% '%  
          (s, e,  T10_mod[i].mean(), 
           np.sum(T10_mod[i].notnull().values)/np.sum(elevation.notnull().values)*100,
          T10_mod[i].where(msk).mean(),
          np.sum(msk.values)/np.sum(elevation.notnull().values)*100)) 

# %% 3d plot
target_var = "temperatureObserved"
fig, ax = plt.subplots(1,3,subplot_kw={"projection": "3d"})
ax[1] = plt.subplot(1,1,1, projection = '3d')
for i in range(3):
    surf = ax[i].plot_surface(xx,yy,zz[i], cmap=cm.coolwarm,
                            linewidth=0, alpha=0.5,
                            vmin=-35, vmax=0)
    s=start[i]
    e=end[i]
    df = df_save.reset_index()
    df = df.loc[(df.date>=str(s)) & (df.date<str(e)),:]
    cNorm = matplotlib.colors.Normalize(vmin=min(df[target_var]), vmax=max(df[target_var]))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
    ax[i].scatter(df.latitude, df.elevation, df[target_var], 
                  c=scalarMap.to_rgba(df[target_var]))

#%% T10m or era_dev for given periods
ftsz=9
matplotlib.rcParams.update({"font.size": ftsz})

ABC = 'ABCDEFGH'
import matplotlib.patheffects as pe
for target_var in ['temperatureObserved']:
    if target_var == "era_dev":
        colorbar_label = "$\Delta T_{10m -air}$ ($^o$C)"
        vmin = -5
        vmax = 5
        contour_levels = []
        order=3
        cmap = matplotlib.cm.get_cmap('Spectral', int(vmax-vmin)).reversed()
        zz_tmp = zz_delta_t
        T10_tmp = delta_t
    
    elif target_var == "temperatureObserved":
        colorbar_label = "$T_{10m}$ ($^o$C)"
        contour_levels = np.arange(-50,0,5)
        vmin = -35
        vmax = 0
        order=2
        cmap ='coolwarm'
        zz_tmp = zz
        T10_tmp = T10_mod
        
    fig = plt.figure(figsize=(5, 25))
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.9, hspace = 0.1, wspace=0.04)
            
    for i, (s, e) in enumerate(zip(start, end)):
        df = df_save.reset_index()
        df = df.loc[(df.date>=str(s)) & (df.date<str(e)),:]
    
        # subsurface temperature regression latitude elevation
        ax = fig.add_subplot(3,2, i*2+1)
        im = plot_latitude_elevation_space(ax, zz_tmp[i], df, target_var, 
                                           vmin = vmin, vmax = vmax, 
                                           contour_levels=contour_levels, 
                                           cmap = cmap)
        
        ax.annotate(ABC[i*2]+'. '+str(s)+'-'+str(e-1) , (0.03, 0.9), 
                    xycoords = 'axes fraction', color = 'black', fontsize = ftsz, 
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                    zorder = 10)
        
        ax = fig.add_subplot(3,2, i*2+2, projection = ccrs.NorthPolarStereo(-45))
        plot_greenland_map(ax, T10_tmp[i], df, target_var, 
                           vmin = vmin, vmax = vmax, 
                           colorbar_label=colorbar_label,
                           colorbar = False, cmap = cmap)
        ax.annotate(ABC[i*2+1]+'. '+str(s)+'-'+str(e-1) , (0.03, 0.92), 
                    xycoords = 'axes fraction', color = 'black', fontsize = ftsz,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                    zorder = 10)
    cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
    cbar_ax.xaxis.tick_top()
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.set_xlabel(colorbar_label, fontsize=ftsz)
    if target_var == 'era_dev':
        plt.annotate('radiative cooling', [0.16, 0.917],xycoords='figure fraction',fontsize = ftsz, color='white')
        plt.annotate('meltwater refreezing', [0.62, 0.917],xycoords='figure fraction',fontsize = ftsz, color='white')

    plt.savefig("figures/" +target_var+ "_lat_elev_model_deviation_Ta_T10m_periods.png")
    
#%% Change in T10m
colorbar_label = "$\Delta$$T_{10m}$ ($^o$C)"
contour_levels = np.arange(-50,0,5)
vmin = -1
vmax = 3

fig = plt.figure(figsize=(8, 18))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.87, wspace=0.3, hspace=0.3)
cmap = matplotlib.cm.get_cmap('Reds', int(vmax-vmin)*2)
# subsurface temperature regression latitude elevation
ax = fig.add_subplot(2, 2, 1)
im = plot_latitude_elevation_space(ax, zz_diff_1,
                                   vmin = vmin, vmax = vmax, # contour_levels=np.arange(-5, 5),
                                   cmap = cmap)
ax.set_title('A. 1990-2009 minus 1950-1989')

ax = fig.add_subplot(2, 2, 3, projection = ccrs.NorthPolarStereo(-45))
plot_greenland_map(ax, T10_mod_diff_1, [], target_var, 
                   vmin = vmin, vmax = vmax, 
                   colorbar_label=colorbar_label,
                   colorbar = False,
                   cmap = cmap)
ax.set_title('B. 1990-2009 minus 1950-1989')

ax = fig.add_subplot(2, 2, 2)
im = plot_latitude_elevation_space(ax, zz_diff_2,
                                   vmin = vmin, vmax = vmax,# contour_levels=np.arange(-5, 5),
                                   cmap = cmap)
ax.set_title('C. 2010-2022 minus 1990-2009')

ax = fig.add_subplot(2, 2, 4, projection = ccrs.NorthPolarStereo(-45))
plot_greenland_map(ax, T10_mod_diff_2, [], target_var, 
                   vmin = vmin, vmax = vmax, 
                   colorbar_label=colorbar_label,
                   colorbar = False,
                   cmap = cmap)
ax.set_title('D. 2010-2021 minus 1990-2009')

cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cbar_ax.xaxis.tick_top()
cbar_ax.xaxis.set_label_position('top') 
cb.set_label(label = colorbar_label, size=15)
plt.savefig("figures/temperatureObserved_lat_elev_model_difference.png")

# %% Evaluation of the map on three transects

transects = gpd.GeoDataFrame.from_file("Data/misc/transects.shp")
transects = transects.to_crs("EPSG:3413")

fig, ax = plt.subplots(3,1)

for i in transects.index:
    df = gpd.GeoDataFrame(df_save.reset_index()).to_crs("EPSG:3413")
    tran = transects.loc[transects.index == i,:]

    points_within = gpd.sjoin(df, tran, op='within')
    print(i)
    # period 1
    msk = (points_within.year >= 1950) & (points_within.year < 1990)
    ax[i].scatter(points_within.loc[msk, 'elevation'],
                points_within.loc[msk, 'temperatureObserved'],
                marker = '^',linestyle = 'None',
                color = 'tab:blue')
    
    T10_clipped = T10_mod[0].rio.clip(tran.geometry.values, transects.crs)
    elevation_clipped = elevation.rio.clip(tran.geometry.values, transects.crs)
    T10_clipped = T10_clipped.where(T10_clipped>-100)
    elevation_clipped = elevation_clipped.where(T10_clipped>-100)
    
    ax[i].plot(elevation_clipped.values[T10_clipped.notnull().values],
             T10_clipped.values[T10_clipped.notnull().values],
             color = 'tab:blue',  marker = '.', linestyle = 'None')
    
    # period 2
    msk = (points_within.year >= 1990) & (points_within.year < 2010)
    ax[i].scatter(points_within.loc[msk, 'elevation']+30,
                points_within.loc[msk, 'temperatureObserved'],
                marker = '^',linestyle = 'None',
                color = 'tab:red')

    T10_clipped = T10_mod[1].rio.clip(tran.geometry.values, tran.crs)
    T10_clipped = T10_clipped.where(T10_clipped>-100)
    
    ax[i].plot(elevation_clipped.values[T10_clipped.notnull().values],
             T10_clipped.values[T10_clipped.notnull().values],
             color = 'tab:red',   marker = '.', linestyle = 'None')
    
    # period 3
    T10_clipped = T10_mod[2].rio.clip(tran.geometry.values, tran.crs)
    T10_clipped = T10_clipped.where(T10_clipped>-100)
    
    ax[i].plot(elevation_clipped.values[T10_clipped.notnull().values],
             T10_clipped.values[T10_clipped.notnull().values],
             color = 'tab:orange',  marker = '.', linestyle = 'None')
    msk = (points_within.year >= 2010) & (points_within.year < 2022)
    ax[i].scatter(points_within.loc[msk, 'elevation']+60,
                points_within.loc[msk, 'temperatureObserved'],
                marker = 'x',linestyle = 'None',
                color = 'tab:orange')

ax[0].set_title('EGIG')
ax[1].set_title('North Greenland Traverse')
ax[2].set_title('K-transect')

# %% Loading GC-Net AWS data
df = df_save.reset_index()
station_list = ['CP1', 'DYE-2','Saddle','NASA-SE','SouthDome','TUNU-N','NASA-E','NASA-U','Summit']

df_gcnet = df_save.reset_index().loc[np.isin(df.site,station_list),:]

df_aws = pd.DataFrame()
for site in station_list:
    print(site)
    tmp = xr.open_dataset('C:/Data_save/Data JoG 2020/Corrected/'+site+'_0_SiCL_pr0.001_Ck1.00_darcy_wh0.10/'+site+'_surf-bin-1.nc').to_dataframe()
    tmp['site'] = site
    tmp['melt_mm'] = -tmp.H_melt.diff()
    df_aws = df_aws.append(tmp)

df_sum = df_aws.groupby('site').resample('Y').sum()
df_mean = df_aws.groupby('site').resample('Y').mean()
df_y = df_aws.groupby('site').resample('Y').first()[['LRin', 'LRout_mdl', 'SHF', 'LHF', 'GF',
       'GFsubsurf', 'rainHF', 'meltflux', 'melt_mm', 'SRin', 'SRout', 'Tsurf', 
       'theta_2m', 'site']]

df_y.melt_mm = df_sum.melt_mm
df_y[['LRin', 'LRout_mdl', 'SHF', 'LHF', 'GF',
       'GFsubsurf', 'rainHF', 'meltflux', 'SRin', 'SRout', 'Tsurf', 
       'theta_2m']] = df_mean[['LRin', 'LRout_mdl', 'SHF', 'LHF', 'GF',
       'GFsubsurf', 'rainHF', 'meltflux', 'SRin', 'SRout', 'Tsurf', 
       'theta_2m']] 
df_gcnet['melt_mm'] = np.nan
df_gcnet['netRad'] = np.nan
for i in progressbar.progressbar(df_gcnet.index):
    site = df_gcnet.loc[i,'site']
    date = df_gcnet.loc[i,'date']
    if site not in df_sum.index.get_level_values(0).unique():
        continue
    
    df_y_site = df_sum.loc[site,:]
    msk = (df_y_site.index < pd.to_datetime(date)) & (df_y_site.index >  pd.to_datetime(date)-pd.Timedelta(str(5*365)+' days'))
    if np.sum(msk)<5:
        continue
    df_gcnet.loc[i, 'melt_mm'] = df_y_site.loc[msk,:].melt_mm.sum()
    df_gcnet.loc[i, 'netRad'] = (df_y_site.loc[msk,:].LRin \
        - df_y_site.loc[msk,:].LRout_mdl \
        + df_y_site.loc[msk,:].SRin \
        - df_y_site.loc[msk,:].SRout).mean()

# %% Plotting T10m vs Ta scatter and DeltaT at GC-Net sites
import scipy
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace = 0.5, wspace = 0.1)
ax = plt.subplot(2,2,1)
plt.plot(df.era_avg_t2m.values, df.temperatureObserved.values, 'ok',markersize=2)
plt.plot([-33, 0], [-33, 0], color='black')
msk = ~np.isnan(df.era_avg_t2m.values) & ~np.isnan(df.temperatureObserved.values)
_, _, r, _, _ = scipy.stats.linregress(df.era_avg_t2m.values[msk],
                                              df.temperatureObserved.values[msk])
RMSD = np.sqrt(np.nanmean((df.era_avg_t2m.values- df.temperatureObserved.values)**2))
MD = (np.nanmean((df.era_avg_t2m.values- df.temperatureObserved.values)))
plt.annotate('$R^2$ = %0.2f \nRMSD = %0.1f $^o$C \nMD = %0.1f $^o$C'%(r**2, RMSD, MD), 
             (0.65, 0.1), xycoords = 'axes fraction')
plt.xlabel('$\overline{Ta}$ ($^o$C)')
plt.ylabel('Observed $T_{10m}$ ($^o$C)')
ax.annotate('A' , (0.03, 0.92),   
            xycoords = 'axes fraction', color = 'black', fontsize = 14, zorder = 10)
ax.set_xlim(-33,0)
ax.set_ylim(-33,0)

ax = plt.subplot(2,2,2)
T_interp = df_save.temperatureObserved - df_save.T10m_res
plt.plot(T_interp.values, df.temperatureObserved.values, 'ok',markersize=2)
msk = ~np.isnan(T_interp.values) & ~np.isnan(df.temperatureObserved.values)
_, _, r, _, _ = scipy.stats.linregress(T_interp.values[msk],
                                              df.temperatureObserved.values[msk])
RMSD = np.sqrt(np.nanmean((T_interp.values- df.temperatureObserved.values)**2))
MD = (np.nanmean((T_interp.values- df.temperatureObserved.values)))
plt.annotate('$R^2$ = %0.2f \nRMSD = %0.1f $^o$C \nMD = %0.1f $^o$C'%(r**2, RMSD, MD), 
             (0.65, 0.1), xycoords = 'axes fraction')

plt.plot([-33, 0], [-33, 0], color='black')
plt.xlabel('Observed $T_{10m}$ ($^o$C)')
plt.ylabel('Interpolated $T_{10m}$ ($^o$C)')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.annotate('B' , (0.03, 0.92),   
            xycoords = 'axes fraction', color = 'black', fontsize = 14, zorder = 10)
ax.set_xlim(-33,0)
ax.set_ylim(-33,0)

ax = plt.subplot(2,2,3)
ax.axvline(0,color = 'k', linestyle='--')
for site in df_gcnet.site.unique():
    ax.plot(df_gcnet.loc[df_gcnet.site == site, 'era_dev'],
           df_gcnet.loc[df_gcnet.site == site, 'melt_mm'], 
           marker='o', linestyle = '')
    
msk = ~np.isnan(df_gcnet.era_dev.values) & ~np.isnan(df_gcnet.melt_mm.values)
slope, intercept, r, p, _ = scipy.stats.linregress(df_gcnet.era_dev.values[msk],
                                              df_gcnet.melt_mm.values[msk])
print(r, p)
ax.plot([-5, 4], np.array([-5, 4])*slope+intercept,color = 'gray', linestyle='--')
ax.set_ylabel('Average annual melt (mm)')
ax.set_xlabel('$\Delta T_{10m - air}$ ($^o$C)')
ax.annotate('C' , (0.03, 0.92),   
            xycoords = 'axes fraction', color = 'black', fontsize = 14, zorder = 10)
ax.set_xlim(-5, 4)

ax2 = plt.subplot(2,2,4)
ax2.axvline(0,color = 'k', linestyle='--')
for site in df_gcnet.site.unique():
    ax2.plot(df_gcnet.loc[df.site == site, 'era_dev'],
           df_gcnet.loc[df.site == site, 'netRad']/1000, 
           marker='o', linestyle = '',
           label=site)
msk = ~np.isnan(df_gcnet.era_dev.values) & ~np.isnan(df_gcnet.netRad.values)
slope, intercept, r, p, _ = scipy.stats.linregress(df_gcnet.era_dev.values[msk],
                                              df_gcnet.netRad.values[msk]/1000)
print(r, p)
ax2.plot([-5, 4], np.array([-5, 4])*slope+intercept,color = 'gray', linestyle='--')
ax2.set_ylabel('Average annual radiative \nbudget at the surface  (kW)')
ax2.set_xlabel('$\Delta T_{10m - air}$ ($^o$C)')
ax2.axes.yaxis.set_ticks_position('right')
ax2.axes.yaxis.set_label_position('right')
ax2.annotate('D' , (0.03, 0.92),   
            xycoords = 'axes fraction', color = 'black', fontsize = 14, zorder = 10)
ax2.set_xlim(-5, 4)
plt.legend(ncol=5, loc='upper right', bbox_to_anchor=(0.85, 1.28), title='Site:')

fig.savefig('figures/scatter_T10m_Ta_gcnet_analysis.png')

# %% Plotting observed air temp compared to T10m
df = df_save.reset_index()
df =df.replace({'site': {'EGP':'EastGRIP', 'NSU':'NASA-U', 'KAN-U':'KAN_U',
 'EastGrip':'EastGRIP', 'Camp Century':'CampCentury', 'CEN':'CampCentury'}})
df['date'] = pd.to_datetime(df.date).dt.tz_localize(0)

df_obs = pd.read_csv('data/AWS_monthly_temperatures.csv')
df_obs.time = pd.to_datetime(df_obs.time)
df_obs['Ta'] = df_obs[['TA1','TA2']].mean(axis=1)
df_obs = df_obs.drop(columns=['TA1','TA2'])

df['Ta_obs'] = np.nan
# calculating average temp backward
for ind in df.loc[np.isin(df.site, df_obs.site.unique()),:].index:
    if ind%500 == 0:
        print(ind)
    date = df.loc[ind,'date']
    site = df.loc[ind,'site']
    if site not in df_obs.site.unique():
        continue

    date0 = date - pd.DateOffset(years=1)
    if date0 < df_obs.loc[df_obs.site == site].time.min():
        continue   
    df_tmp = df_obs.loc[df_obs.site == site].set_index('time').resample('M').mean().loc[date0:date,:]
    
    if df_tmp.Ta.notnull().sum() < 11:
        continue
    df_tmp.Ta.mean()
    df.loc[ind, 'Ta_obs'] = df_tmp.Ta.mean()
#%% Plotting
df_tmp = df.loc[df.Ta_obs.notnull(),:]
plt.figure()
plt.subplots_adjust(right=0.6)
for site in df_tmp.site.unique():
    plt.plot(df_tmp.loc[df_tmp.site == site, 'Ta_obs'],
             df_tmp.loc[df_tmp.site == site, 'temperatureObserved'],
             marker = 'o', linestyle='None',
             label = site)
plt.plot([-40, 2],[-40, 2],'-k')
plt.ylabel('Observed $T_{10m}$ ($^o$C)')
plt.xlabel('Annual average air temperature from AWS ($^o$C)')
plt.xlim(-40, 2)
plt.ylim(-40, 2)
plt.legend(ncol=2, bbox_to_anchor = (1.05,0.8))
    