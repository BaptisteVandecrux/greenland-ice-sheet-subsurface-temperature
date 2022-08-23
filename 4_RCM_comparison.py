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
df = pd.read_csv("10m_temperature_dataset_monthly.csv")

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
crs_racmo_proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +o_lon_p=0'
crs_racmo = CRS.from_string(crs_racmo_proj)
# target_crs = crs_racmo
target_crs = CRS.from_string('EPSG:3413')

print('loading RACMO')
ds_racmo = xr.open_dataset("C:/Data_save/RCM/RACMO/FDM_T10m_FGRN055_1957-2020_GrIS_GIC.nc").set_coords('lat').set_coords('lon')
ds_racmo = ds_racmo.rename({'rlat': 'y', 'rlon': 'x'})
# adding crs info and lat lon grids
ds_racmo = ds_racmo.rio.write_crs(crs_racmo).drop_vars(['lat', 'lon'])
# ds_racmo = ds_racmo.rio.reproject(target_crs)
# x, y = np.meshgrid(ds_racmo.x.values, ds_racmo.y.values)
# lon, lat = transform({'init': 'EPSG:3413'}, {'init': 'EPSG:4326'}, x.flatten(), y.flatten())
# lat = np.reshape(lat, x.shape)
# lon = np.reshape(lon, x.shape)
# ds_racmo['lat'] = (('y', 'x'), lat)
# ds_racmo['lon'] = (('y', 'x'), lon)

print('loading MAR')
ds_mar = xr.open_dataset("C:/Data_save/RCM/MAR/MARv3.12.0.4 fixed/MARv3.12.0.4-ERA5-20km-T10m_2.nc")
ds_mar = ds_mar.rename({'X10_85': 'x', 'Y20_155': 'y'})
ds_mar = ds_mar.drop_vars('OUTLAY')
crs_mar_proj = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs'
crs_mar = CRS.from_string(crs_mar_proj)
ds_mar = ds_mar.rio.write_crs(crs_mar).drop_vars(['lat', 'lon'])
# ds_mar = ds_mar.rio.reproject(target_crs)

print('loading HIRHAM')
# HIRHAM has no projection, just a lat-lon grid
ds_hh = xr.open_dataset("C:/Data_save/RCM/RetMIP/RetMIP_2D_T10m/RetMIP_2Doutput_Daily_DMIHH_T10m.nc")
ds_hh['time'] = pd.to_datetime([int(s) for s in np.floor(ds_hh["time"].values)], format="%Y%m%d")
x, y = transform( {'init': 'EPSG:4326'}, crs_racmo,
                     ds_hh.lon.values.flatten(), ds_hh.lat.values.flatten())
x = np.reshape(x,ds_hh.lon.values.shape)
x = np.round(x[0,:],2)
y = np.reshape(y,ds_hh.lat.values.shape)
y= np.round(y[:,0],2)
ds_hh['x'] = x
ds_hh['y'] = y
ds_hh = ds_hh.rio.write_crs(crs_racmo).drop_vars(['lat', 'lon'])
# ds_hh = ds_hh.drop_vars(['lat', 'lon']).rio.reproject(target_crs)

print('loading ANN')
ds_ann = xr.open_dataset("predicted_T10m.nc")
ds_ann['time'] = pd.to_datetime(ds_ann["time"])
crs_ann = CRS.from_string('EPSG:4326')
ds_ann = ds_ann.rio.write_crs(crs_ann)
# ds_ann = ds_ann.rio.reproject(target_crs)

# finding observation coordinate in RACMO's CRS
df.date = pd.to_datetime(df.date)
df['x'], df['y'] = transform( {'init': 'EPSG:4326'}, crs_racmo, df.longitude.values, df.latitude.values)
df['x_mar'], df['y_mar'] = transform( {'init': 'EPSG:4326'}, crs_mar, df.longitude.values, df.latitude.values)
df['T10m_RACMO'] = np.nan
df['T10m_MAR'] = np.nan
df['T10m_HIRHAM'] = np.nan
df['T10m_ANN'] = np.nan

# interpolating HIRHAM output at the observation locations

def extract_T10m_values(ds, df, dim1='x', dim2='y', name_out='out'):
    coords_uni = np.unique(df[[dim1, dim2]].values, axis=0)
    x = xr.DataArray(coords_uni[:, 0], dims='points')
    y = xr.DataArray(coords_uni[:, 1], dims='points')
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
        index_point = np.where((coords_uni ==  query_point).all(axis=1))[0][0]
        tmp = ds_interp.T10m.isel(points = index_point).sel(time=df.date.values[i], method='nearest')
        if tmp.isnull().all():
            tmp = ds_interp_2.T10m.isel(points = index_point).sel(time=df.date.values[i], method='nearest')
        if ((tmp[dim1.replace('_mar','')].values, tmp[dim2.replace('_mar','')].values) != query_point):
            print(wtf)
        if np.size(tmp)>1:
            print(wtf)
        df.iloc[i, df.columns.get_loc(name_out)] = tmp.values
    return df

print('extracting from RACMO')
df = extract_T10m_values(ds_racmo, df, dim1='x', dim2='y', name_out='T10m_RACMO')
df.T10m_RACMO = df.T10m_RACMO-273.15
print('extracting from MAR')
df = extract_T10m_values(ds_mar, df, dim1='x_mar', dim2='y_mar', name_out='T10m_MAR')
print('extracting from HIRHAM')
df = extract_T10m_values(ds_hh, df, dim1='x', dim2='y', name_out='T10m_HIRHAM')
df.T10m_HIRHAM = df.T10m_HIRHAM-273.15
print('extracting from ANN')
df = extract_T10m_values(ds_ann, df, dim1='longitude', dim2='latitude', name_out='T10m_ANN')

df_save = df.copy()  
    
# %% Plotting RCM performance

df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["reference_short"].unique()
df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2,figsize=(9, 9))
ax = ax.flatten()
plt.subplots_adjust(left=0.09, bottom=0.1, right=0.97, top=0.9, wspace=0.2, hspace=0.3)
cmap = matplotlib.cm.get_cmap("tab20b")
version =['', 'v3.12','2.3p2','5']
sym = ['o','^','v','d']

model_list = ['ANN', 'MAR','RACMO','HIRHAM']
for i, model in enumerate(model_list):
    sym_i = 0
    ax[i].plot(
        df_10m['T10m_'+model],
        df_10m['temperatureObserved'],
        marker='+',
        linestyle="none",
        # markeredgecolor="lightgray",
        markeredgewidth=0.5,
        markersize=5,
        color='k',
    )
    RMSE = np.sqrt(np.mean((df_10m['T10m_'+model] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m['T10m_'+model] - df_10m.temperatureObserved)
    RMSE2 = np.sqrt(np.mean((df_10m['T10m_'+model].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()]) ** 2))
    ME2 = np.mean(df_10m['T10m_'+model].loc[df_10m.T10m_HIRHAM.notnull()] - df_10m.temperatureObserved.loc[df_10m.T10m_HIRHAM.notnull()])
    if model == 'HIRHAM':
        textstr = '\n'.join((
            r'MD = %.2f $^o$C' % (ME,),
            r'RMSD=%.2f $^o$C' % (RMSE,),
            r'N=%.0f' % (np.sum(~np.isnan(df_10m['T10m_'+model])),)))
    else:
        textstr = '\n'.join((
            r'MD = %.2f  (%.2f) $^o$C' % (ME, ME2),
            r'RMSD=%.2f (%.2f) $^o$C' % (RMSE, RMSE2),
            r'N=%.0f  (%.0f)' % (np.sum(~np.isnan(df_10m['T10m_'+model])),
                                  np.sum(df_10m.T10m_HIRHAM.notnull()))))
    ax[i].text(0.01, 0.95, 'All sites\n'+textstr, transform=ax[i].transAxes, fontsize=11,
            verticalalignment='top')

    ax[i].set_title(ABC[i]+'. '+model, loc='left')
    ax[i].plot([-35, 2], [-35, 2], c="black")
    ax[i].set_xlim(-35, 2)
    ax[i].set_ylim(-35, 2)
    ax[i].grid()
    
# Comparison for ablation datasets
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
df_10m = df_10m.loc[(df_10m["reference_short"].astype(str) == "PROMICE")
                    | (df_10m["reference_short"].astype(str) == "Hills et al. (2018)") 
                    | (df_10m["reference_short"].astype(str) == "Hills et al. (2017)")
                    | (df_10m["site"].astype(str) == "SwissCamp"), :]
df_10m = df_10m.loc[(df_10m["site"].astype(str) != "CEN")
                    & (df_10m["site"].astype(str) != "EGP")
                    & (df_10m["site"].astype(str) != "KAN_U"),:]
df_10m = df_10m.reset_index()
df_10m = df_10m.sort_values("year")
ref_list = df_10m["site"].unique()

df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference"]]

for i,model in enumerate(model_list):
    ax[i].plot(
        df_10m['T10m_'+model],
        df_10m['temperatureObserved'],
        marker='+',
        linestyle="none",
        # markeredgecolor="lightgray",
        markeredgewidth=0.5,
        markersize=5,
        color='tab:red',
    )
    RMSE = np.sqrt(np.mean((df_10m['T10m_'+model] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m['T10m_'+model] - df_10m.temperatureObserved)

    textstr = '\n'.join((
        r'$MD=%.2f ^o$C ' % (ME, ),
        r'$RMSD=%.2f ^o$C' % (RMSE, ),
        r'$N=%.0f$' % (np.sum(~np.isnan(df_10m['T10m_'+model])), )))

    ax[i].text(0.63, 0.25, 'Ablation sites\n'+textstr, transform=ax[i].transAxes, fontsize=11,
            verticalalignment='top', color='tab:red')

fig.text(0.5, 0.04, 'Simulated 10 m subsurface temperature ($^o$C)', ha='center', va='center',fontsize=12)
fig.text(0.03, 0.5, 'Observed 10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
fig.savefig("figures/model_comp_all_no_legend.png")

# %% Plotting at selected sites
from math import sin, cos, sqrt, atan2, radians

def get_distance(point1, point2):
    R = 6370
    lat1 = radians(point1[0])  #insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

all_points = df[['latitude', 'longitude']].values

site_list = pd.DataFrame(np.array([
    ['CP1', '1955', '2021', 69.91666666666667, -46.93333333333333, 2012.0],
    ['DYE-2', '1964', '2019', 66.46166666666667, -46.23333333333333, 2100.0],
    ['Camp Century', '1954', '2019', 77.21833333333333, -61.025, 1834.0],
    ['SwissCamp', '1990', '2005', 69.57, -49.3, 1174.0],
    ['Summit', '1997', '2018', 72.5667, -38.5, 3248.0],
    ['KAN-U', '2011', '2019', 67.0003, -47.0253, 1840.0],
    ['NASA-SE', '1998', '2021', 66.47776999999999, -42.493634, 2385.0],
    ['NASA-U', '2003', '2021', 73.840558, -49.531539, 2340.0],
    ['Saddle', '1998', '2017', 65.99947, -44.50016, 2559.0],
    ['THU_U', '1954', '2021', 76.4197, -68.1460, 770],
    ['French Camp VI', '1950', '2021', 69.706167,  -48.344967, 1555.],
    ['FA_13', '1950', '2021', 66.181 ,  -39.0435, 1563.]
    ]))
site_list.columns = ['site','date start','date end','lat','lon','elev']
site_list = site_list.set_index('site')
site_list.lon = site_list.lon.astype(float)
site_list.lat = site_list.lat.astype(float)
site_list['x'], site_list['y'] = transform( {'init': 'EPSG:4326'}, crs_racmo, site_list.lon.astype(float).values, site_list.lat.astype(float).values)
site_list['x_mar'], site_list['y_mar'] = transform( {'init': 'EPSG:4326'}, crs_mar, site_list.lon.astype(float).values, site_list.lat.astype(float).values)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

fig, ax = plt.subplots(6,2, figsize=(9,17))
fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom= 0.03, hspace=0.2)
ax = ax.flatten()

for i, site in enumerate(site_list.index):
    # plotting observations
    coords = np.expand_dims(site_list.loc[site, ['lat','lon']].values.astype(float),0)
    dm = cdist(all_points, coords, get_distance)
    df_select = df.loc[dm<5,:]
    ax[i].plot(df_select.date,df_select.temperatureObserved, '.',
               markersize=9, color='darkgray',markeredgecolor='lightgray', linestyle='None',
               label = 'observations')
    
    # plotting ANN
    df_ANN = ds_ann.T10m.interp(longitude = site_list.loc[site, 'lon'],
                        latitude = site_list.loc[site, 'lat'],
                        method = 'linear').to_dataframe()
    df_ANN_interp = np.interp(df_select.date, df_ANN.index, df_ANN.T10m)
    MD = np.sqrt(np.mean((df_ANN_interp-df_select.temperatureObserved)**2))
    RMSD = np.mean((df_ANN_interp-df_select.temperatureObserved))
    N = df_select.temperatureObserved.notnull().sum()
    print('%s, %i, %0.2f, %0.2f'%(site,N,RMSD,MD))
    # ax[i].fill_between(df_era.index, bs_out[0]-bs_out[1], bs_out[0]+bs_out[1], color = 'lightgray')
    # df_ANN.T10m.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_ANN.T10m.resample('Y').mean().plot(ax=ax[i], drawstyle='steps-post',
                                          color=CB_color_cycle[0], 
                                          linewidth=2, label='ANN')

    # plotting MAR
    df_mar = ds_mar.T10m.interp(x = site_list.loc[site, 'x_mar'], 
                             y = site_list.loc[site, 'y_mar'],
                             method = 'linear').to_dataframe().T10m
    if df_mar.isnull().all():
        df_mar = ds_mar.T10m.interp(x = site_list.loc[site, 'x_mar'], 
                                 y = site_list.loc[site, 'y_mar'],
                                 method = 'nearest').to_dataframe().T10m
    # df_mar.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_mar.resample('Y').mean().plot(ax=ax[i], drawstyle='steps-post',
                                     color=CB_color_cycle[1],
                                     linewidth=2, label='MAR')
    
    # plotting RACMO
    df_racmo = ds_racmo.T10m.interp(x = site_list.loc[site, 'x'], 
                                    y = site_list.loc[site, 'y'],
                                    method = 'linear').to_dataframe().T10m-273.15
    if df_racmo.isnull().all():
        df_racmo = ds_racmo.T10m.interp(x = site_list.loc[site, 'x'], 
                                    y = site_list.loc[site, 'y'],
                                    method = 'nearest').to_dataframe().T10m-273.15
    # df_racmo.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_racmo.resample('Y').mean().plot(ax=ax[i], drawstyle='steps-post',
                                       color=CB_color_cycle[2],
                                       linewidth=2, label='RACMO')
    
    # plotting HIRHAM
    df_hh = ds_hh.T10m.interp(x = site_list.loc[site, 'x'],
                              y = site_list.loc[site, 'y'],
                              method = 'linear').to_dataframe().T10m-273.15
    if df_hh.isnull().all():
        df_hh = ds_hh.T10m.interp(x = site_list.loc[site, 'x'], 
                                 y = site_list.loc[site, 'y'],
                                 method = 'nearest').to_dataframe().T10m
    # df_hh.plot(ax=ax[i], color='k', alpha=0.3, label='_no_legend_')
    df_hh.resample('Y').mean().plot(ax=ax[i], drawstyle='steps-post',
                                    color=CB_color_cycle[3],
                                    linewidth=2, label='HIRHAM')
    

    # ax[i].set_ylim(np.nanmean(bs_out[0])-4, np.nanmean(bs_out[0])+4)
    ax[i].set_title(ABC[i]+'. '+site, loc='left')
    ax[i].grid()
    ax[i].set_xlim(pd.to_datetime('1950-01-01'),pd.to_datetime('2022-01-01'))
for i, site in enumerate(site_list.index):
    if i < 10:
        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].set_xlabel('')
    else:
        ax[i].set_xlabel('Year')
ax[0].legend(ncol=5, loc='lower right', bbox_to_anchor=(2, 1.13), fontsize=13)
fig.text(0.5, 0.01, 'Year', ha='center', va='center',fontsize=12)
fig.text(0.02, 0.5, '10 m subsurface temperature ($^o$C)', ha='center', va='center', rotation='vertical',fontsize=12)
fig.savefig("figures/model_comp_selected_sites.png")

# %% trend analysis
from datetime import datetime as dt

def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

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
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)
 
    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n     = y.notnull().sum(dim='time')
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)
    
    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats
    
    from scipy.stats import t
    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cov,cor,slope,intercept,pval,stderr

year_ranges = np.array([[1954, 1985],
                        [1985, 2022],
                        [1954, 2022]])

import statsmodels.api as sm
import rioxarray # for the extension to load
import scipy.odr
import scipy.stats
from scipy import optimize

land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
land = land.to_crs("EPSG:3413")


def plot_trend_analysis(ds_T10m, T10m_GrIS, model, year_ranges):
    # ds_T10m = ds_T10m.transpose("time", 'x', 'y')
    land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
    land = land.to_crs(ds_T10m.rio.crs)
        
    fig, ax = plt.subplots(1,3,figsize=(9,7))
    plt.subplots_adjust(hspace = 0.1,wspace=0.1, right=0.8, left = 0.01, bottom=-0.02, top = 0.8)
    ax = ax.flatten()
    ax_bot = fig.add_axes([0.12, 0.80, 0.87, 0.15])
    ax_bot.set_title(ABC[0]+'. Ice-sheet-wide average',fontweight = 'bold')
    ax_bot.plot(T10m_GrIS.index, T10m_GrIS,color='lightgray')
    ax_bot.step(T10m_GrIS.resample('Y').mean().index, T10m_GrIS.resample('Y').mean())
    ds_T10m_dy = ds_T10m.copy()
    ds_T10m_dy['time'] = [toYearFraction(d) for d in pd.to_datetime(ds_T10m_dy.time.values)]
    ds_T10m_dy = ds_T10m_dy.T10m.transpose('time','y','x')
    for i in range(3):
        # tmp = T10m_GrIS.resample('Y').mean().loc[str(year_ranges[i,0]):str (year_ranges[i,1])]
        tmp = ds_T10m_dy.sel(time=slice(year_ranges[i][0], year_ranges[i][1]))
        _, _,slope, _, pval, _ = linregress_3D(ds_T10m_dy.time, ds_T10m_dy)
    

        land.plot(ax=ax[i], zorder=0, color="black")
        vmin = -0.2; vmax = 0.2
            
        im = slope.plot(ax=ax[i],vmin = vmin, vmax = vmax,
                                            cmap='coolwarm',add_colorbar=False)
        significant = slope.where(pval < 0.05)
        X, Y = np.meshgrid(slope.x, slope.y)
        plt.hexbin(X.reshape(-1), Y.reshape(-1), significant.data[0,:,:].reshape(-1), gridsize=(40,20))
        
        year_start = max(year_ranges[i,0], tmp.index.min().year)
        year_end = min(year_ranges[i,1]-1, tmp.index.max().year)
        ax[i].set_title(ABC[i+1]+'. '+str(year_start)+' to '+str(year_end),fontweight = 'bold')
        if i in [1,3]:
            if i == 1:
                cbar_ax = fig.add_axes([0.85, 0.07, 0.03, 0.6])
            cb = plt.colorbar(im, ax = ax[i], cax=cbar_ax)
            cb.ax.get_yaxis().labelpad = 18
            cb.set_label('Trend in 10 m subsurface temperature ($^o$C yr $^{-1}$)', 
                         fontsize = 12, rotation=270)
        ax[i].set_axis_off()

        ax[i].set_xlim(land.bounds.minx.min(), land.bounds.maxx.max())
        ax[i].set_ylim(land.bounds.miny.min(),land.bounds.maxy.max())
        
    X = np.array([toYearFraction(d) for d in T10m_GrIS.loc[T10m_GrIS.notnull()].index])
    y = T10m_GrIS.loc[T10m_GrIS.notnull()].values
    
    # calculating trend on entire period
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print('%s, %i-%i, %0.3f, %0.3f'% (model, X[0], X[-1], est2.params[1]*10, est2.pvalues[1]))

    # calculating piecewise function, trends and pvalues of the trends
    def piecewise_linear(x, y0, k1, k2):
        x0 = 1985.33
        return np.piecewise(x, [x < x0], [lambda x: k1*x + y0-k1*x0, lambda x: k2*x + y0-k2*x0])
    p , e = optimize.curve_fit(piecewise_linear, X, y)


    def f_wrapper_for_odr(beta, x): # parameter order for odr
        return piecewise_linear(x, *beta)
    
    func = scipy.odr.odrpack.Model(f_wrapper_for_odr)
    data = scipy.odr.odrpack.Data(X,y)
    myodr = scipy.odr.odrpack.ODR(data, func, beta0=p,  maxit=0)
    myodr.set_job(fit_type=2)
    ptatistics = myodr.run()
    df_e = len(X) - len(p) # degrees of freedom, error
    cov_beta = ptatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta = ptatistics.sd_beta * ptatistics.sd_beta
    ci = []
    t_df = scipy.stats.t.ppf(0.975, df_e)
    ci = []
    for i in range(len(p)):
        ci.append([p[i] - t_df * ptatistics.sd_beta[i], p[i] + t_df * ptatistics.sd_beta[i]])
    
    tstat_beta = p / ptatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values
    print('%s, %i-%i, %0.3f, %0.3f'% (model, X[0], 1985, p[1]*10, pstat_beta[1]))
    print('%s, %i-%i, %0.3f, %0.3f'% (model, 1985, X[-1], p[2]*10, pstat_beta[2]))


    y_pred = piecewise_linear(X, *p)
    ax_bot.plot(T10m_GrIS.loc[T10m_GrIS.notnull()].index, y_pred, color='k', linestyle='--')
    ax_bot.autoscale(enable=True, axis='x', tight=True)
    ax_bot.set_ylabel('10 m subsurface \ntemperature ($^o$C)', fontsize=12)
    fig.savefig( 'figures/'+model+'_trend_map.png',dpi=300)
    
    # trend over common period
    X = np.array([toYearFraction(d) for d in T10m_GrIS.loc['1980':'2016'].index])
    y = T10m_GrIS.loc['1980':'2016'].values
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print('%s, %i-%i, %0.3f, %0.3f'% (model, X[0], X[-1], est2.params[1]*10, est2.pvalues[1]))
    
if 'ds_ann_3413' not in locals():
    ds_ann_3413 = ds_ann.rio.reproject("EPSG:3413")
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")
ice_4326 = ice.to_crs("EPSG:4326")
ds_ann = ds_ann.rio.clip(ice_4326.geometry.values, ice_4326.crs)
ds_ann_3413 = ds_ann_3413.rio.clip(ice.geometry.values, ice.crs)
# ds_ann_3413_2 = ds_ann_3413.rio.clip(ice.geometry.values, ice.crs, all_touched=True)

ds_ann_3413['T10m'] = ds_ann_3413.T10m.where(ds_ann_3413['T10m'] != 3.4028234663852886e+38)
weights = np.cos(np.deg2rad(ds_ann.latitude))
weights.name = "weights"
ds_ann_weighted = ds_ann.weighted(weights)
ds_ann_GrIS = (ds_ann_weighted.mean(('latitude','longitude'))).to_pandas().T10m

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

plot_trend_analysis(ds_ann_3413, ds_ann_GrIS, 'ANN', year_ranges)

#%%

# RACMO
# if 'ds_racmo_3413' not in locals():
#     ds_racmo_3413 = ds_racmo.rio.reproject("EPSG:3413")
ds_racmo = ds_racmo.rio.clip(ice.to_crs(ds_hh.rio.crs).geometry.values, ds_racmo.rio.crs)
ds_racmo_GrIS = (ds_racmo.mean(dim=('x','y'))).to_pandas().T10m-273.15
plot_trend_analysis(ds_racmo, ds_racmo_GrIS, 'RACMO', year_ranges)

# HIRHAM
# if 'ds_hh_3413' not in locals():
#     ds_hh_3413 = ds_hh.rio.reproject("EPSG:3413")
ds_hh = ds_hh.rio.clip(ice.to_crs(ds_hh.rio.crs).geometry.values, ds_hh.rio.crs)
# ds_hh_3413['T10m'] = ds_hh_3413.T10m.where(ds_hh_3413.T10m!=3.4028234663852886e+38)
ds_hh_GrIS = (ds_hh.mean(dim=('x','y'))).to_pandas().T10m-273.15
plot_trend_analysis(ds_hh, ds_hh_GrIS, 'HIRHAM', year_ranges)
ds_hh.to_netcdf('HIRHAM_T10m.nc')

# MAR
# if 'ds_mar_3413' not in locals():
#     ds_mar_3413 = ds_mar.rio.reproject("EPSG:3413")
ds_mar = ds_mar.rio.clip(ice.to_crs(ds_mar.rio.crs).geometry.values, ds_mar.rio.crs)
ds_mar['T10m'] = ds_mar.T10m.where(ds_mar.T10m > -200)
ds_mar_GrIS = (ds_mar.mean(dim=('x','y'))).to_pandas().T10m
plot_trend_analysis(ds_mar, ds_mar_GrIS, 'MAR', year_ranges)

# %% Firn area averages analysis
DSA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/DSA_MAR_4326.shp")
HAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/HAPA_MAR_4326.shp")
LAPA = gpd.GeoDataFrame.from_file("Data/misc/firn areas/LAPA_MAR_4326.shp")
firn = gpd.GeoDataFrame.from_file("Data/misc/firn areas/FirnLayer2000-2017_final_4326.shp")
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
def plot_selected_ds(ds, shape, ax, label, mask = 0):
    ds_select = ds.rio.clip(shape.to_crs(ds.rio.crs).geometry.values, 
                                                 ds.rio.crs,
                                                 all_touched=True)
    if label == 'ANN':
        col = CB_color_cycle[0]
    elif label == 'RACMO':
        col = CB_color_cycle[1]
    elif label == 'MAR':
        col = CB_color_cycle[2]
    elif label == 'HIRHAM':
        col = CB_color_cycle[3]
    else:
        col = CB_color_cycle[4]
        
    if mask:
        ds_select = ds.where(ds_select.T10m.isel(time=100).isnull())
    if label in ['HIRHAM','RACMO', 'ERA5 $T_{2m}$']:
        offset = -273.15
    else:
        offset = 0
    try:
        tmp = (ds_select.mean(dim=('x','y'))).to_pandas().T10m + offset
    except:
        weights = np.cos(np.deg2rad(ds_select.latitude))
        weights.name = "weights"
        tmp_w = ds_select.weighted(weights)
        tmp = (tmp_w.mean(dim=('latitude','longitude'))).to_pandas().T10m + offset
    
    tmp.plot(ax=ax, color='black', label='_no_legend_',alpha=0.3)
    tmp = tmp.resample('Y').mean()
    tmp.plot(ax=ax, label=label, drawstyle='steps-post', linewidth=3, color=col)

    # print(tmp.loc['1980-01-01':'1990-01-01'].mean())
    # print(tmp.loc['2010-01-01':'2020-01-01'].mean())
    # print(tmp.loc['1998-01-01':'2009-01-01'].mean())
    # print(tmp.loc['2010-01-01':'2018-01-01'].mean())

    tmp = tmp.loc['1991':]
    X = np.array([toYearFraction(d) for d in tmp.index])
    y = tmp.values
    
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()

    print('%s, %i-%i, %0.3f, %0.3f'% (label, X[0], X[-1], est2.params[1]*10, est2.pvalues[1]))

ds_era = xr.open_dataset('Data/ERA5/m_era5_t2m_all.nc').rename({'t2m':'T10m'})
ds_era = ds_era.rio.write_crs("epsg:4326")

fig, ax = plt.subplots(5,1, figsize=(10,9))
fig.subplots_adjust(left=0.3,right=0.99,top=0.95,bottom=0.03, hspace=0.25)
ax=ax.flatten()
print('bare ice')
plot_selected_ds(ds_ann, firn, ax[0], 'ANN', mask=1)
plot_selected_ds(ds_racmo, firn, ax[0], 'RACMO', mask=1)
plot_selected_ds(ds_mar, firn, ax[0], 'MAR', mask=1)
plot_selected_ds(ds_hh, firn,ax[0], 'HIRHAM', mask=1)
plot_selected_ds(ds_era, firn,ax[0], 'ERA5 $T_{2m}$', mask=1)
ax[0].set_title('B. Bare ice area', loc='left')
ax[0].legend(ncol=4, bbox_to_anchor=(1,1), loc="lower right",fontsize=11)
print('DSA')
plot_selected_ds(ds_ann, DSA, ax[1], 'ANN')
plot_selected_ds(ds_racmo, DSA, ax[1], 'RACMO')
plot_selected_ds(ds_mar, DSA, ax[1], 'MAR')
plot_selected_ds(ds_hh, DSA, ax[1], 'HIRHAM')
plot_selected_ds(ds_era, DSA, ax[1], 'ERA5 $T_{2m}$')
ax[1].set_title('C. Dry snow area', loc='left')
print('LAPA')
plot_selected_ds(ds_ann, LAPA, ax[2], 'ANN')
plot_selected_ds(ds_racmo, LAPA, ax[2], 'RACMO')
plot_selected_ds(ds_mar, LAPA, ax[2], 'MAR')
plot_selected_ds(ds_hh, LAPA, ax[2], 'HIRHAM')
plot_selected_ds(ds_era, LAPA, ax[2], 'ERA5 $T_{2m}$')
ax[2].set_title('D. Low accumulation percolation area', loc='left')
print('HAPA')
plot_selected_ds(ds_ann, HAPA, ax[3], 'ANN')
plot_selected_ds(ds_racmo, HAPA, ax[3], 'RACMO')
plot_selected_ds(ds_mar, HAPA, ax[3], 'MAR')
plot_selected_ds(ds_hh, HAPA, ax[3], 'HIRHAM')
plot_selected_ds(ds_era, HAPA, ax[3], 'ERA5 $T_{2m}$')
ax[3].set_title('E. High accumulation percolation area', loc='left')
print('All Greenland')
plot_selected_ds(ds_ann, ice, ax[4], 'ANN')
plot_selected_ds(ds_racmo, ice, ax[4], 'RACMO')
plot_selected_ds(ds_mar, ice, ax[4], 'MAR')
plot_selected_ds(ds_hh, ice, ax[4], 'HIRHAM')
plot_selected_ds(ds_era, ice, ax[4], 'ERA5 $T_{2m}$')
ax[4].set_title('F. All Greenland ice', loc='left')

ax[4].set_ylim(-29,-18)
ax[1].set_ylim(-29,-18)
ax[2].set_ylim(-19,-7)
ax[3].set_ylim(-19,-7)
ax[0].set_ylim(-19,-7)
for i in range(5):
    ax[i].set_ylim(-35,0)
    ax[i].set_xlim(pd.to_datetime('1954'),pd.to_datetime('2021'))
    ax[i].tick_params(axis='y', labelsize= 10.5)
    ax[i].grid()

    if i <4:
        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].set_xlabel('')
    else:
        ax[i].set_xlabel('Year', fontsize=11)
ax[2].set_ylabel('Average 10 m subsurface temperature ($^o$C)',fontsize=12)
ax[4].tick_params(axis='x', labelsize= 10.5)
ax_map = fig.add_axes([0.02, 0.32, 0.18, 0.7])
land.to_crs('EPSG:3413').plot(ax=ax_map, color='k')
ice.to_crs('EPSG:3413').plot(ax=ax_map, color='lightblue')
DSA.to_crs('EPSG:3413').plot(ax=ax_map, color='tab:blue')
LAPA.to_crs('EPSG:3413').plot(ax=ax_map, color='m')
HAPA.to_crs('EPSG:3413').plot(ax=ax_map, color='tab:red')
ax_map.axes.get_xaxis().set_ticks([])
ax_map.axes.get_yaxis().set_ticks([])
from matplotlib import patches as mpatches
h=[np.nan,np.nan,np.nan,np.nan,np.nan]
h[0] = mpatches.Patch(facecolor='k', label='Land')
h[1] = mpatches.Patch(facecolor='lightblue', label='Bare ice area')
h[2] = mpatches.Patch(facecolor='tab:blue', label='Dry snow area')
h[3] = mpatches.Patch(facecolor='m', label='Low accumulation\npercolation area')
h[4] = mpatches.Patch(facecolor='tab:red', label='High accumulation\npercolation area')
ax_map.legend(handles=h, bbox_to_anchor=(-0.1,-0.6), loc="lower left",fontsize=11)
ax_map.set_title('A. Ice sheet areas')
fig.subplots_adjust(left=0.3,right=0.99,top=0.95,bottom=0.07, hspace=0.25)
fig.savefig( 'figures/model_comp_ice_sheet_areas.png',dpi=300)


