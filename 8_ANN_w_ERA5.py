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
import rasterio as rio
# import GIS_lib as gis
import progressbar
import matplotlib


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

land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

dates = pd.DatetimeIndex(df.date)

years = pd.DatetimeIndex(dates).year.unique().values
years.sort()
df = df.reset_index(drop=True)

# Extracting ERA5 data

ds_era = xr.open_dataset('Data/ERA5/ERA5_monthly_temp_snowfall.nc')

time_era = ds_era.time

for i in range(6):
    df['t2m_'+str(i)] = np.nan
    df['sf_'+str(i)] = np.nan
    
df['time_era'] = np.nan
df['t2m_amp'] = np.nan

coords_uni = np.unique(df[['latitude','longitude']].values, axis=0)
print('interpolating ERAat the observation sites')
x = xr.DataArray(coords_uni[:,0],dims='points')
y = xr.DataArray(coords_uni[:,1],dims='points')
ds_interp = ds_era.interp(latitude=x, longitude=y, method="linear")

for i in progressbar.progressbar(range(df.shape[0])):
    query_point = (df.iloc[i,:].latitude,  df.iloc[i,:].longitude)
    index_point = np.where((coords_uni ==  query_point).all(axis=1))[0][0]
    tmp = ds_interp.isel(points = index_point)
    if ((tmp.latitude.values, tmp.longitude.values) != query_point):
        print('whf')
    for k in range(6):
        if (pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=0-k)) < tmp.time.min():
            continue
        time_end = tmp.sel(time = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=0-k), method='ffill').time.values
        time_start = tmp.sel(time = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=-1-k) +  pd.DateOffset(days=1), method='bfill').time.values
        
        if k ==0:
            df.iloc[i, df.columns.get_loc('t2m_amp')] = tmp.sel(time=slice(time_start, time_end)).t2m.max().values - tmp.sel(time=slice(time_start, time_end)).t2m.min().values
        df.iloc[i, df.columns.get_loc('t2m_'+str(k))] = tmp.sel(time=slice(time_start, time_end)).t2m.mean().values
        df.iloc[i, df.columns.get_loc('sf_'+str(k))] = tmp.sel(time=slice(time_start, time_end)).sf.sum().values
        if k == 0:
            df.iloc[i, df.columns.get_loc('time_era')] = tmp.sel(time=slice(time_start, time_end)).time.max().values
        
df['t2m_avg'] = df[['t2m_'+str(i) for i in range(6)]].mean(axis = 1)
df['sf_avg'] = df[['sf_'+str(i) for i in range(6)]].mean(axis = 1)

fig,ax = plt.subplots(2,2)
ax = ax.flatten()
ax[0].plot(df.t2m_0, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[1].plot(df.sf_0, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[2].plot(df.t2m_avg, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)
ax[3].plot(df.sf_avg, df.temperatureObserved,marker='o',linestyle='None',markersize=1.5)

df = df.loc[df.time_era.notnull(),:]
df['date'] = pd.to_datetime(df.date)
df['time_era'] = pd.to_datetime(df.time_era)
df['year'] = df['date'].dt.year
df['month'] = (df['date'].dt.month-1)/12

df = df.loc[df.t2m_avg.notnull(),:]
df = df.loc[df.sf_avg.notnull(),:]
for i in range(6):
    df = df.loc[df['sf_'+str(i)].notnull(),:]
    df = df.loc[df['t2m_'+str(i)].notnull(),:]
    
# %% ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TargetVariable=['temperatureObserved']
Predictors = ['t2m_amp', 't2m_avg', 'sf_avg'] + ['sf_'+str(i) for i in range(6)] + ['t2m_'+str(i) for i in range(6)]
 
X=df[Predictors].values
y=df[TargetVariable].values

# Sandardization of data
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)

X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and fit ANN model
model = Sequential()
model.add(Dense(units=40, input_dim=len(Predictors), kernel_initializer='normal', activation='relu'))
model.add(Dense(units=40, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, batch_size=30, epochs=30, verbose=0)

# bootstrapping:
model_list = [model, model, model, model, model, model, model, model, model, model]
print('fitting bootstrapped models')
np.random.seed(0)
random_index = np.random.permutation(X.shape[0])

for i in range(10):
    # removing consecutive samples
    ind_min = int(i*X.shape[0]/10)
    ind_max = min(int((i+1)*X.shape[0]/10),X.shape[0])
    print(i+1,'/10')
    msk = (np.arange(X.shape[0])<ind_min) | (np.arange(X.shape[0])>ind_max)
    
    # removing random samples
    msk = random_index[msk]
    
    X_bs = X[msk, :]
    y_bs = y[msk, :]
    model_list[i] = Sequential()
    model_list[i].add(Dense(units=40, input_dim=len(Predictors), kernel_initializer='normal', activation='relu'))
    model_list[i].add(Dense(units=40, kernel_initializer='normal', activation='relu'))
    model_list[i].add(Dense(1, kernel_initializer='normal'))
    model_list[i].compile(loss='mean_squared_error', optimizer='adam')
    model_list[i].fit(X_bs, y_bs, batch_size=30, epochs=30, verbose=0)

Predictions = model.predict(X_test)
Predictions = TargetVarScalerFit.inverse_transform(Predictions)

y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
Test_Data = PredictorScalerFit.inverse_transform(X_test)
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['temperatureObserved'] = y_test_orig
TestingData['temperaturePredicted']=Predictions
TestingData.head()

# Computing the absolute  error
AE = abs(TestingData['temperatureObserved']-TestingData['temperaturePredicted'])
TestingData['APE'] = AE

print('The MAE of ANN model is:', np.mean(AE))
TestingData.head()

# defining function that makes the ANN estimate from predictor raw values
df = df.sort_values('date')
def ANN_model(df, model):
    X=PredictorScalerFit.transform(df.values)
    Y=model.predict(X)
    Predictions = pd.DataFrame(data = TargetVarScalerFit.inverse_transform(Y), index = df.index, columns=['temperaturePredicted']).temperaturePredicted
    Predictions.loc[Predictions>0] = 0
    return Predictions
    

def ANN_model_bs(df_pred, model_list):
    pred = pd.DataFrame() 
    for i in range(10):
        pred['T10m_pred_'+str(i)] = ANN_model(df_pred, model_list[i])
    df_mean = pred.mean(axis=1)
    df_std = pred.std(axis=1)
    return df_mean.values, df_std.values


plt.figure()
plt.plot(df.temperatureObserved,
         ANN_model(df[Predictors], model),
         'o',linestyle= 'None')
ME = np.mean(df.temperatureObserved - ANN_model(df[Predictors], model))
RMSE = np.sqrt(np.mean((df.temperatureObserved - ANN_model(df[Predictors], model))**2))
plt.title('N = %i ME = %0.2f RMSE = %0.2f'%(len(df.temperatureObserved), ME, RMSE))

fig, ax = plt.subplots(2,2, figsize=(15,15))
ax = ax.flatten()
for k, site in enumerate(['DYE-2', 'Summit', 'KAN_U','EKT']):
    df_select = df.loc[df.site ==site,:]
    df_select['bs_mean'], df_select['bs_std'] = ANN_model_bs(df_select[Predictors], model_list)

    for i in range(10):
        ax[k].plot(df_select.date, ANN_model(df_select[Predictors], model_list[i]), 
                 '-', color = 'lightgray', alpha = 0.5)
    ax[k].plot(df_select.date, ANN_model(df_select[Predictors], model), 'o-', label='best model')
    ax[k].plot(df_select.date, df_select['bs_mean'], 'o-', label='average of bootstrapped models')
        
    ax[k].plot(df_select.date,df_select.temperatureObserved, 'x', linestyle='None', label='observations')
    ax[k].set_title(site)
ax[k].legend()

# saving model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# %% Producing input grids:
produce_input_grid = 0

if produce_input_grid:
    print('Initializing array')
    for pred in Predictors:
        print(pred)
        ds_era[pred] = ds_era.t2m*np.nan
    for k in range(6):
        ds_era['t2m_'+str(k)] = ds_era.t2m*np.nan
    for k in range(6):
        ds_era['sf_'+str(k)] = ds_era.t2m*np.nan
    
    
    for time in progressbar.progressbar(ds_era.time.to_dataframe().values):
        for k in range(6):
            time_end = pd.to_datetime(time) +  pd.DateOffset(years=0-k) +  pd.DateOffset(days=-1)
            time_start = pd.to_datetime(time) +  pd.DateOffset(years=-1-k) 
            tmp = ds_era.sel(time=slice(time_start.values[0], time_end.values[0]))
            if tmp.t2m.shape[0] == 0:
                continue
            if k == 0:
                ds_era['t2m_amp'].loc[dict(time=time)] = tmp.t2m.mean(dim = 'time').values - tmp.t2m.min(dim = 'time').values
            ds_era['t2m_'+str(k)].loc[dict(time=time)] = tmp.t2m.mean(dim = 'time').values
            ds_era['sf_'+str(k)].loc[dict(time=time)] = tmp.sf.sum(dim = 'time').values
            
        ds_era['t2m_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['t2m_'+str(k) for k in range(6)]].to_array(dim='new').mean('new').values
        ds_era['sf_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['sf_'+str(k) for k in range(6)]].to_array(dim='new').mean('new').values
    
    ds_era.to_netcdf('era5_monthly_plus_predictors.nc')

ds_era = xr.open_dataset('era5_monthly_plus_predictors.nc')

# % Predicting T10m
predict = 0
if predict:
    print('predicting T10m over entire ERA5 dataset')
    ds_T10m = ds_era['t2m'].copy().rename('T10m')*np.nan
    for time in progressbar.progressbar(ds_T10m.time.to_dataframe().values[12*5:]):
        tmp = ds_era.sel(time=time).to_dataframe()
        tmp['month'] = (pd.to_datetime(time[0]).month-1)/12
        out = ANN_model(tmp[Predictors], model)
        ds_T10m.loc[dict(time=time)] = out.to_frame().to_xarray()['temperaturePredicted'].transpose("time", "latitude", "longitude").values
    
    ds_T10m.to_netcdf('predicted_T10m.nc')
ds_T10m = xr.open_dataset('predicted_T10m.nc')['T10m']

#%% Analysis at specific sites
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
ds_T10m['time'] = pd.to_datetime(ds_T10m.time, utc=True)
ds_era['time'] = pd.to_datetime(ds_era.time, utc=True)
df.date = pd.to_datetime(df.date,utc=True)
plt.close('all')
fig, ax = plt.subplots(3,4, figsize=(15,15))
fig.subplots_adjust(left=0.05, right=0.99, top=0.95)
ax = ax.flatten()
for i, site in enumerate(site_list.index):
    coords = np.expand_dims(site_list.loc[site, ['lat','lon']].values.astype(float),0)
    dm = cdist(all_points, coords, get_distance)
    df_select = df.loc[dm<5,:]
    lat = df_select.latitude.values[0]
    lon = df_select.longitude.values[0]
    
    df_era = ds_era.sel(latitude = lat, longitude = lon, method = 'nearest').to_dataframe()
    df_era['month'] = (df_era.index.month - 1)/12

    predicted = ds_T10m.sel(latitude=lat, longitude=lon, method='nearest').to_dataframe().T10m
    predicted_y = predicted.resample('Y').mean()
    
    bs_out = ANN_model_bs(df_era[Predictors], model_list)
    ax[i].fill_between(df_era.index, bs_out[0]-bs_out[1], bs_out[0]+bs_out[1], color = 'lightgray')
    ax[i].plot(df_era.index, ANN_model(df_era[Predictors], model),  label='predicted, monthly')
    # ax[i].plot(predicted.index, predicted.values, marker='o', markersize=5, label='predicted, monthly')
    # ax[i].step(predicted_y.index, predicted_y.values, label='predicted, yearly')

    ax[i].plot(df_select.date,df_select.temperatureObserved, 'o',
               markersize=5, linestyle='None', label = 'observations')
    # ax[i].set_ylim(np.nanmean(bs_out[0])-4, np.nanmean(bs_out[0])+4)
    ax[i].set_title(site)
plt.legend()

# %% Converting to epsg:3413 and cropping to ice extent
ds_T10m['time'] = pd.to_datetime(ds_T10m.time)
ds_T10m_y = ds_T10m.resample(time='Y').mean()

ds_T10m = ds_T10m.to_dataset(name='T10m')


ds_T10m = ds_T10m.rio.write_crs('epsg:4326')
ds_T10m = ds_T10m.rio.reproject("EPSG:3413")

ds_T10m = ds_T10m.rio.clip(ice.geometry.values, ice.crs)
ds_T10m['T10m'] = ds_T10m.T10m.where(ds_T10m.T10m!=3.4028234663852886e+38)
ds_T10m.T10m.resample(time='Y').mean().to_netcdf('predicted_T10m_y.nc')

T10m_GrIS = (ds_T10m.mean(dim=('y','x'))).to_pandas().T10m
from pandas.tseries.frequencies import to_offset

plt.figure()
T10m_GrIS.plot()
T10m_GrIS_y = T10m_GrIS.resample('Y').mean()
T10m_GrIS_y.plot(drawstyle='steps-post')

# %% 
from datetime import datetime as dt

def toYearFraction(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def calculate_trend(ds_T10m, year_start,year_end):
    ds_T10m_period = ds_T10m.loc[dict(time=slice(str(year_start)+"-01-01", str(year_end)+"-12-31"))]
    
    vals = ds_T10m_period['T10m'].values 
    time = np.array([toYearFraction(d) for d in pd.to_datetime(ds_T10m_period.time.values)])
    # Reshape to an array with as many rows as years and as many columns as there are pixels
    vals2 = vals.reshape(len(time), -1)
    # Do a first-degree polyfit
    regressions = np.nan * vals2[:2,:]
    ind_nan = np.all(np.isnan(vals2), axis=0)
    regressions[:,~ind_nan] = np.polyfit(time, vals2[:,~ind_nan], 1)
    # Get the coefficients back
    trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
    dims = [key for key in ds_T10m_period.dims.keys() if key!='time']
    return (dims,  trends)

# Determining the breakpoints:
    
# import jenkspy
# T10m_GrIS_y = T10m_GrIS_y.loc['1957':]
# breaks = jenkspy.jenks_breaks(T10m_GrIS_y.values, nb_class=3)
# breaks_jkp = []
# for v in breaks:
#     idx = T10m_GrIS_y.index[T10m_GrIS_y == v]
#     breaks_jkp.append(idx)
# breaks_jkp = [k.year.values[0] for k in breaks_jkp]
# year_ranges = np.array([[breaks_jkp[0], breaks_jkp[-1]],
#                         [breaks_jkp[0], breaks_jkp[1]], 
#                         [breaks_jkp[1], breaks_jkp[2]],
#                         [breaks_jkp[2], breaks_jkp[3]]])
# import ruptures as rpt
# model = rpt.Dynp(model="l1",min_size=5)
# model.fit(np.array(T10m_GrIS_y.tolist()))
# n_breaks=2
# breaks = model.predict(n_bkps=n_breaks-1)
# breaks_rpt = []
# for i in breaks:
#     breaks_rpt.append(T10m_GrIS_y.index[i-1])
# breaks_rpt = pd.to_datetime(breaks_rpt)
# year_ranges = np.array([[T10m_GrIS_y.index[0], breaks_rpt[0]],
#                         [T10m_GrIS_y.index[0], breaks_rpt[1]], 
#                         [breaks_rpt[1], T10m_GrIS_y.index[-1]],
#                         [T10m_GrIS_y.index[0], T10m_GrIS_y.index[-1]]])

year_ranges = np.array([[1956, 1992],
                        [1992, 2013], 
                        [2013, 2022],
                        [1956, 2022]])
for i in range(4):
    ds_T10m["trend_"+str(i)] = calculate_trend(ds_T10m, year_ranges[i][0],year_ranges[i][1])


#% Plotting trend analysis
import statsmodels.api as sm

import rioxarray # for the extension to load
import xarray
import rasterio
ABC = 'ABCDEF'
fig, ax = plt.subplots(2,2,figsize=(7,20))
plt.subplots_adjust(hspace = 0.1,wspace=0.1, right=0.8, left = 0, bottom=0.05, top = 0.8)
ax = ax.flatten()

for i in range(4):
    land.plot(ax=ax[i], zorder=0, color="black")
    print(i, year_ranges[i][0], year_ranges[i][1])
    vmin = -0.2; vmax = 0.2
        
    im = ds_T10m['trend_'+str(i)].plot(ax=ax[i],vmin = vmin, vmax = vmax,
                                        cmap='coolwarm',add_colorbar=False)
                
    ax[i].set_title(ABC[i+1]+'. '+str(year_ranges[i][0])+'-'+str(year_ranges[i][1]),fontweight = 'bold')
    if i in [1,3]:
        if i == 1:
            cbar_ax = fig.add_axes([0.85, 0.07, 0.03, 0.7])
        cb = plt.colorbar(im, ax = ax[i], cax=cbar_ax)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label('Trend in 10 m subsurface temperature ($^o$C yr $^{-1}$)', rotation=270)

    ax[i].set_xlim(-700000.0, 900000.0)
    ax[i].set_ylim(-3400000, -600000)
    ax[i].set_axis_off()
    
ax_bot = fig.add_axes([0.2, 0.85, 0.75, 0.12])
ax_bot.set_title(ABC[0]+'. Ice-sheet-wide average',fontweight = 'bold')
ax_bot.plot(T10m_GrIS.index, T10m_GrIS,color='lightgray')
ax_bot.step(T10m_GrIS_y.index, T10m_GrIS_y)

for r in range(4):
    tmp = T10m_GrIS_y.loc[str(year_ranges[r,0]):str (year_ranges[r,1])]
    X = np.array([toYearFraction(d) for d in tmp.index])
    y = tmp.values
    
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    x_plot = np.array([np.nanmin(X), np.nanmax(X)])
    if r ==3:
        linestyle = '--'
    else:
        linestyle = '-'
        
    ax_bot.plot([tmp.index.min(), tmp.index.max()], 
                est2.params[0] + est2.params[1] * x_plot,
                color = 'tab:red',
                linestyle=linestyle)
    print(str (year_ranges[r,0])+'-'+ str(year_ranges[r,1])+' slope (p-value): %0.3f (%0.3f)'% (est2.params[1], est2.pvalues[1]))
    
ax_bot.autoscale(enable=True, axis='x', tight=True)
ax_bot.set_ylabel('10 m subsurface \ntemperature ($^o$C yr $^{-1}$)')
fig.savefig( 'figures/ANN_trend_map.png',dpi=300)

# %% Supporting plots
ds_era_GrIS = ds_era.mean(dim=('latitude','longitude')).to_pandas()

fig, ax = plt.subplots(4,4, figsize=(15,15))
# (ds_era_GrIS.t2m-273.15).plot(ax=ax[0])
(ds_era_GrIS.t2m.resample('Y').mean()-273.15).plot(ax=ax[0],drawstyle='steps')
(ds_era_GrIS.t2m.resample('5Y').mean()-273.15).plot(ax=ax[0],drawstyle='steps')
# ds_era_GrIS.sf.plot(ax=ax[1])
ds_era_GrIS.sf.resample('Y').mean().plot(ax=ax[1],drawstyle='steps')
ds_era_GrIS.sf.resample('5Y').mean().plot(ax=ax[1],drawstyle='steps')
ds_era_GrIS.t2m_amp.resample('Y').mean().plot(ax=ax[2],drawstyle='steps')
ds_era_GrIS.t2m_amp.resample('5Y').mean().plot(ax=ax[2],drawstyle='steps')
