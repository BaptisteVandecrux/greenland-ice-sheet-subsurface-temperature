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
import wandb

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
firn_memory = 4

for i in range(firn_memory):
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
        print(wtf)
    for k in range(firn_memory):
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
        
df['t2m_avg'] = df[['t2m_'+str(i) for i in range(firn_memory)]].mean(axis = 1)
df['sf_avg'] = df[['sf_'+str(i) for i in range(firn_memory)]].mean(axis = 1)

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
for i in range(firn_memory):
    df = df.loc[df['sf_'+str(i)].notnull(),:]
    df = df.loc[df['t2m_'+str(i)].notnull(),:]
    
# % Producing or loading input grids:
Predictors = ['t2m_amp', 't2m_avg', 'sf_avg'] + ['sf_'+str(i) for i in range(firn_memory)] + ['t2m_'+str(i) for i in range(firn_memory)]

produce_input_grid = 0

if produce_input_grid:
    print('Initializing array')
    for pred in Predictors:
        print(pred)
        ds_era[pred] = ds_era.t2m*np.nan
    for k in range(firn_memory):
        ds_era['t2m_'+str(k)] = ds_era.t2m*np.nan
    for k in range(firn_memory):
        ds_era['sf_'+str(k)] = ds_era.t2m*np.nan
    
    
    for time in progressbar.progressbar(ds_era.time.to_dataframe().values):
        for k in range(firn_memory):
            time_end = pd.to_datetime(time) +  pd.DateOffset(years=0-k) +  pd.DateOffset(days=-1)
            time_start = pd.to_datetime(time) +  pd.DateOffset(years=-1-k) 
            tmp = ds_era.sel(time=slice(time_start.values[0], time_end.values[0]))
            if tmp.t2m.shape[0] == 0:
                continue
            if tmp.time.shape[0]<12:
                continue
            if k == 0:
                ds_era['t2m_amp'].loc[dict(time=time)] = tmp.t2m.max(dim = 'time').values - tmp.t2m.min(dim = 'time').values
            ds_era['t2m_'+str(k)].loc[dict(time=time)] = tmp.t2m.mean(dim = 'time').values
            ds_era['sf_'+str(k)].loc[dict(time=time)] = tmp.sf.sum(dim = 'time').values
            
        ds_era['t2m_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['t2m_'+str(k) for k in range(firn_memory)]].to_array(dim='new').mean('new').values
        ds_era['sf_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['sf_'+str(k) for k in range(firn_memory)]].to_array(dim='new').mean('new').values
    ds_era.to_netcdf('era5_monthly_plus_predictors.nc')

ds_era = xr.open_dataset('era5_monthly_plus_predictors.nc')

# %% Representativity of the dataset

bins_temp = np.linspace(230, 275,15)
bins_sf = np.linspace(0, 0.15,15)
bins_amp = np.linspace(0, 50, 15)
pred_name = ['$T_{2m, amp.}$ (K)', 
             '$T_{2m, avg.}$ (last 4 years, K)', '$SF_{avg.}$ (last 4 years, m w.e.)']  \
        +['$SF_{avg.}$ (year-%i, m w.e.)'%(i+1) for i in range(firn_memory)]  \
            +['$T_{2m, avg.}$ (year-%i, K)'%(i+1) for i in range(firn_memory)]

# first calculating the target histograms (slow)
target_hist = [None] * len(Predictors) 
for i in range(11):
    if 't2m' in Predictors[i]:
        bins = bins_temp
    if 'amp' in Predictors[i]:
        bins = bins_amp
    if 'sf' in Predictors[i]:
        bins = bins_sf
    target_hist[i], _ = np.histogram(ds_era[Predictors[i]].values, bins = bins)
    target_hist[i] = target_hist[i].astype(np.float32) / target_hist[i].sum()

# %%plotting (less slow)
for k in range(2):
    fig, ax = plt.subplots(4,3, figsize=(10,8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, wspace = 0.25, hspace = 0.35)
    ax=ax.flatten()
    for i in range(len(Predictors)):
        if 't2m' in Predictors[i]:
            bins = bins_temp
        if 'amp' in Predictors[i]:
            bins = bins_amp
        if 'sf' in Predictors[i]:
            bins = bins_sf
    
        hist1 = target_hist[i]
        ax[i].bar(bins[:-1], hist1, width=(bins[1]-bins[0]),
                    alpha= 0.5, color='tab:blue',
                    edgecolor = 'lightgray',
                    label='ERA5 values for the entire ice sheet')
        if k == 0:
            hist2, _ = np.histogram(df[Predictors[i]].values, bins = bins)
        elif k == 1:
            hist2, _ = np.histogram(df[Predictors[i]].values, bins = bins, weights = df['weights'].values)

        hist2 =  hist2.astype(np.float32) / hist2.sum()
        ax[i].bar(bins[:-1][(hist1!=0)&(hist2==0)], hist1[(hist1!=0)&(hist2==0)], width=(bins[1]-bins[0]),
                    alpha= 0.8, color='tab:blue',
                    edgecolor = 'gray', hatch='/',
                    label='bins where no training data is available')
        ax[i].bar(bins[:-1],hist2, width=(bins[1]-bins[0]),
                    alpha= 0.5, color='tab:orange',
                    edgecolor = 'lightgray',
                    label='ERA5 values at the observation sites')
    
        
        weights_bins =  0*hist1
        weights_bins[hist2!=0] = hist1[hist2!=0]/hist2[hist2!=0]
        ind = np.digitize(df[Predictors[i]].values, bins)
        df[Predictors[i]+'_w'] = weights_bins[ind-1]
        ax[i].annotate('MRD: %0.2f'%np.mean((hist1[hist2!=0] - hist2[hist2!=0])/hist1[hist2!=0]),
                       xy=(0.7, 0.87),  xycoords='axes fraction')

        ax[i].set_xlabel(pred_name[i])
        ax[i].set_ylabel('Probability (-)')
        ax[i].set_title('')
    ax[i+1].set_axis_off() 
    if k == 0:
        ttl = 'Without weights'
    else:
        ttl = 'With weights'
        
    ax[i].legend(loc='upper right', bbox_to_anchor=(2.3,0.9), title=ttl)
    fig.savefig('figures/histograms_'+str(k)+'.png')
    if k == 0:
        df['weights'] = df[[p+'_w' for p in Predictors]].mean(axis=1)
    

# %% ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_ANN(df, Predictors, TargetVariable = 'temperatureObserved', calc_weights = False):
    if calc_weights:
        for i in range(len(Predictors)):
            hist1 = target_hist[i]
            hist2, _ = np.histogram(df[Predictors[i]].values, bins = bins)
            weights_bins =  0*hist1
            weights_bins[hist2!=0] = hist1[hist2!=0]/hist2[hist2!=0]
            ind = np.digitize(df[Predictors[i]].values, bins)
            df[Predictors[i]+'_w'] = weights_bins[ind-1]
        df['weights'] = df[[p+'_w' for p in Predictors]].mean(axis=1)
    
    w = df['weights'].values
    X=df[Predictors].values
    y=df[TargetVariable].values.reshape(-1, 1)
    
    # Sandardization of data   
    PredictorScalerFit=StandardScaler().fit(X)
    TargetVarScalerFit=StandardScaler().fit(y)
    
    X=PredictorScalerFit.transform(X)
    y=TargetVarScalerFit.transform(y)
    
    
    # create and fit ANN model
    from keras.layers import GaussianNoise
    model = Sequential()
    model.add(Dense(units=40, input_dim=len(Predictors)))
    model.add(GaussianNoise(0.1))
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, batch_size=30, epochs=30, verbose=0, sample_weight=w)
    return model, PredictorScalerFit, TargetVarScalerFit

model, PredictorScalerFit, TargetVarScalerFit = train_ANN(df, Predictors, TargetVariable = 'temperatureObserved', calc_weights = False)

# %% spatial cross-validation:
print('fitting cross-validation models')
zwally = gpd.GeoDataFrame.from_file("Data/misc/Zwally_10_zones_3413.shp")
zwally = zwally.set_crs('EPSG:3413').to_crs('EPSG:4326')
zwally = zwally.drop(columns=['name'])
df = df.reset_index(drop=True)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude)).set_crs(4326)
points_within = gpd.sjoin(gdf, zwally, op='within')
df['zwally_zone'] = np.nan
model_list = [None] * zwally.shape[0]

for i in range(zwally.shape[0]):
    msk = points_within.loc[points_within.index_right == i, :].index
    df.loc[msk, 'zwally_zone'] = i
    print(i, len(msk), '%0.2f'%(len(msk)/df.shape[0]*100))
    
    df_cv = df.loc[df.zwally_zone != i,].copy()
    
    model_list[i], _, _ = train_ANN(df_cv, Predictors, TargetVariable = 'temperatureObserved', calc_weights = True)

# defining function that makes the ANN estimate from predictor raw values
df = df.sort_values('date')
def ANN_model(df, model):
    X=PredictorScalerFit.transform(df.values)
    Y=model.predict(X)
    Predictions = pd.DataFrame(data = TargetVarScalerFit.inverse_transform(Y), index = df.index, columns=['temperaturePredicted']).temperaturePredicted
    Predictions.loc[Predictions>0] = 0
    return Predictions
    

def ANN_model_cv(df_pred, model_list):
    pred = pd.DataFrame() 
    for i in range(zwally.shape[0]):
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
    df_select = df.loc[df.site ==site,:].copy()
    df_select['cv_mean'], df_select['cv_std'] = ANN_model_cv(df_select[Predictors], model_list)


    best_model = ANN_model(df_select[Predictors], model)
    ax[k].fill(np.append(df_select.date, df_select.date[::-1]), 
               np.append(best_model-df_select['cv_std'],
                         (best_model + df_select['cv_std'])[::-1]), 
               'grey', label='CV standard deviation')
    for i in range(zwally.shape[0]):
        if i == 0:
            lab = 'CV models'
        else:
            lab = '_no_legend_'
        ax[k].plot(df_select.date, ANN_model(df_select[Predictors], model_list[i]), 
                 '-', color = 'lightgray', alpha = 0.8, label=lab)
    ax[k].plot(df_select.date, best_model, '-', label='best model')
        
    ax[k].plot(df_select.date,df_select.temperatureObserved, 'x', linestyle='None', label='observations')
    ax[k].set_title(site)
ax[k].legend()

# %% Predicting T10m
predict = 0
if predict:
    print('predicting T10m over entire ERA5 dataset')
    ds_T10m = ds_era['t2m'].copy().rename('T10m')*np.nan
    for time in progressbar.progressbar(ds_T10m.time.to_dataframe().values[12*firn_memory:]):
        tmp = ds_era.sel(time=time).to_dataframe()
        tmp['month'] = (pd.to_datetime(time[0]).month-1)/12
        out = ANN_model(tmp[Predictors], model)
        ds_T10m.loc[dict(time=time)] = out.to_frame().to_xarray()['temperaturePredicted'].transpose("time", "latitude", "longitude").values
    
    ds_T10m.to_netcdf('predicted_T10m.nc')
ds_T10m = xr.open_dataset('predicted_T10m.nc')['T10m']

# %% Predicting T10m uncertainty
predict = 0
if predict:
    print('predicting T10m uncertainty over entire ERA5 dataset')
    ds_T10m_std = ds_era['t2m'].copy().rename('T10m')*np.nan
    for time in progressbar.progressbar(ds_T10m_std.time.to_dataframe().values[12*firn_memory:]):
        tmp = ds_era.sel(time=time).to_dataframe()
        tmp['month'] = (pd.to_datetime(time[0]).month-1)/12
        _, tmp['cv_std'] = ANN_model_cv(tmp[Predictors], model_list)

        ds_T10m_std.loc[dict(time=time)] = tmp['cv_std'].to_frame().to_xarray()['cv_std'].transpose("time", "latitude", "longitude").values
    
    ds_T10m_std.to_netcdf('predicted_T10m_std.nc')
ds_T10m_std = xr.open_dataset('predicted_T10m_std.nc')['T10m']

# ========= Supporting plots ============
# %% Selection and overview of the predictors
ds_era_GrIS = ds_era.mean(dim=('latitude','longitude')).to_pandas()

fig, ax = plt.subplots(4,3, figsize=(15,15), sharex=True)
ax=ax.flatten()
for i in range(len(Predictors)):
    ds_era_GrIS[Predictors[i]].resample('Y').mean().plot(ax=ax[i],drawstyle='steps')
    ax[i].set_ylabel(Predictors[i])

ds_era_GrIS['t2m'].resample('Y').mean().plot(ax=ax[i+1],drawstyle='steps')
ax[i+1].set_ylabel('t2m')

# %% overview of the cross validation analysis
ds_era = xr.open_dataset('era5_monthly_plus_predictors.nc')
ds_T10m = xr.open_dataset('predicted_T10m.nc')['T10m']
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:4326")
ds_era = ds_era.rio.write_crs('EPSG:4326').rio.clip(ice.geometry.values, ice.crs)

ds_T10m_std_avg = ds_T10m_std.mean(dim='time').rio.write_crs('EPSG:4326').rio.clip(ice.geometry.values, ice.crs)
ds_T10m_std_avg = ds_T10m_std_avg.rio.reproject("EPSG:3413")

fig, ax= plt.subplots(1,1)
land.plot(color='k',ax=ax)
im = ds_T10m_std_avg.where(ds_T10m_std_avg<1000).plot(ax=ax, cbar_kwargs={'label': 'Average uncertainty of the ANN ($^o$C)'})
gpd.GeoDataFrame.from_file("Data/misc/Zwally_10_zones_3413.shp").boundary.plot(ax=ax, color = 'w', linestyle='-', linewidth=0.5)
ax.set_axis_off()
plt.title('')
plt.xlim(-700000, 900000)
plt.ylim(-3400000, -574000)
plt.savefig("uncertainty_map.png", bbox_inches='tight')
plt.show()