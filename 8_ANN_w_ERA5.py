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

land = gpd.GeoDataFrame.from_file("Data/misc/Ice-free_GEUS_GIMP.shp")
land = land.to_crs("EPSG:3413")

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

for i in progressbar.progressbar(range(df.shape[0])):
    tmp = ds_era.sel(latitude = df.iloc[i,:].latitude, 
                  longitude = df.iloc[i,:].longitude, method= 'nearest')
    for k in range(6):
        if (pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=0-k)) < tmp.time.min():
            continue
        time_end = tmp.sel(time = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=0-k), method='ffill').time.values
        time_start = tmp.sel(time = pd.to_datetime(df.iloc[i,:].date) +  pd.DateOffset(years=-1-k) +  pd.DateOffset(days=1), method='bfill').time.values
        # print(tmp.sel(time=slice(time_start, time_end)).t2m.shape)
        df.iloc[i, df.columns.get_loc('t2m_'+str(k))] = tmp.sel(time=slice(time_start, time_end)).t2m.mean().values
        df.iloc[i, df.columns.get_loc('sf_'+str(k))] = tmp.sel(time=slice(time_start, time_end)).sf.sum().values
        if k == 0:
            df.iloc[i, df.columns.get_loc('time_era')] = tmp.sel(time=slice(time_start, time_end)).time.max().values
        
df['t2m_avg'] = df[['t2m_'+str(i) for i in range(6)]].mean(axis = 1)
df['sf_avg'] = df[['sf_'+str(i) for i in range(6)]].mean(axis = 1)
#%% plot
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
# Separate Target Variable and Predictor Variables
TargetVariable=['temperatureObserved']
Predictors=['month'] +['t2m_avg', 'sf_avg']# ['sf_'+str(i) for i in range(6)] + ['t2m_'+str(i) for i in range(6)] + ['t2m_avg', 'sf_avg']
 
X=df[Predictors].values
y=df[TargetVariable].values
 
### Sandardization of data ###
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)
 
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# create ANN model
model = Sequential()
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=40, input_dim=len(Predictors), kernel_initializer='normal', activation='relu'))
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=40, kernel_initializer='normal', activation='relu'))
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=0)

# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[5, 10, 15, 20]
    epoch_list  =   [5, 10, 50, 100]
    
    import pandas as pd
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))

            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)

            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)


######################################################
# Calling the function
# ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)

# %matplotlib inline
# ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')
# best accuracy: batch_size: 20 - epochs: 10 Accuracy: 104.99074018952209
# %%
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 30, epochs = 10, verbose=0)

# Generating Predictions on testing data
Predictions=model.predict(X_test)

# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)


# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)

TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['temperatureObserved']=y_test_orig
TestingData['temperaturePredicted']=Predictions
TestingData.head()

# Computing the absolute percent error
APE=100*(abs(TestingData['temperatureObserved']-TestingData['temperaturePredicted'])/TestingData['temperatureObserved'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.head()

# %% 
df = df.sort_values('date')
def ANN_model(df):
    X=PredictorScalerFit.transform(df.values)
    Y=model.predict(X)
    Predictions = pd.DataFrame(data = TargetVarScalerFit.inverse_transform(Y), index = df.index, columns=['temperaturePredicted']).temperaturePredicted
    Predictions.loc[Predictions>0] = 0
    return Predictions
    
plt.figure()
plt.plot(df.temperatureObserved,
         ANN_model(df[Predictors]),
         'o',linestyle= 'None')
ME = np.mean(df.temperatureObserved - ANN_model(df[Predictors]))
RMSE = np.sqrt(np.mean((df.temperatureObserved - ANN_model(df[Predictors]))**2))
plt.title('N = %i ME = %0.2f RMSE = %0.2f'%(len(df.temperatureObserved), ME, RMSE))

#%%
for site in ['DYE-2', 'Summit', 'KAN_U','EKT']:
    df_select = df.loc[df.site ==site,:]
    plt.figure()
    plt.plot(df_select.date,df_select.temperatureObserved, 'o', linestyle='None')
    
    Predictions=TargetVarScalerFit.inverse_transform(model.predict(PredictorScalerFit.transform(df_select[Predictors].values)))
    plt.plot(df_select.date, Predictions, 'o-')
    plt.title(site)
    
# %% Producing input grids:
print('Initializing array')
for pred in Predictors[1:]:
    print(pred)
    ds_era[pred] = ds_era.t2m*np.nan

for time in progressbar.progressbar(ds_era.time.to_dataframe().values):
    for k in range(6):
        time_end = pd.to_datetime(time) +  pd.DateOffset(years=0-k) +  pd.DateOffset(days=-1)
        time_start = pd.to_datetime(time) +  pd.DateOffset(years=-1-k) 
        tmp = ds_era.sel(time=slice(time_start.values[0], time_end.values[0]))
        ds_era['t2m_'+str(k)].loc[dict(time=time)] = tmp.t2m.mean(dim = 'time').values
        ds_era['sf_'+str(k)].loc[dict(time=time)] = tmp.sf.sum(dim = 'time').values
        
    ds_era['t2m_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['t2m_'+str(k) for k in range(6)]].to_array(dim='new').mean('new').values
    ds_era['sf_avg'].loc[dict(time=time)] = ds_era.sel(time=time)[['sf_'+str(k) for k in range(6)]].to_array(dim='new').mean('new').values
    # for pred in Predictors[1:]:
    #     plt.figure()
    #     ds_era.sel(time=time)[pred].plot()
        
    #     plt.figure()
    #     ds_era.sel(time=time)['t2m_'+str(k)].plot()
ds_era.to_netcdf('era5_monthly_plus_predictors.nc')

# ds_era = xr.open_dataset('era5_monthly_plus_predictors.nc')

# % Predicting T10m
print('predicting T10m over entire ERA5 dataset')
ds_T10m = ds_era['t2m'].copy().rename('T10m')*np.nan
for time in progressbar.progressbar(ds_T10m.time.to_dataframe().values[12*5:]):
    tmp = ds_era.sel(time=time).to_dataframe()
    tmp['month'] = (pd.to_datetime(time[0]).month-1)/12
    out = ANN_model(tmp[Predictors])
    ds_T10m.loc[dict(time=time)] = out.to_frame().to_xarray()['temperaturePredicted'].transpose("time", "latitude", "longitude").values

ds_T10m.to_netcdf('predicted_T10m.nc')
ds_T10m = xr.open_dataset('predicted_T10m.nc')['T10m']
# %% 
# for i in range(5*12,5*12+40):
#     fig = plt.figure()
#     ds_T10m.isel(time=i).plot(add_colorbar=True, vmin = -30, vmax = 0)
#     fig.savefig('T10m_pred_'+str(i))

T10m_GrIS_p = (ds_T10m.mean(dim=('latitude','longitude'))).to_pandas()

plt.figure()
T10m_GrIS_p.plot()
#%%
for site in ['DYE-2', 'Summit', 'KAN_U','EKT']:
    df_select = df.loc[df.site ==site,:]
    lat = df_select.latitude.values[0]
    lon = df_select.longitude.values[0]
    
    # df_era = ds_era.sel(latitude = lat, longitude = lon, method = 'nearest').to_dataframe()
    
    # for time in progressbar.progressbar(df_era.index):
    #     for k in range(6):
    #         time_end = pd.to_datetime(time) +  pd.DateOffset(years=0-k) +  pd.DateOffset(days=-1)
    #         time_start = pd.to_datetime(time) +  pd.DateOffset(years=-1-k)# +  pd.DateOffset(days=1)
    #         tmp = df_era.loc[time_start:time_end]
    #         df_era.loc[time, 't2m_'+str(k)] = tmp.t2m.mean()
    #         df_era.loc[time, 'sf_'+str(k)] = tmp.sf.sum()

    #     df_era.loc[time, 't2m_avg'] = df_era.loc[time, ['t2m_'+str(k) for k in range(6)]].mean()
    #     df_era.loc[time, 'sf_avg'] = df_era.loc[time, ['sf_'+str(k) for k in range(6)]].mean()

    # df_era['month'] = (df_era.index.month-1)/12
    plt.figure()
    ds_T10m.sel(latitude = lat, longitude = lon, method = 'nearest').plot(marker='o', label = 'predicted 1')
    
    # plt.plot(df_era.index, ANN_model(df_era[Predictors]), 'o-', label = 'predicted 2')
    plt.plot(df_select.date,df_select.temperatureObserved, 'o', linestyle='None', label = 'observations')
    plt.plot(df_select.date, ANN_model(df_select[Predictors]), 'o-', label = 'predicted 1')
    plt.title(site)
    plt.legend()


