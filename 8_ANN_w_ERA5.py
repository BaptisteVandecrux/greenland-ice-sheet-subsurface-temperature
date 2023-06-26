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
import rasterio as rio
import dask


# import GIS_lib as gis
import matplotlib

# loading T10m dataset
df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")
df = df.loc[~pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :]
df['date'] = pd.to_datetime(df.date, utc=True).dt.tz_localize(None)
df["year"] = pd.DatetimeIndex(df.date).year
df.loc[df.site=='CEN1','site'] = 'Camp Century'  
df.loc[df.site=='CEN2','site'] = 'Camp Century'   
df.loc[df.site=='CEN','site'] = 'Camp Century'   

# spatial selection to the contiguous ice sheet
print(len(df), 'observation in dataset')
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")
ice_4326 = ice.to_crs(4326)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
ind_in = gpd.sjoin(gdf, ice_4326, predicate='within').index

# ice_4326.plot()
# gdf.loc[[i for i in gdf.index if i not in ind_in]].plot(color='r', ax=plt.gca())

print(len(df)-len(ind_in), 'observations outside ice sheet mask')
df = df.loc[ind_in, :]

# temporal selection
print(((df.date<'1949-12-31') | (df.date>'2022')).sum(),'observations outside of 1950-2023')
df = df.loc[(df.date>'1949-12-01') & (df.date<'2023'),:]

land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")

years = df["year"].values
years.sort()
df = df.reset_index(drop=True)

firn_memory = 5
Predictors = (
    ["t2m_amp", "t2m_10y_avg", "sf_10y_avg"]
    + ["sf_anomaly_" + str(i) for i in range(firn_memory)]
    + ["t2m_anomaly_" + str(i) for i in range(firn_memory)]
    + ["month"]
)

# Extracting ERA5 data
print('loading ERA5 data')
produce_input_grid = 1
if produce_input_grid:
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_era = xr.open_mfdataset(("Data/ERA5/ERA5_monthly_temp_snowfall.nc",
                                    'Data/ERA5/m_era5_t2m_sf_2022_proc.nc')).load()
    ds_era_1940 = xr.open_dataset('Data/ERA5/ERA5_monthly_temperature_precipitation_1940_1950.nc')[['t2m']]
    ds_era_1940['sf'] = xr.open_dataset('Data/ERA5/ERA5_monthly_snowfall_1940_1950.nc')['sf']
    
    ds_era_1940 = ds_era_1940.interp_like(ds_era.t2m.isel(time=0), method='nearest')
    ds_era = ds_era.combine_first(ds_era_1940)
    
    ds_era = ds_era.interpolate_na(dim = 'latitude', use_coordinate='latitude',
                          method='nearest', fill_value='extrapolate')
    ds_era = ds_era.interpolate_na(dim = 'latitude', use_coordinate='latitude',
                          method='nearest', fill_value='extrapolate')

    print('t2m_avg and sf_avg')
    ds_era["t2m_10y_avg"] = ds_era.t2m.rolling(time=10*12).mean()
    ds_era["sf_10y_avg"] = ds_era.sf.rolling(time=10*12).mean()
    
    print('t2m_amp')
    ds_era["t2m_amp"] = ds_era.t2m.rolling(time=12).max() - ds_era.t2m.rolling(time=12).min()
    
    print('t2m_0 and sf_0')    
    ds_era["t2m_0"] = ds_era.t2m.rolling(time=12).mean()
    ds_era["sf_0"] = ds_era.sf.rolling(time=12).mean()*12
    
    for k in range(0, firn_memory):
        print(k+1,'/',firn_memory)
        ds_era["t2m_anomaly_" + str(k)] = ds_era["t2m_10y_avg"] - ds_era["t2m_0"].shift(time=12*k)
        ds_era["sf_anomaly_" + str(k)] = ds_era["sf_10y_avg"] - ds_era["sf_0"].shift(time=12*k)
    ds_era['month'] = np.cos((ds_era.time.dt.month-1)/11*2*np.pi)
    
    ds_era = ds_era.rio.write_crs(4326)
    ice_4326 = ice.to_crs(4326)
    msk = ds_era.t2m_amp.isel(time=12*5).rio.clip(ice_4326.geometry, ice_4326.crs)
    ds_era = ds_era.where(msk.notnull())
    ds_era.to_netcdf("output/era5_monthly_plus_predictors.nc")

else: 
    ds_era = xr.open_dataset("output/era5_monthly_plus_predictors.nc")

time_era = ds_era.time

for i in range(firn_memory):
    df["t2m_" + str(i)] = np.nan
    df["sf_" + str(i)] = np.nan

df["time_era"] = np.nan
df["t2m_amp"] = np.nan


print("interpolating ERA at the observation sites")
tmp = ds_era.interp(longitude = xr.DataArray(df.longitude, dims="points"), 
                            latitude = xr.DataArray(df.latitude, dims="points"), 
                            time = xr.DataArray(df.date, dims="points"), 
                            method="nearest").load()
df[Predictors] = tmp[Predictors].to_dataframe()[Predictors].values

df["date"] = pd.to_datetime(df.date)
df["year"] = df["date"].dt.year

df = df.loc[df.t2m_10y_avg.notnull(), :]
df = df.loc[df.sf_10y_avg.notnull(), :]
for i in range(firn_memory):
    df = df.loc[df["sf_" + str(i)].notnull(), :]
    df = df.loc[df["t2m_" + str(i)].notnull(), :]

print('plotting')
fig, ax = plt.subplots(3, 4, figsize=(13,13))
ax = ax.flatten()
for i, ax in enumerate(ax):
    ax.plot( df[Predictors[i]], df.temperatureObserved, 
               marker="o", linestyle="None", markersize=1.5)
    
    ax.set_xlabel(Predictors[i])
    ax.set_ylabel('temperatureObserved')

# %% Representativity of the dataset
print('calculating weights based on representativity')
bins_temp = np.linspace(230, 275, 15)
bins_sf = np.append(np.linspace(0, 0.10, 15),3)
bins_amp = np.linspace(0, 50, 15)
pred_name = (
    [
        "$T_{2m, amp.}$ (K)",
        "$T_{2m, 10 y avg.}$ (K)",
        "$SF_{annual, 10 y avg.}$ (m w.e.)",
    ]
)

# first calculating the target histograms (slow)
pred_list = ["t2m_amp", "t2m_10y_avg", "sf_10y_avg"]
target_hist = [None] * len(pred_list)
for i in range(len(pred_list)):
    if "t2m" in pred_list[i]:
        bins = bins_temp
    if "amp" in pred_list[i]:
        bins = bins_amp
    if "sf" in pred_list[i]:
        bins = bins_sf
    print(pred_list[i])
    target_hist[i], _ = np.histogram(ds_era[pred_list[i]].values, bins=bins)
    target_hist[i] = target_hist[i].astype(np.float32) / target_hist[i].sum()

    hist1 = target_hist[i]
    hist2, _ = np.histogram(df[pred_list[i]].values, bins=bins)
    weights_bins = 0 * hist1
    weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
    ind = np.digitize(df[pred_list[i]].values, bins)
    df[pred_list[i] + "_w"] = weights_bins[ind - 1]
df["weights"] = df[[p + "_w" for p in pred_list]].mean(axis=1)

# % plotting histograms
for k in range(2):
    fig, ax = plt.subplots(4, 3, figsize=(10, 8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, wspace=0.25, hspace=0.35)
    ax = ax.flatten()
    for i in range(len(pred_list)):
        if "t2m" in pred_list[i]:
            bins = bins_temp
        if "amp" in pred_list[i]:
            bins = bins_amp
        if "sf" in pred_list[i]:
            bins = bins_sf

        hist1 = target_hist[i]
        ax[i].bar(
            bins[:-1],
            hist1,
            width=(bins[1] - bins[0]),
            alpha=0.5,
            color="tab:blue",
            edgecolor="lightgray",
            label="ERA5 values for the entire ice sheet",
        )
        if k == 0:
            hist2, _ = np.histogram(df[pred_list[i]].values, bins=bins)
        elif k == 1:
            hist2, _ = np.histogram(
                df[pred_list[i]].values, bins=bins, weights=df["weights"].values
            )

        hist2 = hist2.astype(np.float32) / hist2.sum()
        ax[i].bar(
            bins[:-1][(hist1 != 0) & (hist2 == 0)],
            hist1[(hist1 != 0) & (hist2 == 0)],
            width=(bins[1] - bins[0]),
            alpha=0.8,
            color="tab:blue",
            edgecolor="gray",
            hatch="/",
            label="bins where no training data is available",
        )
        ax[i].bar(
            bins[:-1],
            hist2,
            width=(bins[1] - bins[0]),
            alpha=0.5,
            color="tab:orange",
            edgecolor="lightgray",
            label="ERA5 values at the observation sites",
        )

        weights_bins = 0 * hist1
        weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
        ind = np.digitize(df[pred_list[i]].values, bins)
        df[pred_list[i] + "_w"] = weights_bins[ind - 1]
        ax[i].annotate(
            "MRD: %0.2f"
            % np.mean((hist1[hist2 != 0] - hist2[hist2 != 0]) / hist1[hist2 != 0]),
            xy=(0.7, 0.87),
            xycoords="axes fraction",
        )

        ax[i].set_xlabel(pred_name[i])
        ax[i].set_ylabel("Probability (-)")
        ax[i].set_title("")
    ax[i + 1].set_axis_off()
    if k == 0:
        ttl = "Without weights"
    else:
        ttl = "With weights"

    ax[i].legend(loc="upper right", bbox_to_anchor=(2.3, 0.9), title=ttl)
    fig.savefig("figures/histograms_" + str(k) + ".png", dpi =300)
    if k == 0:
        df["weights"] = df[[p + "_w" for p in pred_list]].mean(axis=1)
        
# %% ANN functions
# definition of the useful function to fit an ANN and run the trained model
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.layers import GaussianNoise
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def train_ANN(df, Predictors, TargetVariable="temperatureObserved",
              num_nodes = 32, num_layers = 2,
              epochs = 1000, batch_size=200):

    w = df["weights"].values
    X = df[Predictors].values
    y = df[TargetVariable].values.reshape(-1, 1)

    # Sandardization of data
    PredictorScalerFit = StandardScaler().fit(X)
    TargetVarScalerFit = StandardScaler().fit(y)

    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    # create and fit ANN model

    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(len(Predictors),)))
    for i in range(num_layers):
        model.add(Dense(units=num_nodes, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, sample_weight=w)
    return model, PredictorScalerFit, TargetVarScalerFit

def ANN_predict(df, model, PredictorScalerFit, TargetVarScalerFit):
    X = PredictorScalerFit.transform(df.values)
    Y = model.predict(X, verbose=0)
    Predictions = pd.DataFrame(
        data=TargetVarScalerFit.inverse_transform(Y),
        index=df.index,
        columns=["temperaturePredicted"],
    ).temperaturePredicted
    Predictions.loc[Predictions > 0] = 0
    return Predictions


def create_model(n_layers, num_nodes):
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(len(Predictors),)))
    for i in range(n_layers):
        model.add(Dense(units=num_nodes, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# %% Testing the stability of the ANN
test_stability=0
    
if test_stability:
    filename="./output/batch_size_epochs_grid.txt"
    f = open(filename, "w")
    def Msg(txt):
        f = open(filename, "a")
        print(txt)
        f.write(txt + "\n")

    df_train = df.loc[~np.isin(df.site,['NASA-E','SwissCamp','SWC','Swiss Camp', 'NASA-SE','KAN_M']), :].copy()
    df_test = df.loc[np.isin(df.site,['NASA-E','SwissCamp','SWC','Swiss Camp', 'NASA-SE','KAN_M']), :].copy()
    df_sub = df.loc[np.isin(df.site, ["Swiss Camp", "DYE-2", "KAN_U", 
                                      "Camp Century", 'KPC_U', 'QAS_U',
                                      'NASA-E', 'NASA-SE','KAN_M']),:].copy()
    for batch_size in [100,  200,  500, 1000, 2000, 3000, 4000, 5000]:
        for epochs in [e for e in range(5,100,5)]+[e for e in range(100,1000,20)]: #:
            # print(epochs, 'epochs')
            num_models = 10
            model = [None]*num_models
            PredictorScalerFit = [None]*num_models
            TargetVarScalerFit = [None]*num_models
            # training the models
            for i in range(num_models):
                # print(i)
                model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
                    df_train, Predictors, num_nodes = 32, num_layers=2, epochs=epochs,
                    batch_size=batch_size,
                )
                df_test['out_mod_'+str(i)] = ANN_predict(
                    df_test[Predictors], 
                    model[i], 
                    PredictorScalerFit[i], 
                    TargetVarScalerFit[i]).values
                Msg('%i, %i, %0.2f, %0.2f'%(batch_size, epochs,
                    (df_test['temperatureObserved'] - df_test['out_mod_'+str(i)]).mean(),
                    np.sqrt(((df_test['temperatureObserved'] - df_test['out_mod_'+str(i)])**2).mean())))

plt.close('all')
fig, ax = plt.subplots(2,1,figsize=(9,5), sharex=True)
ls=['-','--',':','-.','-','--',':','-.','-','--',':','-.']
df_epochs = pd.read_csv('./output/batch_size_epochs_grid.txt',header=None)
df_epochs.columns=['batch_size','epochs','MD','RMSD']
tmp = df_epochs.groupby(['batch_size','epochs']).mean()
tmp[['MD_std','RMSD_std']] = df_epochs.groupby(['batch_size','epochs']).std().mean()
for k, batch_size in enumerate(np.unique(tmp.index.get_level_values('batch_size'))):
    ax[0].errorbar(tmp.loc[batch_size].index + k/2 - 2, tmp.loc[batch_size, 'MD'],
                   yerr=tmp.loc[batch_size, 'MD_std'], ls=ls[k],label = 'batch size '+str(batch_size))
    ax[1].errorbar(tmp.loc[batch_size].index + k/2 -2, tmp.loc[batch_size, 'RMSD'],
                   ls=ls[k], yerr=tmp.loc[batch_size, 'RMSD_std'])
ax[0].set_ylabel('MD (°C)')
ax[1].set_ylabel('RMSD (°C)')
ax[1].set_xlabel('Number of epochs')
ax[0].set_xlim(0,200)
ax[0].grid()
ax[1].grid()
ax[0].legend(loc="lower center",ncol=2, bbox_to_anchor=(0.5,1))
fig.savefig('figures/epochs_test.png',dpi=300)
# %% Testing the stability of the ANN
test_stability=0
    
if test_stability:
    filename="./output/epochs_sigma.txt"
    f = open(filename, "w")
    def Msg(txt):
        f = open(filename, "a")
        print(txt)
        f.write(txt + "\n")

    import matplotlib
    matplotlib.use('Agg')

    df_train = df.loc[~np.isin(df.site,['NASA-E','SwissCamp','SWC','Swiss Camp', 'NASA-SE','KAN_M']), :].copy()
    df_test = df.loc[np.isin(df.site,['NASA-E','SwissCamp','SWC','Swiss Camp', 'NASA-SE','KAN_M']), :].copy()
    df_sub = df.loc[np.isin(df.site, ["Swiss Camp", "DYE-2", "KAN_U", 
                                      "Camp Century", 'KPC_U', 'QAS_U',
                                      'NASA-E', 'NASA-SE','KAN_M']),:].copy()
    for sigma in [0, 0.01, 0.05, 0.1, 0.5]:
        
        def train_ANN(df, Predictors, TargetVariable="temperatureObserved",
                      num_nodes = 32, num_layers = 2,
                      epochs = 1000, batch_size=200):
        
            w = df["weights"].values
            X = df[Predictors].values
            y = df[TargetVariable].values.reshape(-1, 1)
        
            # Sandardization of data
            PredictorScalerFit = StandardScaler().fit(X)
            TargetVarScalerFit = StandardScaler().fit(y)
        
            X = PredictorScalerFit.transform(X)
            y = TargetVarScalerFit.transform(y)
        
            # create and fit ANN model
        
            model = Sequential()
            model.add(GaussianNoise(sigma, input_shape=(len(Predictors),)))
            for i in range(num_layers):
                model.add(Dense(units=num_nodes, activation="relu"))
            model.add(Dense(1))
            model.compile(loss="mean_squared_error", optimizer="adam")
            model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, sample_weight=w)
            return model, PredictorScalerFit, TargetVarScalerFit
        # print(epochs, 'epochs')
        num_models = 10
        model = [None]*num_models
        PredictorScalerFit = [None]*num_models
        TargetVarScalerFit = [None]*num_models
        # training the models
        for i in range(num_models):
            # print(i)
            model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
                df_train, Predictors, num_nodes = 32, num_layers=2, epochs=epochs,
                batch_size=2000,
            )
            df_test['out_mod_'+str(i)] = ANN_predict(
                df_test[Predictors], 
                model[i], 
                PredictorScalerFit[i], 
                TargetVarScalerFit[i]).values
            Msg('%s, %0.2f, %0.2f'%(sigma,
                (df_test['temperatureObserved'] - df_test['out_mod_'+str(i)]).mean(),
                np.sqrt(((df_test['temperatureObserved'] - df_test['out_mod_'+str(i)])**2).mean())))
            
        #  Plotting model outputs
        fig, ax = plt.subplots(3, 3, figsize=(10, 14))
        plt.subplots_adjust(top=0.93,hspace=0.3)
        ax = ax.flatten()
        for k, site in enumerate(df_sub.site.unique()):
            for i in range(num_models):
                if i == 2:
                    label = 'ANN models'
                else:
                    label = ' '
                df_sub['out_mod_'+str(i)]  = np.nan
                df_sub.loc[df_sub.site == site, 'out_mod_'+str(i)] = ANN_predict(
                    df_sub.loc[df_sub.site == site, Predictors], 
                    model[i], 
                    PredictorScalerFit[i], 
                    TargetVarScalerFit[i]).values

                df_sub.loc[df_sub.site == site, :]  = df_sub.loc[df_sub.site == site, :] .sort_values('date')
                ax[k].plot(df_sub.loc[df_sub.site == site, 'date'], 
                           df_sub.loc[df_sub.site == site, 'out_mod_'+str(i)],
                           alpha=0.3, marker='o',linestyle='None',
                           # markerfacecolor='gray', markeredgecolor='gray',
                           label = label)
            ax[k].plot(np.nan,np.nan,'w',label=' ')
            ax[k].plot(pd.to_datetime(df_sub.loc[df_sub.site == site, 'date']).values,
                         df_sub.loc[df_sub.site == site, 'temperatureObserved'].values, 
                         marker="o", markerfacecolor='w',  markeredgecolor='gray',
                         linestyle='None', label="observations")
            ax[k].set_title(site+' ($\overline{\sigma}$= %0.2f °C)'%\
                            df_sub.loc[df_sub.site == site, 
                           ['out_mod_'+str(i) for i in range(5)]].std(axis=1).mean())
            ax[k].set_ylim(-32, 0)
            ax[k].grid()
            for label in ax[k].get_xticklabels():
                  label.set_rotation(45)
                  label.set_ha('right')
            # print('Avg. st. dev. at',site, ': %0.2f'%df_sub.loc[df_sub.site == site, ['out_mod_'+str(i) for i in range(5)]].std(axis=1).mean(), '°C')
        fig.suptitle(str(epochs)+' epochs')
        ax[0].legend(labelspacing=-0.2)
        
        fig.savefig('figures/stability test/stability_sigma_'+str(sigma)+'.png', dpi=300)
# Average standard deviation between 10 models trained on the same data				
# epochs        20 	    100 	1000 	5000 	10000 
# SwissCamp	    0.69	0.58	0.59	0.64	0.57
# DYE-2	        0.79	0.7  	0.45	0.44	0.42
# KAN_U         0.68	0.61	0.52	0.51	0.46
# Camp Century  0.7	    0.42	0.28	0.25	0.28
# KPC_U         0.69	0.52	0.43	0.44	0.36
# QAS_U	        0.45	0.53	0.37	0.39	0.37

# %% GridSearch

run_gridsearch = 0
if run_gridsearch:   
    w = df["weights"].values
    X = df[Predictors].values
    y = df["temperatureObserved"].values.reshape(-1, 1)
    
    # Sandardization of data
    PredictorScalerFit = StandardScaler().fit(X)
    TargetVarScalerFit = StandardScaler().fit(y)
    
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)
      
    # create model
    model = KerasRegressor(model=create_model, verbose=0,
                           batch_size = [5000],
                           epochs = [100],
                           n_layers=[1, 2, 3], 
                           num_nodes = [16, 32, 64])
    
    # define the grid search parameters
    param_grid = dict(batch_size = [5000],
                    epochs = [100],
                    n_layers=[1, 2, 3], 
                    num_nodes = [16, 32, 64])
    grid = GridSearchCV(estimator = model , param_grid = param_grid, n_jobs=-1, verbose=2)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# Gridsearch output for:
# batch_size = [100, 1000], epochs = [20, 60, 500],
# n_layers=[1, 2],  num_nodes = [16, 32, 64]
# Best: 0.879104 using {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}

# Gridsearch output for:
# batch_size = [10, 50, 100, 200, 500, 1000], epochs = [20, 60, 500],
# n_layers=[1, 2, 3],  num_nodes = [16, 32, 64]
# Best: 0.923820 using {'batch_size': 10, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}

# Gridsearch output for:
# batch_size = [10, 100, 500, 1000], epochs = [500, 1000, 1500, 5000],
# n_layers=[1, 2, 3],   num_nodes = [16, 32, 64]
# Best: 0.868316 using {'batch_size': 500, 'epochs': 5000, 'n_layers': 2, 'num_nodes': 16}

# Gridsearch output for:
# batch_size = [5000], epochs = [100],
# n_layers=[1, 2, 3],   num_nodes = [16, 32, 64]
# Best: 0.765404 using {'batch_size': 5000, 'epochs': 100, 'n_layers': 3, 'num_nodes': 64}
# print(wtf)
# %% best model
print('Training model on entire dataset')
best_model, best_PredictorScalerFit, best_TargetVarScalerFit = train_ANN(
    df, Predictors, num_nodes = 32, num_layers=2, epochs=1000)

# %% evaluating overfitting
print('Training model on entire dataset')
for epochs in [100, 500, 1000, 5000, 10000]:
    model, PredictorScalerFit, TargetVarScalerFit = train_ANN(
        df, Predictors, num_nodes = 32, num_layers=2, epochs=1000)

# %% Predicting T10m over ERA5 dataset
predict = 0

if predict:    
    print("predicting T10m over entire ERA5 dataset")
    print('converting to dataframe')
    tmp = ds_era[Predictors].to_dataframe()
    tmp = tmp.loc[tmp.notnull().all(1),:][Predictors]
    
    print('applying ANN (takes ~45min on my laptop)')
    out = ANN_predict(tmp, best_model, best_PredictorScalerFit, best_TargetVarScalerFit)

    ds_T10m =  out.to_frame().to_xarray()["temperaturePredicted"].sortby(['time','latitude','longitude'])
    ds_T10m = ds_T10m.rio.write_crs(4326).rio.clip(ice.to_crs(4326).geometry).rename('T10m')
    ds_T10m.attrs['author'] = 'Baptiste Vandecrux'
    ds_T10m.attrs['contact'] = 'bav@geus.dk'
    ds_T10m.attrs['description'] = 'Monthly grids of Greenland ice sheet 10 m subsurface temperature for 1954-2022 as predicted by an artifical neural network trained on more than 4500 in situ observations and using ERA5 snowfall and air temperature as input.'
    ds_T10m.to_netcdf("output/T10m_prediction.nc")
    
ds_T10m = xr.open_dataset("output/T10m_prediction.nc")["T10m"]


# %% spatial cross-validation:
print("fitting cross-validation models")
zwally = gpd.GeoDataFrame.from_file("Data/misc/Zwally_10_zones_3413.shp")
zwally = zwally.set_crs("EPSG:3413").to_crs("EPSG:4326")
zwally = zwally.drop(columns=["name"])
df = df.reset_index(drop=True)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
).set_crs(4326)
points_within = gpd.sjoin(gdf, zwally, op="within")
df["zwally_zone"] = np.nan
df["T10m_pred_unseen"] = np.nan
model_list = [None] * zwally.shape[0]
PredictorScalerFit_list = [None] * zwally.shape[0]
TargetVarScalerFit_list = [None] * zwally.shape[0]

for i in range(zwally.shape[0]):
    msk = points_within.loc[points_within.index_right == i, :].index
    df.loc[msk, "zwally_zone"] = i
    print(i, len(msk), "%0.2f" % (len(msk) / df.shape[0] * 100))

    df_cv = df.loc[df.zwally_zone != i, ].copy()

    model_list[i], PredictorScalerFit_list[i], TargetVarScalerFit_list[i] \
                            = train_ANN(df_cv, Predictors, 
                                           TargetVariable="temperatureObserved",  
                                           num_nodes = 32, 
                                           num_layers=2, 
                                           epochs=1000)
    # saving the model estimate for the unseen data
    df_unseen = df.loc[df.zwally_zone == i, :].copy()

    df.loc[df.zwally_zone == i,"T10m_pred_unseen"] = ANN_predict(df_unseen[Predictors],
                                              model_list[i],
                                              PredictorScalerFit_list[i],
                                              TargetVarScalerFit_list[i])

# defining function that makes the ANN estimate from predictor raw values
df = df.sort_values("date")

def ANN_model_cv(df_pred, model_list, PredictorScalerFit_list, TargetVarScalerFit_list):
    pred = pd.DataFrame()
    for i in range(zwally.shape[0]):
        pred["T10m_pred_" + str(i)] = ANN_predict(df_pred,
                                                  model_list[i],
                                                  PredictorScalerFit_list[i],
                                                  TargetVarScalerFit_list[i])
    df_mean = pred.mean(axis=1)
    df_std = pred.std(axis=1)
    return df_mean.values, df_std.values

# %% checking best model corresponds to saved T10m dataset
best_model_pred = ANN_predict(df[Predictors], 
                              best_model, 
                              best_PredictorScalerFit, 
                              best_TargetVarScalerFit)

fig, ax = plt.subplots(4, 3, figsize=(15, 15))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.4, top = 0.9,left =0.15,right=0.95)
for k, site in enumerate(["CP1","DYE-2", "Summit","Camp Century",
                          "Swiss Camp", "NASA-SE", "KAN_U", "EKT",
                          "French Camp VI","FA_13", "NASA-E", 'KPC_U']):
    print(site)
    df_select = df.loc[df.site == site, :].set_index('date')
    best_model_pred = ANN_predict(df_select[Predictors], 
                                  best_model, 
                                  best_PredictorScalerFit, 
                                  best_TargetVarScalerFit)
    best_model_pred.plot(ax=ax[k])
    ds_T10m.sel(dict(latitude=df_select.latitude.unique()[0], 
                         longitude=df_select.longitude.unique()[0]),
                    method='nearest').to_dataframe().T10m.plot(ax=ax[k])

# %% Predicting T10m uncertainty
predict = 0
if predict:
    print("predicting T10m uncertainty over entire ERA5 dataset")
    for year in range(2022,2023): #np.unique(ds_era.time.dt.year):
        ds_T10m_std = ds_era["t2m"].sel(time=str(year)).copy().rename("T10m_std") * np.nan
        for time in ds_T10m_std.time:
            if (time.dt.month==1) & (time.dt.day==1):
                print(time.dt.year.values)
            tmp = ds_era.sel(time=time).to_dataframe()
            _, tmp["cv_std"] = ANN_model_cv(tmp[Predictors], model_list,
                                            PredictorScalerFit_list,
                                            TargetVarScalerFit_list)
    
            ds_T10m_std.loc[dict(time=time)] = (
                tmp["cv_std"].to_frame().to_xarray()["cv_std"]
                .transpose("latitude", "longitude").values
                )
    
        ds_T10m_std.to_netcdf("output/uncertainty/predicted_T10m_std_"+str(year)+".nc")
    ds_T10m_std = xr.open_mfdataset(
        ["output/uncertainty/predicted_T10m_std_"+str(year)+".nc" for year in range(1954,2023)])
    ds_T10m_std = ds_T10m_std.rio.write_crs(4326)
    ice_4326 = ice.to_crs(4326)
    msk = ds_T10m_std.T10m_std.isel(time=0).rio.clip(ice_4326.geometry, ice_4326.crs)
    ds_T10m_std = ds_T10m_std.where(msk.notnull())

    ds_T10m_std.attrs['author'] = 'Baptiste Vandecrux'
    ds_T10m_std.attrs['contact'] = 'bav@geus.dk'
    ds_T10m_std.attrs['description'] = 'Monthly grids of the uncertainty of the ANN Greenland ice sheet 10 m subsurface temperature for 1954-2022 calculated from the standard deviation between the predictions of 10 spatial cross-validation models.'
    ds_T10m_std.to_netcdf("output/T10m_uncertainty.nc")
ds_T10m_std = xr.open_dataset("output/T10m_uncertainty.nc")['T10m_std']

#%% checking that CV std
fig, ax = plt.subplots(4, 3, figsize=(15, 15))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.4, top = 0.9,left =0.15,right=0.95)
for k, site in enumerate(["CP1","DYE-2", "Summit","Camp Century",
                          "Swiss Camp", "NASA-SE", "KAN_U", "EKT",
                          "French Camp VI","FA_13", "NASA-E", 'KPC_U']):
    print(site)
    df_select = ds_era.sel(df.loc[df.site == site, ['latitude',
                                                    'longitude']].iloc[0,:].to_dict(),
                           method='nearest').to_dataframe()
    df_select["cv_mean"], df_select["cv_std"] = ANN_model_cv(
        df_select[Predictors],
        model_list, 
        PredictorScalerFit_list, 
        TargetVarScalerFit_list
    )
    df_select["cv_std"].plot(ax=ax[k])
    ds_T10m_std.sel(dict(latitude=df_select.latitude.unique()[0], 
                         longitude=df_select.longitude.unique()[0]),
                    method='nearest').T10m_std.to_dataframe().T10m_std.plot(ax=ax[k])
    plt.title(site)


# %% Preparing for figure 2
# averaging and reprojecting uncertainty map
# ds_era = xr.open_dataset("era5_monthly_plus_predictors.nc")
# ds_T10m = xr.open_dataset("predicted_T10m.nc")["T10m"]
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:4326")
ds_era = ds_era.rio.write_crs("EPSG:4326").rio.clip(ice.geometry.values, ice.crs)

ds_T10m_std_avg = (
    ds_T10m_std.mean(dim="time")
    .rio.write_crs("EPSG:4326")
    .rio.clip(ice.geometry.values, ice.crs)
)
ds_T10m_std_avg = ds_T10m_std_avg.rio.reproject("EPSG:3413")

# Extracting T10m for all dataset
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

    for i in (range(df.shape[0])):
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
df['T10m_ANN'] = np.nan
df = extract_T10m_values(
    ds_T10m.to_dataset(), df, dim1="longitude", dim2="latitude", name_out="T10m_ANN"
)

# %% Plotting Figure 2
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]

plt.close('all')
matplotlib.rcParams.update({'font.size': 13})
df.loc[df.site=='NAE','site'] = 'NASA-E'
df.loc[df.site=='SWC','site'] = 'Swiss Camp'
df.loc[df.site=='DY2','site'] = 'DYE-2'
abc='abcdefghijklmno'

fig, ax = plt.subplots(11, 1, figsize=(10, 14),sharex=(True))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.06, top = 1,bottom=0.05, left =0.12,right=0.9)
for k in range(5):
    ax[k].set_axis_off()

# =============== scatterplot =======================
for ax1, model in zip( [plt.subplot(4,2,1), plt.subplot(4,2,3)],
                      ["ANN", "pred_unseen"]):
    ax1.plot(
        df_10m["temperatureObserved"],
        df_10m["T10m_" + model],
        marker="+",
        linestyle="none",
        # markeredgecolor="lightgray",
        markeredgewidth=0.5,
        markersize=5,
        color="k",
    )
    RMSE = np.sqrt(np.mean((df_10m["T10m_" + model] - df_10m.temperatureObserved) ** 2))
    ME = np.mean(df_10m["T10m_" + model] - df_10m.temperatureObserved)
    if ME<0.05:
        ME=abs(ME)
    textstr = "\n".join(
        (
            r"MD = %.1f °C" % (ME,),
            r"RMSD=%.1f °C" % (RMSE,),
            r"N=%.0f" % (np.sum(~np.isnan(df_10m["T10m_" + model])),),
        )
    )
    
    t1 = ax1.text(0.02, 0.95, "All sites\n" + textstr,
        transform=ax1.transAxes, fontsize=13, verticalalignment="top")
    t1.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='w'))
    
    if model == "ANN":
        ax1.set_title("(a) best ANN model vs. its training data", loc="left",fontsize=13)
        pos1 = ax1.get_position()
        pos2 = [pos1.x0, pos1.y0+0.03,  pos1.width*1.1, pos1.height*0.7] 
        ax1.set_position(pos2)
    else:
        ax1.set_title("(b) CV ANN models vs. their unseen data", loc="left",fontsize=13)
        pos1 = ax1.get_position()
        pos2 = [pos1.x0, pos1.y0+0.04,  pos1.width*1.1, pos1.height*0.7] 
        ax1.set_position(pos2)
    ax1.plot([-35, 2], [-35, 2], c="black")
    ax1.set_xlim(-35, 2)
    ax1.set_ylim(-35, 2)
    ax1.grid()
    
    # Comparison for ablation datasets
    df_ablation = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]
    df_ablation = df_ablation.loc[
        np.isin(df.site, ['KPC_L', 'SCO_U', 'KAN_L', 'KPC_U', 'SCO_L', 'QAS_U', 'NUK_U',
               'UPE_L', 'UPE_U', 'NUK_L', 'TAS_L', 'KAN_M', 'THU_L', 'TAS_U',
               'T-11b', 'THU_U', 'T-11a', 'T-14', 'TAS_A', 'T-15a', 'T-15b',
               'T-16', 'QAS_M', 'T-15c', 'THU_U2', 'Swiss Camp','SwissCamp',
               'QAS_Uv3', 'KPC_Uv3', 'SWC_O', 'JAR_O']),:]
    
    ax1.plot( df_ablation["temperatureObserved"], df_ablation["T10m_" + model],
        marker="+", linestyle="none", markeredgewidth=0.5, 
        markersize=5, color="tab:red",
    )
    RMSE = np.sqrt(np.mean((df_ablation["T10m_" + model] - df_ablation.temperatureObserved) ** 2))
    ME = np.mean(df_ablation["T10m_" + model] - df_ablation.temperatureObserved)
    
    textstr = "\n".join((r"MD=%.1f °C " % (ME,),
                         r"RMSD=%.1f °C" % (RMSE,),
                         r"N=%.0f" % (np.sum(~np.isnan(df_ablation["T10m_" + model]))),
                         ))
    t = ax1.text(0.63, 0.4, "Ablation sites\n" + textstr, transform=ax1.transAxes,
        fontsize=13, verticalalignment="top", color="tab:red")
    t.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='w'))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    ax1.set_xlabel("Observed $T_{10m}$ (°C)")
    ax1.set_ylabel("Predicted $T_{10m}$ (°C)")


# =============== map =======================
ax_map = plt.subplot(2,2,2)

land.plot(color="k", ax=ax_map)

im = ds_T10m_std_avg.where(ds_T10m_std_avg < 1000).plot(
    ax=ax_map, add_colorbar=False)
cbar = fig.colorbar(im, label="Average uncertainty of the ANN (°C)", shrink=0.8)
gpd.GeoDataFrame.from_file("Data/misc/Zwally_10_zones_3413.shp").boundary.plot(
    ax=ax_map, color="w", linestyle="-", linewidth=0.5
)
ax_map.set_axis_off()
ax_map.set_title("")
ax_map.text(0.1, 0.9, '(c)',
 horizontalalignment='left',
 verticalalignment='center',
 fontsize=14,
 transform = ax_map.transAxes)
ax_map.set_xlim(-700000, 900000)
ax_map.set_ylim(-3400000, -574000)
# pos1 = ax_map.get_position()
# pos2 = [pos1.x0, pos1.y0 + 0.1,  pos1.width, pos1.height] 
# ax_map.set_position(pos2)
import matplotlib.patheffects as pe

# =============== site examples =======================
for k, site in enumerate(["NASA-E", "DYE-2", "KAN_L", 'KPC_U',"FA_13"]):
    k = k+6
    print(site)
    
    df_site = df.loc[df.site == site, :].iloc[0:1]
    gdf_site = gpd.GeoDataFrame(df_site,
                                geometry=gpd.points_from_xy(df_site.longitude,
                                                            df_site.latitude),
                                crs="EPSG:4326").to_crs(3413)
    gdf_site.plot(ax=ax_map, color='w', edgecolor='k')
    xytext=(5, 3)
    if site == 'DYE-2': xytext=(-5, 8)
    if site == 'KAN_L': xytext=(-40, 10)
    if site == 'FA_13': xytext=(5, -20)
        
        
    for x, y in zip(gdf_site.geometry.x, gdf_site.geometry.y):
        ax_map.annotate(site, xy=(x, y), xytext=xytext, 
                        textcoords="offset points",
                        fontsize=15,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
    df_select = ds_era.sel(df.loc[df.site == site, ['latitude',
                                                    'longitude']].iloc[0,:].to_dict(),
                           method='nearest').to_dataframe()
    cv_std = ds_T10m_std.sel(dict(latitude=df_select.latitude.unique()[0], 
                         longitude=df_select.longitude.unique()[0]),
                    method='nearest').to_dataframe().T10m_std

    best_model_pred = ANN_predict(df_select[Predictors], 
                                  best_model, 
                                  best_PredictorScalerFit, 
                                  best_TargetVarScalerFit)
    
    ax[k].step(df_select.index, best_model_pred, color="tab:blue", where='post', label="ANN")
    
    ax[k].fill_between(df_select.index, best_model_pred - cv_std, 
        best_model_pred + cv_std, color="turquoise", step='post',
        label="ANN uncertainty", zorder=0)
    
    for i in range(zwally.shape[0]):
        if i == 0:
            lab = "cross-validation models"
        else:
            lab = "_no_legend_"
        ax[k].step(
            df_select.index,
            ANN_predict(df_select[Predictors], model_list[i],  
                        PredictorScalerFit_list[i], TargetVarScalerFit_list[i]),
            "-", color="gray", where='post', alpha=0.7, lw=0.5, label=lab,
        )

    ax[k].plot(
        df.loc[df.site == site, :].date,
        df.loc[df.site == site, :].temperatureObserved,
        ".", color="orange",alpha=0.9,  linestyle="None",  label="_nolegend_",
    )
    ax[k].plot(np.nan, np.nan,
        "o", color="orange",alpha=0.9,  linestyle="None",  label="observations",
    )
    if site == 'DYE-2':
        ax[k].text(0.01, 0.8, '('+abc[k-3]+') '+site,
         horizontalalignment='left', verticalalignment='center',
         fontsize=14, transform = ax[k].transAxes)
    else:
        ax[k].text(0.01, 0.1, '('+abc[k-3]+') '+site,
         horizontalalignment='left', verticalalignment='center',
         fontsize=14, transform = ax[k].transAxes)
    ax[k].set_xlim(pd.to_datetime('1998-01-01'),pd.to_datetime('2023-01-01'))
    ax[k].grid()
ax[6].legend(ncol=2, bbox_to_anchor=(0,0.95))

fig.text(0.5,  0.02,  "Year",
    ha="center", va="center", fontsize=16)
fig.text(0.03, 0.3, "10 m subsurface temperature (°C)",
    ha="center", va="center", rotation="vertical", fontsize=16)
fig.savefig('figures/figure2.png')

# %% Selection and overview of the predictors
ds_era_GrIS = ds_era.mean(dim=("latitude", "longitude")).to_pandas()

fig, ax = plt.subplots(4, 3, figsize=(15, 15), sharex=True)
ax = ax.flatten()
for i in range(len(Predictors)):
    ds_era_GrIS[Predictors[i]].resample("Y").mean().plot(ax=ax[i], drawstyle="steps")
    ax[i].set_ylabel(Predictors[i])

ds_era_GrIS["t2m"].resample("Y").mean().plot(ax=ax[i + 1], drawstyle="steps")
ax[i + 1].set_ylabel("t2m")