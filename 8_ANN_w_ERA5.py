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
import dask
import progressbar
bar = progressbar.ProgressBar()

# import GIS_lib as gis
from progressbar import progressbar
import matplotlib

df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")
df = df.loc[~pd.to_numeric(df.depthOfTemperatureObservation, errors="coerce").isnull(), :]

df['date'] = pd.to_datetime(df.date, utc=True).dt.tz_localize(None)
df["year"] = pd.DatetimeIndex(df.date).year
df.loc[df.site=='CEN1','site'] = 'Camp Century'  
df.loc[df.site=='CEN2','site'] = 'Camp Century'   
df.loc[df.site=='CEN','site'] = 'Camp Century'   


land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")

years = df["year"].values
years.sort()
df = df.reset_index(drop=True)

firn_memory = 4
Predictors = (
    ["t2m_amp", "t2m_avg", "sf_avg"]
    + ["sf_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
)

# Extracting ERA5 data
print('loading ERA5 data')
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    ds_era = xr.open_mfdataset(("Data/ERA5/ERA5_monthly_temp_snowfall.nc",
                                'Data/ERA5/m_era5_t2m_sf_2022_proc.nc')).load()

produce_input_grid = 0
if produce_input_grid:
    print('t2m_avg and sf_avg')
    ds_era["t2m_avg"] = ds_era.t2m.rolling(time=firn_memory*12).mean()
    ds_era["sf_avg"] = ds_era.sf.rolling(time=firn_memory*12).mean()
    print('t2m_amp')
    ds_era["t2m_amp"] = ds_era.t2m.rolling(time=12).max() - ds_era.t2m.rolling(time=12).min()
    print('t2m_0 and sf_0')    
    ds_era["t2m_0"] = ds_era.t2m.rolling(time=12).mean()
    ds_era["sf_0"] = ds_era.sf.rolling(time=12).mean()
    
    for k in range(1, firn_memory):
        print(k,'/',firn_memory-1)
        ds_era["t2m_" + str(k)] = ds_era["t2m_0"].shift(time=12*k)
        ds_era["sf_" + str(k)]  = ds_era["sf_0"].shift(time=12*k)

    ds_era.to_netcdf("output/era5_monthly_plus_predictors.nc")

else: 
    ds_era = xr.open_dataset("output/era5_monthly_plus_predictors.nc")

time_era = ds_era.time

for i in range(firn_memory):
    df["t2m_" + str(i)] = np.nan
    df["sf_" + str(i)] = np.nan

df["time_era"] = np.nan
df["t2m_amp"] = np.nan

df = df.loc[(df.date>'1950') & (df.date<'2023'),:]

print("interpolating ERA at the observation sites")
tmp = ds_era.interp(longitude = xr.DataArray(df.longitude, dims="points"), 
                            latitude = xr.DataArray(df.latitude, dims="points"), 
                            time = xr.DataArray(df.date, dims="points"), 
                            method="nearest").load()
df[Predictors] = tmp[Predictors].to_dataframe()[Predictors].values

df["date"] = pd.to_datetime(df.date)
df["year"] = df["date"].dt.year
df["month"] = (df["date"].dt.month - 1) / 12

df = df.loc[df.t2m_avg.notnull(), :]
df = df.loc[df.sf_avg.notnull(), :]
for i in range(firn_memory):
    df = df.loc[df["sf_" + str(i)].notnull(), :]
    df = df.loc[df["t2m_" + str(i)].notnull(), :]


print('plotting')
fig, ax = plt.subplots(2, 2, figsize=(13,13))
ax = ax.flatten()
ax[0].plot( df.t2m_0, df.temperatureObserved, 
           marker="o", linestyle="None", markersize=1.5)
ax[1].plot(df.sf_0, df.temperatureObserved,
           marker="o", linestyle="None", markersize=1.5)
ax[2].plot(df.t2m_avg, df.temperatureObserved, 
           marker="o", linestyle="None", markersize=1.5)
ax[3].plot(df.sf_avg, df.temperatureObserved, 
           marker="o", linestyle="None", markersize=1.5)
ax[0].set_xlabel('t2m_0')
ax[1].set_xlabel('sf_0')
ax[2].set_xlabel('t2m_avg')
ax[3].set_xlabel('sf_avg')
for i in range(4): ax[i].set_ylabel('temperatureObserved')

# %% Representativity of the dataset
print('calculating weights based on representativity')
bins_temp = np.linspace(230, 275, 15)
bins_sf = np.linspace(0, 0.15, 15)
bins_amp = np.linspace(0, 50, 15)
pred_name = (
    [
        "$T_{2m, amp.}$ (K)",
        "$T_{2m, avg.}$ (last 4 years, K)",
        "$SF_{avg.}$ (last 4 years, m w.e.)",
    ]
    + ["$SF_{avg.}$ (year-%i, m w.e.)" % (i + 1) for i in range(firn_memory)]
    + ["$T_{2m, avg.}$ (year-%i, K)" % (i + 1) for i in range(firn_memory)]
)

# first calculating the target histograms (slow)
target_hist = [None] * len(Predictors)
for i in range(len(Predictors)):
    if "t2m" in Predictors[i]:
        bins = bins_temp
    if "amp" in Predictors[i]:
        bins = bins_amp
    if "sf" in Predictors[i]:
        bins = bins_sf
    print(Predictors[i])
    target_hist[i], _ = np.histogram(ds_era[Predictors[i]].values, bins=bins)
    target_hist[i] = target_hist[i].astype(np.float32) / target_hist[i].sum()

    hist1 = target_hist[i]
    hist2, _ = np.histogram(df[Predictors[i]].values, bins=bins)
    weights_bins = 0 * hist1
    weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
    ind = np.digitize(df[Predictors[i]].values, bins)
    df[Predictors[i] + "_w"] = weights_bins[ind - 1]
df["weights"] = df[[p + "_w" for p in Predictors]].mean(axis=1)

# %% plotting histograms
for k in range(2):
    fig, ax = plt.subplots(4, 3, figsize=(10, 8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, wspace=0.25, hspace=0.35)
    ax = ax.flatten()
    for i in range(len(Predictors)):
        if "t2m" in Predictors[i]:
            bins = bins_temp
        if "amp" in Predictors[i]:
            bins = bins_amp
        if "sf" in Predictors[i]:
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
            hist2, _ = np.histogram(df[Predictors[i]].values, bins=bins)
        elif k == 1:
            hist2, _ = np.histogram(
                df[Predictors[i]].values, bins=bins, weights=df["weights"].values
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
        ind = np.digitize(df[Predictors[i]].values, bins)
        df[Predictors[i] + "_w"] = weights_bins[ind - 1]
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
    fig.savefig("figures/histograms_" + str(k) + ".png")
    if k == 0:
        df["weights"] = df[[p + "_w" for p in Predictors]].mean(axis=1)


# %% ANN functions
# definition of the useful function to fit an ANN and run the trained model
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.layers import GaussianNoise
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def train_ANN(df, Predictors, TargetVariable="temperatureObserved",
              num_nodes = 32, num_layers = 2,
              epochs = 1000):

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
    model.fit(X, y, batch_size=200, epochs=epochs, verbose=0, sample_weight=w)
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
# Predictors = (
#     ["month", "t2m_amp", "t2m_avg", "sf_avg"]
#     + ["sf_" + str(i) for i in range(firn_memory)]
#     + ["t2m_" + str(i) for i in range(firn_memory)]
# )
# df['month'] = np.cos((pd.DatetimeIndex(df['date']).month-1)/11/12*2*np.pi) 

# %% Testing the stability of the ANN
test_stability=0
if test_stability:
    df_sub = df.loc[np.isin(df.site, ["Swiss Camp", "DYE-2", "KAN_U", "Camp Century", 'KPC_U', 'QAS_U']),:].copy()
    for epochs in [20, 100, 200]:
        print(epochs, 'epochs')
        num_models = 10
        model = [None]*num_models
        PredictorScalerFit = [None]*num_models
        TargetVarScalerFit = [None]*num_models
        # training the models
        for i in range(num_models):
            # print(i)
            model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
                df, Predictors, num_nodes = 32, num_layers=2, epochs=epochs,
            )
            
        # % Plotting model outputs
        fig, ax = plt.subplots(2, 3, figsize=(15, 15))
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
                                                PredictorScalerFit[i], TargetVarScalerFit[i]).values
                df_sub.loc[df_sub.site == site, :]  = df_sub.loc[df_sub.site == site, :] .sort_values('date')
                ax[k].plot(df_sub.loc[df_sub.site == site, 'date'], 
                           df_sub.loc[df_sub.site == site, 'out_mod_'+str(i)],
                           alpha=0.5, label = label)
            ax[k].plot(np.nan,np.nan,'w',label=' ')
            ax[k].plot(pd.to_datetime(df_sub.loc[df_sub.site == site, 'date']).values,
                         df_sub.loc[df_sub.site == site, 'temperatureObserved'].values, 
                         marker="x", linestyle='None', label="observations")
            ax[k].set_title(site)
            ax[k].set_ylim(-25, 0)
            print('Avg. st. dev. at',site, ': %0.2f'%df_sub.loc[df_sub.site == site, ['out_mod_'+str(i) for i in range(5)]].std(axis=1).mean(), '°C')
        fig.suptitle(str(epochs)+' epochs')
        ax[0].legend()
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
                           batch_size = [10, 50, 100, 200, 500, 1000],
                           epochs = [20, 60, 500, 1000, 1500],
                           n_layers=[1, 2, 3, 4, 8, 10, 15], 
                           num_nodes = [16, 32, 64])
    
    # define the grid search parameters
    param_grid = dict(batch_size = [10, 50, 100, 200, 500, 1000],
                    epochs = [20, 60, 500, 1000, 1500],
                    n_layers=[1, 2, 3, 4, 8, 10, 15], 
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

# %% best model
best_model, best_PredictorScalerFit, best_TargetVarScalerFit = train_ANN(
    df, Predictors, num_nodes = 32, num_layers=2, epochs=1000,
)
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
model_list = [None] * zwally.shape[0]

for i in range(zwally.shape[0]):
    msk = points_within.loc[points_within.index_right == i, :].index
    df.loc[msk, "zwally_zone"] = i
    print(i, len(msk), "%0.2f" % (len(msk) / df.shape[0] * 100))

    df_cv = df.loc[
        df.zwally_zone != i,
    ].copy()

    model_list[i], _, _ = train_ANN(
        df_cv, Predictors, TargetVariable="temperatureObserved", calc_weights=True
    )

# defining function that makes the ANN estimate from predictor raw values
df = df.sort_values("date")

def ANN_model_cv(df_pred, model_list):
    pred = pd.DataFrame()
    for i in range(zwally.shape[0]):
        pred["T10m_pred_" + str(i)] = ANN_predict(df_pred, model_list[i])
    df_mean = pred.mean(axis=1)
    df_std = pred.std(axis=1)
    return df_mean.values, df_std.values


plt.figure()
plt.plot(
    df.temperatureObserved, ANN_predict(df[Predictors], model), "o", linestyle="None"
)
ME = np.mean(df.temperatureObserved - ANN_predict(df[Predictors], model))
RMSE = np.sqrt(
    np.mean((df.temperatureObserved - ANN_predict(df[Predictors], model)) ** 2)
)
plt.title("N = %i ME = %0.2f RMSE = %0.2f" % (len(df.temperatureObserved), ME, RMSE))

fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax = ax.flatten()
for k, site in enumerate(["DYE-2", "Summit", "KAN_U", "EKT"]):
    df_select = df.loc[df.site == site, :].copy()
    df_select["cv_mean"], df_select["cv_std"] = ANN_model_cv(
        df_select[Predictors], model_list
    )

    best_model = ANN_predict(df_select[Predictors], model)
    ax[k].fill(
        np.append(df_select.date, df_select.date[::-1]),
        np.append(
            best_model - df_select["cv_std"], (best_model + df_select["cv_std"])[::-1]
        ),
        "grey",
        label="CV standard deviation",
    )
    for i in range(zwally.shape[0]):
        if i == 0:
            lab = "CV models"
        else:
            lab = "_no_legend_"
        ax[k].plot(
            df_select.date,
            ANN_predict(df_select[Predictors], model_list[i]),
            "-",
            color="lightgray",
            alpha=0.8,
            label=lab,
        )
    ax[k].plot(df_select.date, best_model, "-", label="best model")

    ax[k].plot(
        df_select.date,
        df_select.temperatureObserved,
        "x",
        linestyle="None",
        label="observations",
    )
    ax[k].set_title(site)
ax[k].legend()

# %% Predicting T10m over ERA5 dataset
predict = 1

if predict:    
    print("predicting T10m over entire ERA5 dataset")
    print('converting to dataframe')
    tmp = ds_era[Predictors].to_dataframe()
    tmp = tmp.loc[tmp.notnull().all(1),:]
    
    print('applying ANN (takes ~45min on my laptop)')
    out = ANN_predict(tmp, best_model, best_PredictorScalerFit, best_TargetVarScalerFit)

    print(time.now() - t1)
    ds_T10m = out.to_frame().to_xarray()["temperaturePredicted"]
        .transpose("time", "latitude", "longitude")
        .values
    )

    ds_T10m.to_netcdf("predicted_T10m.nc")
ds_T10m = xr.open_dataset("predicted_T10m.nc")["T10m"]

# %% Predicting T10m uncertainty
predict = 0
if predict:
    print("predicting T10m uncertainty over entire ERA5 dataset")
    ds_T10m_std = ds_era["t2m"].copy().rename("T10m") * np.nan
    for time in progressbar.progressbar(
        ds_T10m_std.time.to_dataframe().values[12 * firn_memory :]
    ):
        tmp = ds_era.sel(time=time).to_dataframe()
        tmp["month"] = (pd.to_datetime(time[0]).month - 1) / 12
        _, tmp["cv_std"] = ANN_model_cv(tmp[Predictors], model_list)

        ds_T10m_std.loc[dict(time=time)] = (
            tmp["cv_std"]
            .to_frame()
            .to_xarray()["cv_std"]
            .transpose("time", "latitude", "longitude")
            .values
        )

    ds_T10m_std.to_netcdf("predicted_T10m_std.nc")
ds_T10m_std = xr.open_dataset("predicted_T10m_std.nc")["T10m"]

# ========= Supporting plots ============
# %% Selection and overview of the predictors
ds_era_GrIS = ds_era.mean(dim=("latitude", "longitude")).to_pandas()

fig, ax = plt.subplots(4, 3, figsize=(15, 15), sharex=True)
ax = ax.flatten()
for i in range(len(Predictors)):
    ds_era_GrIS[Predictors[i]].resample("Y").mean().plot(ax=ax[i], drawstyle="steps")
    ax[i].set_ylabel(Predictors[i])

ds_era_GrIS["t2m"].resample("Y").mean().plot(ax=ax[i + 1], drawstyle="steps")
ax[i + 1].set_ylabel("t2m")

# %% overview of the cross validation analysis
ds_era = xr.open_dataset("era5_monthly_plus_predictors.nc")
ds_T10m = xr.open_dataset("predicted_T10m.nc")["T10m"]
ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:4326")
ds_era = ds_era.rio.write_crs("EPSG:4326").rio.clip(ice.geometry.values, ice.crs)

ds_T10m_std_avg = (
    ds_T10m_std.mean(dim="time")
    .rio.write_crs("EPSG:4326")
    .rio.clip(ice.geometry.values, ice.crs)
)
ds_T10m_std_avg = ds_T10m_std_avg.rio.reproject("EPSG:3413")

fig, ax = plt.subplots(1, 1)
land.plot(color="k", ax=ax)
im = ds_T10m_std_avg.where(ds_T10m_std_avg < 1000).plot(
    ax=ax, cbar_kwargs={"label": "Average uncertainty of the ANN ($^o$C)"}
)
gpd.GeoDataFrame.from_file("Data/misc/Zwally_10_zones_3413.shp").boundary.plot(
    ax=ax, color="w", linestyle="-", linewidth=0.5
)
ax.set_axis_off()
plt.title("")
plt.xlim(-700000, 900000)
plt.ylim(-3400000, -574000)
plt.savefig("uncertainty_map.png", bbox_inches="tight")
plt.show()
