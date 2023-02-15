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
from progressbar import progressbar
import matplotlib

df = pd.read_csv("output/10m_temperature_dataset_monthly.csv")

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

df_too_cold = df.loc[df.temperatureObserved<-42, :]
df = df.loc[df.temperatureObserved>=-42, :]
df['date'] = pd.to_datetime(df.date, utc=True).dt.tz_localize(None)
df["year"] = pd.DatetimeIndex(df.date).year

land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")

ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
ice = ice.to_crs("EPSG:3413")


years = df["year"].values
years.sort()
df = df.reset_index(drop=True)

# Extracting ERA5 data

ds_era = xr.open_dataset("Data/ERA5/ERA5_monthly_temp_snowfall.nc")
tmp2 = xr.open_dataset('Data/ERA5/m_era5_t2m_2016_2021.nc')
tmp = xr.open_dataset('Data/ERA5/m_era5_t2m_2022.nc')
tmp = tmp.isel(expver=0).combine_first(tmp.isel(expver=1))
ds_era = xr.concat((ds_era,tmp), dim='time')

time_era = ds_era.time
firn_memory = 4

for i in range(firn_memory):
    df["t2m_" + str(i)] = np.nan
    df["sf_" + str(i)] = np.nan

df["time_era"] = np.nan
df["t2m_amp"] = np.nan

coords_uni = np.unique(df[["latitude", "longitude"]].values, axis=0)
print("interpolating ERA at the observation sites")
x = xr.DataArray(coords_uni[:, 0], dims="points")
y = xr.DataArray(coords_uni[:, 1], dims="points")
ds_interp = ds_era.interp(latitude=x, longitude=y, method="linear")

for i in (range(df.shape[0])):
    query_point = (df.iloc[i, :].latitude, df.iloc[i, :].longitude)
    index_point = np.where((coords_uni == query_point).all(axis=1))[0][0]
    tmp = ds_interp.isel(points=index_point)
    if (tmp.latitude.values, tmp.longitude.values) != query_point:
        print(wtf)
    for k in range(firn_memory):
        if (
            df.iloc[i, :].date + pd.DateOffset(years=0 - k)
        ) < tmp.time.min().values:
            continue
        if (
            df.iloc[i, :].date + pd.DateOffset(years=0 - k)
        ) > tmp.time.max().values:
            continue
        time_end = tmp.sel(
            time=df.iloc[i, :].date + pd.DateOffset(years=0 - k),
            method="ffill",
        ).time.values
        time_start = tmp.sel(
            time=df.iloc[i, :].date + pd.DateOffset(years=-1 - k) + pd.DateOffset(days=1),
            method="bfill",
        ).time.values

        if k == 0:
            df.iloc[i, df.columns.get_loc("t2m_amp")] = (
                tmp.sel(time=slice(time_start, time_end)).t2m.max().values
                - tmp.sel(time=slice(time_start, time_end)).t2m.min().values
            )
        df.iloc[i, df.columns.get_loc("t2m_" + str(k))] = (
            tmp.sel(time=slice(time_start, time_end)).t2m.mean().values
        )
        df.iloc[i, df.columns.get_loc("sf_" + str(k))] = (
            tmp.sel(time=slice(time_start, time_end)).sf.sum().values
        )
        if k == 0:
            df.iloc[i, df.columns.get_loc("time_era")] = (
                tmp.sel(time=slice(time_start, time_end)).time.max().values
            )

df["t2m_avg"] = df[["t2m_" + str(i) for i in range(firn_memory)]].mean(axis=1)
df["sf_avg"] = df[["sf_" + str(i) for i in range(firn_memory)]].mean(axis=1)

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

df = df.loc[df.time_era.notnull(), :]
df["date"] = pd.to_datetime(df.date)
df["time_era"] = pd.to_datetime(df.time_era)
df["year"] = df["date"].dt.year
df["month"] = (df["date"].dt.month - 1) / 12

df = df.loc[df.t2m_avg.notnull(), :]
df = df.loc[df.sf_avg.notnull(), :]
for i in range(firn_memory):
    df = df.loc[df["sf_" + str(i)].notnull(), :]
    df = df.loc[df["t2m_" + str(i)].notnull(), :]

# % Producing or loading input grids:
Predictors = (
    ["t2m_amp", "t2m_avg", "sf_avg"]
    + ["sf_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
)

produce_input_grid = 0
import progressbar
if produce_input_grid:
    print("Initializing array")
    for pred in Predictors:
        print(pred)
        ds_era[pred] = ds_era.t2m * np.nan
    for k in range(firn_memory):
        ds_era["t2m_" + str(k)] = ds_era.t2m * np.nan
    for k in range(firn_memory):
        ds_era["sf_" + str(k)] = ds_era.t2m * np.nan
    bar = progressbar.ProgressBar()
    for time in bar(ds_era.time.to_dataframe().values):
        for k in range(firn_memory):
            time_end = (
                pd.to_datetime(time)
                + pd.DateOffset(years=0 - k)
                + pd.DateOffset(days=-1)
            )
            time_start = pd.to_datetime(time) + pd.DateOffset(years=-1 - k)
            tmp = ds_era.sel(time=slice(time_start.values[0], time_end.values[0]))
            if tmp.t2m.shape[0] == 0:
                continue
            if tmp.time.shape[0] < 12:
                continue
            if k == 0:
                ds_era["t2m_amp"].loc[dict(time=time)] = (
                    tmp.t2m.max(dim="time").values - tmp.t2m.min(dim="time").values
                )
            ds_era["t2m_" + str(k)].loc[dict(time=time)] = tmp.t2m.mean(
                dim="time"
            ).values
            ds_era["sf_" + str(k)].loc[dict(time=time)] = tmp.sf.sum(dim="time").values

        ds_era["t2m_avg"].loc[dict(time=time)] = (
            ds_era.sel(time=time)[["t2m_" + str(k) for k in range(firn_memory)]]
            .to_array(dim="new")
            .mean("new")
            .values
        )
        ds_era["sf_avg"].loc[dict(time=time)] = (
            ds_era.sel(time=time)[["sf_" + str(k) for k in range(firn_memory)]]
            .to_array(dim="new")
            .mean("new")
            .values
        )
    ds_era.to_netcdf("output/era5_monthly_plus_predictors.nc")

ds_era = xr.open_dataset("output/era5_monthly_plus_predictors.nc")

# %% Representativity of the dataset

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

# %%plotting (less slow)
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


# %% ANN
# definition of the useful function to fit an ANN and run the trained model
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.layers import GaussianNoise
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def train_ANN(df, Predictors, TargetVariable="temperatureObserved",
              calc_weights=False, num_nodes = 40, num_layers = 2):
    if calc_weights:
        for i in range(len(Predictors)):
            hist1 = target_hist[i]
            hist2, _ = np.histogram(df[Predictors[i]].values, bins=bins)
            weights_bins = 0 * hist1
            weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
            ind = np.digitize(df[Predictors[i]].values, bins)
            df[Predictors[i] + "_w"] = weights_bins[ind - 1]
        df["weights"] = df[[p + "_w" for p in Predictors]].mean(axis=1)

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
    model.fit(X, y, batch_size=200, epochs=1000, verbose=0, sample_weight=w)
    return model, PredictorScalerFit, TargetVarScalerFit

def ANN_model(df, model, PredictorScalerFit, TargetVarScalerFit):
    X = PredictorScalerFit.transform(df.values)
    Y = model.predict(X)
    Predictions = pd.DataFrame(
        data=TargetVarScalerFit.inverse_transform(Y),
        index=df.index,
        columns=["temperaturePredicted"],
    ).temperaturePredicted
    Predictions.loc[Predictions > 0] = 0
    return Predictions

    
Predictors = (
    ["month", "t2m_amp", "t2m_avg", "sf_avg"]
    + ["sf_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
)
df['month'] = (pd.DatetimeIndex(df['date']).month-1)/11

# %% Testing the stability of the ANN
model = [None,None,None,None,None]
PredictorScalerFit = [None,None,None,None,None]
TargetVarScalerFit = [None,None,None,None,None]
for i in range(5):
    print(i)
    model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
        df, Predictors, num_nodes = 20, num_layers=2
    )
    
# %% Plotting stability test
df.loc[df.site=='CEN1','site'] = 'Camp Century'  
df.loc[df.site=='CEN2','site'] = 'Camp Century'   
df.loc[df.site=='CEN','site'] = 'Camp Century'   

fig, ax = plt.subplots(2, 3, figsize=(15, 15))
ax = ax.flatten()
for k, site in enumerate(["SwissCamp", "DYE-2", "KAN_U", "Camp Century", 'KPC_U', 'QAS_U']):
    df_select = df.loc[df.site == site, :].copy()

    ax[k].plot(pd.to_datetime(df_select.date).values,
                 df_select.temperatureObserved.values, 
                 marker="x", linestyle='None', label="observations")
    ax[k].set_title(site)

    for i in range(4):
        df_select['T10m_pred'] = ANN_model(df_select[Predictors], model[i], 
                                        PredictorScalerFit[i], TargetVarScalerFit[i]).values
        df_select = df_select.sort_values('date')
        ax[k].plot(df_select.date, df_select.T10m_pred)


# %% GridSearch
run_gridsearch = 1
if run_gridsearch:
    def create_model(n_layers, num_nodes):
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(len(Predictors),)))
        for i in range(n_layers):
            model.add(Dense(units=num_nodes, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model
    
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
    
# Best: 0.879104 using {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.857241 (0.065473) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.860077 (0.080005) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.879104 (0.072659) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.860805 (0.094318) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.874895 (0.063966) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.853947 (0.100261) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.861025 (0.078887) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.856441 (0.077439) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.860730 (0.091668) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.843159 (0.094143) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.858215 (0.097231) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.859242 (0.114457) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.863666 (0.075298) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.859792 (0.099163) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.869833 (0.080074) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.852131 (0.132142) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.876816 (0.058891) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.818858 (0.209782) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.674435 (0.184390) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.792447 (0.082326) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.834502 (0.057394) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.774601 (0.077686) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.817849 (0.108512) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.861159 (0.064054) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.805611 (0.070307) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.843098 (0.058637) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.870107 (0.072686) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.851074 (0.087104) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.850538 (0.097900) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.852706 (0.096214) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.862727 (0.075749) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.874594 (0.062164) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.866744 (0.075013) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.819494 (0.146188) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.823967 (0.161196) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.811046 (0.196999) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}

# Best: 0.923820 using {'batch_size': 10, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.844529 (0.075284) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.873701 (0.066676) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.858027 (0.077007) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.826090 (0.142378) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.864002 (0.077259) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.841072 (0.119865) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.881552 (0.062824) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.861627 (0.088106) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.912169 (0.036335) with: {'batch_size': 10, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.881386 (0.058688) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.840390 (0.125936) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.874093 (0.070518) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.887282 (0.057080) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.885417 (0.077705) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.918250 (0.048708) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.851278 (0.155154) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.863428 (0.116801) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.899332 (0.059853) with: {'batch_size': 10, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.877228 (0.052393) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.818159 (0.159584) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.813186 (0.189483) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.867902 (0.074434) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.858507 (0.106467) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.659186 (0.521866) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.897671 (0.069333) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.923820 (0.029341) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.913571 (0.040977) with: {'batch_size': 10, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
# 0.877004 (0.063679) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.862982 (0.084786) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.863718 (0.083048) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.832459 (0.120279) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.841155 (0.110298) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.853312 (0.112504) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.895317 (0.038178) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.887245 (0.060791) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.885847 (0.057671) with: {'batch_size': 50, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.872639 (0.072207) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.833624 (0.115754) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.854699 (0.081778) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.872223 (0.071181) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.835930 (0.130616) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.857792 (0.115435) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.838157 (0.122343) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.870870 (0.091012) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.813470 (0.212124) with: {'batch_size': 50, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.878629 (0.057566) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.850936 (0.073761) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.863573 (0.096596) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.675813 (0.469033) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.792314 (0.248194) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.817995 (0.190426) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.811142 (0.222227) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.807597 (0.222850) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.908239 (0.036493) with: {'batch_size': 50, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
# 0.812218 (0.129054) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.865773 (0.071175) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.868434 (0.071689) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.866596 (0.059748) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.880620 (0.068524) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.875904 (0.062100) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.839311 (0.102595) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.882874 (0.055681) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.868455 (0.081426) with: {'batch_size': 100, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.863460 (0.065367) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.866599 (0.064661) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.857613 (0.074432) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.808258 (0.176927) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.882571 (0.062809) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.830703 (0.157012) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.852125 (0.097893) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.884170 (0.072567) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.867169 (0.109768) with: {'batch_size': 100, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.848692 (0.091882) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.874964 (0.047478) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.855084 (0.121206) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.903659 (0.046769) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.845632 (0.125261) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.837465 (0.164635) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.592948 (0.657222) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.857328 (0.130543) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.844296 (0.155092) with: {'batch_size': 100, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
# 0.836834 (0.051308) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.828838 (0.082399) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.855467 (0.090524) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.827237 (0.097426) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.855404 (0.095285) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.875770 (0.071608) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.874165 (0.046712) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.892649 (0.062619) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.863691 (0.088192) with: {'batch_size': 200, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.822967 (0.127743) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.847282 (0.093362) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.855494 (0.086676) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.855420 (0.094188) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.867574 (0.077291) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.850076 (0.118099) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.836635 (0.112296) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.861122 (0.095094) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.851315 (0.118450) with: {'batch_size': 200, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.858401 (0.076864) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.890832 (0.050454) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.879069 (0.066883) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.851814 (0.111734) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.837296 (0.142046) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.818494 (0.178636) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.857226 (0.122708) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.666383 (0.490050) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.863838 (0.103008) with: {'batch_size': 200, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
# 0.812110 (0.104727) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.859718 (0.057583) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.844393 (0.053225) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.853451 (0.061320) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.838093 (0.112257) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.864915 (0.083922) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.848703 (0.087141) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.843624 (0.082051) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.863965 (0.084753) with: {'batch_size': 500, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.854150 (0.074751) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.854836 (0.080819) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.864091 (0.076044) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.884499 (0.061321) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.852858 (0.095270) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.863864 (0.101577) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.866821 (0.068214) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.848448 (0.105475) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.831504 (0.130678) with: {'batch_size': 500, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.849783 (0.094818) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.877055 (0.071126) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.885074 (0.061617) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.875603 (0.058841) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.865739 (0.095970) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.891768 (0.066455) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.863147 (0.108773) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.728113 (0.371166) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.756440 (0.330313) with: {'batch_size': 500, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
# 0.491443 (0.493817) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 16}
# 0.790247 (0.118426) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 32}
# 0.832660 (0.062930) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 1, 'num_nodes': 64}
# 0.694265 (0.125558) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 16}
# 0.867550 (0.067293) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 32}
# 0.833219 (0.112063) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 2, 'num_nodes': 64}
# 0.815469 (0.099156) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 3, 'num_nodes': 16}
# 0.864070 (0.080145) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 3, 'num_nodes': 32}
# 0.860014 (0.090787) with: {'batch_size': 1000, 'epochs': 20, 'n_layers': 3, 'num_nodes': 64}
# 0.809860 (0.049927) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 16}
# 0.853416 (0.073050) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 32}
# 0.844215 (0.092269) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 1, 'num_nodes': 64}
# 0.846142 (0.068664) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 16}
# 0.874897 (0.074476) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 32}
# 0.876098 (0.078126) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 2, 'num_nodes': 64}
# 0.878405 (0.066481) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 3, 'num_nodes': 16}
# 0.871540 (0.072514) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 3, 'num_nodes': 32}
# 0.869007 (0.080054) with: {'batch_size': 1000, 'epochs': 60, 'n_layers': 3, 'num_nodes': 64}
# 0.859760 (0.080814) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 16}
# 0.863719 (0.073954) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 32}
# 0.857214 (0.085003) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 1, 'num_nodes': 64}
# 0.891409 (0.057571) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 16}
# 0.879160 (0.077732) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 32}
# 0.827773 (0.159106) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 2, 'num_nodes': 64}
# 0.870521 (0.086786) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 3, 'num_nodes': 16}
# 0.852455 (0.117129) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 3, 'num_nodes': 32}
# 0.788944 (0.259407) with: {'batch_size': 1000, 'epochs': 500, 'n_layers': 3, 'num_nodes': 64}
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
        pred["T10m_pred_" + str(i)] = ANN_model(df_pred, model_list[i])
    df_mean = pred.mean(axis=1)
    df_std = pred.std(axis=1)
    return df_mean.values, df_std.values


plt.figure()
plt.plot(
    df.temperatureObserved, ANN_model(df[Predictors], model), "o", linestyle="None"
)
ME = np.mean(df.temperatureObserved - ANN_model(df[Predictors], model))
RMSE = np.sqrt(
    np.mean((df.temperatureObserved - ANN_model(df[Predictors], model)) ** 2)
)
plt.title("N = %i ME = %0.2f RMSE = %0.2f" % (len(df.temperatureObserved), ME, RMSE))

fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax = ax.flatten()
for k, site in enumerate(["DYE-2", "Summit", "KAN_U", "EKT"]):
    df_select = df.loc[df.site == site, :].copy()
    df_select["cv_mean"], df_select["cv_std"] = ANN_model_cv(
        df_select[Predictors], model_list
    )

    best_model = ANN_model(df_select[Predictors], model)
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
            ANN_model(df_select[Predictors], model_list[i]),
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

# %% Predicting T10m
predict = 0
if predict:
    print("predicting T10m over entire ERA5 dataset")
    ds_T10m = ds_era["t2m"].copy().rename("T10m") * np.nan
    for time in progressbar.progressbar(
        ds_T10m.time.to_dataframe().values[12 * firn_memory :]
    ):
        tmp = ds_era.sel(time=time).to_dataframe()
        tmp["month"] = (pd.to_datetime(time[0]).month - 1) / 12
        out = ANN_model(tmp[Predictors], model)
        ds_T10m.loc[dict(time=time)] = (
            out.to_frame()
            .to_xarray()["temperaturePredicted"]
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
