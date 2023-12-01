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
plt.figure()
ice_4326.plot()
gdf.loc[[i for i in gdf.index if i not in ind_in]].plot(color='r', ax=plt.gca())
print(gdf.loc[[i for i in gdf.index if i not in ind_in]])
print(len(df)-len(ind_in), 'observations outside ice sheet mask')
df = df.loc[ind_in, :]

# temporal selection
print(((df.date<'1949-12-31') | (df.date>'2023')).sum(),'observations outside of 1950-2022')
print(df.loc[((df.date<'1949-12-31') | (df.date>'2023')),:])
df.loc[(df.date<='1949-12-01') | (df.date>='2023'),:]
df = df.loc[(df.date>'1949-12-01') & (df.date<'2023'),:]

print(len(df), 'observations kept')

land = gpd.GeoDataFrame.from_file("Data/misc/Land_3413.shp")

years = df["year"].values
years.sort()
df = df.reset_index(drop=True)

firn_memory = 5
Predictors = (
    ["t2m_amp", "t2m_10y_avg", "sf_10y_avg"]
    + ["sf_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
    + ["month"]
)

# Extracting ERA5 data
print('loading ERA5 data')
produce_input_grid = 0
if produce_input_grid:
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_era = xr.open_mfdataset(("Data/ERA5/ERA5_monthly_temp_snowfall.nc",
                                    'Data/ERA5/m_era5_t2m_sf_2022_proc.nc')).load()
    ds_era_1940 = xr.open_dataset('Data/ERA5/ERA5_monthly_temperature_precipitation_1940_1950.nc')[['t2m']]
    ds_era_1940['sf'] = xr.open_dataset('Data/ERA5/ERA5_monthly_snowfall_1940_1950.nc')['sf']
    
    ds_era_1940 = ds_era_1940.interp_like(ds_era.t2m.isel(time=0), method='linear')
    ds_era = ds_era.combine_first(ds_era_1940)
    
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
        ds_era["t2m_" + str(k)] = ds_era["t2m_0"].shift(time=12*k)
        ds_era["sf_" + str(k)] = ds_era["sf_0"].shift(time=12*k)
    ds_era['month'] = np.cos((ds_era.time.dt.month-1)/12*2*np.pi)
    
    #cropping the first 10 years
    ds_era = ds_era.isel(time=slice(10*12,None))
    ds_era.to_netcdf("output/era5_monthly_plus_predictors.nc")
else: 
    ds_era = xr.open_dataset("output/era5_monthly_plus_predictors.nc")

time_era = ds_era.time

for i in range(firn_memory):
    df["t2m_" + str(i)] = np.nan
    df["sf_" + str(i)] = np.nan

df["time_era"] = np.nan
df["t2m_amp"] = np.nan

# extrapolating ERA5
tmp = ds_era.isel(time=-1)
tmp['time'] = tmp.time + pd.Timedelta('31 days')
ds_era = xr.concat((ds_era, tmp), dim='time')

print("interpolating ERA at the observation sites")
tmp = ds_era.interp(longitude = xr.DataArray(df.longitude, dims="points"), 
                            latitude = xr.DataArray(df.latitude, dims="points"), 
                            time = xr.DataArray(df.date, dims="points"), 
                            method="nearest").load()
df[Predictors] = tmp[Predictors].to_dataframe()[Predictors].values

df["date"] = pd.to_datetime(df.date)
df["year"] = df["date"].dt.year

df.loc[df.t2m_amp.isnull()]
print('plotting')
fig, ax = plt.subplots(3, 4, figsize=(13,13))
ax = ax.flatten()
for i, ax in enumerate(ax):
    ax.plot( df[Predictors[i]], df.temperatureObserved, 
               marker="o", linestyle="None", markersize=1.5)
    
    ax.set_xlabel(Predictors[i])
    ax.set_ylabel('temperatureObserved')

# masking on ice sheet
ds_era = ds_era.rio.write_crs(4326)
ice_4326 = ice.to_crs(4326)
msk = ds_era.t2m_amp.isel(time=12*5).rio.clip(ice_4326.geometry, ice_4326.crs)
ds_era = ds_era.where(msk.notnull())

# %% Representativity of the dataset
print('calculating weights based on representativity')
bins_temp = np.linspace(230, 275, 15) - 273.15
bins_sf = np.append(np.linspace(0, 0.06, 15),3)*1000
bins_amp = np.linspace(0, 55, 15)

# first calculating the target histograms (slow)
pred_list = ["t2m_10y_avg", "sf_10y_avg", "t2m_amp"]
target_hist = [None] * len(pred_list)
for i in range(len(pred_list)):
    c=1
    d=0
    if pred_list[i] == 't2m_10y_avg': 
        bins = bins_temp
        d = -273.15
    if "amp" in pred_list[i]: bins = bins_amp
    if "sf" in pred_list[i]: 
        c=1000*12
        bins = bins_sf
    print(pred_list[i])
    target_hist[i], _ = np.histogram(ds_era[pred_list[i]].values*c + d, bins=bins)   
    target_hist[i] = target_hist[i].astype(np.float32) / target_hist[i].sum()

    hist1 = target_hist[i]
    hist2, _ = np.histogram(df[pred_list[i]].values*c + d, bins=bins)

    weights_bins = 0 * hist1 + 1
    weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
    ind = np.digitize(df[pred_list[i]].values*c + d, bins)
    df[pred_list[i] + "_w"] = weights_bins[ind - 1]
df["weights"] = df[[p + "_w" for p in pred_list]].mean(axis=1)

#  plotting histograms
abc='abcdef'
pred_name = [r"$\overline{T_{2m,\ 10\ y}}$ (°C)",
             r"$\overline{SF_{10\ y}}$ (mm w.e.)",
             r"$T_{2m, amp.}$ (°C)"]

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median
h=['','','','']

fig, ax = plt.subplots(2, 3, figsize=(11, 8), sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.8, wspace=0.05, hspace=1)
for k in range(2):
    if k == 0:
        ttl = "All observations weigh equally"
    else:
        ttl = "With weights based on representativity"
    for i in range(len(pred_list)):
        c=1
        d=0
        if pred_list[i] == 't2m_10y_avg': 
            bins = bins_temp
            d = -273.15
        if "amp" in pred_list[i]: bins = bins_amp
        if "sf" in pred_list[i]: 
            c=1000*12
            bins = bins_sf

        hist1 = target_hist[i]

        if k == 0:
            hist2, _ = np.histogram(df[pred_list[i]].values*c + d, bins=bins)
        elif k == 1:
            hist2, _ = np.histogram(
                df[pred_list[i]].values*c + d, bins=bins, weights=df["weights"].values
            )

        hist2 = hist2.astype(np.float32) / hist2.sum()

        h[0] = ax[k,i].bar(bins[:-1], hist2,
            width=(bins[1] - bins[0]),
            alpha=0.5, color="tab:blue", edgecolor="lightgray",
            label="ERA5 values at the observation sites",
        )
        if k == 0:
            med1=np.median(df[pred_list[i]].values*c + d)
        else:
            med1=weighted_median(df[pred_list[i]].values*c + d, df["weights"].values)
        h[1] = ax[k,i].axvline(med1,
            color="tab:blue", ls='--', lw=3, label="median")
        h[2] = ax[k,i].bar(bins[:-1], hist1,
            width=(bins[1] - bins[0]),
            alpha=0.5, color="tab:orange", edgecolor="lightgray",
            label="ERA5 values for the entire ice sheet",
        )
        med2 =np.nanmedian(ds_era[pred_list[i]].values*c + d)
        print(pred_list[i]+' %0.1f'%(med1- med2))
        h[3] = ax[k,i].axvline(med2,
            color="tab:orange", ls='--', lw=3,
            label="median",
        )

        ind = (hist1 + hist2)>0
        d_canberra = np.sum(np.abs(hist1[ind] - hist2[ind]) / (hist1[ind]))
        ax[k,i].annotate(r"$d_{Canberra}= %0.1f$"% d_canberra,
            xy=(0.65, 0.89), xycoords="axes fraction")

        ax[k,i].set_ylim(0,0.35)
        ax[k,i].set_xlabel(pred_name[i],fontsize=14)
        if i ==0:
            ax[k,i].set_ylabel("Probability (-)",fontsize=14)
        else:
            ax[k,i].set_ylabel("")
        ax[k,i].set_title('(%s)'%abc[k*3+i], loc='left',weight='bold')
        ax[k,i].grid(axis='y')
        if i ==1:
            ax[k,i].legend(handles=h,
                           loc="lower center",
                           title=ttl,
                           title_fontproperties={'weight':'bold'},
                           ncol=2,
                           bbox_to_anchor=(0.5, 1.2))

fig.savefig("figures/figure2_histograms.tif", dpi =900)
# fig.savefig("figures/figure2_histograms.pdf")
        

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
              epochs = 50, batch_size=3000, plot=False):

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
    model.compile(loss="mean_squared_error", optimizer="adam",weighted_metrics=[])
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, sample_weight=w)
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


# %% Testing batch size and epoch numbers
test_stability=0
plt.close('all')
if test_stability:
    filename="./output/batch_size_epochs_grid_final.txt"
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
    for epochs in [10, 25]+[e for e in range(50,300,50)]+[e for e in range(200,1000,100)]: #:
        for batch_size in [100, 500, 1000, 2000, 3000, 4000, 5000]:
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
                    batch_size=batch_size, plot=False,
                )
                df_test['out_mod_'+str(i)] = ANN_predict(df_test[Predictors], 
                    model[i],  PredictorScalerFit[i],  TargetVarScalerFit[i]).values
                df_train['out_mod_'+str(i)] = ANN_predict(df_train[Predictors], 
                    model[i],  PredictorScalerFit[i],  TargetVarScalerFit[i]).values
                Msg('%i, %i, %0.2f, %0.2f, %0.2f, %0.2f'%(batch_size, epochs,
                    (df_test['temperatureObserved'] - df_test['out_mod_'+str(i)]).mean(),
                    np.sqrt(((df_test['temperatureObserved'] - df_test['out_mod_'+str(i)])**2).mean()),
                        (df_train['temperatureObserved'] - df_train['out_mod_'+str(i)]).mean(),
                        np.sqrt(((df_train['temperatureObserved'] - df_train['out_mod_'+str(i)])**2).mean())))

 
# plt.close('all')
abc='abcdefghijklmnopqrst'
ls=['-','--',':','-.','-','--',':','-.','-','--',':','-.']
df_epochs = pd.read_csv('./output/batch_size_epochs_grid_final.txt',header=None)
df_epochs.columns=['batch_size','epochs','MD_val','RMSD_val', 'MD_train','RMSD_train']
tmp = df_epochs.groupby(['batch_size','epochs']).mean()
tmp[['MD_val_std','RMSD_val_std','MD_train_std','RMSD_train_std']] = df_epochs.groupby(['batch_size','epochs']).std().mean()
fig, ax = plt.subplots(7,2,figsize=(15,15), sharex=True)

plt.subplots_adjust(hspace=0.5)
for k, batch_size in enumerate(np.unique(tmp.index.get_level_values('batch_size'))):

    ax[k,0].errorbar(tmp.loc[batch_size].index + k/2 - 2, tmp.loc[batch_size, 'MD_val'],
                    yerr=tmp.loc[batch_size, 'MD_val_std'], label = 'validation')
    ax[k,1].errorbar(tmp.loc[batch_size].index + k/2 -2, tmp.loc[batch_size, 'RMSD_val'],
                     yerr=tmp.loc[batch_size, 'RMSD_val_std'])
    ax[k,0].errorbar(tmp.loc[batch_size].index + k/2 - 2, tmp.loc[batch_size, 'MD_train'],
                    yerr=tmp.loc[batch_size, 'MD_train_std'], label = 'training')
    ax[k,1].errorbar(tmp.loc[batch_size].index + k/2 -2, tmp.loc[batch_size, 'RMSD_train'],
                    yerr=tmp.loc[batch_size, 'RMSD_train_std'])
    ax[k,0].set_ylabel('MD (°C)')
    ax[k,1].set_ylabel('RMSD (°C)')
    # ax[0].set_xlim(0,200)
    ax[k,0].grid()
    ax[k,1].grid()
    if k==0:
        ax[k,0].legend(loc="lower center",ncol=2, bbox_to_anchor=(1,1.3))
    ax[k,0].set_title('('+abc[k*2]+') Batch size '+str(batch_size), loc='left')
    ax[k,1].set_title('('+abc[k*2+1]+') Batch size '+str(batch_size), loc='left')
    
    ax[k,0].set_yticks([-1, 0, 1])
    ax[k,0].set_ylim(-1,1)
    ax[k,1].set_ylim(0,5)
    ax[k,1].set_yticks([0, 2, 4])
ax[-1,1].set_xlabel('Number of epochs')
ax[-1,0].set_xlabel('Number of epochs')

fig.savefig('figures/learning_curve_all.png',dpi=300)


# %% Testing number of layers and nodes
test_stability=0
# plt.close('all')
if test_stability:
    filename="./output/layers_nodes_grid_final.txt"
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
    for layers in [1, 2, 3]:
        for nodes in [8, 16, 32, 64, 128, 256]:
            # print(epochs, 'epochs')
            num_models = 10
            model = [None]*num_models
            PredictorScalerFit = [None]*num_models
            TargetVarScalerFit = [None]*num_models
            # training the models
            for i in range(num_models):
                # print(i)
                model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
                    df_train, Predictors, num_nodes = nodes, num_layers=layers, epochs=150,
                    batch_size=4000, plot=False,
                )
                df_test['out_mod_'+str(i)] = ANN_predict(df_test[Predictors], 
                    model[i],  PredictorScalerFit[i],  TargetVarScalerFit[i]).values
                df_train['out_mod_'+str(i)] = ANN_predict(df_train[Predictors], 
                    model[i],  PredictorScalerFit[i],  TargetVarScalerFit[i]).values
                Msg('%i, %i, %0.2f, %0.2f, %0.2f, %0.2f'%(layers, nodes,
                    (df_test['temperatureObserved'] - df_test['out_mod_'+str(i)]).mean(),
                    np.sqrt(((df_test['temperatureObserved'] - df_test['out_mod_'+str(i)])**2).mean()),
                        (df_train['temperatureObserved'] - df_train['out_mod_'+str(i)]).mean(),
                        np.sqrt(((df_train['temperatureObserved'] - df_train['out_mod_'+str(i)])**2).mean())))


# plt.close('all')
abc='abcdefghijklmnopqrst'
ls=['-','--',':','-.','-','--',':','-.','-','--',':','-.']
df_nodes = pd.read_csv('./output/layers_nodes_grid_final.txt',header=None)
df_nodes.columns=['layers','nodes','MD_val','RMSD_val', 'MD_train','RMSD_train']
tmp = df_nodes.groupby(['layers','nodes']).mean()
tmp[['MD_val_std','RMSD_val_std','MD_train_std','RMSD_train_std']] = df_nodes.groupby(['layers','nodes']).std().mean()
fig, ax = plt.subplots(3,2,figsize=(8, 8), sharex=True)

plt.subplots_adjust(hspace=0.2)
for k, layers in enumerate([1,2,3]):

    ax[k,0].errorbar(tmp.loc[layers].index + k/2 - 2, tmp.loc[layers, 'MD_val'],
                    yerr=tmp.loc[layers, 'MD_val_std'], label = 'validation')
    ax[k,1].errorbar(tmp.loc[layers].index + k/2 -2, tmp.loc[layers, 'RMSD_val'],
                     yerr=tmp.loc[layers, 'RMSD_val_std'])
    ax[k,0].errorbar(tmp.loc[layers].index + k/2 - 2, tmp.loc[layers, 'MD_train'],
                    yerr=tmp.loc[layers, 'MD_train_std'], label = 'training')
    ax[k,1].errorbar(tmp.loc[layers].index + k/2 -2, tmp.loc[layers, 'RMSD_train'],
                    yerr=tmp.loc[layers, 'RMSD_train_std'])
    ax[k,0].set_ylabel('MD (°C)')
    ax[k,1].set_ylabel('RMSD (°C)')
    # ax[0].set_xlim(0,200)
    ax[k,0].grid()
    ax[k,1].grid()
    if k==0:
        ax[k,0].legend(loc="lower center",ncol=2, bbox_to_anchor=(1,1.3))
    ax[k,0].set_title('('+abc[k*2]+') Number of layers: '+str(layers), loc='left')
    ax[k,1].set_title('('+abc[k*2+1]+') Number of layers: '+str(layers), loc='left')
    
    ax[k,0].set_zorder(10)
    ax[k,1].set_zorder(10)
ax[-1,1].set_xlabel('Number of nodes')
ax[-1,0].set_xlabel('Number of nodes')

fig.savefig('figures/layers_and_nodes.png',dpi=300)



# %% best model
print('Training model on entire dataset')
best_model, best_PredictorScalerFit, best_TargetVarScalerFit = train_ANN(
    df, Predictors, num_nodes = 64, num_layers=2, epochs=150, batch_size=4000)

# %% synthetic monthly output
plt.figure()
ax=plt.gca()
for i in [1, 1000, 3000, 4000]:
    df_in = df.iloc[[i], :].copy()
    df_test = pd.DataFrame()
    for m in range(1,13):
        df_in.month = np.cos((m-1)/12*2*np.pi)
        df_in['month_true'] = m
        
        
        df_in['T10m'] = ANN_predict(df_in[Predictors],
                                    best_model,
                                    best_PredictorScalerFit,
                                    best_TargetVarScalerFit).item()
        
        df_test = pd.concat((df_test, df_in), ignore_index=True)
    label = '%0.3f °N, %0.3f °E'%tuple(df_in[['latitude','longitude']].iloc[0,:].to_list())
    ax.plot(df_test.month_true, df_test.T10m, marker='o', ls='None', label=label)
plt.legend()

plt.figure()
ax=plt.gca()
df_in = df.iloc[[1000], :].copy()
df_test = pd.DataFrame()
for m in  [-15, -10, -5, 0, 5, 10, 15]:
    df_in['t2m_10y_avg'] = df_in['t2m_10y_avg'] + m
    
    df_in['T10m'] = ANN_predict(df_in[Predictors],
                                best_model,
                                best_PredictorScalerFit,
                                best_TargetVarScalerFit).item()
    
    df_test = pd.concat((df_test, df_in), ignore_index=True)
ax.plot(df_test.t2m_10y_avg, df_test.T10m, marker='o', ls='None')
plt.xlabel('t2m_10y_avg')
plt.ylabel('T10m')
plt.xlabel('month')
plt.ylabel('T10m')

plt.figure()
ax=plt.gca()
df_in = df.iloc[[1000], :].copy()
df_test = pd.DataFrame()
for m in  [0.2, 0.5, 1, 1.5, 2]:
    df_in['sf_10y_avg'] = df_in['sf_10y_avg'] * m
    
    df_in['T10m'] = ANN_predict(df_in[Predictors],
                                best_model,
                                best_PredictorScalerFit,
                                best_TargetVarScalerFit).item()
    
    df_test = pd.concat((df_test, df_in), ignore_index=True)
ax.plot(df_test.sf_10y_avg, df_test.T10m, marker='o', ls='None')
plt.xlabel('sf_10y_avg')
plt.ylabel('T10m')
         
         
# %% SHAP analysis
make_shap_analysis = 0
if make_shap_analysis:
    import shap
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df[Predictors].values, 
                                                        df['temperatureObserved'].values,
                                                        test_size=0.1, random_state=42)
    explainer = shap.KernelExplainer(best_model.predict, best_PredictorScalerFit.transform(X_train))
    shap_values = explainer.shap_values(X_test, nsamples=100)
    shap_values_scaled = shap_values[0]
    for i in range(shap_values[0].shape[0]):
        shap_values_scaled[i,:] = best_TargetVarScalerFit.transform(shap_values[0][i,:].reshape(-1,1)).reshape(1,-1)
        
    shap.summary_plot(shap_values, best_PredictorScalerFit.transform(X_train), feature_names=Predictors)
    shap.summary_plot([shap_values_scaled], best_PredictorScalerFit.transform(X_train), feature_names=Predictors)
    
    
    features = Predictors
    groups = {
        't2m': [f for f in features if 't2m' in f],
        'sf': [f for f in features if 'sf' in f],
        'month': ['month']
    }
    
    from itertools import repeat, chain
    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))
    mapgroup = revert_dict(groups)
    
    def grouped_shap(shap_vals, features, groups):
        groupmap = revert_dict(groups)
        shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
        shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
        shap_grouped = shap_Tdf.groupby('group').sum().T
        return shap_grouped
    shap_vals = np.array(shap_values)[0,:,:]
    shap_group = grouped_shap(shap_vals, features, groups)
    Predictors_name = [
                        'T$_{2m}$ amp. in last year',
                        'T$_{2m}$ 10 year average',
                        'SF 10 year average',
                        'SF in year Y-1',
                        'SF in year Y-2',
                        'SF in year Y-3',
                        'SF in year Y-4',
                        'SF in year Y-5',
                        'T$_{2m}$ in year Y-1',
                        'T$_{2m}$ in year Y-2',
                        'T$_{2m}$ in year Y-3',
                        'T$_{2m}$ in year Y-4',
                        'T$_{2m}$ in year Y-5',
                        'month',
                        ]
    shap.summary_plot(shap_values_scaled, best_PredictorScalerFit.transform(X_test),
                      plot_type='violin',
                      feature_names=Predictors_name)
    plt.savefig('figures/SHAP_1.png')
    
    shap.summary_plot(shap_group.values, 
                      plot_type='violin',
                      feature_names=['month',
                                     'SF-dependent inputs',
                                     'T$_{2m}$-dependent inputs',])
    plt.savefig('figures/SHAP_2.png')

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


# %% Predicting T10m uncertainty
predict = 0
if predict:
    print("predicting T10m uncertainty over entire ERA5 dataset")
    print("...takes about 10 hours")
    for year in range(1950,2023): #np.unique(ds_era.time.dt.year):
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
        ["output/uncertainty/predicted_T10m_std_"+str(year)+".nc" for year in range(1950,2023)])
    ds_T10m_std = ds_T10m_std.rio.write_crs(4326)
    ice_4326 = ice.to_crs(4326)
    msk = ds_T10m_std.T10m_std.isel(time=0).rio.clip(ice_4326.geometry, ice_4326.crs)
    ds_T10m_std = ds_T10m_std.where(msk.notnull())
    ds_T10m_std['T10m_std'] = xr.where(ds_T10m_std.T10m_std<0.5, 0.5, ds_T10m_std.T10m_std)
    ds_T10m_std.attrs['author'] = 'Baptiste Vandecrux'
    ds_T10m_std.attrs['contact'] = 'bav@geus.dk'
    ds_T10m_std.attrs['description'] = 'Monthly grids of the uncertainty of the ANN Greenland ice sheet 10 m subsurface temperature for 1954-2022 calculated from the standard deviation between the predictions of 10 spatial cross-validation models.'
    ds_T10m_std.to_netcdf("output/T10m_uncertainty.nc")
ds_T10m_std = xr.open_dataset("output/T10m_uncertainty.nc")['T10m_std']

# %% checking that CV std
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
                    method='nearest').to_dataframe().T10m_std.plot(ax=ax[k])
    plt.title(site)


# %% Preparing for figure 3
# averaging and reprojecting uncertainty map

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

    ds_interp = ds.bfill(dim='latitude').interp(longitude=x, 
                                                latitude=y, 
                                                method="nearest")

    for i in (range(df.shape[0])):
        query_point = (df[dim1].values[i], df[dim2].values[i])
        index_point = np.where((coords_uni == query_point).all(axis=1))[0][0]
        tmp = ds_interp.T10m.isel(points=index_point).sel(
            time=df.date.values[i], method="nearest"
        )

        df.iloc[i, df.columns.get_loc(name_out)] = tmp.values
    return df

# extracting from T_10m dataset
df['T10m_ANN'] = np.nan
df = extract_T10m_values(
    ds_T10m.to_dataset(), df, dim1="longitude", dim2="latitude", name_out="T10m_ANN"
)


# %% Plotting Figure 3
df_10m = df.loc[df.depthOfTemperatureObservation.astype(float) == 10, :]

# plt.close('all')
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
    if abs(ME)<0.05:
        ME=abs(ME)
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

land.plot(color="lightgray", ax=ax_map)

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

    # best_model_pred = ANN_predict(df_select[Predictors], 
    #                               best_model, 
    #                               best_PredictorScalerFit, 
    #                               best_TargetVarScalerFit)
                                
    df_interp = ds_T10m.interp(longitude=df_site.longitude.item(), 
                          latitude=df_site.latitude.item(), method="linear").to_dataframe()['T10m']

        
    ax[k].step(df_interp.index, df_interp, color="tab:blue", where='post', label="ANN")
    
    ax[k].fill_between(df_interp.index, df_interp - cv_std, 
        df_interp + cv_std, color="turquoise", step='post',
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
fig.savefig('figures/figure3.tif', dpi =300)
fig.savefig('figures/figure3.png', dpi=120)
