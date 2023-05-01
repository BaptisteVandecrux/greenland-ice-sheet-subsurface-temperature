# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.seterr(invalid="ignore")


df_info = pd.read_csv('Data/PROMICE/AWS_station_locations.csv')
df_info = df_info.loc[df_info.location_type == 'ice sheet',:]
path_to_PROMICE = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/L4/'

for site in df_info['stid']:
    break
        # %%
    # plt.close('all')
    site = 'KAN_U'
    
    df_aws = pd.read_csv(path_to_PROMICE+site+'_L4.csv')
    df_aws['time'] = pd.to_datetime(df_aws.time, utc=True)
    df_aws = df_aws.set_index('time')
    
    depth_cols_name = [v for v in df_aws.columns if "depth" in v]
    temp_cols_name = [v for v in df_aws.columns if "t_i_" in v and "depth" not in v and "10m" not in v]

    print(site)

    if df_aws[temp_cols_name].isnull().all().all():
        print('No thermistor data')
        # continue
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    plt.subplots_adjust(left=0.08, right=0.98, wspace=0.2, top=0.9)
    df_aws["z_surf_combined"].plot(
        ax=ax[0], color="black", label="surface", linewidth=3
    )
    (df_aws["z_surf_combined"] - 10).plot(
        ax=ax[0],
        color="red",
        linestyle="-",
        linewidth=4,
        label="10 m depth",
    )


    for i, col in enumerate(depth_cols_name):
        (-df_aws[col] + df_aws["z_surf_combined"]).plot(
            ax=ax[0],
            label="_nolegend_",alpha=0.5
        )

    ax[0].set_ylim(
        df_aws["z_surf_combined"].min() - 11,
        df_aws["z_surf_combined"].max() + 1,
    )

    for i in range(len(temp_cols_name)):
        df_aws[temp_cols_name[i]].interpolate(limit=14).plot(
            ax=ax[1], label="_nolegend_", alpha=0.5
        )

    if len(df_aws["t_i_10m"]) == 0:
        print("No 10m temp for ", site)
    else:
        df_aws["t_i_10m"].resample(
            "W"
        ).mean().plot(ax=ax[1], color="red", linewidth=5, label="10 m temperature")
    for i in range(2):
        ax[i].plot(np.nan, np.nan, c='w', label=' ') 
        ax[i].plot(np.nan, np.nan, c='w', label='individual sensors') 
        ax[i].plot(np.nan, np.nan, c='w', label=' ') 
        ax[i].legend()
    ax[0].set_ylabel("Height (m)")
    ax[1].set_ylabel("Subsurface temperature ($^o$C)")
    fig.suptitle(site)
    fig.savefig("figures/string processing/PROMICE_" + site + ".png", dpi=300)


# %% 10 m firn temp
df_PROMICE = pd.DataFrame()

for i in df_info.index:
    site = df_info.loc[i, 'stid']
    # break
    #     # %%
    # plt.close('all')
    # site = 'SWC_O'
    print(site)
    df_aws = pd.read_csv(path_to_PROMICE+site+'_L4.csv')
    df_aws['time'] = pd.to_datetime(df_aws.time, utc=True)
    df_aws = df_aws.set_index('time')
    
    if df_aws['t_i_10m'].isnull().all():
        print('No 10m temperature data')
        # continue

    df_10 = df_aws[['t_i_10m']].copy()
    df_10.columns = ['temperatureObserved']
    df_10.index = df_10.index.rename('date')
    
    df_10["latitude"] = df_info.loc[i, "lat"]
    df_10["longitude"] = df_info.loc[i, "lon"]
    df_10["elevation"] = df_info.loc[i, "alt"]
    df_10["site"] = site
    df_10["depthOfTemperatureObservation"] = 10
    df_10["note"] = ""

    # filtering
    df_10.loc[df_10["temperatureObserved"] > 0.1, "temperatureObserved"] = np.nan
    df_10.loc[df_10["temperatureObserved"] < -70, "temperatureObserved"] = np.nan
    
    df_PROMICE = pd.concat((df_PROMICE, df_10.reset_index()))

df_PROMICE = df_PROMICE.set_index("date")
df_PROMICE_month_mean = df_PROMICE.groupby("site").resample("M").mean().reset_index("site")
df_PROMICE = df_PROMICE.loc[df_PROMICE.temperatureObserved.notnull(), :]
df_PROMICE_month_mean = df_PROMICE_month_mean.loc[df_PROMICE_month_mean.temperatureObserved.notnull(), :]

df_PROMICE_month_mean.to_csv("Data/PROMICE/PROMICE_10m_firn_temperature.csv", sep=";")


