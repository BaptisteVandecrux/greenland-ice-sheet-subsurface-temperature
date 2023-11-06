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
import xarray as xr

np.seterr(invalid="ignore")
needed_cols = ["date", "site", "latitude", "longitude", "elevation", "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note", "error", "durationOpen", "durationMeasured", "method"]

df_info = pd.read_csv('Data/PROMICE/AWS_station_locations.csv')
df_info = df_info.loc[df_info.location_type == 'ice sheet',:]
path_to_PROMICE = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/L4/'
plotting = 0
if plotting:
    for site in df_info['stid']:
        # break
        #     # %%
        # # plt.close('all')
        # site = 'THU_U2'
        
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


# %% exporting all subsurface temperatures to netcdf
export_nc = True
if export_nc:   

    ref_promice = "Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022."
    ref_gcnet = "Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022."
    
    df_info["reference_short"] = "PROMICE: Fausto et al. (2021); How et al. (2022)"
    df_info.loc[ df_info.station_type == 'two booms', "reference_short"] = "GC-Net continuation: Fausto et al. (2021); How et al. (2022)"
    df_info["reference"] = ref_promice
    df_info.loc[ df_info.station_type == 'two booms', "reference"] = ref_gcnet
    df_info = df_info.set_index('stid')
    
    df_all = pd.DataFrame()
    for i, site in enumerate(df_info.index):
        print(site)
        df_aws = pd.read_csv(path_to_PROMICE+site+'_L4.csv')
        df_aws['time'] = pd.to_datetime(df_aws.time, utc=True)

        temp_var = ['t_i_'+str(i) for i in range(1,12) if 't_i_'+str(i) in df_aws.columns]
        depth_var = ['depth_t_i_'+str(i) for i in range(1,12) if 'depth_t_i_'+str(i) in df_aws.columns]
    
        df_aws = df_aws[['time']+temp_var+depth_var]
        df_aws['site'] = site
        for v in temp_var + depth_var:
            if v not in df_aws:
                df_aws[v] = np.nan
        df_all = pd.concat((df_all, df_aws),ignore_index=True)
    df_save = df_all.copy()
#%%
    import firn_temp_lib as ftl
    df_all = df_save.copy()

    df_all["latitude"] = df_info.loc[df_all.site, "lat"].values
    df_all["longitude"] = df_info.loc[df_all.site, "lon"].values
    df_all["elevation"] = df_info.loc[df_all.site, "alt"].values
    df_all["reference"] = df_info.loc[df_all.site, "reference"].values
    df_all["reference_short"] = df_info.loc[df_all.site, "reference_short"].values
    
    ds_all = ftl.df_to_xarray(df_all, temp_var, depth_var)

    ftl.write_netcdf(ds_all, 'Data/netcdf/PROMICE_GC-Net_GEUS_subsurface_temperatures.nc')

# %% 10 m firn temp
df_PROMICE = pd.DataFrame()

for i, site in enumerate(df_info.index):
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

    df_10["reference"] = "Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022."
    
    df_10["reference_short"] = "PROMICE: Fausto et al. (2021); How et al. (2022)"
    if df_info.loc[i,'station_type'] == 'two booms':
        df_10["reference"] = "Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022."
        
        df_10["reference_short"] = "GC-Net continuation: Fausto et al. (2021); How et al. (2022)"
        
    # filtering
    df_10.loc[df_10["temperatureObserved"] > 0.1, "temperatureObserved"] = np.nan
    df_10.loc[df_10["temperatureObserved"] < -70, "temperatureObserved"] = np.nan
    
    df_PROMICE = pd.concat((df_PROMICE, df_10.reset_index()))

df_PROMICE.index = pd.to_datetime(df_PROMICE.index)
df_PROMICE = df_PROMICE.set_index("date")
df_PROMICE = df_PROMICE.loc[df_PROMICE.temperatureObserved.notnull(), :]

col_to_avg = ['site', 'temperatureObserved',  'latitude', 'longitude', 'elevation']
df_PROMICE_month_mean = df_PROMICE[col_to_avg].groupby("site").resample("M").mean().reset_index("site")

col_to_fill = ['site', 'depthOfTemperatureObservation', 'note', 'reference', 'reference_short']
tmp = df_PROMICE[col_to_fill].groupby("site").resample("M").first().drop(columns='site').reset_index("site")
if (tmp.site != df_PROMICE_month_mean.site).any():
    print(wtf)
df_PROMICE_month_mean[col_to_fill] = tmp
df_PROMICE_month_mean.to_csv("Data/PROMICE/PROMICE_10m_firn_temperature.csv", sep=";")


