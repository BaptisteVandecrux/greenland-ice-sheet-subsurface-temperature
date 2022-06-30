# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import firn_temp_lib as ftl
# from sklearn.linear_model import LinearRegression

# %% Old GC-Net stations not processed in Vandecrux et al. 2020
meta = pd.read_csv("Data/GC-Net/Gc-net_documentation_Nov_10_2000.csv")
df_gcn = pd.DataFrame()

for site_id in [1, 4, 5, 9, 13, 14, 22]:  # , 23]:
    site = meta.loc[meta.ID == site_id, "Station Name"].values[0]
    header = 52

    if site_id == 23:
        header = 34
        # 18 - 29
    df = pd.read_csv(
        "Data/GC-Net/" + str(site_id).zfill(2) + "c.dat_Req1957",
        sep=" ",
        skiprows=header + 1,
        header=None,
        index_col=None,
    )
    df = df.iloc[:, [1, 2] + [i for i in range(17, 29)]]
    df[df == 999] = np.nan
    temp_cols = ["TS" + str(i) for i in range(1, 11)]
    df.columns = ["year", "doy", "HS1", "HS2"] + temp_cols
    df["time"] = (np.asarray(df["year"], dtype="datetime64[Y]") - 1970) + (
        np.asarray((df["doy"] - 1) * 86400 * 1e9, dtype="timedelta64[ns]")
    )
    df = df.set_index("time")
    df = df.resample("M").mean()

    ind_first = df[temp_cols].mean(axis=1, skipna=True).first_valid_index()
    if site == "GITS":
        ind_first = pd.to_datetime("2001-03-01")

    ind_last = df[temp_cols].mean(axis=1, skipna=True).last_valid_index()
    if site == "Humboldt":
        ind_last = pd.to_datetime("2008-03-01")
    if site == "JAR":
        ind_last = pd.to_datetime("2003-03-01")
    if site == "PET-ELA":
        ind_last = pd.to_datetime("2006-05-01")
    if site == "SwissCamp":
        ind_last = pd.to_datetime("2011-01-01")
    df = df.loc[ind_first:ind_last, :]

    df_save = df.copy()

    if site == "NGRIP":
        df.loc["2005-06-01":, "HS1"] = df.loc["2005-06-01":, "HS1"].values + 1.5
        df.loc["2005-06-01":, "HS2"] = df.loc["2005-06-01":, "HS2"].values + 1.5
        df.loc["2006-06-01":, "HS1"] = df.loc["2006-06-01":, "HS1"].values + 0.5
        df.loc["2006-06-01":, "HS2"] = df.loc["2006-06-01":, "HS2"].values + 0.5
    if site == "SwissCamp":
        df.loc["2004-03-01":, "HS1"] = df.loc["2004-03-01":, "HS1"].values - 2
        df.loc["2004-03-01":, "HS2"] = df.loc["2004-03-01":, "HS2"].values - 2
        df.loc["2011-03-01":, "HS1"] = df.loc["2011-03-01":, "HS1"].values - 5
        df.loc["2011-03-01":, "HS2"] = df.loc["2011-03-01":, "HS2"].values - 5
        df.loc["2012-01-01":, "HS1"] = df.loc["2012-01-01":, "HS1"].values + 3
        df.loc["2012-01-01":, "HS2"] = df.loc["2012-01-01":, "HS2"].values + 3
        df.loc["2016-05-01":, "HS1"] = df.loc["2016-05-01":, "HS1"].values - 3
        df.loc["2016-05-01":, "HS2"] = df.loc["2016-05-01":, "HS2"].values - 3
    # if site == 'GITS':
    #     df.loc['1999-01-01':,'HS1'] = df.loc['1999-01-01':,'HS1'].values+3
    #     df.loc['1999-01-01':,'HS2'] = df.loc['1999-01-01':,'HS2'].values+3
    #     df.loc['2001-01-01':,'HS1'] = df.loc['2001-01-01':,'HS1'].values+2
    #     df.loc['2001-01-01':,'HS2'] = df.loc['2001-01-01':,'HS2'].values+2
    if site == "Humboldt":
        df.loc["2007-02-01":, "HS1"] = df.loc["2007-02-01":, "HS1"].values + 1.4
        df.loc["2007-02-01":, "HS2"] = df.loc["2007-02-01":, "HS2"].values + 1.4
        df.loc["2003-04-01":"2003-05-01", "HS1"] = np.nan
        df.loc["2003-04-01":"2003-05-01", "HS2"] = np.nan

    df["HS_summary"] = df[["HS1", "HS2"]].mean(axis=1, skipna=True)
    if all(df["HS_summary"].isnull()):
        df["HS_summary"] = 0
    df["HS_summary"] = (
        df["HS_summary"] - df["HS_summary"][df["HS_summary"].first_valid_index()]
    )
    df["HS_summary"] = df["HS_summary"].interpolate()

    fig, ax = plt.subplots()
    df_save["HS1"].plot(ax=ax)
    df_save["HS2"].plot(ax=ax)
    plt.title(site)
    df["HS1"].plot(ax=ax)
    df["HS2"].plot(ax=ax)
    df["HS_summary"].plot(ax=ax, linewidth=3)
    plt.title(site)

    depth_cols = ["depth_" + str(i) for i in range(1, 11)]
    depth_ini_val = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]
    depth_ini_val = np.flip(depth_ini_val)

    if site == "PET-ELA":
        depth_ini_val = np.flip(depth_ini_val)
    if site == "Humboldt":
        depth_ini_val = [0.1, 9.1, 8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1]

    if site == "GITS":
        df_gits = df
        df = pd.read_csv(
            "Data/GC-Net/data_GITS_combined_hour.txt", sep="\t", index_col=False
        )
        df = df.loc[df.Year > 1998, :]
        df = df.loc[df.Year < 2003, :]
        df[df == -999] = np.nan

        df["HS_summary"] = df["SurfaceHeightm"]

        df["date"] = pd.to_datetime(df["time"] - 719529, unit="d").round("s")
        df = df.set_index("date")
        df = df.resample("M").mean()

        temp_cols = ["IceTemperature" + str(i) + "C" for i in range(1, 11)]

    for i, depth in enumerate(depth_cols):
        df[depth] = depth_ini_val[i] + df["HS_summary"]

    df_10 = ftl.interpolate_temperature(
        df.index, df[depth_cols].values, df[temp_cols].values, kind="linear", title=site
    )
    df_10 = df_10.reset_index()
    df_10["latitude"] = meta.loc[meta["Station Name"] == site, "Northing"].values[0]
    df_10["longitude"] = meta.loc[meta["Station Name"] == site, "Easting"].values[0]
    df_10["elevation"] = meta.loc[meta["Station Name"] == site, "Elevation"].values[0]
    df_10["reference"] = "GC-Net"
    df_10["reference_short"] = "GC-Net"
    df_10["site"] = site
    df_10["note"] = "depth unsure"
    df_10["depthOfTemperatureObservation"] = 10
    df_gcn = df_gcn.append(
        df_10[
            [
                "date",
                "site",
                "latitude",
                "longitude",
                "elevation",
                "depthOfTemperatureObservation",
                "temperatureObserved",
                "reference",
                "reference_short",
                "note",
            ]
        ]
    )

df_gcn.to_csv("Data/GC-Net/10m_firn_temperature.csv")

# %% New GC-Net stations
meta = pd.read_csv("Data/GCNv2/metadata.txt", sep="\t")
df_gcn2 = pd.DataFrame()
import shutil

for site in meta.site_short: # 
    print(site)
    if site in ["JAR", "CEN2"]:
        print("no thermistor string intalled")
        continue

        
    IMEI = meta.loc[meta.site_short == site, "IMEI"].values[0]
    if site in ["SDL"]:
        filename = 'Data/GCNv2//TOA5_23810.DataTable10min.dat'
    else:
        filename = 'Data/GCNv2/AWS_'+str(IMEI)+'.txt'
    # break
    # trying to copy most recent files:
    try:
        shutil.copyfile('G:/test/aws_data/AWS_'+str(IMEI)+'.txt', 'Data/GCNv2/AWS_'+str(IMEI)+'.txt')
    except: 
        print('cannot copy latest file, using local file instead')    
        

    cols = [ "time", "counter", "Pressure_L", "Pressure_U", "Asp_temp_L", "Asp_temp_U","Humidity_L", "Humidity_U", "WindSpeed_L", "WindDirection_L", "WindSpeed_U","WindDirection_U", "SWUpper", "SWLower", "LWUpper", "LWLower","TemperatureRadSensor","SR1", "SR2", "thermistorstring_1", "thermistorstring_2","thermistorstring_3", "thermistorstring_4", "thermistorstring_5","thermistorstring_6", "thermistorstring_7", "thermistorstring_8","thermistorstring_9", "thermistorstring_10", "thermistorstring_11", "Roll","Pitch", "Heading", "Rain_amount_L", "Rain_amount_U", "Gtime", "latitude", "longitude","altitude", "HDOP", "FanCurrent_L", "FanCurrent_U", "BattVolt", "PressureMinus1000_L","Asp_temp_L2", "Humidity_L", "WindSpeed_S_L", "WindDirection_S_L", "?" ]

    values = [str, str, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float ]
    
    firn_temp_cols = [  "thermistorstring_1",  "thermistorstring_2",  "thermistorstring_3", "thermistorstring_4", "thermistorstring_5", "thermistorstring_6", "thermistorstring_7", "thermistorstring_8", "thermistorstring_9",  "thermistorstring_10", "thermistorstring_11",
    ]
    depth_cols = ["depth_" + str(i) for i in range(1, 1 + len(firn_temp_cols))]
    depth_ini_val = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]
    if site in ["SWC", "JAR"]:
        # SWC and JAR are PROMICE-type stations
        cols = [ "time", "counter", "Pressure", "Asp_temp", "Humidity", "WindSpeed","WindDirection", "SWUpper", "SWLower", "LWUpper", "LWLower", "TemperatureRadSensor","SR1", "SR2", "IceHeight", "thermistorstring_1", "thermistorstring_2","thermistorstring_3", "thermistorstring_4", "thermistorstring_5","thermistorstring_6", "thermistorstring_7", "thermistorstring_8", "Roll","Pitch", "Rain_amount", "TimeGPS", "Heading", "latitude", "longitude", "altitude","Rain_amount", "Rain_amount2", "counterx", "ss", "Giodal", "GeoUnit", "Battery","NumberSatellites", "HDOP", "FanCurrent", "FanCurrent2", "Quality", "LoggerTemp", "?1","?2"]

        values = [ str, str, float, float, float, float, float, float, float, float, float, float,float, float, float, float, float, float, float, float, float, float, float, float, float,float, float, float, float, float, float, float, float, float, float, float, float, float,float, float, float, float, float, float, float, float]
        firn_temp_cols = [
            "thermistorstring_1",  "thermistorstring_2", "thermistorstring_3",
            "thermistorstring_4",  "thermistorstring_5",  "thermistorstring_6",
            "thermistorstring_7", "thermistorstring_8",
        ]
        depth_cols = ["depth_" + str(i) for i in range(1, 1 + len(firn_temp_cols))]
        depth_ini_val = [1, 2, 3, 4, 5, 6, 7, 10]
        
    try:
        if site == "SWC":
            df = pd.read_csv(
                "Data/GCNv2/AWS_" + str(IMEI) + ".txt",
                index_col=None,
                header=None,
                names=cols,
            )
        elif site == "SDL":
            df = pd.read_csv(filename, skiprows=5, low_memory=False)
            cols.append('?')
            df.columns = cols
            df.time = pd.to_datetime(df.time) - pd.Timedelta(days=365*102-88, minutes=8, seconds=16)
        else:
            df = pd.read_csv(filename,  index_col=None, header=None)
            df.columns = cols

        df = df.astype(dict(zip(cols, values)), errors="ignore")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        continue

    if site == "NSE":
        df = df.iloc[1:, :]
    if site == "SWC":
        df = df.iloc[2:, :]

    lat_deg = np.trunc(df["latitude"] / 100)
    lat_min = df["latitude"] - lat_deg * 100
    df["latitude"] = lat_deg.values + lat_min / 60
    lon_deg = np.trunc(df["longitude"] / 100)
    lon_min = df["longitude"] - lon_deg * 100
    df["longitude"] = -(lon_deg.values + lon_min / 60)

    print(
        meta.loc[meta.site_short == site, ["latitude", "longitude", "elevation"]].values
    )
    print(
        df["latitude"].iloc[24:].mean(),
        df["longitude"].iloc[24:].mean(),
        df["altitude"].iloc[24:].mean(),
    )

    # print(df[firn_temp_cols].mean(skipna=True))
    df["time"] = pd.to_datetime(df["time"])
    df = df.reset_index(drop=True).set_index("time")
    df = df.resample("H").mean().interpolate(limit=24 * 7)
    df = df.iloc[24 * 7 :, :]

    # filtering temperature
    for col in firn_temp_cols:
        df.loc[df[col] > -0.1, col] = np.nan
        if site in ["SWC", "SDM"]:
            ind_filter = (
                df[col]
                .interpolate(limit=24 * 7)
                .rolling(24 * 3, center=True)
                .min()
                .values
                - df[col].interpolate(limit=24 * 7)
            ).abs() > 1
            df.loc[ind_filter, col] = np.nan
        df[col] = df[col].interpolate(limit=24 * 7).values

    # creating depth columns
    hs1 = df.SR1 * np.sqrt((df.Asp_temp_L + 273.15) / 273.15)
    hs2 = df.SR2 * np.sqrt((df.Asp_temp_U + 273.15) / 273.15)
    hs1.loc[hs1<1] = np.nan
    hs2.loc[hs2<1] = np.nan
    if site == 'NEEM':
        hs1.loc[hs1>3] = np.nan
        hs2.loc[hs2>4] = np.nan
    elif site == 'NSU':
        df.loc['2021-12-08':'2021-12-17',firn_temp_cols[-1]] = np.nan
    hs_summary = (hs1 + hs2) / 2
    hs_summary = hs_summary[hs_summary.first_valid_index()] - hs_summary 
    hs_summary = hs_summary.rolling(24, center=True, min_periods=1).mean()

    plt.figure()
    hs1.plot()
    hs2.plot()
    hs_summary.plot(linewidth=3)

    for i, depth in enumerate(depth_cols):
        df[depth] = depth_ini_val[i] + hs_summary.values
    df_10 = ftl.interpolate_temperature(
        df.index,
        df[depth_cols].values,
        df[firn_temp_cols].values,
        kind="linear",
        title=site,
    )
    df_10 = df_10.set_index("date").resample("M").mean().reset_index()
    df_10["latitude"] = meta.loc[meta.site_short == site, "latitude"].values[0]
    df_10["longitude"] = meta.loc[meta.site_short == site, "longitude"].values[0]
    df_10["elevation"] = meta.loc[meta.site_short == site, "elevation"].values[0]
    df_10["reference"] = "GC-Net_v2 by GEUS"
    df_10["reference_short"] = "GC-Net_v2 by GEUS"
    df_10["site"] = site
    df_10["note"] = "depth unsure"
    df_10["depthOfTemperatureObservation"] = 10
    df_gcn2 = df_gcn2.append(
        df_10[
            [
                "date",
                "site",
                "latitude",
                "longitude",
                "elevation",
                "depthOfTemperatureObservation",
                "temperatureObserved",
                "reference",
                "reference_short",
                "note",
            ]
        ]
    )

df_gcn2.to_csv("Data/GCNv2/10m_firn_temperature.csv")
