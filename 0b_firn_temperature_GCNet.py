# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import firn_temp_lib as ftl


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

