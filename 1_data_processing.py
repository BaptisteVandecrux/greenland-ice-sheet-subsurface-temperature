# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
import progressbar
import matplotlib.pyplot as plt
import firn_temp_lib as ftl
import time
import os
import xarray as xr
import time


def interp_pandas(s, kind="quadratic"):
    # A mask indicating where `s` is not null
    m = s.notna().values
    s_save = s.copy()
    # Construct an interpolator from the non-null values
    # NB 'kind' instead of 'method'!
    kw = dict(kind=kind, fill_value="extrapolate")
    f = interp1d(s[m].index, s.loc[m].values.reshape(1, -1)[0], **kw)

    # Apply this to the indices of the nulls; reconstruct a series
    s[~m] = f(s[~m].index)[0]

    plt.figure()
    s.plot(marker='o',linestyle='none')
    s_save.plot(marker='o',linestyle='none')
    # plt.xlim(0, 60)
    return s


# %% Mock and Weeks
print("Loading Mock and Weeks")
df_all = pd.DataFrame(
    columns=[
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
        "error",
        "durationOpen",
        "durationMeasured",
        "method",
    ]
)

df_MW = pd.read_excel("Data/MockandWeeks/CRREL RR- 170 digitized.xlsx")
df_MW.loc[df_MW.Month.isnull(), "Day"] = 1
df_MW.loc[df_MW.Month.isnull(), "Month"] = 1
df_MW["date"] = pd.to_datetime(df_MW[["Year", "Month", "Day"]])
df_MW["note"] = "as reported in Mock and Weeks 1965"
df_MW["depthOfTemperatureObservation"] = 10
df_MW = df_MW.rename(
    columns={
        "Refstationnumber": "site",
        "Lat(dec_deg)": "latitude",
        "Lon(dec_deg)": "longitude",
        "Elevation": "elevation",
        "10msnowtemperature(degC)": "temperatureObserved",
        "Reference": "reference",
    }
)
df_MW = df_MW.loc[df_MW["temperatureObserved"].notnull()]

df_MW['durationOpen'] = 'NA'
df_MW['durationMeasured'] = 'NA'
df_MW['error'] = 0.5
df_MW['method'] = 'thermohms and a Wheats tone bridge, standard mercury or alcohol thermometers'

df_all = df_all.append(
    df_MW[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Benson (not reported in Mock and Weeks)
print("Loading Benson 1962")
df_benson = pd.read_excel("Data/Benson 1962/Benson_1962.xlsx")
df_benson.loc[df_benson.Month.isnull(), "Day"] = 1
df_benson.loc[df_benson.Month.isnull(), "Month"] = 1
df_benson["date"] = pd.to_datetime(df_MW[["Year", "Month", "Day"]])
df_benson["note"] = ""
df_benson["depthOfTemperatureObservation"] = 10
df_benson = df_benson.rename(
    columns={
        "Refstationnumber": "site",
        "Lat(dec_deg)": "latitude",
        "Lon(dec_deg)": "longitude",
        "Elevation": "elevation",
        "10msnowtemperature(degC)": "temperatureObserved",
        "Reference": "reference",
    }
)

# only keeping measurements not in Mock and Weeks
msk = (df_benson.site == '0-35') | (df_benson.site == 'French Camp VI')
df_benson = df_benson.loc[msk,:]

df_benson['durationOpen'] = 'measured in pit wall or borehole bottom after excavation'
df_benson['durationMeasured'] = 'few minutes'
df_benson['error'] = 'NA'
df_benson['method'] = 'Weston bimetallic thermometers'


df_all = df_all.append(
    df_benson[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Polashenski
print("Loading Polashenski")
df_Pol = pd.read_csv("Data/Polashenski/2013_10m_Temperatures.csv")
df_Pol.columns = df_Pol.columns.str.replace(" ", "")
df_Pol.date = pd.to_datetime(df_Pol.date, format="%m/%d/%y")
df_Pol[
    "reference"
] = "Polashenski, C., Z. Courville, C. Benson, A. Wagner, J. Chen, G. Wong, R. Hawley, and D. Hall (2014), Observations of pronounced Greenland ice sheet firn warming and implications for runoff production, Geophys. Res. Lett., 41, 4238–4246, doi:10.1002/2014GL059806."
df_Pol["reference_short"] = "Polashenski et al. (2014)"
df_Pol["note"] = ""
df_Pol["longitude"] = -df_Pol["longitude"]
df_Pol["depthOfTemperatureObservation"] = (
    df_Pol["depthOfTemperatureObservation"].str.replace("m", "").astype(float)
)
df_Pol['durationOpen'] = 'string lowered in borehole and left 30min for equilibrating with surrounding firn prior measurement start'
df_Pol['durationMeasured'] = 'overnight ~10 hours'
df_Pol['error'] = 0.1
df_Pol['method'] = 'thermistor string'

df_all = df_all.append(
    df_Pol[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",        
        ]
    ],
    ignore_index=True,
)

# %% Ken's dataset
print("Loading Kens dataset")
df_Ken = pd.read_excel(
    "Data/greenland_ice_borehole_temperature_profiles-main/data_filtered.xlsx"
)


df_all = df_all.append(
    df_Ken[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",        
        ]
    ],
    ignore_index=True,
)
# %% Sumup
df_sumup = pd.read_csv('Data/Sumup/SUMup_temperature_2022.csv')
df_sumup = df_sumup.loc[df_sumup.Latitude>0]
df_sumup = df_sumup.loc[df_sumup.Citation!=30]   # Redundant with Miege and containing positive temperatures

df_sumup['date'] = pd.to_datetime(df_sumup.Timestamp)
df_sumup['note']='as reported in Sumup'
df_sumup['reference'] = ''
df_sumup['site'] = ''
df_sumup.loc[df_sumup.Citation == 32, 'reference'] = 'Graeter, K., Osterberg, E. C., Ferris, D., Hawley, R. L., Marshall, H. P. and Lewis, G.: Ice Core Records of West Greenland Surface Melt and Climate Forcing, Geophys. Res. Lett., doi:10.1002/2017GL076641, 2018.'
df_sumup.loc[df_sumup.Citation == 32, 'reference_short'] = 'Graeter et al. (2018) as in SUMup'
df_sumup.loc[df_sumup.Citation == 33, 'reference'] = 'Lewis, G., Osterberg, E., Hawley, R., Marshall, H. P., Meehan, T., Graeter, K., McCarthy, F., Overly, T., Thundercloud, Z. and Ferris, D.: Recent precipitation decrease across the western Greenland ice sheet percolation zone, Cryosph., 13(11), 2797–2815, doi:10.5194/tc-13-2797- 2019, 2019.'
df_sumup.loc[df_sumup.Citation == 33, 'reference_short'] = 'Lewis et al. (2019) as in SUMup'

site_list = pd.DataFrame(np.array([[44, 'GTC01'], [45, 'GTC02'], [46, 'GTC04'],
                      [47, 'GTC05'], [48, 'GTC06'], [49, 'GTC07'],
                      [50, 'GTC08'], [51, 'GTC09'], [52, 'GTC11'],
                      [53, 'GTC12'], [54, 'GTC13'], [55, 'GTC14'],
                      [56, 'GTC15'], [57, 'GTC16']]), columns=['id','site']).set_index('id')
for ind in site_list.index:
    df_sumup.loc[df_sumup.Name == int(ind), 'site'] = site_list.loc[ind,'site']
    

df_sumup = df_sumup.drop(['Name','Citation', 'Timestamp'], axis=1)

df_sumup = df_sumup.rename(columns={'Latitude':'latitude',
                                    'Longitude':'longitude',
                                    'Elevation':'elevation',
                                    'Depth':'depthOfTemperatureObservation',
                                    'Temperature':'temperatureObserved',
                                    'Duration':'durationMeasured',
                                    'Error':'error',
                                    'Open_Time':'durationOpen',
                                    'Method':'method'})
df_all = df_all.append(df_sumup[['date',
            'site', 
            'latitude', 'longitude', 
            'elevation', 'depthOfTemperatureObservation', 
            'temperatureObserved', 
            'reference','reference_short', 
            "note",
            "error",
            "durationOpen",
            "durationMeasured",
            "method"]],ignore_index=True)

# ====> only temperature at 18m depth


# %% McGrath
print("Loading McGrath")
df_mcgrath = pd.read_excel(
    "Data/McGrath/McGrath et al. 2013 GL055369_Supp_Material.xlsx"
)
df_mcgrath = df_mcgrath.loc[df_mcgrath["Data Type"] != "Met Station"]
df_mcgrath["depthOfTemperatureObservation"] = np.array(
    df_mcgrath["Data Type"].str.split("m").to_list()
)[:, 0].astype(int)

df_mcgrath = df_mcgrath.rename(
    columns={
        "Observed\nTemperature (°C)": "temperatureObserved",
        "Latitude\n(°N)": "latitude",
        "Longitude (°E)": "longitude",
        "Elevation\n(m)": "elevation",
        "Reference": "reference",
        "Location": "site",
    }
)
df_mcgrath["note"] = "as reported in McGrath et al. (2013)"

df_mcgrath["date"] = pd.to_datetime(
    (df_mcgrath.Year * 10000 + 101).apply(str), format="%Y%m%d"
)
df_mcgrath["site"] = df_mcgrath["site"].str.replace("B4", "4")
df_mcgrath["site"] = df_mcgrath["site"].str.replace("B5", "5")
df_mcgrath["site"] = df_mcgrath["site"].str.replace("4-425", "5-0")

df_mcgrath['method'] = 'digital Thermarray system from RST©'
df_mcgrath['durationOpen'] = 0
df_mcgrath['durationMeasured'] = 30
df_mcgrath['error'] = 0.07
df_mcgrath
df_all = df_all.append(
    df_mcgrath[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

#  adding real date to Benson's measurement
df_fausto = pd.read_excel(
    "Data/misc/Data_Sheet_1_ASnowDensityDatasetforImprovingSurfaceBoundaryConditionsinGreenlandIceSheetFirnModeling.XLSX",
    skiprows=[0],
)

df_fausto.Name = df_fausto.Name.str.replace("station ", "")
for site in df_fausto.Name:
    if any(df_all.site == site):
        if np.sum(df_all.site == site) > 1:
            print(
                df_all.loc[
                    df_all.site == site,
                    ["temperatureObserved", "depthOfTemperatureObservation"],
                ]
            )
            print("duplicate, removing from McGrath")
            df_all = df_all.loc[
                ~np.logical_and(
                    df_all.site == site,
                    df_all["note"] == "as reported in McGrath et al. (2013)",
                )
            ]

        print(
            site,
            df_all.loc[df_all.site == site].date.values,
            df_fausto.loc[df_fausto.Name == site, ["year", "Month", "Day"]].values,
        )
        if (
            df_all.loc[df_all.site == site].date.iloc[0].year
            == df_fausto.loc[df_fausto.Name == site, ["year"]].iloc[0].values
        ):
            df_all.loc[df_all.site == site, "date"] = (
                pd.to_datetime(
                    df_fausto.loc[df_fausto.Name == site, ["year", "Month", "Day"]]
                )
                .astype("datetime64[ns]")
                .iloc[0]
            )
            print("Updating date")
        else:
            print("Different years")

# %% Hawley GrIT
print("Loading Hawley GrIT")
df_hawley = pd.read_excel("Data/Hawley GrIT/GrIT2011_9m-borehole_calc-temps.xlsx")
df_hawley = df_hawley.rename(
    columns={
        "Pit name (tabs)": "site",
        "Date": "date",
        "Lat (dec.degr)": "latitude",
        "Long (dec.degr)": "longitude",
        "Elevation": "elevation",
        "9-m temp": "temperatureObserved",
    }
)
df_hawley["depthOfTemperatureObservation"] = 9
df_hawley["note"] = ""
df_hawley[
    "reference"
] = "Bob Hawley. 2014. Traverse physical, chemical, and weather observations. arcitcdata.io, doi:10.18739/A2W232. "
df_hawley["reference_short"] = "Hawley (2014) GrIT"

df_hawley = df_hawley.loc[[isinstance(x, float) for x in df_hawley.temperatureObserved]]
df_hawley = df_hawley.loc[df_hawley.temperatureObserved.notnull()]

df_hawley['method'] = 'thermistor'
df_hawley['durationOpen'] = 2
df_hawley['durationMeasured'] = 0
df_hawley['error'] = 'not reported'

df_all = df_all.append(
    df_hawley[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% PROMICE
print("Loading PROMICE")
df_promice = pd.read_csv("Data/PROMICE/PROMICE_10m_firn_temperature.csv", sep=";")
df_promice = df_promice.loc[df_promice.temperatureObserved.notnull()]
df_promice["reference_short"] = "PROMICE"
df_promice = df_promice.loc[df_promice.site!='QAS_A',:]
df_promice.loc[(df_promice.site=='CEN') & (df_promice.temperatureObserved>-18),'temperatureObserved'] = np.nan

df_promice['method'] = 'RS Components thermistors 151-243'
df_promice['durationOpen'] = 0
df_promice['durationMeasured'] = 30*24
df_promice['error'] = 0.2
df_promice['note'] = ''
df_promice[
    "reference"
] = "Fausto, R.S. and van As, D., (2019). Programme for monitoring of the Greenland ice sheet (PROMICE): Automatic weather station data. Version: v03, Dataset published via Geological Survey of Denmark and Greenland. DOI: https://doi.org/10.22008/promice/data/aws"

df_all = df_all.append(
    df_promice[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %%  GC-Net
print("Loading GC-Net")
time.sleep(0.2)

sites = [
    "CP1",
    "NASA-U",
    "Summit",
    "TUNU-N",
    "DYE-2",
    "Saddle",
    "SouthDome",
    "NASA-E",
    "NASA-SE",
]
lat = [
    69.87975,
    73.84189,
    72.57972,
    78.01677,
    66.48001,
    65.99947,
    63.14889,
    75,
    66.4797,
]
lon = [
    -46.98667,
    -49.49831,
    -38.50454,
    -33.99387,
    -46.27889,
    -44.50016,
    -44.81717,
    -29.99972,
    -42.5002,
]
elev = [2022, 2369, 3254, 2113, 2165, 2559, 2922, 2631, 2425]

df_gcnet = pd.DataFrame()
for ii, site in progressbar.progressbar(enumerate(sites)):
    ds = xr.open_dataset("Data/Vandecrux et al. 2020/" + site + "_T_firn_obs.nc")
    df = ds.to_dataframe()
    df = df.reset_index(1).groupby("level").resample("D").mean()
    df.reset_index(0, inplace=True, drop=True)
    df.reset_index(inplace=True)

    df_d = pd.DataFrame()
    df_d["date"] = df.loc[df["level"] == 1, "time"]
    for i in range(1, 11):
        df_d["rtd" + str(i)] = df.loc[df["level"] == i, "T_firn"].values
    for i in range(1, 11):
        df_d["depth_" + str(i)] = df.loc[df["level"] == i, "Depth"].values

    df_10 = ftl.interpolate_temperature(
        df_d["date"],
        df_d[["depth_" + str(i) for i in range(1, 11)]].values,
        df_d[["rtd" + str(i) for i in range(1, 11)]].values,
        title=site+ " GC-Net",
    )
    df_10["site"] = site
    df_10["latitude"] = lat[ii]
    df_10["longitude"] = lon[ii]
    df_10["elevation"] = elev[ii]
    df_10 = df_10.set_index("date").resample("M").first().reset_index()

    df_gcnet = df_gcnet.append(df_10)

df_gcnet["reference"] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103."

df_gcnet["reference_short"] = "Steffen et al. (1996)"

df_gcnet["note"] = ""
df_gcnet["depthOfTemperatureObservation"] = 10

df_gcnet['method'] = 'thermocouple'
df_gcnet['durationOpen'] = 0
df_gcnet['durationMeasured'] = 30*24
df_gcnet['error'] = 0.5

df_all = df_all.append(
    df_gcnet[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)


# %% GC-Net
print("Loading GC-Net")
df_GCN = pd.read_csv("Data/GC-Net/10m_firn_temperature.csv")
df_GCN = df_GCN.loc[df_GCN.temperatureObserved.notnull()]
df_GCN.reference = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103."
df_GCN.reference_short = 'Steffen et al. (1996)'
df_GCN['method'] = 'thermocouple'
df_GCN['durationOpen'] = 0
df_GCN['durationMeasured'] = 30*24
df_GCN['error'] = 0.5
df_all = df_all.append(
    df_GCN[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Steffen 2001 table
df = pd.read_excel('Data/GC-Net/steffen2001.xlsx')
df['depthOfTemperatureObservation'] = 10
df['temperatureObserved'] = df['temperature']
df['note'] = 'annual average'
df['date'] = [pd.to_datetime(str(yr)+'-12-01') for yr in df.year]
df[
    "reference"
] = "Steffen, K. and J. Box (2001), Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972"
df["reference_short"] = "Steffen et al. (2001)"
df['error'] = 0.5
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)
# %% GCNv2
print("Loading GCNv2")
df_GCNv2 = pd.read_csv("Data/GCNv2/10m_firn_temperature.csv")
df_GCNv2 = df_GCNv2.loc[df_GCNv2.temperatureObserved.notnull()]
df_GCNv2['method'] = 'thermocouple'
df_GCNv2['durationOpen'] = 0
df_GCNv2['durationMeasured'] = 30*24
df_GCNv2['error'] = 0.05
df_all = df_all.append(
    df_GCNv2[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Miege aquifer
print("Loading firn aquifer data")
time.sleep(0.2)
metadata = np.array(
    [
        ["FA-13", 66.181, 39.0435, 1563],
        ["FA-15-1", 66.3622, 39.3119, 1664],
        ["FA-15-2", 66.3548, 39.1788, 1543],
    ]
)

# mean_accumulation = 1 # m w.e. from Miege et al. 2014
# thickness_accum = 2.7 # thickness of the top 1 m w.e. in the FA13 core
thickness_accum = 1.4  # Burial of the sensor between their installation in Aug 2015 and revisit in Aug 2016
df_miege = pd.DataFrame()

for k, site in enumerate(["FA_13", "FA_15_1", "FA_15_2"]):
    depth = pd.read_csv(
        "Data/Miege firn aquifer/" + site + "_Firn_Temperatures_Depths.csv"
    ).transpose()
    if k == 0:
        depth = depth.iloc[5:].transpose()
    else:
        depth = depth.iloc[5:, 0]
    temp = pd.read_csv("Data/Miege firn aquifer/" + site + "_Firn_Temperatures.csv")
    dates = pd.to_datetime(
        (
            temp.Year * 1000000
            + temp.Month * 10000
            + temp.Day * 100
            + temp["Hours (UTC)"]
        ).apply(str),
        format="%Y%m%d%H",
    )
    temp = temp.iloc[:, 4:]

    ellapsed_hours = (dates - dates[0]).astype("timedelta64[h]")
    accum_depth = ellapsed_hours.values * thickness_accum / 365 / 24
    depth_cor = pd.DataFrame()
    depth_cor = depth.values.reshape((1, -1)).repeat(
        len(dates), axis=0
    ) + accum_depth.reshape((-1, 1)).repeat(len(depth.values), axis=1)

    df_10 = ftl.interpolate_temperature(dates, depth_cor, temp.values, title=site + " Miller et al. (2020)")
    df_10.loc[np.greater(df_10["temperatureObserved"], 0), "temperatureObserved"] = 0
    df_10 = df_10.set_index("date", drop=False).resample("M").first()
    df_10["site"] = site
    df_10["latitude"] = float(metadata[k, 1])
    df_10["longitude"] = -float(metadata[k, 2])
    df_10["elevation"] = float(metadata[k, 3])
    df_10["depthOfTemperatureObservation"] = 10
    df_10[
        "reference"
    ] = "Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W"
    df_10["reference_short"] = "Miller et al. (2020)"
    df_10["note"] = "interpolated to 10 m, monthly snapshot"
    # plt.figure()
    # df_10.temperatureObserved.plot()
    df_miege = df_miege.append(df_10)
    
df_miege['method'] = 'digital Thermarray system from RST©'
df_miege['durationOpen'] = 0
df_miege['durationMeasured'] = 30
df_miege['error'] = 0.07


df_all = df_all.append(
    df_miege[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Harper ice temperature
print("Loading Harper ice temperature")
df_harper = pd.read_csv(
    "Data/Harper ice temperature/harper_iceTemperature_2015-2016.csv"
)
num_row = df_harper.shape[0]
df_harper["date"] = np.nan
df_harper["temperatureObserved"] = np.nan
df_harper["note"] = ""
df_harper = df_harper.append(df_harper)
df_harper["borehole"].iloc[:num_row] = df_harper["borehole"].iloc[:num_row] + "_2015"
df_harper["borehole"].iloc[num_row:] = df_harper["borehole"].iloc[num_row:] + "_2016"
df_harper["date"].iloc[:num_row] = pd.to_datetime("2015-01-01")
df_harper["date"].iloc[num_row:] = pd.to_datetime("2016-01-01")
df_harper["temperatureObserved"].iloc[:num_row] = df_harper[
    "temperature_2015_celsius"
].iloc[:num_row]
df_harper["temperatureObserved"].iloc[num_row:] = df_harper[
    "temperature_2016_celsius"
].iloc[num_row:]
df_harper["depth"] = df_harper.depth_m - df_harper.height_m
df_harper = df_harper.loc[df_harper.depth < 100]
df_harper = df_harper.drop(
    columns=[
        "height_m",
        "temperature_2015_celsius",
        "temperature_2016_celsius",
        "yearDrilled",
        "dateDrilled",
        "depth_m",
    ]
)
df_harper = df_harper.loc[df_harper.temperatureObserved.notnull()]

for borehole in df_harper["borehole"].unique():
    # print(borehole, df_harper.loc[df_harper["borehole"] == borehole, "depth"].min())
    if df_harper.loc[df_harper["borehole"] == borehole, "depth"].min() > 20:
        df_harper = df_harper.loc[df_harper["borehole"] != borehole]
        continue

    new_row = df_harper.loc[df_harper["borehole"] == borehole].iloc[0, :].copy()
    new_row["depth"] = 10
    new_row["temperatureObserved"] = np.nan
    df_harper = df_harper.append(new_row)

df_harper = df_harper.set_index(["depth"]).sort_index()

# not interpolating anymore
# plt.figure()
# for borehole in df_harper["borehole"].unique():
#     s = df_harper.loc[df_harper["borehole"] == borehole, "temperatureObserved"]
#     s.iloc[0] = interp1d(
#         s.iloc[1:].index, s.iloc[1:], kind="linear", fill_value="extrapolate"
#     )(10)
#     s.plot(marker='o',label='_no_legend_')
#     df_harper.loc[df_harper['borehole']==borehole,'temperatureObserved'].plot(marker='o',label=borehole)
#     plt.legend()
#     df_harper.loc[df_harper["borehole"] == borehole, "temperatureObserved"] = s.values
#     df_harper.loc[df_harper["borehole"] == borehole, "note"] = (
#         "interpolated from " + str(s.iloc[1:].index.values) + " m depth"
#     )

df_harper = df_harper.reset_index()
# df_harper = df_harper.loc[df_harper.depth == 10]

df_harper[
    "reference"
] = "Hills, B. H., Harper, J. T., Humphrey, N. F., & Meierbachtol, T. W. (2017). Measured horizontal temperature gradients constrain heat transfer mechanisms in Greenland ice. Geophysical Research Letters, 44. https://doi.org/10.1002/2017GL074917;  https://doi.org/10.18739/A24746S04"

df_harper["reference_short"] = "Hills et al. (2017)"

df_harper = df_harper.rename(
    columns={
        "borehole": "site",
        "latitude_WGS84": "latitude",
        "longitude_WGS84": "longitude",
        "Elevation_m": "elevation",
        "depth": "depthOfTemperatureObservation",
    }
)
df_harper['method'] = 'TMP102 digital temperature sensor'
df_harper['durationOpen'] = 0
df_harper['durationMeasured'] = 30*24
df_harper['error'] = 0.1

df_all = df_all.append(
    df_harper[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %%  FirnCover
print("Loading FirnCover")
time.sleep(0.2)

filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

rtd_df = rtd_df.reset_index()
rtd_df = rtd_df.set_index(["sitename", "date"])
df_firncover = pd.DataFrame()
for site in sites:
    df_d = rtd_df.xs(site, level="sitename").reset_index()
    # df_d.to_csv('FirnCover_'+site+'.csv')
    df_10 = ftl.interpolate_temperature(
        df_d["date"],
        df_d[["depth_" + str(i) for i in range(24)]].values,
        df_d[["rtd" + str(i) for i in range(24)]].values,
        title=site+" FirnCover",
    )
    df_10["site"] = site
    if site == "Crawford":
        df_10["site"] = "CP1"
    df_10["latitude"] = statmeta_df.loc[site, "latitude"]
    df_10["longitude"] = statmeta_df.loc[site, "longitude"]
    df_10["elevation"] = statmeta_df.loc[site, "elevation"]
    df_10 = df_10.set_index("date").resample("M").first().reset_index()

    df_firncover = df_firncover.append(df_10)
df_firncover[
    "reference"
] = "MacFerrin, M., Stevens, C.M., Vandecrux, B., Waddington, E., Abdalati, W.: The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) Dataset, 2013-2019, submitted to ESSD"
df_firncover["reference_short"] = "FirnCover"
df_firncover["note"] = ""
df_firncover["depthOfTemperatureObservation"] = 10

# Correction of FirnCover bias
p = np.poly1d([ 1.03093649, -0.49950273])
df_firncover['temperatureObserved']  = p(df_firncover['temperatureObserved'].values) 

df_firncover['method'] = 'Resistance Temperature Detectors + correction'
df_firncover['durationOpen'] = 0
df_firncover['durationMeasured'] = 30*24
df_firncover['error'] = 0.5

df_all = df_all.append(
    df_firncover[
        [
            "date",
            "site",
            "latitude",
            "longitude",
            "elevation",
            "depthOfTemperatureObservation",
            "temperatureObserved",
            "reference",
            "note",
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% SPLAZ KAN_U
print("Loading SPLAZ at KAN-U")
site = "KAN_U"
num_therm = [32, 12, 12]
df_splaz = pd.DataFrame()
for k, note in enumerate(["SPLAZ_main", "SPLAZ_2", "SPLAZ_3"]):

    ds = xr.open_dataset("Data/SPLAZ/T_firn_KANU_" + note + ".nc")
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df2 = pd.DataFrame()
    df2["date"] = df.loc[df["level"] == 1, "time"]
    for i in range(1, num_therm[k] + 1):
        df2["rtd" + str(i - 1)] = df.loc[df["level"] == i, "Firn temperature"].values
    for i in range(1, num_therm[k] + 1):
        df2["depth_" + str(i - 1)] = df.loc[df["level"] == i, "depth"].values
    df2[df2 == -999] = np.nan
    df2 = df2.set_index(["date"]).resample("D").mean()

    df_10 = ftl.interpolate_temperature(
        df2.index,
        df2[["depth_" + str(i) for i in range(num_therm[k])]].values,
        df2[["rtd" + str(i) for i in range(num_therm[k])]].values,
        min_diff_to_depth=1.5,
        kind="linear",
        title='KAN_U '+note,
    )
    # for i in range(10):
    #     plt.figure()
    #     plt.plot(df2.iloc[i*20,0:12].values,-df2.iloc[i*20,12:].values)
    #     plt.plot(df_10.iloc[i*20,1],-10,marker='o')
    #     plt.title(df2.index[i*20])
    #     plt.xlim(-15,0)

    df_10["note"] = note
    df_10["latitude"] = 67.000252
    df_10["longitude"] = -47.022999
    df_10["elevation"] = 1840
    df_10 = df_10.set_index("date").resample("M").first().reset_index()
    df_splaz = df_splaz.append(df_10)
df_splaz[
    "reference"
] = "Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10."
df_splaz["reference_short"] = "SPLAZ"
df_splaz["site"] = site
df_splaz["depthOfTemperatureObservation"] = 10

df_splaz['method'] = 'RS 100 kΩ negative-temperature coefficient thermistors'
df_splaz['durationOpen'] = 0
df_splaz['durationMeasured'] = 30*24
df_splaz['error'] = 0.2


df_all = df_all.append(
    df_splaz[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Load Humphrey data
print("loading Humphrey")
df = pd.read_csv("Data/Humphrey string/location.txt", delim_whitespace=True)
df_humphrey = pd.DataFrame(
    columns=["site", "latitude", "longitude", "elevation", "date", "T10m"]
)
for site in df.site:
    try:
        df_site = pd.read_csv(
            "Data/Humphrey string/" + site + ".txt",
            header=None,
            delim_whitespace=True,
            names=["doy"] + ["IceTemperature" + str(i) + "(C)" for i in range(1, 33)],
        )
    except:  # Exception as e:
        # print(e)
        continue

    print(site)
    temp_label = df_site.columns[1:]
    # the first column is a time stamp and is the decimal days after the first second of January 1, 2007.
    df_site["time"] = [
        datetime(2007, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]
    ]
    if site == "T1old":
        df_site["time"] = [
            datetime(2006, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]
        ]
    df_site = df_site.loc[df_site["time"] <= df_site["time"].values[-1], :]
    df_site = df_site.set_index("time")
    df_site = df_site.resample("H").mean()

    depth = [
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
        3.75,
        4.00,
        4.25,
        4.50,
        4.75,
        5.00,
        5.25,
        5.00,
        5.50,
        6.00,
        6.50,
        7.00,
        7.50,
        8.00,
        8.50,
        9.00,
        9.50,
        10.0,
    ]

    if site != "H5":
        df_site = df_site.iloc[24 * 30 :, :]
    if site == "T4":
        df_site = df_site.loc[:"2007-12-05"]
    if site == "H2":
        depth = np.array(depth) - 1
    if site == "H4":
        depth = np.array(depth) - 0.75
    if site in ["H3", "G165", "T1new"]:
        depth = np.array(depth) - 0.50

    df_hs = pd.read_csv("Data/Humphrey string/" + site + "_surface_height.csv")
    df_hs.time = pd.to_datetime(df_hs.time)
    df_hs = df_hs.set_index("time")
    df_hs = df_hs.resample("H").mean()
    df_site["surface_height"] = np.nan

    df_site["surface_height"] = df_hs.iloc[
        [df_hs.index.get_loc(index, method="nearest") for index in df_site.index]
    ].values

    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        df_site[depth_label[i]] = (
            depth[i]
            + df_site["surface_height"].values
            - df_site["surface_height"].iloc[0]
        )
        if site != "H5":
            df_site[temp_label[i]] = (
                df_site[temp_label[i]].rolling(24 * 3, center=True).mean().values
            )

    df_10 = ftl.interpolate_temperature(
        df_site.index,
        df_site[depth_label].values,
        df_site[temp_label].values,
        title=site + " Humphrey et al. (2012)",
    )
    df_10 = df_10.set_index("date").resample("M").mean().reset_index()

    df_10["site"] = site
    df_10["latitude"] = df.loc[df.site == site, "latitude"].values[0]
    df_10["longitude"] = df.loc[df.site == site, "longitude"].values[0]
    df_10["elevation"] = df.loc[df.site == site, "elevation"].values[0]

    df_humphrey = df_humphrey.append(df_10)
df_humphrey = df_humphrey.reset_index(drop=True)
df_humphrey = df_humphrey.loc[df_humphrey.temperatureObserved.notnull()]
df_humphrey["depthOfTemperatureObservation"] = 10
df_humphrey[
    "reference"
] = "Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/"
df_humphrey["reference_short"] = "Humphrey et al. (2012)"
df_humphrey[
    "note"
] = "no surface height measurements, using interpolating surface height using CP1 and SwissCamp stations"

df_humphrey['method'] = 'sealed 50K ohm thermistors'
df_humphrey['durationOpen'] = 0
df_humphrey['durationMeasured'] = 30*24
df_humphrey['error'] = 0.5


df_all = df_all.append(
    df_humphrey[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% loading Hills
print("Loading Hills")
df_meta = pd.read_csv("Data/Hills/metadata.txt", sep=" ")
df_meta.date_start = pd.to_datetime(df_meta.date_start, format="%m/%d/%y")
df_meta.date_end = pd.to_datetime(df_meta.date_end, format="%m/%d/%y")

df_meteo =  pd.read_csv("Data/Hills/Hills_33km_meteorological.txt", sep="\t")
df_meteo['date'] = [pd.to_datetime('2014-07-18') + pd.Timedelta(int(f*24*60*60), 'seconds') for f in (df_meteo.Time.values-197)]
df_meteo = df_meteo.set_index('date').resample('D').mean()
df_meteo['surface_height'] =  df_meteo.DistanceToTarget.iloc[0] - df_meteo.DistanceToTarget

df_hills = pd.DataFrame()
for site in df_meta.site[:-1]:
    print(site)
    df = pd.read_csv("Data/Hills/Hills_" + site + "_IceTemp.txt", sep="\t")
    df['date'] = [df_meta.loc[df_meta.site==site, 'date_start'].values[0] + pd.Timedelta(int(f*24*60*60), 'seconds') for f in (df.Time.values-int(df.Time.values[0]))]

    df = df.set_index('date')
    
    df['surface_height'] = np.nan
    df.loc[[i in df_meteo.index for i in df.index], 'surface_height'] = df_meteo.loc[[i in df.index for i in df_meteo.index],'surface_height']
    df['surface_height'] = df.surface_height.interpolate(method = 'linear',limit_direction = 'both')
    if all(np.isnan(df.surface_height)):
        df['surface_height'] = 0
    plt.figure()
    df.surface_height.plot()
    plt.title(site)
    
    depth = df.columns[1:-1].str.replace("Depth_", "").values.astype(float)
    temp_label = ['temp_'+str(len(depth)-i) for i in range(len(depth))]
    depth_label = ['depth_'+str(len(depth)-i) for i in range(len(depth))]

    for i in range(len(temp_label)):
        df = df.rename(columns={df.columns[i + 1]: temp_label[i]})
        df.iloc[:14, i+1] = np.nan
        if site in ['T-14', 'T-11b']:
            df.iloc[:30, i+1] = np.nan
            
        df[depth_label[i]] = (
            depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
        )

    df=df.resample('D').mean()
    
    df_10 = ftl.interpolate_temperature(
        df.index,
        df[depth_label].values,
        df[temp_label].values,
        title=site,
    )

    df_10 = df_10.set_index('date').resample("M").mean().reset_index()

    df_10["latitude"] = df_meta.latitude[df_meta.site == site].iloc[0]
    df_10["longitude"] = df_meta.longitude[df_meta.site == site].iloc[0]
    df_10["elevation"] = df_meta.elevation[df_meta.site == site].iloc[0]
    df_10["depthOfTemperatureObservation"] = 10
    df_10["site"] = site
    df_hills = df_hills.append(df_10)

df_hills["note"] = "monthly mean, interpolated at 10 m"
df_hills[
    "reference"
] = "Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418"

df_hills["reference_short"] = "Hills et al. (2018)"
df_hills['method'] = 'digital temperature sensor model DS18B20 from Maxim Integrated Products, Inc.'
df_hills['durationOpen'] = 0
df_hills['durationMeasured'] = 30*24
df_hills['error'] = 0.0625


df_all = df_all.append(
    df_hills[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Achim Dye-2
from datetime import datetime
print("Loading Achim Dye-2")

# loading temperature data
df = pd.read_csv("Data/Achim/CR1000_PT100.txt", header=None)
df.columns= ['time_matlab', 'temp_1',  'temp_2', 'temp_3', 'temp_4', 'temp_5', 'temp_6', 'temp_7', 'temp_8']
df['time'] = pd.to_datetime([datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366) for matlab_datenum in df.time_matlab])
df = df.set_index('time')
df = df.resample('D').mean().drop(columns = 'time_matlab')

# loading surface height data
df_surf = pd.read_csv("Data/Achim/CR1000_SR50.txt", header=None)
df_surf.columns = ['time_matlab', 'sonic_m', 'height_above_upgpr']
df_surf['time'] = pd.to_datetime([datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366) for matlab_datenum in df_surf.time_matlab])
df_surf = df_surf.set_index('time')


df_surf = df_surf.resample('D').mean().drop(columns = ['time_matlab', 'height_above_upgpr'])

# loading surface height data from firncover
filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
_, sonic_df, _, _, _ = ftl.load_metadata(filepath, sites)

sonic_df = sonic_df.xs("DYE-2", level="sitename").reset_index()
sonic_df = sonic_df.set_index('date').drop(columns='delta').resample('D').mean()

sonic_df = sonic_df.append(df_surf.loc[sonic_df.index[-1]:]-1.83 )

plt.figure()
sonic_df.sonic_m.plot()
df_surf.sonic_m.plot()

sonic_df = sonic_df.resample('D').mean()
df['surface_height'] = -(sonic_df.loc[df.index[0]:df.index[-1]] - sonic_df.loc[df.index[0]]).values

depth = 3.4 - np.array([3,2,1,0,-1,-2,-4,-6])
temp_label = ['temp_'+str(i+1) for i in range(len(depth))]
depth_label = ['depth_'+str(i+1) for i in range(len(depth))]

for i in range(len(depth)):
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df.loc['2018-05-18':, 'depth_1'] = df.loc['2018-05-18':, 'depth_1'].values - 1.5
df.loc['2018-05-18':, 'depth_2'] = df.loc['2018-05-18':, 'depth_2'].values - 1.84

# ds = xr.open_dataset('Data/Achim/T_firn_DYE-2_16.nc')
# df = ds.to_dataframe()
# df = df.reset_index(0).groupby('level').resample('D').mean()
# df.reset_index(0,inplace=True, drop=True)  
# df.reset_index(inplace=True)  

# df_d = pd.DataFrame()
# df_d['date'] = df.loc[df['level']==1,'time']
# for i in range(1,9):
#     df_d['rtd'+str(i)] = df.loc[df['level']==i,'Firn temperature'].values    
# for i in range(1,9):
#     df_d['depth_'+str(i)] = df.loc[df['level']==i,'depth'].values

# df.to_csv('Heilig_Dye-2_thermistor.csv')

df_achim = ftl.interpolate_temperature(df.index,
                             df[depth_label].values,
                             df[temp_label].values,
                                 title='Dye-2 Achim')
df_achim['site'] = 'DYE-2'
df_achim['latitude'] = 66.4800
df_achim['longitude'] = -46.2789
df_achim['elevation'] = 2165.0
df_achim['depthOfTemperatureObservation'] = 10
df_achim['note'] = 'interpolated at 10 m, monthly mean, using surface height from FirnCover station'
df_achim['reference'] = 'Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018. '
df_achim['reference_short'] = 'Heilig et al. (2018)'
df_achim['method'] = 'thermistors'
df_achim['durationOpen'] = 0
df_achim['durationMeasured'] = 30*24
df_achim['error'] = 0.25
df_achim = df_achim.set_index('date').resample('M').first().reset_index()
    
df_all = df_all.append(df_achim[[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
            ]],ignore_index=True)


# %% Camp Century Climate
print('Loading Camp Century data')
df = pd.read_csv("Data/Camp Century Climate/data_long.txt", sep=",", header=None)
df = df.rename(columns={0: "date"})
df["date"] = pd.to_datetime(df.date)
df[df == -999] = np.nan
df = df.set_index("date").resample("D").first()
df = df.iloc[:, :-2]

df_promice = pd.read_csv(
    "C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3/EGP_hour_v03_L3.txt",
    sep="\t",
)
df_promice[df_promice == -999] = np.nan
df_promice = df_promice.rename(columns={"time": "date"})
df_promice["date"] = pd.to_datetime(df_promice.date)
df_promice = df_promice.set_index("date").resample("D").mean()
df_promice = df_promice["SurfaceHeight_summary(m)"]
temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]
df["surface_height"] = df_promice[
    np.array([x for x in df_promice.index if x in df.index])
].values

depth = [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    21,
    24,
    27,
    30,
    33,
    36,
    39,
    42,
    45,
    48,
    53,
    58,
    63,
    68,
    73,
]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df_10 = ftl.interpolate_temperature(
    df.index,
    df[depth_label].values,
    df[temp_label].values,
    title="Camp Century Climate (long)",
)
df_10.loc[np.greater(df_10["temperatureObserved"], -15), "temperatureObserved"] = np.nan
df_10 = df_10.set_index("date", drop=False).resample("M").first()
df_10["site"] = "CEN"
df_10["latitude"] = 77.1333
df_10["longitude"] = -61.0333
df_10["elevation"] = 1880
df_10["depthOfTemperatureObservation"] = 10
df_10["note"] = "THM_long, interpolated at 10 m"
df_10[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
df_10["reference_short"] = "Camp Century Climate"
df_10['method'] = 'thermistors'
df_10['durationOpen'] = 0
df_10['durationMeasured'] = 30*24
df_10['error'] = 0.2
df_all = df_all.append(
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

df = pd.read_csv("Data/Camp Century Climate/data_short.txt", sep=",", header=None)
df = df.rename(columns={0: "date"})
df["date"] = pd.to_datetime(df.date)
df[df == -999] = np.nan
df = df.set_index("date").resample("D").first()

df_promice = pd.read_csv(
    "C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3/EGP_hour_v03_L3.txt",
    sep="\t",
)
df_promice[df_promice == -999] = np.nan
df_promice = df_promice.rename(columns={"time": "date"})
df_promice["date"] = pd.to_datetime(df_promice.date)
df_promice = df_promice.set_index("date").resample("D").mean()
df_promice = df_promice["SurfaceHeight_summary(m)"]
temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]
df["surface_height"] = df_promice[
    np.array([x for x in df_promice.index if x in df.index])
].values

# plt.figure()
# df_10.temperatureObserved.plot()
depth = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    13,
    15,
    17,
    19,
    22,
    25,
    28,
    31,
    34,
    38,
    42,
    46,
    50,
    54,
]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df_10 = ftl.interpolate_temperature(
    df.index,
    df[depth_label].values,
    df[temp_label].values,
    title="Camp Century Climate (short)",
)
df_10.loc[np.greater(df_10["temperatureObserved"], -15), "temperatureObserved"] = np.nan
df_10 = df_10.set_index("date", drop=False).resample("M").first()
df_10["site"] = "CEN"
df_10["latitude"] = 77.1333
df_10["longitude"] = -61.0333
df_10["elevation"] = 1880
df_10["depthOfTemperatureObservation"] = 10
df_10["note"] = "THM_short, interpolated at 10 m"
df_10[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
# df_10.temperatureObserved.plot()
df_10["reference_short"] = "Camp Century Climate"
df_10['method'] = 'thermistors'
df_10['durationOpen'] = 0
df_10['durationMeasured'] = 30*24
df_10['error'] = 0.2

df_all = df_all.append(
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Camp Century historical
# % In 1977, firn density was measured to a depth of 100.0 m, and firn temperature was measured at 10 m depth (Clausen and Hammer, 1988). In 1986, firn density was measured to a depth of 12.0 m, and the 10 m firn temperature was again measured (Gundestrup et al., 1987).

df_cc_hist = pd.DataFrame()
df_cc_hist["date"] = pd.to_datetime(["1977-07-01"])
df_cc_hist["site"] = "Camp Century"
df_cc_hist["temperatureObserved"] = -24.29
df_cc_hist["depthOfTemperatureObservation"] = 10
df_cc_hist["note"] = "original data cannot be found"
df_cc_hist[
    "reference"
] = "Clausen, H., and Hammer, C. (1988). The laki and tambora eruptions as revealed in Greenland ice cores from 11 locations. J. Glaciology 10, 16–22. doi:10.1017/s0260305500004092"
df_cc_hist["reference_short"] = "Clausen and Hammer (1988)"

df_cc_hist["latitude"] = 77.1333
df_cc_hist["longitude"] = -61.0333
df_cc_hist["elevation"] = 1880
df_cc_hist['method'] = 'NA'
df_cc_hist['durationOpen'] = 'NA'
df_cc_hist['durationMeasured'] = 'NA'
df_cc_hist['error'] = 'NA'
df_all = df_all.append(
    df_cc_hist[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Davies dataset
df_davies = pd.read_excel("Data/Davies South Dome/table_3_digitized.xlsx")
df_davies["depthOfTemperatureObservation"] = 10
df_davies["note"] = ""
df_davies[
    "reference"
] = "Davies, T.C., Structures in the upper snow layers of the southern Dome Greenland ice sheet, CRREL research report 115, 1954"
df_davies["reference_short"] = "Davies (1954)"

df_davies['method'] = 'Weston bimetallic thermometers'
df_davies['durationOpen'] = 0
df_davies['durationMeasured'] = 0
df_davies['error'] = 0.5

df_all = df_all.append(
    df_davies[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %%  Echelmeyer Jakobshavn isbræ
df_echel = pd.read_excel("Data/Echelmeyer jakobshavn/Fig5_points.xlsx")
df_echel.date = pd.to_datetime(df_echel.date)
df_echel[
    "reference"
] = "Echelmeyer Κ, Harrison WD, Clarke TS and Benson C (1992) Surficial Glaciology of Jakobshavns Isbræ, West Greenland: Part II. Ablation, accumulation and temperature. Journal of Glaciology 38(128), 169–181 (doi:10.3189/S0022143000009709)"
df_echel["reference_short"] = "Echelmeyer et al. (1992)"
df_echel = df_echel.rename(
    columns={"12 m temperature": "temperatureObserved", "Name": "site"}
)
df_echel["depthOfTemperatureObservation"] = 12

df_profiles = pd.read_excel("Data/Echelmeyer jakobshavn/Fig3_profiles.xlsx")
df_profiles = df_profiles[pd.DatetimeIndex(df_profiles.date).month > 3]
df_profiles["date"] = pd.to_datetime(df_profiles.date)
df_profiles = df_profiles.set_index(["site", "date"], drop=False)
for site in ["L20", "L23"]:
    dates = df_profiles.loc[site, "date"].unique()
    for date in dates:
        df_profiles.loc[(site, date), "depth"] = (
            -df_profiles.loc[(site, date), "depth"].values
            + df_profiles.loc[(site, date), "depth"].max()
        )
        tmp = df_echel.iloc[0, :]
        tmp.Name = df_profiles.loc[(site, date), "site"].iloc[0]
        tmp.longitude = df_profiles.loc[(site, date), "longitude"].iloc[0]
        tmp.latitude = df_profiles.loc[(site, date), "latitude"].iloc[0]
        tmp.elevation = df_profiles.loc[(site, date), "elevation"].iloc[0]
        f = interp1d(
            df_profiles.loc[(site, date), "depth"].values,
            df_profiles.loc[(site, date), "temperature"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = f(10)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "digitized, interpolated at 10 m"
        df_echel = df_echel.append(tmp)
        
df_echel['method'] = 'thermistors and thermocouples'
df_echel['durationOpen'] = 0
df_echel['durationMeasured'] = 0
df_echel['error'] = 0.3
df_all = df_all.append(
    df_echel[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Fischer de Quervain EGIG
df_fischer = pd.read_excel('Data/Fischer EGIG/fischer_90_91.xlsx')

df_fischer[
    "reference"
] = "Fischer, H., Wagenbach, D., Laternser, M. & Haeberli, W., 1995. Glacio-meteorological and isotopic studies along the EGIG line, central Greenland. Journal of Glaciology, 41(139), pp. 515-527."
df_fischer["reference_short"] = "Fischer et al. (1995)"

df_all = df_all.append(
    df_fischer[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

df_dequervain = pd.read_csv("Data/Fischer EGIG/DeQuervain.txt", index_col=False)
df_dequervain.date = pd.to_datetime(
    df_dequervain.date.str[:-2] + "19" + df_dequervain.date.str[-2:]
)
df_dequervain = df_dequervain.rename(
    columns={"depth": "depthOfTemperatureObservation", "temp": "temperatureObserved"}
)
df_dequervain["note"] = "as reported in Fischer et al. (1995)"
df_dequervain[
    "reference"
] = "de Quervain, M, 1969. Schneekundliche Arbeiten der Internationalen Glaziologischen Grönlandexpedition (Nivologie). Medd. Grønl. 177(4)"
df_dequervain["reference_short"] = "de Quervain (1969)"

df_dequervain['method'] = 'bimetallic, mercury, Wheastone bridge, platinium resistance thermometers'
df_dequervain['durationOpen'] = 0
df_dequervain['durationMeasured'] = 0
df_dequervain['error'] = 0.2

df_all = df_all.append(
    df_dequervain[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Larternser EGIG
df_laternser = pd.read_excel('Data/Laternser 1992/Laternser94.xlsx')

df_laternser[
    "reference"
] = 'Laternser, M., 1994 Firn temperature measurements and snow pit studies on the EGIG traverse of central Greenland, 1992. Eidgenössische Technische Hochschule.  Versuchsanstalt für Wasserbau  Hydrologie und Glaziologic. (Arbeitsheft 15).'
df_laternser["reference_short"] = "Laternser (1994)"

# interpolating the profiles that do not have 10 m depth
tmp, ind = np.unique([str(x)+str(y) for (x,y) in zip(df_laternser.site,
                                                     df_laternser.date)],
                     return_inverse = True)
df_interp = pd.DataFrame()
for i in np.unique(ind):
    core_df = df_laternser.loc[ind==i,:]
    if 10 in core_df.depthOfTemperatureObservation.values:
        continue
    
    tmp = core_df.iloc[0, :].copy()
    f = interp1d(
        core_df["depthOfTemperatureObservation"].values,
        core_df["temperatureObserved"].values,
        fill_value="extrapolate",
    )
    tmp.temperatureObserved = f(10)
    tmp.depthOfTemperatureObservation = 10
    tmp.note = "interpolated at 10 m, "+tmp.note
    df_interp = df_interp.append(tmp)
    
    plt.figure()
    plt.plot(core_df.temperatureObserved,
             -core_df.depthOfTemperatureObservation,
             marker='o')
    plt.plot(tmp.temperatureObserved, 
             -tmp.depthOfTemperatureObservation,  marker='o')
    plt.title(tmp.site)

df_laternser = df_laternser.append(df_interp)

df_laternser['method'] = 'Fenwal 197-303 KAG-401 thermistors'
df_laternser['durationOpen'] = 0
df_laternser['error'] = 0.02

df_all = df_all.append(
    df_laternser[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)
# %% Wegener 1929-1930
df1 = pd.read_csv("Data/Wegener 1930/200mRandabst_firtemperature_wegener.csv", sep=";")
df3 = pd.read_csv("Data/Wegener 1930/ReadMe.txt", sep=";")

df1["depth"] = df1.depth / 100
df1 = df1.append({"Firntemp": np.nan, "depth": 10}, ignore_index=True)
df1 = interp_pandas(df1.set_index("depth"))

df_wegener = pd.DataFrame.from_dict(
    {
        "date": [df3.date.iloc[0]],
        "temperatureObserved": df1.loc[10].values[0],
        "depthOfTemperatureObservation": 10,
        "latitude": [df3.latitude.iloc[0]],
        "longitude": [df3.longitude.iloc[0]],
        "elevation": [df3.elevation.iloc[0]],
        "reference": [df3.reference.iloc[0]],
        "site": [df3.name.iloc[0]],
    }
)
df2 = pd.read_csv(
    "Data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv", sep=";"
)
date = "1930-" + df2.month.astype(str).apply(lambda x: x.zfill(2)) + "-15"
df2 = -df2.iloc[:, 1:].transpose().reset_index(drop=True)
df2.iloc[:, 0] = interp_pandas(df2.iloc[:, 0])
df2 = df2.iloc[10, :].transpose()
df_new = pd.DataFrame()
df_new["temperatureObserved"] = df2.values
df_new["date"] = date.values
df_new["depthOfTemperatureObservation"] = 10
df_new["latitude"] = df3.latitude.iloc[1]
df_new["longitude"] = df3.longitude.iloc[1]
df_new["elevation"] = df3.elevation.iloc[1]
df_new["site"] = df3.name.iloc[1]
df_new["reference"] = df3.reference.iloc[1]
df_wegener = df_wegener.append(df_new)
df_wegener["reference_short"] = "Wegener (1930)"
df_wegener["note"] = ""

df_wegener['method'] = 'electric resistance thermometer'
df_wegener['durationOpen'] = 'NA'
df_wegener['durationMeasured'] = 'NA'
df_wegener['error'] = 0.2

df_all = df_all.append(
    df_wegener[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Japanese stations
df = pd.read_excel("Data/Japan/Qanaaq.xlsx")
df.date = pd.to_datetime(df.date)
df["note"] = "interpolated to 10 m"
df['method'] = 'thermistor'
df['durationOpen'] = 'NA'
df['durationMeasured'] = 'NA'
df['error'] = 0.1
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

df = pd.read_excel("Data/Japan/Sigma.xlsx")
df["note"] = ""
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Ambach

meta = pd.read_csv(
    "Data/Ambach1979b/metadata.txt",
    sep="\t",
    header=None,
    names=["site", "file", "date", "latitude", "longitude", "elevation"],
)
meta.date = pd.to_datetime(meta.date)
for file in meta.file:
    df = pd.read_csv(
        "Data/Ambach1979b/" + file + ".txt", header=None, names=["temperature", "depth"]
    )
    if df.depth.max() < 7.5:
        meta.loc[meta.file == file, "temperatureObserved"] = df.temperature.iloc[-1]
        meta.loc[meta.file == file, "depthOfTemperatureObservation"] = df.index.values[
            -1
        ]
    else:
        df.loc[df.shape[0]] = [np.nan, 10]  # adding a row
        df = df.set_index("depth")
        df = interp_pandas(df)
        plt.figure()
        df.plot()
        plt.title(file)
        meta.loc[meta.file == file, "temperatureObserved"] = df.temperature.iloc[-1]
        meta.loc[meta.file == file, "depthOfTemperatureObservation"] = df.index.values[
            -1
        ]

meta[
    "reference"
] = "Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979"
meta["reference_short"] = "Ambach EGIG 1959"
meta["note"] = "digitized, interpolated at 10 m"
meta['method'] = 'NA'
meta['durationOpen'] = 'NA'
meta['durationMeasured'] = 'NA'
meta['error'] = 'NA'

df_all = df_all.append(
    meta[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Kjær 2020 TCD data
df = pd.read_excel('Data/Kjær/tc-2020-337.xlsx')
df[
    "reference"
] = "Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337 , 2021."
df["reference_short"] = "Kjær 2015"
df["note"] = ""
df['method'] = 'thermistor'
df['durationOpen'] = 0
df['durationMeasured'] = 0
df['error'] = 0.1
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Covi
sites = ['DYE-2', 'EKT','SiteJ']
filenames = ['AWS_Dye2_20170513_20190904_CORR_daily', 'AWS_EKT_20170507_20190904_CORR_daily', 'AWS_SiteJ_20170429_20190904_CORR_daily']
for site, filename in zip(sites, filenames):
    df = pd.read_csv('Data/Covi/'+filename+'.dat', skiprows=1)
    depth = np.flip(16 - np.concatenate((np.arange(16, 4.5,-0.5), np.arange(4,-1,-1))))
    depth_ini = 1
    depth = depth+depth_ini
    df['date'] = df.TIMESTAMP
    temp_label = ['Tfirn'+str(i)+'(C)' for i in range(1, 29)]
    
    df.date = pd.to_datetime(df.date)
    df = df.set_index("date").resample('D').mean().reset_index()
    
    if site in ['DYE-2', 'EKT']:
        # loading surface height from FirnCover station
        filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
        sites =   ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
        statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
        statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]
        
        df["surface_height"] = -sonic_df.loc[site].resample('D').mean().loc[df.date.iloc[0]:df.date.iloc[-1]].sonic_m.values
        
        
    else:
        df["surface_height"] = df['SR50_corr(m)']
    
    df["surface_height"] = df["surface_height"]-df["surface_height"].iloc[0]
    df["surface_height"] = df["surface_height"].interpolate().values
    plt.figure()
    df.surface_height.plot()
    
    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        df[depth_label[i]] = depth[i] + df["surface_height"].values
    df = df.set_index("date")
    df_10 = ftl.interpolate_temperature(
        df.index,
        df[depth_label].values,
        df[temp_label].values,
        title=site+' Covi et al.',
    )
    df_10 = df_10.set_index("date").resample("M").mean().reset_index()
    
    df_10["site"] = site
    if site == 'SiteJ':
        df_10["latitude"] = 66.864952
        df_10["longitude"] = -46.265141
        df_10["elevation"] = 2060
    else:
        df_10["latitude"] = statmeta_df.loc[site, "latitude"]
        df_10["longitude"] = statmeta_df.loc[site, "longitude"]
        df_10["elevation"] = statmeta_df.loc[site, "elevation"]
    df_10["depthOfTemperatureObservation"] = 10
    df_10["note"] = ''
    
    df_10['reference'] = 'Covi, F., Hock, R., Rennermalm, A: Firn temperatures at Dye-2, EKT and Site J'
    df_10['reference_short'] = 'Covi et al.'
    df_10["note"] = ""
    df_10['method'] = 'thermistor'
    df_10['durationOpen'] = 0
    df_10['durationMeasured'] = 0
    df_10['error'] = 0.1
    df_all = df_all.append(
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
                "error",
                "durationOpen",
                "durationMeasured",
                "method",
            ]
        ],
        ignore_index=True,
    )
    
# %% Stauffer and Oeschger 1979
df_s_o = pd.read_excel('Data/Stauffer and Oeschger 1979/Stauffer&Oeschger1979.xlsx')

df_s_o[
    "reference"
] = "Stauffer, B, Oeschger, H 1979 Temperaturprofile in Bohrlöchern am Rande des grönländischen Inlandeises. Mitteilungen der Versuchsanstalt für Wasserbau, Hydrologie und Glaziologie an der Eidgenössischen Technischen Hochschule (Zürich) 41: 301—313 and Clausen HB and Stauffer B (1988) Analyses of Two Ice Cores Drilled at the Ice-Sheet Margin in West Greenland. Annals of Glaciology 10, 23–27 (doi:10.3189/S0260305500004109)"
df_s_o["reference_short"] = "Stauffer and Oeschger (1979)"
df_s_o["note"] = "site location estimated by M. Luethi"
df_s_o['method'] = 'Fenwal Thermistor UUB 31-J1'
df_s_o['durationOpen'] = 0
df_s_o['durationMeasured'] = 0
df_s_o['error'] = 0.1
df_all = df_all.append(
    df_s_o[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Schwager EGIG
df_schwager = pd.read_excel('Data/Schwager/schwager.xlsx')

df_schwager[
    "reference"
] = "Schwager, M. (2000): Eisbohrkernuntersuchungen zur räumlichen und zeitlichen Variabilität von Temperatur und Niederschlagsrate im Spätholozän in Nordgrönland - Ice core analysis on the spatial and temporal variability of temperature and precipitation during the late Holocene in North Greenland , Berichte zur Polarforschung (Reports on Polar Research), Bremerhaven, Alfred Wegener Institute for Polar and Marine Research, 362 , 136 p. . doi: 10.2312/BzP_0362_2000"
df_schwager["reference_short"] = "Schwager (2000)"
df_schwager["note"] = ""
df_schwager.date=pd.to_datetime([str(y)+'-07-01' for y in df_schwager.date])
df_schwager['method'] = 'custom thermistors'
df_schwager['durationOpen'] = 0
df_schwager['durationMeasured'] = 12
df_schwager['error'] = 0.5
df_all = df_all.append(
    df_schwager[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Giese & Hawley
from datetime import datetime, timedelta

df = pd.read_excel('Data/Giese and Hawley/giese_hawley.xlsx')

df1= df.iloc[:,:2]
df1 = df1.loc[~np.isnan(df1.time1.values),:]
df1['time'] = [datetime(int(d_y), 1, 1) + timedelta(seconds= (d_y - int(d_y)) * timedelta(days=365).total_seconds()) for d_y in df1.time1.values]
df1 = df1.set_index('time').resample('D').mean().interpolate(method='cubic')
 
df2= df.iloc[:,2:4]
df2 = df2.loc[~np.isnan(df2.time2.values),:]
df2['time'] = [datetime(int(d_y), 1, 1) + timedelta(seconds= (d_y - int(d_y)) * timedelta(days=365).total_seconds()) for d_y in df2.time2.values]
df2 = df2.set_index('time').resample('D').mean().interpolate(method='cubic')
 
df_giese= df.iloc[:,4:]
df_giese = df_giese.loc[~np.isnan(df_giese.time3.values),:]
df_giese['time'] = [datetime(int(d_y), 1, 1) + timedelta(seconds= (d_y - int(d_y)) * timedelta(days=365).total_seconds()) for d_y in df_giese.time3.values]

df_giese = df_giese.set_index('time').resample('D').mean().interpolate(method='cubic')

df_giese['temp_8'] = df2.temp_8.values
df_giese['depth_7'] = 6.5 + df1.depth1.values - df1.depth1.min()
df_giese['depth_8'] = 9.5 + df1.depth1.values - df1.depth1.min()


df_giese['temperatureObserved'] = np.nan

for date in df_giese.index:
    f = interp1d(
        df_giese.loc[(date), ['depth_7','depth_8']].values,
        df_giese.loc[(date), ['temp_7','temp_8']].values,
        fill_value="extrapolate",
    )
    df_giese.loc[date,'temperatureObserved'] = f(10)
    
df_giese = df_giese.resample('M').mean().reset_index()
df_giese['date'] = df_giese.time
df_giese['site'] = 'Summit'
df_giese['latitude'] = 72+35/60
df_giese['longitude'] = -38-30/60
df_giese['elevation'] = 3208
df_giese['depthOfTemperatureObservation'] = 10

df_giese[
    "reference"
] = "Giese AL and Hawley RL (2015) Reconstructing thermal properties of firn at Summit, Greenland, from a temperature profile time series. Journal of Glaciology 61(227), 503–510 (doi:10.3189/2015JoG14J204)"
df_giese["reference_short"] = "Giese and Hawley (2015)"
df_giese["note"] = "digitized and interpolated at 10m"

df_giese['method'] = 'thermistors'
df_giese['durationOpen'] = 0
df_giese['durationMeasured'] = 24*30
df_giese['error'] = 0.5

df_all = df_all.append(
    df_giese[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

#%% Historical summit
df_summit = pd.DataFrame()
df_summit['temperatureObserved'] = -32.0
df_summit['depthOfTemperatureObservation'] = 10
df_summit['site'] = 'Summit'
df_summit['latitude'] = 72+18/60
df_summit['longitude'] = -37-55/60
df_summit['elevation'] = 3260
df_summit['note'] = ''
df_summit['date'] = pd.to_datetime('1974-06-01')
df_summit['reference_short'] = 'Hodge et al. (1990)'
df_summit['reference'] = 'Hodge SM, Wright DL, Bradley JA, Jacobel RW, Skou N and Vaughn B (1990) Determination of the Surface and Bed Topography in Central Greenland. Journal of Glaciology 36(122), 17–30 (doi:10.3189/S0022143000005505) as reported in  Firestone J, Waddington ED and Cunningham J (1990) The Potential for Basal Melting Under Summit, Greenland. Journal of Glaciology 36(123), 163–168 (doi:10.3189/S0022143000009400)'

df_summit['method'] = 'NA'
df_summit['durationOpen'] = 'NA'
df_summit['durationMeasured'] = 'NA'
df_summit['error'] ='NA'

df_all = df_all.append(
    df_summit[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% IMAU
df_meta = pd.read_csv('data/IMAU/meta.csv')
depth_label = ['depth_'+str(i) for i in range(1,6)]
temp_label = ['temp_'+str(i) for i in range(1,6)]
df_imau = pd.DataFrame()
for i, site in enumerate(['s5','s6','s9']):
    df = pd.read_csv('data/IMAU/'+site+'_tsub.txt')
    df['date'] = pd.to_datetime(df['year'], format='%Y') + pd.to_timedelta(df['doy'] - 1, unit='d')
    df = df.set_index('date').drop(columns=['year','doy']).resample('D').mean()

    for dep, temp in zip(depth_label,temp_label):   
        df.loc[df[dep]<0.2, temp] = np.nan
    if site == 's5':
        surface_height = df.depth_5-df.depth_5.values[0]
        surface_height.loc['2011-09-01':] = surface_height.loc['2011-08-31':]-9.38
        surface_height=surface_height.values
        df.loc['2011-08-29':'2011-09-05', temp_label] = np.nan
    else:
        surface_height = []
    if site == 's6':
        for dep, temp in zip(depth_label,temp_label):   
            df.loc[df[dep]<1.5, temp] = np.nan
        min_diff_to_depth=3
    else:
        min_diff_to_depth=3
    df_10 = ftl.interpolate_temperature(
                df.index,
                df[depth_label].values,
                df[temp_label].values,
                title=site,
                min_diff_to_depth=min_diff_to_depth,
                kind='slinear',
                surface_height = surface_height
                )
    df_10 = df_10.set_index("date").resample("M").mean().reset_index()

    df_10['site'] = site
    df_10['note'] = ''
    df_10['latitude'] = df_meta.loc[df_meta.site==site, 'latitude'].values[0]
    df_10['longitude'] = df_meta.loc[df_meta.site==site, 'longitude'].values[0]
    df_10['elevation'] = df_meta.loc[df_meta.site==site, 'elevation'].values[0]
    df_imau=df_imau.append(df_10)
df_imau['reference'] = ' Paul C. J. P. Smeets, Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954'
df_imau['reference_short'] = 'IMAU'
df_imau['depthOfTemperatureObservation'] = 10
df_imau['method'] = 'thermistor'
df_imau['durationOpen'] = 0
df_imau['durationMeasured'] = 24*30
df_imau['error'] = 0.2
df_all = df_all.append(
    df_imau[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Braithwaite
df = pd.read_excel('Data/Braithwaite/data.xlsx')
df['temperatureObserved'] = df[10].values
df['depthOfTemperatureObservation'] = 10

df[
    "reference"
] = "Braithwaite, R. (1993). Firn temperature and meltwater refreezing in the lower accumulation area of the Greenland ice sheet, Pâkitsoq, West Greenland. Rapport Grønlands Geologiske Undersøgelse, 159, 109–114. https://doi.org/10.34194/rapggu.v159.8218"
df["reference_short"] = "Braithwaite (1993)"
df["note"] = "from table"
df['method'] = 'thermistor'
df['durationOpen'] = 0
df['durationMeasured'] = 0
df['error'] = 0.5
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Clement
df = pd.read_excel('Data/Clement/data.xlsx')
df['depthOfTemperatureObservation'] = df.depth
df['temperatureObserved'] = df.temperature

df[
    "reference"
] = "Clement, P. “Glaciological Activities in the Johan Dahl Land Area, South Greenland, As a Basis for Mapping Hydropower Potential”. Rapport Grønlands Geologiske Undersøgelse, vol. 120, Dec. 1984, pp. 113-21, doi:10.34194/rapggu.v120.7870."
df["reference_short"] = "Clement (1984)"
df["note"] = "digitized"
df['method'] = 'thermistor'
df['durationOpen'] = 0
df['durationMeasured'] = 0
df['error'] = 0.5

df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Nobles
df = pd.read_excel('Data/Nobles Nuna Ramp/data.xlsx')
df['depthOfTemperatureObservation'] = 10
df['temperatureObserved'] = df['annual temp at 8m']
df['date'] = pd.to_datetime('1954-07-01')
df[
    "reference"
] = "Nobles, L. H., Glaciological investigations, Nunatarssuaq ice ramp, Northwestern Greenland, Tech. Rep. 66, U.S. Army Snow, Ice and Permafrost Research Establishment, Corps of Engineers, 1960."
df["reference_short"] = "Nobles (1960)"
df["note"] = "digitized"

df['method'] = 'iron-constantan thermocouples'
df['durationOpen'] = 0
df['durationMeasured'] = 0
df['error'] = 0.5

df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Schytt
df = pd.read_excel('Data/Schytt Tuto/data.xlsx')
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date==date]
        if df_date.shape[0]<2:
            continue
        tmp = df_date.iloc[0, :].copy()

        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "digitized, interpolated at 10 m"
        df_interp = df_interp.append(tmp)
    
        plt.figure()
        plt.plot(df_date.temperatureObserved,
                  -df_date.depth,
                  marker='o')
        plt.plot(tmp.temperatureObserved, 
                  -tmp.depthOfTemperatureObservation,  marker='o')
        plt.title(tmp.site+' '+tmp.date)


df_interp['depthOfTemperatureObservation'] = 10
df_interp['date'] = pd.to_datetime(df_interp.date)
df_interp[
    "reference"
] = "Schytt, V. (1955) Glaciological investigations in the Thule Ramp area, U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 28, 88 pp."
df_interp["reference_short"] = "Schytt (1955)"
df_interp["note"] = "from table"

df_interp['method'] = 'copper-constantan thermocouples'
df_interp['durationOpen'] = 0
df_interp['durationMeasured'] = 0
df_interp['error'] = 0.5

df_all = df_all.append(
    df_interp[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Griffiths & Schytt
df = pd.read_excel('Data/Griffiths Tuto/data.xlsx')
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date==date]
        if df_date.shape[0]<2:
            continue
        tmp = df_date.iloc[0, :].copy()
        if df_date["depth"].max()<8:
            continue
        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "digitized, interpolated at 10 m"
        df_interp = df_interp.append(tmp)
    
        plt.figure()
        plt.plot(df_date.temperatureObserved,
                  -df_date.depth,
                  marker='o')
        plt.plot(tmp.temperatureObserved, 
                  -tmp.depthOfTemperatureObservation,  marker='o')
        plt.title(tmp.site+' '+tmp.date)

df_interp['depthOfTemperatureObservation'] = 10
df_interp['date'] = pd.to_datetime(df_interp.date)
df_interp[
    "reference"
] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp."
df_interp["reference_short"] = "Griffiths (1960)"
df_interp["note"] = "from table"
df_interp['method'] = 'copper-constantan thermocouples'
df_interp['durationOpen'] = 0
df_interp['durationMeasured'] = 0
df_interp['error'] = 0.5

df_all = df_all.append(
    df_interp[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Griffiths & Meier
df = pd.read_excel('Data/Griffiths Tuto/data_crevasse3.xlsx')
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date==date]
        if df_date.shape[0]<2:
            continue
        tmp = df_date.iloc[0, :].copy()
        if df_date["depth"].max()<8:
            continue
        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "measurement made close to an open crevasse, digitized, interpolated at 10 m"
        df_interp = df_interp.append(tmp)
    
        plt.figure()
        plt.plot(df_date.temperatureObserved,
                  -df_date.depth,
                  marker='o')
        plt.plot(tmp.temperatureObserved, 
                  -tmp.depthOfTemperatureObservation,  marker='o')
        plt.title(tmp.site+' '+tmp.date)

df_interp['depthOfTemperatureObservation'] = 10
df_interp['date'] = pd.to_datetime(df_interp.date)
df_interp[
    "reference"
] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. and Meier, M. F., Conel, J. E., Hoerni, J. A., Melbourne, W. G., & Pings, C. J. (1957). Preliminary Study of Crevasse Formation. Blue Ice Valley, Greenland, 1955. OCCIDENTAL COLL LOS ANGELES CALIF."
df_interp["reference_short"] = "Griffiths (1960)"

df_interp["latitude"] = 76.43164
df_interp["longitude"] = -67.54949
df_interp["elevation"] = 800
df_interp['method'] = 'copper-constantan thermocouples'
df_interp['durationOpen'] = 0
df_interp['durationMeasured'] = 0
df_interp['error'] = 0.5
# only keeping measurements more than 1 m into the crevasse wall
df_interp = df_interp.loc[df_interp['distance from crevasse'] >= 1,:]
df_all = df_all.append(
    df_interp[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)


# %% Vanderveen
df = pd.read_excel('Data/Vanderveen et al. 2001/summary.xlsx')
df = df.loc[df.date1.notnull(), :]
df = df.loc[df.Temperature_celsius.notnull(), :]
df['site'] = [str(s) for s in df.site]
df['date'] = pd.to_datetime(df.date1) + (pd.to_datetime(df.date2)-pd.to_datetime(df.date1))/2
df['note'] = ''
df.loc[np.isnan(df['date']),'note'] = 'only year available'
df.loc[np.isnan(df['date']),'date'] = pd.to_datetime([str(y)+'-07-01' for y in df.loc[np.isnan(df['date']),'date1'].values])

df['temperatureObserved'] = df['Temperature_celsius']
df['depthOfTemperatureObservation'] = df['Depth_centimetre']/100
tmp, ind = np.unique([str(x)+str(y) for (x,y) in zip(df.site, df.date)],
                      return_inverse = True)
df_interp = pd.DataFrame()
for i in np.unique(ind):
    core_df = df.loc[ind==i,:]
    if core_df['depthOfTemperatureObservation'].max()<9:
        continue
    if core_df['depthOfTemperatureObservation'].min()>11:
        continue
    if len(core_df['depthOfTemperatureObservation'])<2:
        continue
    # # if core_df.site.unique() in [
    # print(core_df.site.unique() )
    tmp = core_df.iloc[0, :].copy()
    f = interp1d(
        np.log(core_df["depthOfTemperatureObservation"].values),
        core_df["temperatureObserved"].values,
        fill_value="extrapolate",
    )
    tmp.temperatureObserved = f(np.log(10))
    tmp.depthOfTemperatureObservation = 10
    tmp.note = "digitized, interpolated at 10 m"
    df_interp = df_interp.append(tmp)
    
    plt.figure()
    plt.plot((core_df.temperatureObserved),
              -(core_df.depthOfTemperatureObservation),
              marker='o', linestyle='None')
    plt.plot(tmp.temperatureObserved, 
              -tmp.depthOfTemperatureObservation,  marker='o')
    x = np.arange(0,20,0.1)
    plt.plot(f(np.log(x)),-x)
    plt.title(tmp.site)
    
    # core_df  = interp_pandas(core_df)
df = df.append(df_interp)

df['reference'] = 'van der Veen, C. J., Mosley-Thompson, E., Jezek, K. C., Whillans, I. M., and Bolzan, J. F.: Accumulation rates in South and Central Greenland, Polar Geography, 25, 79–162, https://doi.org/10.1080/10889370109377709, 2001.'
df['reference_short'] = 'van der Veen et al. (2001)'
df['method'] = 'thermistor'
df['durationOpen'] = 8*24
df['durationMeasured'] = 0
df['error'] = 0.1
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Koch Wegener 1913
df = pd.read_excel('Data/Koch Wegener/data.xlsx')

df['depthOfTemperatureObservation'] = 10
df['date'] = pd.to_datetime(df.date)
df[
    "reference"
] = "Koch, Johann P., and Alfred Wegener. Wissenschaftliche Ergebnisse Der Dänischen Expedition Nach Dronning Louises-Land Und Quer über Das Inlandeis Von Nordgrönland 1912 - 13 Unter Leitung Von Hauptmann J. P. Koch : 1 (1930). 1930."
df["reference_short"] = "Koch (1913)"
df["site"] = "Koch 1912-13 winter camp"
df['method'] = 'electric resistance thermometer'
df['durationOpen'] = 'NA'
df['durationMeasured'] = 'NA'
df['error'] = 0.2
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)

# %% Thomsen shallow thermistor
df = pd.read_excel('Data/Thomsen/data-formatted.xlsx')
# df = df.set_index('d').interpolate(method= 'index')

for date in df.date.unique():
    tmp = df.loc[df.date==date]
    if tmp.temperature.isnull().any():
        df.loc[df.date==date,'temperature'] = interp_pandas(tmp.set_index('depth').temperature).values


df['depthOfTemperatureObservation'] = df['depth']
df['temperatureObserved'] = df['temperature']
df['note'] = 'from unpublished pdf'
df['date'] = pd.to_datetime(df.date)
df[
    "reference"
] = "Thomsen, H. ., Olesen, O. ., Braithwaite, R. . and Bøggild, C. .: Ice drilling and mass balance at Pâkitsoq, Jakobshavn, central West Greenland, Rapp. Grønlands Geol. Undersøgelse, 152, 80–84, doi:10.34194/rapggu.v152.8160, 1991."
df["reference_short"] = "Thomsen et al. (1991)"
df['method'] = 'thermistor'
df['durationOpen'] = 'NA'
df['durationMeasured'] = 'NA'
df['error'] = 0.2
df_all = df_all.append(
    df[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]
    ],
    ignore_index=True,
)
# %% Checking values
df_all = df_all.loc[~df_all.temperatureObserved.isnull(),:]

df_ambiguous_date = df_all.loc[pd.to_datetime(df_all.date,errors='coerce').isnull(),:]
df_bad_long = df_all.loc[df_all.longitude.astype(float)>0,:]
df_no_coord = df_all.loc[np.logical_or(df_all.latitude.isnull(), df_all.latitude.isnull()),:]
df_invalid_depth =  df_all.loc[pd.to_numeric(df_all.depthOfTemperatureObservation,errors='coerce').isnull(),:]
df_no_elev =  df_all.loc[df_all.elevation.isnull(),:]
# df_no_elev.to_csv('missing_elev.csv')

# %% Removing nan and saving
tmp = df_all.loc[np.isnan(df_all.temperatureObserved.astype(float).values)]
df_all = df_all.loc[~np.isnan(df_all.temperatureObserved.astype(float).values)]

df_all.to_csv("output/subsurface_temperature_summary.csv")

# %% avereaging to monthly

df = pd.read_csv("output/subsurface_temperature_summary.csv")

df_ambiguous_date = df.loc[pd.to_datetime(df.date, errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, errors="coerce").isnull(), :]

df_bad_long = df.loc[df.longitude > 0, :]
df["longitude"] = -df.longitude.abs().values

df_hans_tausen = df.loc[df.latitude > 82, :]
df = df.loc[~(df.latitude > 82), :]

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

df = df.loc[df.depthOfTemperatureObservation==10, :]
df = df.loc[df.temperatureObserved.notnull(), :]

df['date'] = pd.to_datetime(df.date)
df.loc[df.reference_short.isnull(),'reference_short'] = df.loc[df.reference_short.isnull(),'reference']

df.loc[df.site.isnull(),'site'] = ['unnamed '+str(i) for i in df.loc[df.site.isnull()].index]

df_m = pd.DataFrame()
# generating the monthly file
for ref in df.reference_short.unique():
    for site in df.loc[df.reference_short == ref,'site'].unique():
        df_loc = df.loc[(df.reference_short == ref)&(df.site == site),:].copy()
        
        if np.sum((df_loc.date.diff() < '2 days') & (df_loc.date.diff() > '0 days')) > 5:
            print(ref, site, '... averaging to monthly')
            df_loc_first = df_loc.set_index('date').resample('M').first()
            df_loc = df_loc.set_index('date').resample('M').mean()
            df_loc[['Unnamed: 0', 'site', 'reference',
       'reference_short', 'note']] = df_loc_first[['Unnamed: 0', 'site', 'reference',
       'reference_short', 'note']]
            if any(df_loc.depthOfTemperatureObservation.unique() != 10):
                print('Some non-10 m depth')
                print(df_loc.depthOfTemperatureObservation.unique())
                print(df_loc.loc[df_loc.depthOfTemperatureObservation!=10].head())
                df_loc = df_loc.loc[df_loc.depthOfTemperatureObservation==10,:]
        
            df_m = df_m.append(df_loc.reset_index()[        
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]],ignore_index=True)
        else:
            print(ref, site, '... not averaging')
            df_m = df_m.append(df_loc[
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
            "error",
            "durationOpen",
            "durationMeasured",
            "method",
        ]],ignore_index=True)
                
df_m.to_csv('output/10m_temperature_dataset_monthly.csv',index=False)
