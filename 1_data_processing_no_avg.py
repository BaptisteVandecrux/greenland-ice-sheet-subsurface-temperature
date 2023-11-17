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
import matplotlib
matplotlib.use('Agg')

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
    s.plot(marker="o", linestyle="none")
    s_save.plot(marker="o", linestyle="none")
    # plt.xlim(0, 60)
    return s
plot = True
if plot:
    for f in os.listdir('figures/strings_raw/'):
        os.remove('figures/strings_raw/'+f)
def plot_string_dataframe(df_stack, site):
    f=plt.figure(figsize=(11,7))
    sc=plt.scatter(df_stack.date,
                -df_stack.depthOfTemperatureObservation,
                12, df_stack.temperatureObserved)
    plt.title(site)
    plt.colorbar(sc)
    f.savefig('figures/strings_raw/'+df_stack["reference_short"].iloc[0].replace(' ','_').replace('.','').replace('(','').replace(')','')+'_'+site+'.png')
    

needed_cols = ["date", "site", "latitude", "longitude", "elevation", "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note", "error", "durationOpen", "durationMeasured", "method"]

# %% Mock and Weeks
print("Loading Mock and Weeks")
df_all = pd.DataFrame(
    columns=[ "date", "site", "latitude", "longitude", "elevation", "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note", "error", "durationOpen", "durationMeasured", "method"]
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

df_MW["durationOpen"] = "NA"
df_MW["durationMeasured"] = "NA"
df_MW["error"] = 0.5
df_MW["method"] = "thermohms and a Wheats tone bridge, standard mercury or alcohol thermometers"

df_all = pd.concat((df_all,  df_MW[needed_cols]), ignore_index=True)

# %% Benson (not reported in Mock and Weeks)
del df_MW
print("Loading Benson 1962")
df_benson = pd.read_excel("Data/Benson 1962/Benson_1962.xlsx")
df_benson.loc[df_benson.Month.isnull(), "Day"] = 1
df_benson.loc[df_benson.Month.isnull(), "Month"] = 1
df_benson["date"] = pd.to_datetime(df_benson[["Year", "Month", "Day"]])
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
msk = (df_benson.site == "0-35") | (df_benson.site == "French Camp VI")
df_benson = df_benson.loc[msk, :]

df_benson["durationOpen"] = "measured in pit wall or borehole bottom after excavation"
df_benson["durationMeasured"] = "few minutes"
df_benson["error"] = "NA"
df_benson["method"] = "Weston bimetallic thermometers"

df_all = pd.concat((df_all,  df_benson[needed_cols]), ignore_index=True)

# %% Polashenski
del df_benson, msk
print("Loading Polashenski")
df_Pol = pd.read_csv("Data/Polashenski/2013_10m_Temperatures.csv")
df_Pol.columns = df_Pol.columns.str.replace(" ", "")
df_Pol.date = pd.to_datetime(df_Pol.date, format="%m/%d/%y")
df_Pol["reference"] = "Polashenski, C., Z. Courville, C. Benson, A. Wagner, J. Chen, G. Wong, R. Hawley, and D. Hall (2014), Observations of pronounced Greenland ice sheet firn warming and implications for runoff production, Geophys. Res. Lett., 41, 4238–4246, doi:10.1002/2014GL059806."
df_Pol["reference_short"] = "Polashenski et al. (2014)"
df_Pol["note"] = ""
df_Pol["longitude"] = -df_Pol["longitude"]
df_Pol["depthOfTemperatureObservation"] = (
    df_Pol["depthOfTemperatureObservation"].str.replace("m", "").astype(float)
)
df_Pol[
    "durationOpen"
] = "string lowered in borehole and left 30min for equilibrating with surrounding firn prior measurement start"
df_Pol["durationMeasured"] = "overnight ~10 hours"
df_Pol["error"] = 0.1
df_Pol["method"] = "thermistor string"

df_all = pd.concat((df_all, df_Pol[needed_cols] ), ignore_index=True)

# %% Ken's dataset
del df_Pol
print("Loading Kens dataset")
df_Ken = pd.read_excel("Data/greenland_ice_borehole_temperature_profiles-main/data_filtered.xlsx")

df_all = pd.concat((df_all, df_Ken[needed_cols]), ignore_index=True)
# %% Sumup
del df_Ken
df_sumup = pd.read_csv("Data/Sumup/SUMup_temperature_2022.csv")
df_sumup = df_sumup.loc[df_sumup.Latitude > 0]
df_sumup = df_sumup.loc[
    df_sumup.Citation != 30
]  # Redundant with Miege and containing positive temperatures

df_sumup["date"] = pd.to_datetime(df_sumup.Timestamp)
df_sumup["note"] = "as reported in Sumup"
df_sumup["reference"] = ""
df_sumup["site"] = ""
df_sumup.loc[
    df_sumup.Citation == 32, "reference"
] = "Graeter, K., Osterberg, E. C., Ferris, D., Hawley, R. L., Marshall, H. P. and Lewis, G.: Ice Core Records of West Greenland Surface Melt and Climate Forcing, Geophys. Res. Lett., doi:10.1002/2017GL076641, 2018."
df_sumup.loc[
    df_sumup.Citation == 32, "reference_short"
] = "Graeter et al. (2018) as in SUMup"
df_sumup.loc[
    df_sumup.Citation == 33, "reference"
] = "Lewis, G., Osterberg, E., Hawley, R., Marshall, H. P., Meehan, T., Graeter, K., McCarthy, F., Overly, T., Thundercloud, Z. and Ferris, D.: Recent precipitation decrease across the western Greenland ice sheet percolation zone, Cryosph., 13(11), 2797–2815, doi:10.5194/tc-13-2797- 2019, 2019."
df_sumup.loc[
    df_sumup.Citation == 33, "reference_short"
] = "Lewis et al. (2019) as in SUMup"

site_list = pd.DataFrame(
    np.array(
        [
            [44, "GTC01"],
            [45, "GTC02"],
            [46, "GTC04"],
            [47, "GTC05"],
            [48, "GTC06"],
            [49, "GTC07"],
            [50, "GTC08"],
            [51, "GTC09"],
            [52, "GTC11"],
            [53, "GTC12"],
            [54, "GTC13"],
            [55, "GTC14"],
            [56, "GTC15"],
            [57, "GTC16"],
        ]
    ),
    columns=["id", "site"],
).set_index("id")
for ind in site_list.index:
    df_sumup.loc[df_sumup.Name == int(ind), "site"] = site_list.loc[ind, "site"]


df_sumup = df_sumup.drop(["Name", "Citation", "Timestamp"], axis=1)

df_sumup = df_sumup.rename(
    columns={
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Elevation": "elevation",
        "Depth": "depthOfTemperatureObservation",
        "Temperature": "temperatureObserved",
        "Duration": "durationMeasured",
        "Error": "error",
        "Open_Time": "durationOpen",
        "Method": "method",
    }
)
df_all = pd.concat((df_all,  df_sumup[needed_cols]), ignore_index=True)

# %% McGrath
del df_sumup, ind, site_list
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

df_mcgrath["method"] = "digital Thermarray system from RST©"
df_mcgrath["durationOpen"] = 0
df_mcgrath["durationMeasured"] = 30
df_mcgrath["error"] = 0.07

df_all = pd.concat((df_all,  df_mcgrath[needed_cols]), ignore_index=True)

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
del df_mcgrath, df_fausto, site
print("Loading Hawley GrIT")
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
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

df_hawley["method"] = "thermistor"
df_hawley["durationOpen"] = 2
df_hawley["durationMeasured"] = 0
df_hawley["error"] = "not reported"

df_all = pd.concat((df_all, df_hawley[needed_cols]), ignore_index=True)

# %% PROMICE
del df_hawley
print("Loading PROMICE")
# df_promice = pd.read_csv("Data/PROMICE/PROMICE_10m_firn_temperature.csv", sep=";")
# df_promice = df_promice.loc[df_promice.temperatureObserved.notnull()]
# df_promice = df_promice.loc[df_promice.site != "QAS_A", :]
# df_promice.loc[(df_promice.site == "CEN") & (df_promice.temperatureObserved > -18),
#                "temperatureObserved"] = np.nan

df_promice = xr.open_dataset('Data/netcdf/PROMICE_GC-Net_GEUS_subsurface_temperatures.nc').to_dataframe()
df_promice = df_promice.reset_index()
df_promice["method"] = "RS Components thermistors 151-243"
df_promice["durationOpen"] = 0
df_promice["durationMeasured"] = np.nan
df_promice["error"] = 0.2
df_promice["note"] = ""
df_promice["reference_short"] = "PROMICE and GC-Net (GEUS)"

df_promice=df_promice.loc[df_promice.temperature.notnull(),:]
df_promice = df_promice.rename(columns={'temperature': 'temperatureObserved',
                                        'depth': 'depthOfTemperatureObservation',
                                        'time':'date'})
if plot:
    for site in df_promice.site.unique():
        plot_string_dataframe(df_promice.loc[df_promice.site==site], site)
df_all = pd.concat((df_all, df_promice[needed_cols]), ignore_index=True)

# %% GC-Net
del df_promice
print("Loading GC-Net")
# df_GCN = pd.read_csv("Data/GC-Net/10m_firn_temperature.csv")
df_GCN = xr.open_dataset('Data/netcdf/Historical_GC-Net_subsurface_temperatures.nc').to_dataframe()
df_GCN = df_GCN.reset_index()
df_GCN=df_GCN.loc[df_GCN.temperature.notnull(),:]
df_GCN["method"] = "thermocouple"
df_GCN["durationOpen"] = 0
df_GCN["durationMeasured"] = np.nan
df_GCN["error"] = 0.5
df_GCN["note"] = ""
df_GCN["reference_short"] = "PROMICE and GC-Net (GEUS)"

df_GCN = df_GCN.rename(columns={'temperature': 'temperatureObserved',
                                        'depth': 'depthOfTemperatureObservation',
                                        'time':'date'})
if plot:
    for site in df_GCN.site.unique():
        plot_string_dataframe(df_GCN.loc[df_GCN.site==site], site)

df_all = pd.concat((df_all, df_GCN[needed_cols]), ignore_index=True)

# %% Steffen 2001 table (that could not be found in the GC-Net AWS data)
del df_GCN
df = pd.read_excel("Data/GC-Net/steffen2001.xlsx")
df["depthOfTemperatureObservation"] = 10
df["temperatureObserved"] = df["temperature"]
df["note"] = "annual average"
df["date"] = [pd.to_datetime(str(yr) + "-12-01") for yr in df.year]
df[
    "reference"
] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103. and Steffen, K. and J. Box: Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001 and Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023. and Vandecrux, B., Box, J.E., Ahlstrøm, A.P., Andersen, S.B., Bayou, N., Colgan, W.T., Cullen, N.J., Fausto, R.S., Haas-Artho, D., Heilig, A., Houtz, D.A., How, P., Iosifescu Enescu , I., Karlsson, N.B., Kurup Buchholz, R., Mankoff, K.D., McGrath, D., Molotch, N.P., Perren, B., Revheim, M.K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P.J., Zwally, J., Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented Level 1 dataset, Submitted to ESSD, 2023"
df["reference_short"] = "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)"
df["error"] = 0.5
df_all = pd.concat((df_all[needed_cols], df[needed_cols]), ignore_index=True)

# %% Historical swc
del df
df_swc = pd.DataFrame()
df_swc["depthOfTemperatureObservation"] = [10]*3
df_swc["site"] = ["Swiss Camp"]*3
	
df_swc["latitude"] = [69.57346306]*3
df_swc["longitude"] = [-49.2955275]*3
df_swc["elevation"] = [1155]*3
df_swc["note"] = [""]*3
df_swc["temperatureObserved"] = [-9.1, -9.3, -9.3]
df_swc["date"] = pd.to_datetime(["1990-07-01","1990-08-01","1990-08-24" ])
df_swc["reference_short"] = ["Ohmura et al. (1992)"]*3
df_swc[
    "reference"
] = ["Ohmura, A., Steffen, K., Blatter, H., Greuell, W., Rotach, M., Stober, M., Konzelmann, T., Forrer, J., Abe-Ouchi, A., Steiger, D. and Niederbaumer, G.: Energy and mass balance during the melt season at the equilibrium line altitude. Paakitsoq, Greenland ice sheet: Progress report, 2, 1992."]*3

df_swc["method"] = ["NA"]*3
df_swc["durationOpen"] = ["NA"]*3
df_swc["durationMeasured"] =[ "NA"]*3
df_swc["error"] = ["NA"]*3

df_all = pd.concat((df_all, df_swc[needed_cols]), ignore_index=True)

# %% Miege aquifer
del df_swc
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

for k, site in enumerate(["FA_13", "FA_15_1", "FA_15_2"]):
    depth = pd.read_csv(
        "Data/Miege firn aquifer/" + site + "_Firn_Temperatures_Depths.csv"
    ).transpose()
    print('    ', site)
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

    ellapsed_hours = (dates - dates[0]).dt.total_seconds()/60/60
    accum_depth = ellapsed_hours.values * thickness_accum / 365 / 24
    depth_cor = pd.DataFrame()
    depth_cor = depth.values.reshape((1, -1)).repeat(
        len(dates), axis=0
    ) + accum_depth.reshape((-1, 1)).repeat(len(depth.values), axis=1)

    temp.columns = temp.columns.str.replace('n','').astype(int)
    depth_ds = temp.copy()
    for i, col in enumerate(depth_ds.columns):
        depth_ds[col] = depth_cor[:,i]
        
    df_miege = temp.stack().to_frame()
    df_miege.columns = ['temperatureObserved']
    df_miege['depthOfTemperatureObservation'] = depth_ds.stack()
    df_miege = df_miege.reset_index()
    df_miege['date'] = dates.loc[df_miege.level_0].values
    df_miege = df_miege.drop(columns=['level_0','level_1'])
    
    df_miege = df_miege.loc[df_miege.depthOfTemperatureObservation.notnull(),:]
    df_miege = df_miege.loc[df_miege.temperatureObserved.notnull(),:]
    
    df_miege["site"] = site
    df_miege["latitude"] = float(metadata[k, 1])
    df_miege["longitude"] = -float(metadata[k, 2])
    df_miege["elevation"] = float(metadata[k, 3])
    df_miege[
        "reference"
    ] = "Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W"
    df_miege["reference_short"] = "Miller et al. (2020)"
    df_miege["note"] = ""


    df_miege["method"] = "digital thermarray system from RST©"
    df_miege["durationOpen"] = np.nan
    df_miege["durationMeasured"] = np.nan
    df_miege["error"] = 0.07
    
    if plot:
        plot_string_dataframe(df_miege, site)

    df_all = pd.concat((df_all, 
        df_miege[needed_cols],
        ), ignore_index=True,
    )

# %% Harper ice temperature
del df_miege, dates, col, accum_depth, depth_cor, depth_ds, ellapsed_hours
del site, i, k, thickness_accum, depth, temp, metadata
print("Loading Harper ice temperature")
df_harper = pd.read_csv(
    "Data/Harper ice temperature/harper_iceTemperature_2015-2016.csv"
)
df_harper["temperatureObserved"] = np.nan
df_harper["note"] = ""
df_harper[[pd.to_datetime("2015-01-01"),pd.to_datetime("2016-01-01")]] = df_harper[['temperature_2015_celsius','temperature_2016_celsius']]
df_stack = (
    df_harper[[pd.to_datetime("2015-01-01"),pd.to_datetime("2016-01-01")]]
    .stack().to_frame(name='temperatureObserved')
    .reset_index().rename(columns={'level_1':'date'})
                                   )
                                   
df_stack['site'] = df_harper.loc[df_stack.level_0,'borehole'].values
df_stack['latitude'] = df_harper.loc[df_stack.level_0,'latitude_WGS84'].values
df_stack['longitude'] = df_harper.loc[df_stack.level_0,'longitude_WGS84'].values
df_stack['elevation'] = df_harper.loc[df_stack.level_0,'Elevation_m'].values
df_stack['depthOfTemperatureObservation'] = df_harper.loc[df_stack.level_0,'depth_m'].values-df_harper.loc[df_stack.level_0,'height_m'].values
df_stack = df_stack.loc[df_stack.temperatureObserved.notnull()]

plt.figure()
plt.gca().invert_yaxis()
for borehole in df_stack["site"].unique():
    df_stack.loc[df_stack.site==borehole].plot(ax=plt.gca(),
                                               x='temperatureObserved',
                                               y='depthOfTemperatureObservation', 
                                               label=borehole)
plt.legend()

df_stack[
    "reference"
] = "Hills, B. H., Harper, J. T., Humphrey, N. F., & Meierbachtol, T. W. (2017). Measured horizontal temperature gradients constrain heat transfer mechanisms in Greenland ice. Geophysical Research Letters, 44. https://doi.org/10.1002/2017GL074917; Data: https://doi.org/10.18739/A24746S04"

df_stack["reference_short"] = "Hills et al. (2017)"

df_stack["method"] = "TMP102 digital temperature sensor"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] = 30 * 24
df_stack["error"] = 0.1
df_stack["note"] = ""

df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)

# %%  FirnCover
del df_stack, df_harper, borehole
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
    print('   ',site)
    df_d = rtd_df.xs(site, level="sitename").reset_index()
    df_stack = (
        df_d[[v for v in df_d.columns if v.startswith('rtd')]]
        .stack(dropna=False).to_frame(name='temperatureObserved').reset_index()
        )
    df_stack['depthOfTemperatureObservation'] = df_d[
        [v.replace('rtd','depth_') for v in df_d.columns if v.startswith('rtd')]
        ].stack(dropna=False).values
    
    df_stack['date'] = df_d.loc[df_stack.level_0, 'date'].values
    df_stack["site"] = site
    if site == "Crawford":
        df_stack["site"] = "CP1"
    df_stack["latitude"] = statmeta_df.loc[site, "latitude"]
    df_stack["longitude"] = statmeta_df.loc[site, "longitude"]
    df_stack["elevation"] = statmeta_df.loc[site, "elevation"]

    df_stack[
        "reference"
    ] = "MacFerrin, M. J., Stevens, C. M., Vandecrux, B., Waddington, E. D., and Abdalati, W. (2022) The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) dataset, 2013–2019, Earth Syst. Sci. Data, 14, 955–971, https://doi.org/10.5194/essd-14-955-2022,"
    df_stack["reference_short"] = "MacFerrin et al. (2021, 2022)"
    df_stack["note"] = ""
    
    # Correction of FirnCover bias
    p = np.poly1d([1.03093649, -0.49950273])
    df_stack["temperatureObserved"] = p(df_stack["temperatureObserved"].values)
    
    df_stack["method"] = "Resistance Temperature Detectors + correction"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 24
    df_stack["error"] = 0.5
    
    if plot:
        plot_string_dataframe(df_stack, site)

    df_all = pd.concat((df_all, 
        df_stack[needed_cols],
        ), ignore_index=True,
    )

# %% SPLAZ KAN_U
del df_stack, sonic_df, df_d, metdata_df, df_firncover, rtd_dep, rtd_df, site, sites, statmeta_df

print("Loading SPLAZ at KAN-U")
num_therm = [32, 12, 12]

for k, note in enumerate(["SPLAZ_main", "SPLAZ_2", "SPLAZ_3"]):
    print('   ',note)
    ds = xr.open_dataset("Data/SPLAZ/T_firn_KANU_" + note + ".nc")
    ds=ds.where(ds['Firn temperature'] != -999)  
    ds=ds.where(ds['depth'] != -999)  
    ds=ds.where(ds['depth'] >0.1)  
    ds = ds.resample(time='H').mean()
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df = df.rename(columns={'time':'date',
                            'Firn temperature':'temperatureObserved',
                            'depth':'depthOfTemperatureObservation'})

    df["note"] = ''
    df["latitude"] = 67.000252
    df["longitude"] = -47.022999
    df["elevation"] = 1840
    df["site"] = "KAN_U "+note
    df[
        "reference"
    ] = "Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10."
    df["reference_short"] = "Charalampidis et al. (2016); Charalampidis et al. (2022) "    
    df["method"] = "RS 100 kΩ negative-temperature coefficient thermistors"
    df["durationOpen"] = 0
    df["durationMeasured"] = 1
    df["error"] = 0.2
    
    if plot:
        plot_string_dataframe(df, 'KAN_U_'+note)
    
    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Load Humphrey data
del df, filepath, note, num_therm, k
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
    except: 
        continue

    print(site)
    temp_label = df_site.columns[1:]
    # the first column is a time stamp and is the decimal days after the first second of January 1, 2007.
    df_site["time"] = [datetime(2007, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]]
    if site == "T1old":
        df_site["time"] = [
            datetime(2006, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]
        ]
    df_site = df_site.loc[df_site["time"] <= df_site["time"].values[-1], :]
    df_site = df_site.set_index("time")
    df_site = df_site.resample("H").mean()

    depth = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0]

    if site != "H5": df_site = df_site.iloc[24 * 30 :, :]
    if site == "T4": df_site = df_site.loc[:"2007-12-05"]
    if site == "H2": depth = np.array(depth) - 1
    if site == "H4": depth = np.array(depth) - 0.75
    if site in ["H3", "G165", "T1new"]: depth = np.array(depth) - 0.50

    df_hs = pd.read_csv("Data/Humphrey string/" + site + "_surface_height.csv")
    df_hs.time = pd.to_datetime(df_hs.time)
    df_hs = df_hs.set_index("time")
    df_hs = df_hs.resample("H").mean()
    df_site["surface_height"] = np.nan

    df_site["surface_height"] = df_hs.iloc[
        df_hs.index.get_indexer(df_site.index, method="nearest")
    ].values

    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        plt.figure()
        df_site[depth_label[i]] = (
            depth[i]
            + df_site["surface_height"].values
            - df_site["surface_height"].iloc[0]
        )
        df_site[temp_label[i]].plot(ax=plt.gca())

        df_site.loc[
            df_site[temp_label[i]] <  (df_site[temp_label[i]].rolling(24*7,
                                            min_periods=24*2,
                                            center=True).max()-1),
            temp_label[i]
            ] = np.nan
        df_site[temp_label[i]].plot(ax=plt.gca())
        plt.ylabel(temp_label[i])
        plt.title(site)

    df_stack = ( df_site[temp_label]
                .stack(dropna=False)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df_site[depth_label].stack(dropna=False).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

    df_stack["site"] = site
    df_stack["latitude"] = df.loc[df.site == site, "latitude"].values[0]
    df_stack["longitude"] = df.loc[df.site == site, "longitude"].values[0]
    df_stack["elevation"] = df.loc[df.site == site, "elevation"].values[0]

    df_stack[
        "reference"
    ] = "Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/"
    df_stack["reference_short"] = "Humphrey et al. (2012)"
    df_stack[
        "note"
    ] = "no surface height measurements, using interpolating surface height using CP1 and SwissCamp stations"
    
    df_stack["method"] = "sealed 50K ohm thermistors"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 1
    df_stack["error"] = 0.5
    
    if plot:
        plot_string_dataframe(df_stack, site)
    
    df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)

# %% loading Hills
del df_stack, df_site, df_hs, df_humphrey, depth, depth_label, df,  i, site, temp_label
print("Loading Hills")
df_meta = pd.read_csv("Data/Hills/metadata.txt", sep=" ")
df_meta.date_start = pd.to_datetime(df_meta.date_start, format="%m/%d/%y")
df_meta.date_end = pd.to_datetime(df_meta.date_end, format="%m/%d/%y")

df_meteo = pd.read_csv("Data/Hills/Hills_33km_meteorological.txt", sep="\t")
df_meteo["date"] = [
    pd.to_datetime("2014-07-18") + pd.Timedelta(int(f * 24 * 60 * 60), "seconds")
    for f in (df_meteo.Time.values - 197)
]
df_meteo = df_meteo.set_index("date").resample("D").mean()
df_meteo["surface_height"] = (
    df_meteo.DistanceToTarget.iloc[0] - df_meteo.DistanceToTarget
)

for site in df_meta.site[:-1]:
    print(site)
    df = pd.read_csv("Data/Hills/Hills_" + site + "_IceTemp.txt", sep="\t")
    df["date"] = [
        df_meta.loc[df_meta.site == site, "date_start"].values[0]
        + pd.Timedelta(int(f * 24 * 60 * 60), "seconds")
        for f in (df.Time.values - int(df.Time.values[0]))
    ]

    df = df.set_index("date")
    df["surface_height"] = np.nan
    df.loc[[i in df_meteo.index for i in df.index], "surface_height"] = df_meteo.loc[
        [i in df.index for i in df_meteo.index], "surface_height"
    ]
    df["surface_height"] = df.surface_height.interpolate(
        method="linear", limit_direction="both"
    )
    if all(np.isnan(df.surface_height)):
        df["surface_height"] = 0

    depth = df.columns[1:-1].str.replace("Depth_", "").values.astype(float)
    temp_label = ["temp_" + str(len(depth) - i) for i in range(len(depth))]
    depth_label = ["depth_" + str(len(depth) - i) for i in range(len(depth))]

    for i in range(len(temp_label)):
        df = df.rename(columns={df.columns[i + 1]: temp_label[i]})
        df.iloc[:14, i + 1] = np.nan
        if site in ["T-14", "T-11b"]:
            df.iloc[:30, i + 1] = np.nan

        df[depth_label[i]] = (
            depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
        )

    df_stack = ( df[temp_label]
                .stack(dropna=False)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]


    df_stack["latitude"] = df_meta.latitude[df_meta.site == site].iloc[0]
    df_stack["longitude"] = df_meta.longitude[df_meta.site == site].iloc[0]
    df_stack["elevation"] = df_meta.elevation[df_meta.site == site].iloc[0]
    df_stack["site"] = site

    df_stack["note"] = ''
    df_stack[
        "reference"
    ] = "Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418"
    
    df_stack["reference_short"] = "Hills et al. (2018)"
    df_stack[
        "method"
    ] = "digital temperature sensor model DS18B20 from Maxim Integrated Products, Inc."
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 1
    df_stack["error"] = 0.0625
    
    if plot:
        plot_string_dataframe(df_stack, site)
    
    df_all = pd.concat((df_all, 
        df_stack[needed_cols],
        ), ignore_index=True,
    )

# %% Achim Dye-2
del df_stack, df_meta, df_meteo, df, i, site, temp_label, depth_label, depth
print("Loading Achim Dye-2")

# loading temperature data
df = pd.read_csv("Data/Achim/CR1000_PT100.txt", header=None)
df.columns = [ "time_matlab", "temp_1", "temp_2", "temp_3", "temp_4", "temp_5", "temp_6", "temp_7", "temp_8"]
df["time"] = pd.to_datetime(
    [
        datetime.fromordinal(int(matlab_datenum))
        + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366)
        for matlab_datenum in df.time_matlab
    ]
).round('h')

df = df.set_index("time")
# loading surface height data
df_surf = pd.read_csv("Data/Achim/CR1000_SR50.txt", header=None)
df_surf.columns = ["time_matlab", "sonic_m", "height_above_upgpr"]
df_surf["time"] = pd.to_datetime(
    [
        datetime.fromordinal(int(matlab_datenum))
        + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366)
        for matlab_datenum in df_surf.time_matlab
    ]
).round('h')
df_surf = df_surf.set_index("time")[['sonic_m']]


# loading surface height data from firncover
filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
_, sonic_df, _, _, _ = ftl.load_metadata(filepath, sites)

sonic_df = sonic_df.xs("DYE-2", level="sitename").reset_index().rename(columns={'date':'time'})
sonic_df = sonic_df.set_index("time").drop(columns="delta").resample("H").interpolate()

# aligning and merging the surface heights from Achim and Macferrin
sonic_df = pd.concat(
    (sonic_df.reset_index(),
     (df_surf.loc[sonic_df.index[-1]:] - 1.83).reset_index()),
    ignore_index=True).set_index('time')


df3 = sonic_df.loc[df.index[0]] - sonic_df.loc[df.index[0] : df.index[-1]]
df3=df3[~df3.index.duplicated(keep='first')].loc[df.index.drop_duplicates()]
df["surface_height"] = df3.values

depth = 3.4 - np.array([3, 2, 1, 0, -1, -2, -4, -6])
temp_label = ["temp_" + str(i + 1) for i in range(len(depth))]
depth_label = ["depth_" + str(i + 1) for i in range(len(depth))]

for i in range(len(depth)):
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df.loc["2018-05-18":, "depth_1"] = df.loc["2018-05-18":, "depth_1"].values - 1.5
df.loc["2018-05-18":, "depth_2"] = df.loc["2018-05-18":, "depth_2"].values - 1.84

df_stack = ( df[temp_label]
            .stack(dropna=False)
            .to_frame(name='temperatureObserved')
            .reset_index()
            .rename(columns={'time':'date'}))
df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]
df_stack["site"] = "DYE-2"
df_stack["latitude"] = 66.4800
df_stack["longitude"] = -46.2789
df_stack["elevation"] = 2165.0
df_stack[
    "note"
] = "using surface height from FirnCover station"
df_stack[
    "reference"
] = "Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018. "
df_stack["reference_short"] = "Heilig et al. (2018)"
df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] = 1
df_stack["error"] = 0.25

if plot:
    plot_string_dataframe(df_stack, 'KAN_U_Heilig')
df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)

# %% Camp Century Climate
del df_stack, df, sonic_df, df3, df_surf, depth, depth_label, temp_label, i, filepath, sites
print("Loading Camp Century data")
df = pd.read_csv("Data/Camp Century Climate/data_long.txt", sep=",", header=None)
df = df.rename(columns={0: "date"})
df["date"] = pd.to_datetime(df.date)
df[df == -999] = np.nan
df = df.set_index("date").resample("D").first()
df = df.iloc[:, :-2]

df_promice = pd.read_csv(
    "C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/PROMICE-AWS-toolbox/out/v03_L3/CEN_hour_v03_L3.txt",
    sep="\t",
)
df_promice[df_promice == -999] = np.nan
df_promice = df_promice.rename(columns={"time": "date"})
df_promice["date"] = pd.to_datetime(df_promice.date)
df_promice = df_promice.set_index("date").resample("D").first()
df_promice = df_promice["SurfaceHeight_summary(m)"]

df_cen2 = pd.read_csv('Data/Camp Century Climate/CEN2_day.csv')
df_cen2['time'] = pd.to_datetime(df_cen2.time, utc=True)
df_cen2 = df_cen2.set_index('time')[['z_boom_l']].resample('D').first()
df_cen2["SurfaceHeight_summary(m)"] = 2.1130 - df_cen2.z_boom_l + 1.9 
df_cen2.loc[:'2021-08-24',"SurfaceHeight_summary(m)"] = np.nan
df_cen2.loc['2022-06-17':,"SurfaceHeight_summary(m)"] =df_cen2.loc['2022-06-17':,"SurfaceHeight_summary(m)"] + 0.85


df_promice = pd.concat((df_promice, df_cen2["SurfaceHeight_summary(m)"] ))
df_promice = df_promice.resample('D').mean().interpolate()

temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]

df["surface_height"] = df_promice.loc[df.index[0].strftime('%Y-%m-%d'):df.index[-1].strftime('%Y-%m-%d')].values

depth = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 53, 58, 63, 68, 73]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )
df_stack = ( df[temp_label]
            .stack(dropna=False)
            .to_frame(name='temperatureObserved')
            .reset_index()
            .rename(columns={'time':'date'}))
df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

df_stack.loc[df_stack.temperatureObserved>-15, "temperatureObserved"] = np.nan
df_stack = df_stack.set_index("date").reset_index()
site="CEN_THM_long"
df_stack["site"] = site
df_stack["latitude"] = 77.1333
df_stack["longitude"] = -61.0333
df_stack["elevation"] = 1880
df_stack["note"] = ""
df_stack[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
df_stack["reference_short"] = "Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] =24
df_stack["error"] = 0.2

if plot:
    plot_string_dataframe(df_stack, site)
    
df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

df = pd.read_csv("Data/Camp Century Climate/data_short.txt", sep=",", header=None)
df = df.rename(columns={0: "date"})
df["date"] = pd.to_datetime(df.date)
df[df == -999] = np.nan
df = df.set_index("date").resample("D").mean()

temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]
df["surface_height"] = df_promice.loc[df.index[0].strftime('%Y-%m-%d'):df.index[-1].strftime('%Y-%m-%d')].values

depth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 22, 25, 28, 31, 34, 38, 42, 46, 50, 54]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df_stack = ( df[temp_label]
            .stack(dropna=False)
            .to_frame(name='temperatureObserved')
            .reset_index()
            .rename(columns={'time':'date'}))
df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

df_stack.loc[df_stack.temperatureObserved > -15, "temperatureObserved"] = np.nan
site="CEN_THM_short"
df_stack["site"] = site
df_stack["latitude"] = 77.1333
df_stack["longitude"] = -61.0333
df_stack["elevation"] = 1880
df_stack["note"] = ""
df_stack[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
# df_10.temperatureObserved.plot()
df_stack["reference_short"] = "Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] = 30 * 24
df_stack["error"] = 0.2

if plot:
    plot_string_dataframe(df_stack, site)
    
df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

# %% Camp Century historical
# % In 1977, firn density was measured to a depth of 100.0 m, and firn temperature was measured at 10 m depth (Clausen and Hammer, 1988). In 1986, firn density was measured to a depth of 12.0 m, and the 10 m firn temperature was again measured (Gundestrup et al., 1987).
del df, df_cen2, df_promice, site, temp_label, depth, depth_label, df_stack
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
df_cc_hist["method"] = "NA"
df_cc_hist["durationOpen"] = "NA"
df_cc_hist["durationMeasured"] = "NA"
df_cc_hist["error"] = "NA"
df_all = pd.concat((df_all, 
    df_cc_hist[needed_cols],
    ), ignore_index=True,
)

for site in ['CEN1', 'CEN2', 'CEN','CEN_THM_short','CEN_THM_long']:
    df_all.loc[df_all.site==site,'note'] = df_all.loc[df_all.site==site,'note'] + ' ' +\
                                            df_all.loc[df_all.site==site,'site']
    # df_all.loc[df_all.site==site,'site'] = 'Camp Century'  


# %% Davies dataset
del df_cc_hist, i, site
df_davies = pd.read_excel("Data/Davies South Dome/table_3_digitized.xlsx")
df_davies["depthOfTemperatureObservation"] = 10
df_davies["note"] = ""
df_davies[
    "reference"
] = "Davies, T.C., Structures in the upper snow layers of the southern Dome Greenland ice sheet, CRREL research report 115, 1954"
df_davies["reference_short"] = "Davies (1954)"

df_davies["method"] = "Weston bimetallic thermometers"
df_davies["durationOpen"] = 0
df_davies["durationMeasured"] = 0
df_davies["error"] = 0.5

df_all = pd.concat((df_all,  df_davies[needed_cols]), ignore_index=True)

# %%  Echelmeyer Jakobshavn isbræ
del df_davies
df_echel = pd.read_excel("Data/Echelmeyer jakobshavn/Fig5_points.xlsx")
df_echel.date = pd.to_datetime(df_echel.date)
df_echel = df_echel.rename(
    columns={"12 m temperature": "temperatureObserved", "Name": "site"}
)
df_echel["depthOfTemperatureObservation"] = 12

df_profiles = pd.read_excel("Data/Echelmeyer jakobshavn/Fig3_profiles.xlsx")
df_profiles = df_profiles[pd.DatetimeIndex(df_profiles.date).month > 3]
df_profiles["date"] = pd.to_datetime(df_profiles.date)
df_profiles = df_profiles.set_index(["site", "date"], drop=False).rename(
    columns={"temperature": "temperatureObserved", 
             "depth": "depthOfTemperatureObservation"}
)
df_profiles["note"] = "digitized"

df_echel = pd.concat((df_echel, df_profiles), ignore_index=True)

df_echel["reference"] = "Echelmeyer Κ, Harrison WD, Clarke TS and Benson C (1992) Surficial Glaciology of Jakobshavns Isbræ, West Greenland: Part II. Ablation, accumulation and temperature. Journal of Glaciology 38(128), 169–181 (doi:10.3189/S0022143000009709)"
df_echel["reference_short"] = "Echelmeyer et al. (1992)"

df_echel["method"] = "thermistors and thermocouples"
df_echel["durationOpen"] = 0
df_echel["durationMeasured"] = 0
df_echel["error"] = 0.3

df_all = pd.concat((df_all,  df_echel[needed_cols]), ignore_index=True)

# %% Fischer de Quervain EGIG
del df_echel, df_profiles
df_fischer = pd.read_excel("Data/Fischer EGIG/fischer_90_91.xlsx")

df_fischer[
    "reference"
] = "Fischer, H., Wagenbach, D., Laternser, M. & Haeberli, W., 1995. Glacio-meteorological and isotopic studies along the EGIG line, central Greenland. Journal of Glaciology, 41(139), pp. 515-527."
df_fischer["reference_short"] = "Fischer et al. (1995)"

df_all = pd.concat((df_all, 
    df_fischer[needed_cols],
    ), ignore_index=True,
)

df_dequervain = pd.read_csv("Data/Fischer EGIG/DeQuervain.txt", index_col=False)
df_dequervain.date = pd.to_datetime(
    df_dequervain.date.str[:-2] + "19" + df_dequervain.date.str[-2:]
)
df_dequervain = df_dequervain.rename(
    columns={"depth": "depthOfTemperatureObservation", "temp": "temperatureObserved"}
)
df_dequervain["note"] = "as reported in Fischer et al. (1995)"
df_dequervain["reference"] = "de Quervain, M, 1969. Schneekundliche Arbeiten der Internationalen Glaziologischen Grönlandexpedition (Nivologie). Medd. Grønl. 177(4)"
df_dequervain["reference_short"] = "de Quervain (1969)"

df_dequervain["method"] = "bimetallic, mercury, Wheastone bridge, platinium resistance thermometers"
df_dequervain["durationOpen"] = 0
df_dequervain["durationMeasured"] = 0
df_dequervain["error"] = 0.2

df_all = pd.concat((df_all, df_dequervain[needed_cols]), ignore_index=True)

# %% Larternser EGIG
del df_fischer, df_dequervain
df_laternser = pd.read_excel("Data/Laternser 1992/Laternser94.xlsx")

df_laternser["reference"] = "Laternser, M., 1994 Firn temperature measurements and snow pit studies on the EGIG traverse of central Greenland, 1992. Eidgenössische Technische Hochschule.  Versuchsanstalt für Wasserbau  Hydrologie und Glaziologic. (Arbeitsheft 15)."
df_laternser["reference_short"] = "Laternser (1994)"

df_laternser["method"] = "Fenwal 197-303 KAG-401 thermistors"
df_laternser["durationOpen"] = 0
df_laternser["error"] = 0.02

df_all = pd.concat((df_all,  df_laternser[needed_cols]), ignore_index=True)
# %% Wegener 1929-1930
del df_laternser
df1 = pd.read_csv("Data/Wegener 1930/200mRandabst_firtemperature_wegener.csv", sep=";")
df3 = pd.read_csv("Data/Wegener 1930/ReadMe.txt", sep=";")

df1["depthOfTemperatureObservation"] = df1.depth / 100
df1["temperatureObserved"] = df1.Firntemp
df1["date"] = df3.date.iloc[0]
df1["latitude"] = df3.latitude.iloc[0]
df1["longitude"] = df3.longitude.iloc[0]
df1["elevation"] = df3.elevation.iloc[0]
df1["reference"] = df3.reference.iloc[0]
df1["site"] = df3.name.iloc[0]

df2 = pd.read_csv(
    "Data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv", sep=";"
)
df2['date'] = "1930-" + df2.month.astype(str).apply(lambda x: x.zfill(2)) + "-15"
df2 = df2.set_index('date').drop(columns=['month'])
df_stack = ( df2
            .stack(dropna=False)
            .to_frame(name='temperatureObserved')
            .reset_index()
            .rename(columns={'level_1':'depthOfTemperatureObservation'})
            )
df_stack['depthOfTemperatureObservation'] = df_stack['depthOfTemperatureObservation'].str.replace('m','').astype(float)

df_stack["latitude"] = df3.latitude.iloc[1]
df_stack["longitude"] = df3.longitude.iloc[1]
df_stack["elevation"] = df3.elevation.iloc[1]
df_stack["site"] = df3.name.iloc[1]
df_stack["reference"] = df3.reference.iloc[1]

df_wegener = pd.concat((df1, df_stack), ignore_index=True,)
df_wegener["reference_short"] = "Wegener (1940), Sorge (1940)"
df_wegener["note"] = ""

df_wegener["method"] = "electric resistance thermometer"
df_wegener["durationOpen"] = "NA"
df_wegener["durationMeasured"] = "NA"
df_wegener["error"] = 0.2

df_all = pd.concat((df_all,  df_wegener[needed_cols]), ignore_index=True)

# %% Japanese stations
del df1, df2, df3, df_stack, df_wegener
df = pd.read_excel("Data/Japan/Sigma.xlsx")
df["note"] = ""
df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Ambach
del df
meta = pd.read_csv(
    "Data/Ambach1979b/metadata.txt", sep="\t", header=None,
    names=["site", "file", "date", "latitude", "longitude", "elevation"],
).set_index('file')
meta.date = pd.to_datetime(meta.date)
for file in meta.index:
    df = pd.read_csv(
        "Data/Ambach1979b/" + file + ".txt", header=None,
        names=["temperatureObserved", "depthOfTemperatureObservation"]
    )

    plt.figure()
    df.set_index('depthOfTemperatureObservation').plot(ax=plt.gca(), marker = 'o')
    plt.title(file)
    for v in meta.columns: df[v] = meta.loc[file, v]

    df["reference"] = "Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979"
    df["reference_short"] = "Ambach (1979)"
    df["note"] = "digitized, interpolated at 10 m"
    df["method"] = "NA"
    df["durationOpen"] = "NA"
    df["durationMeasured"] = "NA"
    df["error"] = "NA"

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Kjær 2020 TCD data
del df, file, meta, v
df = pd.read_excel("Data/Kjær/tc-2020-337.xlsx")
df["reference"] = "Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337 , 2021."
df["reference_short"] = "Kjær et al. (2015)"
df["note"] = ""
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.1
df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Covi
del df
sites = ["DYE-2", "EKT", "SiteJ"]
filenames = [
    "AWS_Dye2_20170513_20190904_CORR_daily",
    "AWS_EKT_20170507_20190904_CORR_daily",
    "AWS_SiteJ_20170429_20190904_CORR_daily",
]
for site, filename in zip(sites, filenames):
    df = pd.read_csv("Data/Covi/" + filename + ".dat", skiprows=1)
    depth = np.flip(
        16 - np.concatenate((np.arange(16, 4.5, -0.5), np.arange(4, -1, -1)))
    )
    depth_ini = 1
    depth = depth + depth_ini
    df["date"] = df.TIMESTAMP.astype(str)
    temp_label = ["Tfirn" + str(i) + "(C)" for i in range(1, 29)]

    df.date = pd.to_datetime(df.date)
    # print(site, df.date.diff().unique())
    df = df.drop(columns=['TIMESTAMP'])

    if site in ["DYE-2", "EKT"]:
        # loading surface height from FirnCover station
        filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
        sites = ["Summit",  "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
        statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
        statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

        df["surface_height"] = (
            -sonic_df.loc[site]
            .resample("D")
            .mean()
            .loc[df.date]
            .sonic_m.values
        )
    else:
        df["surface_height"] = df["SR50_corr(m)"]

    df["surface_height"] = df["surface_height"] - df["surface_height"].iloc[0]
    df["surface_height"] = df["surface_height"].interpolate().values

    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        df[depth_label[i]] = depth[i] + df["surface_height"].values
    df = df.set_index("date")
    
    df_stack = ( df[temp_label]
                .stack(dropna=False)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

    df_stack["site"] = site
    if site == "SiteJ":
        df_stack["latitude"] = 66.864952
        df_stack["longitude"] = -46.265141
        df_stack["elevation"] = 2060
    else:
        df_stack["latitude"] = statmeta_df.loc[site, "latitude"]
        df_stack["longitude"] = statmeta_df.loc[site, "longitude"]
        df_stack["elevation"] = statmeta_df.loc[site, "elevation"]
    df_stack["note"] = ""

    df_stack["reference"] = "Covi, F., Hock, R., and Reijmer, C.: Challenges in modeling the energy balance and melt in the percolation zone of the Greenland ice sheet. Journal of Glaciology, 69(273), 164-178. doi:10.1017/jog.2022.54, 2023. and Covi, F., Hock, R., Rennermalm, A., Leidman S., Miege, C., Kingslake, J., Xiao, J., MacFerrin, M., Tedesco, M.: Meteorological and firn temperature data from three weather stations in the percolation zone of southwest Greenland, 2017 - 2019. Arctic Data Center. doi:10.18739/A2BN9X444, 2022."
    df_stack["reference_short"] = "Covi et al. (2022, 2023)"
    df_stack["note"] = ""
    df_stack["method"] = "thermistor"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 24
    df_stack["error"] = 0.1
    
    if plot:
        plot_string_dataframe(df_stack, site)
        
    df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

# %% Stauffer and Oeschger 1979
del depth, depth_ini, depth_label, df, df_stack, filenames, filename, filepath
del i, metdata_df, rtd_dep, rtd_df, site, sites, sonic_df, statmeta_df, temp_label
df_s_o = pd.read_excel("Data/Stauffer and Oeschger 1979/Stauffer&Oeschger1979.xlsx")

df_s_o["reference"] = "Clausen HB and Stauffer B (1988) Analyses of Two Ice Cores Drilled at the Ice-Sheet Margin in West Greenland. Annals of Glaciology 10, 23–27 (doi:10.3189/S0260305500004109)"
df_s_o["reference_short"] = "Stauffer and Oeschger (1979)"
df_s_o["note"] = "site location estimated by M. Luethi"
df_s_o["method"] = "Fenwal Thermistor UUB 31-J1"
df_s_o["durationOpen"] = 0
df_s_o["durationMeasured"] = 0
df_s_o["error"] = 0.1
df_all = pd.concat((df_all, df_s_o[needed_cols]), ignore_index=True)

# %% Schwager EGIG
del df_s_o
df_schwager = pd.read_excel("Data/Schwager/schwager.xlsx")

df_schwager[
    "reference"
] = "Schwager, M. (2000): Eisbohrkernuntersuchungen zur räumlichen und zeitlichen Variabilität von Temperatur und Niederschlagsrate im Spätholozän in Nordgrönland - Ice core analysis on the spatial and temporal variability of temperature and precipitation during the late Holocene in North Greenland , Berichte zur Polarforschung (Reports on Polar Research), Bremerhaven, Alfred Wegener Institute for Polar and Marine Research, 362 , 136 p. . doi: 10.2312/BzP_0362_2000"
df_schwager["reference_short"] = "Schwager (2000)"
df_schwager["note"] = ""
df_schwager.date = pd.to_datetime([str(y) + "-07-01" for y in df_schwager.date])
df_schwager["method"] = "custom thermistors"
df_schwager["durationOpen"] = 0
df_schwager["durationMeasured"] = 12
df_schwager["error"] = 0.5
df_all = pd.concat((df_all,  df_schwager[needed_cols]), ignore_index=True)

# %% Giese & Hawley
del df_schwager
df = pd.read_excel("Data/Giese and Hawley/giese_hawley.xlsx")

df1 = df.iloc[:, :2]
df1 = df1.loc[~np.isnan(df1.time1.values), :]
df1["time"] = [
    datetime(int(d_y), 1, 1)
    + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
    for d_y in df1.time1.values
]
df1 = df1.set_index("time").resample("D").mean().interpolate(method="cubic")

df2 = df.iloc[:, 2:4]
df2 = df2.loc[~np.isnan(df2.time2.values), :]
df2["time"] = [
    datetime(int(d_y), 1, 1)
    + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
    for d_y in df2.time2.values
]
df2 = df2.set_index("time").resample("D").mean().interpolate(method="cubic")

df_giese = df.iloc[:, 4:]
df_giese = df_giese.loc[~np.isnan(df_giese.time3.values), :]
df_giese["time"] = [
    datetime(int(d_y), 1, 1)
    + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
    for d_y in df_giese.time3.values
]

df_giese = df_giese.set_index("time").resample("D").mean().interpolate(method="cubic")

df_giese["temp_8"] = df2.temp_8.values
df_giese["depth_7"] = 6.5 + df1.depth1.values - df1.depth1.min()
df_giese["depth_8"] = 9.5 + df1.depth1.values - df1.depth1.min()

df_stack = ( df_giese[['temp_7','temp_8']]
            .stack(dropna=False)
            .to_frame(name='temperatureObserved')
            .reset_index()
            .rename(columns={'time':'date'}))
df_stack['depthOfTemperatureObservation'] = df_giese[['depth_7','depth_8']].stack(dropna=False).values
df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]
site="Summit"
df_stack["site"] = site
df_stack["latitude"] = 72 + 35 / 60
df_stack["longitude"] = -38 - 30 / 60
df_stack["elevation"] = 3208

df_stack["reference"] = "Giese AL and Hawley RL (2015) Reconstructing thermal properties of firn at Summit, Greenland, from a temperature profile time series. Journal of Glaciology 61(227), 503–510 (doi:10.3189/2015JoG14J204)"
df_stack["reference_short"] = "Giese and Hawley (2015)"
df_stack["note"] = "digitized and interpolated at 10m"

df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] = 24 * 30
df_stack["error"] = 0.5

if plot:
    plot_string_dataframe(df_stack, site)
    
df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

# %% Historical summit
del df1, df2, df, df_giese, df_stack, site
df_summit = pd.DataFrame()
df_summit["temperatureObserved"] = -32.0
df_summit["depthOfTemperatureObservation"] = 10
df_summit["site"] = "Summit"
df_summit["latitude"] = 72 + 18 / 60
df_summit["longitude"] = -37 - 55 / 60
df_summit["elevation"] = 3260
df_summit["note"] = ""
df_summit["date"] = pd.to_datetime("1974-06-01")
df_summit["reference_short"] = "Hodge et al. (1990)"
df_summit["reference"] = "Hodge SM, Wright DL, Bradley JA, Jacobel RW, Skou N and Vaughn B (1990) Determination of the Surface and Bed Topography in Central Greenland. Journal of Glaciology 36(122), 17–30 (doi:10.3189/S0022143000005505) as reported in  Firestone J, Waddington ED and Cunningham J (1990) The Potential for Basal Melting Under Summit, Greenland. Journal of Glaciology 36(123), 163–168 (doi:10.3189/S0022143000009400)"

df_summit["method"] = "NA"
df_summit["durationOpen"] = "NA"
df_summit["durationMeasured"] = "NA"
df_summit["error"] = "NA"

df_all = pd.concat((df_all,  df_summit[needed_cols]), ignore_index=True)

# %% IMAU
del df_summit
df_meta = pd.read_csv("data/IMAU/meta.csv")
depth_label = ["depth_" + str(i) for i in range(1, 6)]
temp_label = ["temp_" + str(i) for i in range(1, 6)]
df_imau = pd.DataFrame()
for i, site in enumerate(["s5", "s6", "s9"]):
    df = pd.read_csv("data/IMAU/" + site + "_tsub.txt")
    df["date"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
        df["doy"] - 1, unit="d"
    )
    df = df.set_index("date").drop(columns=["year", "doy"])

    for dep, temp in zip(depth_label, temp_label):
        df.loc[df[dep] < 0.2, temp] = np.nan
    if site == "s5":
        surface_height = df.depth_5 - df.depth_5.values[0]
        surface_height.loc["2011-09-01":] = surface_height.loc["2011-08-31":] - 9.38
        surface_height = surface_height.values
        df.loc["2011-08-29":"2011-09-05", temp_label] = np.nan
    else:
        surface_height = []
    if site == "s6":
        for dep, temp in zip(depth_label, temp_label):
            df.loc[df[dep] < 1.5, temp] = np.nan
        min_diff_to_depth = 3
    else:
        min_diff_to_depth = 3
        
    df_stack = (df[temp_label]
                .stack(dropna=False)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(dropna=False).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]
    
    df_stack["site"] = site
    df_stack["note"] = ""
    df_stack["latitude"] = df_meta.loc[df_meta.site == site, "latitude"].values[0]
    df_stack["longitude"] = df_meta.loc[df_meta.site == site, "longitude"].values[0]
    df_stack["elevation"] = df_meta.loc[df_meta.site == site, "elevation"].values[0]

    df_stack["reference"] = " Paul C. J. P. Smeets, Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954"
    df_stack["reference_short"] = "Smeets et al. (2018)"
    df_stack["method"] = "thermistor"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 24 * 30
    df_stack["error"] = 0.2
    
    if plot:
        plot_string_dataframe(df_stack, site)
        
    df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

# %% Braithwaite
del df_stack, depth_label, temp_label, df_imau, df_meta, i, min_diff_to_depth, dep
del site, surface_height, temp, df
df = pd.read_excel("Data/Braithwaite/data.xlsx")
df = (
      df.set_index(['site', 'date', 'latitude', 'longitude', 'elevation'])
      .stack().to_frame(name='temperatureObserved').reset_index()
      .rename(columns={'level_5': 'depthOfTemperatureObservation'})
      )
df.site = df.site.astype(str)
df["reference"] = "Braithwaite, R. (1993). Firn temperature and meltwater refreezing in the lower accumulation area of the Greenland ice sheet, Pâkitsoq, West Greenland. Rapport Grønlands Geologiske Undersøgelse, 159, 109–114. https://doi.org/10.34194/rapggu.v159.8218"
df["reference_short"] = "Braithwaite (1993)"
df["note"] = "from table"
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Clement
df = pd.read_excel("Data/Clement/data.xlsx")
df["depthOfTemperatureObservation"] = df.depth
df["temperatureObserved"] = df.temperature

df["reference"] = "Clement, P. “Glaciological Activities in the Johan Dahl Land Area, South Greenland, As a Basis for Mapping Hydropower Potential”. Rapport Grønlands Geologiske Undersøgelse, vol. 120, Dec. 1984, pp. 113-21, doi:10.34194/rapggu.v120.7870."
df["reference_short"] = "Clement (1984)"
df["note"] = "digitized"
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Nobles
df = pd.read_excel("Data/Nobles Nuna Ramp/data.xlsx")
df["depthOfTemperatureObservation"] = 8
df["temperatureObserved"] = df["annual temp at 8m"]
df["date"] = pd.to_datetime("1954-07-01")
df["reference"] = "Nobles, L. H., Glaciological investigations, Nunatarssuaq ice ramp, Northwestern Greenland, Tech. Rep. 66, U.S. Army Snow, Ice and Permafrost Research Establishment, Corps of Engineers, 1960."
df["reference_short"] = "Nobles (1960)"
df["note"] = "digitized"
df["method"] = "iron-constantan thermocouples"
df["durationOpen"] = 0
df["durationMeasured"] = 24*360
df["error"] = 0.5

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)
# %% Schytt
df = pd.read_excel("Data/Schytt Tuto/data.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})

df["date"] = pd.to_datetime(df.date)
df["reference"] = "Schytt, V. (1955) Glaciological investigations in the Thule Ramp area, U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 28, 88 pp. https://hdl.handle.net/11681/5989"
df["reference_short"] = "Schytt (1955)"
df["note"] = "from table"

df["method"] = "copper-constantan thermocouples"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)


# %% Griffiths & Schytt
df = pd.read_excel("Data/Griffiths Tuto/data.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})
df["date"] = pd.to_datetime(df.date)
df["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
df["reference_short"] = "Griffiths (1960)"
df["note"] = "from table"
df["method"] = "copper-constantan thermocouples"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Griffiths & Meier
df = pd.read_excel("Data/Griffiths Tuto/data_crevasse3.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})
df.note = "measurement made close to an open crevasse"
df.temperatureObserved = pd.to_numeric(df.temperatureObserved, errors='coerce')
df["date"] = pd.to_datetime(df.date)
df["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
df["reference_short"] = "Griffiths (1960)"

df.loc[df.date <= '1955-12-31','reference'] = "Meier, M. F., Conel, J. E., Hoerni, J. A., Melbourne, W. G., & Pings, C. J. (1957). Preliminary Study of Crevasse Formation. Blue Ice Valley, Greenland, 1955. OCCIDENTAL COLL LOS ANGELES CALIF. https://hdl.handle.net/11681/6029"
df.loc[df.date <= '1955-12-31',"reference_short"] = "Meier et al. (1957)"

df["latitude"] = 76.43164
df["longitude"] = -67.54949
df["elevation"] = 800
df["method"] = "copper-constantan thermocouples"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5
# only keeping measurements more than 1 m into the crevasse wall
df = df.loc[df["distance from crevasse"] >= 1, :]

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

# %% Vanderveen
df = pd.read_excel("Data/Vanderveen et al. 2001/summary.xlsx")
df = df.loc[df.date1.notnull(), :]
df = df.loc[df.Temperature_celsius.notnull(), :]
df["site"] = [str(s) for s in df.site]
df["date"] = (
    pd.to_datetime(df.date1, errors='coerce') + (pd.to_datetime(df.date2, errors='coerce') - pd.to_datetime(df.date1, errors='coerce')) / 2
)
df["note"] = ""
df.loc[np.isnan(df["date"]), "note"] = "only year available"
df.loc[np.isnan(df["date"]), "date"] = pd.to_datetime(
    [str(y) + "-07-01" for y in df.loc[np.isnan(df["date"]), "date1"].values]
)

df["temperatureObserved"] = df["Temperature_celsius"]
df["depthOfTemperatureObservation"] = df["Depth_centimetre"] / 100

tmp, ind = np.unique(
    [str(x) + ' ' + str(y) for (x, y) in zip(df.site, df.date)], return_inverse=True
)

df["reference"] = "van der Veen, C. J., Mosley-Thompson, E., Jezek, K. C., Whillans, I. M., and Bolzan, J. F.: Accumulation rates in South and Central Greenland, Polar Geography, 25, 79–162, https://doi.org/10.1080/10889370109377709, 2001."
df["reference_short"] = "van der Veen et al. (2001)"
df["method"] = "thermistor"
df["durationOpen"] = 8 * 24
df["durationMeasured"] = 0
df["error"] = 0.1
    
df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

# %% Koch Wegener 1913
df = pd.read_excel("Data/Koch Wegener/data.xlsx")

df["depthOfTemperatureObservation"] = 10
df["date"] = pd.to_datetime(df.date,utc=True)
df["reference"] = "Koch, Johann P., and Alfred Wegener. Wissenschaftliche Ergebnisse Der Dänischen Expedition Nach Dronning Louises-Land Und Quer über Das Inlandeis Von Nordgrönland 1912 - 13 Unter Leitung Von Hauptmann J. P. Koch : 1 (1930). 1930."
df["reference_short"] = "Koch (1913)"
df["site"] = "Koch 1912-13 winter camp"
df["method"] = "electric resistance thermometer"
df["durationOpen"] = "NA"
df["durationMeasured"] = "NA"
df["error"] = 0.2
df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

# %% Thomsen shallow thermistor
df = pd.read_excel("Data/Thomsen/data-formatted.xlsx").rename(
    columns={'depth':"depthOfTemperatureObservation",
             "temperature": "temperatureObserved"})

df["note"] = "from unpublished pdf"
df["date"] = pd.to_datetime(df.date)
df["reference"] = "Thomsen, H. ., Olesen, O. ., Braithwaite, R. . and Bøggild, C. .: Ice drilling and mass balance at Pâkitsoq, Jakobshavn, central West Greenland, Rapp. Grønlands Geol. Undersøgelse, 152, 80–84, doi:10.34194/rapggu.v152.8160, 1991."
df["reference_short"] = "Thomsen et al. (1991)"
df["method"] = "thermistor"
df["durationOpen"] = "NA"
df["durationMeasured"] = "NA"
df["error"] = 0.2

df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

# %% Checking values
# del df
df_all['temperatureObserved'] = df_all.temperatureObserved.astype(float)
df_all['depthOfTemperatureObservation'] = df_all.depthOfTemperatureObservation.astype(float)
df_all = df_all.loc[~df_all.temperatureObserved.isnull(), :]
df_all = df_all.loc[~df_all.depthOfTemperatureObservation.isnull(), :]
df_all = df_all.loc[df_all.depthOfTemperatureObservation>0, :]
df_all = df_all.loc[df_all.temperatureObserved<1, :]

df_ambiguous_date = df_all.loc[pd.to_datetime(df_all.date,utc=True, errors="coerce").isnull(), :]
df_bad_long = df_all.loc[df_all.longitude.astype(float) > 0, :]
# df_bad_long.to_csv('bad_lat.csv')
df_no_coord = df_all.loc[np.logical_or(df_all.latitude.isnull(), df_all.latitude.isnull()), :]
df_invalid_depth = df_all.loc[
    pd.to_numeric(df_all.depthOfTemperatureObservation, errors="coerce").isnull(), :
]
df_no_elev = df_all.loc[df_all.elevation.isnull(), :]
if len(df_no_elev)>0: df_no_elev.to_csv('missing_elev.csv')

tmp = df_all.loc[np.isnan(df_all.temperatureObserved.astype(float).values)]
df_all = df_all.loc[~np.isnan(df_all.temperatureObserved.astype(float).values)]

# some renaming
df_all.loc[df_all.method=='Thermistor', 'method'] = 'thermistors'
df_all.loc[df_all.method=='Thermistors', 'method'] = 'thermistors'
df_all.loc[df_all.method=='thermistor', 'method'] = 'thermistors'
df_all.loc[df_all.method=='thermistor string', 'method'] = 'thermistors'
df_all.loc[df_all.method=='custom thermistors', 'method'] = 'thermistors'
df_all.loc[df_all.method=='NA', 'method'] = 'not available'
df_all.loc[df_all.method.isnull(), 'method'] = 'not available'
df_all.loc[df_all.method=='not_reported', 'method'] = 'not available'
df_all.loc[df_all.method=='digital Thermarray system from RST©', 'method'] = 'RST ThermArray'
df_all.loc[df_all.method=='digital thermarray system from RST©', 'method'] = 'RST ThermArray'

# looking for redundant references
df_all.loc[df_all.reference.str.startswith('Fausto'), 'reference_short'] = 'PROMICE/GC-Net, How et al. (2023)'
df_all.loc[df_all.reference.str.startswith('Fausto'), 'reference_full'] =  'Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022.'

# %% netcdf format
import xarray as xr
# df_all[['elevation','open_time', 'duration']] = \
#     df_all[['elevation','open_time', 'duration']].replace('','-9999').astype(int)
df_all = df_all.rename(columns={'date':'timestamp',
                                'temperatureObserved':'temperature',
                                'depthOfTemperatureObservation': 'depth',
                                'site':'name',
                                'durationMeasured':'duration',
                                'durationOpen':'open_time'
                                })
df_all['timestamp'] = pd.to_datetime(df_all.timestamp, utc = True).dt.tz_localize(None)

print('start netcdf repackaging')
df_all.index.name='measurement_id'

df_names = (df_all[['name']]
                .drop_duplicates()
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index':'name_key'})
                )
df_all['name_key'] = df_names.set_index('name').loc[df_all.name, 'name_key'].values
ds_meta_name = df_names.set_index('name_key').to_xarray()

print('    ds_meta_name created')
df_methods = (df_all[['method']]
                .drop_duplicates()
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index':'method_key'})
                )
df_all['method_key'] = df_methods.set_index('method').loc[df_all.method, 'method_key'].values
ds_meta_method = df_methods.set_index('method_key').to_xarray()

print('    ds_meta_name created')

df_references = (df_all[['reference', 'reference_short']]
                .drop_duplicates(subset='reference')
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index':'reference_key'})
                )

# tmp = df_references.set_index('reference').loc[df_all.reference.values, 'reference_key'].values

df_all['reference_key'] = df_references.set_index('reference').loc[df_all.reference.values,
                                                                   'reference_key'].values
ds_meta_reference = df_references.set_index('reference_key').to_xarray()
print('    ds_meta_name created')

ds_data = df_all.drop(columns=['name', 'method','reference','reference_short']).to_xarray()

ds_meta = xr.merge((ds_meta_name,
                    ds_meta_method,
                    ds_meta_reference))

print('    ds_data and ds_meta created')

ds_data['elevation'] = ds_data.elevation.astype(int)
ds_data['error'] = ('measurement_id', pd.to_numeric(ds_data['error'], errors='coerce')  )
ds_data['duration'] = ('measurement_id', pd.to_numeric(ds_data['duration'], errors='coerce')  )
ds_data['open_time'] = ('measurement_id', pd.to_numeric(ds_data['open_time'], errors='coerce')  )
ds_data['note'] = ds_data['note'].astype(str)      
ds_meta['name'] = ds_meta['name'].astype(str)      
ds_meta['method'] = ds_meta['method'].astype(str)      
ds_meta['reference'] = ds_meta['reference'].astype(str)      
ds_meta['reference_short'] = ds_meta['reference_short'].astype(str)      

ds_data.timestamp.encoding['units'] = 'days since 1900-01-01'
ds_data.attrs['contact'] = 'Baptiste Vandecrux'
ds_data.attrs['email'] = 'bav@geus.dk'
ds_data.attrs['production date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
   
float_encoding = {"dtype": "float32", "zlib": True,"complevel": 9}
int_encoding = {"dtype": "int32", "_FillValue":-9999, "zlib": True,"complevel": 9}
print('    variable types set')
print('    writing to file')
filename = 'Vandecrux_2023_temperature_compilation.nc'

ds_data[['name_key', 'reference_key', 'method_key', 'timestamp',
          'latitude', 'longitude', 'elevation',  
          'temperature',  'depth', 'duration','open_time',
          'error']].to_netcdf(filename, 
                              group='DATA',
                              encoding={
                                 "temperature": float_encoding |{'least_significant_digit':2},
                                 "depth": float_encoding |{'least_significant_digit':2},
                                 "duration": int_encoding,
                                 "open_time": int_encoding,
                                 "error": float_encoding|{'least_significant_digit':2},
                                 "longitude": float_encoding|{'least_significant_digit':6},
                                 "latitude": float_encoding|{'least_significant_digit':6},
                                 "elevation": int_encoding,
                                 "name_key": int_encoding,
                                 "reference_key": int_encoding,
                                 "method_key": int_encoding,
                                 })
ds_meta.to_netcdf(filename, group='METADATA', mode='a',
                   encoding={
                           "name": {"zlib": True,"complevel": 9},
                           "reference": {"zlib": True,"complevel": 9},
                           "reference_short": {"zlib": True,"complevel": 9},
                           "method": {"zlib": True,"complevel": 9},
                       }
                  )
    
print('    writing completed')
