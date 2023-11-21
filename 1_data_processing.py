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
import firn_temp_lib as ftl

def interp_pd_on_index(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

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
    s.plot(ax=plt.gca(), marker="o", linestyle="none")
    s_save.plot(ax=plt.gca(), marker="o", linestyle="none")
    # plt.xlim(0, 60)
    return s

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
df_MW[
    "method"
] = "thermohms and a Wheats tone bridge, standard mercury or alcohol thermometers"

df_all = pd.concat((df_all, 
    df_MW[needed_cols],
    ), ignore_index=True,
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
msk = (df_benson.site == "0-35") | (df_benson.site == "French Camp VI")
df_benson = df_benson.loc[msk, :]

df_benson["durationOpen"] = "measured in pit wall or borehole bottom after excavation"
df_benson["durationMeasured"] = "few minutes"
df_benson["error"] = "NA"
df_benson["method"] = "Weston bimetallic thermometers"


df_all = pd.concat((df_all, 
    df_benson[needed_cols],
    ), ignore_index=True,
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
df_Pol[
    "durationOpen"
] = "string lowered in borehole and left 30min for equilibrating with surrounding firn prior measurement start"
df_Pol["durationMeasured"] = "overnight ~10 hours"
df_Pol["error"] = 0.1
df_Pol["method"] = "thermistor string"

df_all = pd.concat((df_all, 
    df_Pol[needed_cols],
    ), ignore_index=True,
)

# %% Ken's dataset
print("Loading Kens dataset")
df_Ken = pd.read_excel(
    "Data/greenland_ice_borehole_temperature_profiles-main/data_filtered.xlsx"
)

df_Ken['reference_short'] = df_Ken.reference_short+' as in Mankoff et al. (2022)'
df_all = pd.concat((df_all,  df_Ken[needed_cols]), ignore_index=True)
# %% Sumup
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
df_all = pd.concat((df_all, 
    df_sumup[needed_cols],
    ), ignore_index=True,
)

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

df_mcgrath["method"] = "digital Thermarray system from RST©"
df_mcgrath["durationOpen"] = 0
df_mcgrath["durationMeasured"] = 30
df_mcgrath["error"] = 0.07
df_mcgrath
df_all = pd.concat((df_all, 
    df_mcgrath[needed_cols],
    ), ignore_index=True,
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

# %% Steffen 2001 table (that could not be found in the GC-Net AWS data)
print('Loading Steffen 2001 table')
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
df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)


# %% Camp Century historical
# % In 1977, firn density was measured to a depth of 100.0 m, and firn temperature was measured at 10 m depth (Clausen and Hammer, 1988). In 1986, firn density was measured to a depth of 12.0 m, and the 10 m firn temperature was again measured (Gundestrup et al., 1987).
print('Loading CC historical')
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
print('Loading Davies dataset')
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

df_all = pd.concat((df_all, 
    df_davies[needed_cols],
    ), ignore_index=True,
)

# %% Fischer de Quervain EGIG
print('Loading Fischer and deQuervain EGIG data')
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

df_all = pd.concat((df_all, 
    df_dequervain[needed_cols],
    ), ignore_index=True,
)

# %% Japanese stations
# df = pd.read_excel("Data/Japan/Qanaaq.xlsx")
# df.date = pd.to_datetime(df.date)
# df["note"] = "interpolated to 10 m"
# df["method"] = "thermistor"
# df["durationOpen"] = "NA"
# df["durationMeasured"] = "NA"
# df["error"] = 0.1
# df_all = pd.concat((df_all, 
#     df[
#         ["date", "site", "latitude", "longitude", "elevation", "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note", "error", "durationOpen", "durationMeasured", "method"]
#     ],
#     ), ignore_index=True,
# )
print('Loading Japanese measurements')
df = pd.read_excel("Data/Japan/Sigma.xlsx")
df["note"] = ""
df_all = pd.concat((df_all, 
    df[needed_cols],
    ), ignore_index=True,
)


# %% Kjær 2020 TCD data
print('Loading Kjær measurements')

df = pd.read_excel("Data/Kjær/tc-2020-337.xlsx")
df[
    "reference"
] = "Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337 , 2021."
df["reference_short"] = "Kjær et al. (2015)"
df["note"] = ""
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.1
df_all = pd.concat((df_all, 
    df[needed_cols],
    ), ignore_index=True,
)

# %% Stauffer and Oeschger 1979
print('Loading Stauffer and Oeschger measurements')

df_s_o = pd.read_excel("Data/Stauffer and Oeschger 1979/Stauffer&Oeschger1979.xlsx")

df_s_o[
    "reference"
] = "Stauffer B. and H. Oeschger. 1979. Temperaturprofile in bohrloechern am rande des Groenlaendischen Inlandeises. Hydrologie und Glaziologie an der ETH Zurich. Mitteilung Nr. 41."
df_s_o["reference_short"] = "Stauffer and Oeschger (1979)"
df_s_o["note"] = "site location estimated by M. Luethi"
df_s_o["method"] = "Fenwal Thermistor UUB 31-J1"
df_s_o["durationOpen"] = 0
df_s_o["durationMeasured"] = 0
df_s_o["error"] = 0.1
df_all = pd.concat((df_all, 
    df_s_o[needed_cols],
    ), ignore_index=True,
)

# %% Schwager EGIG
print('Loading Schwager EGIG')

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
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

# %% Historical summit
print('Loading Historical summit')

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
df_summit[
    "reference"
] = "Hodge SM, Wright DL, Bradley JA, Jacobel RW, Skou N and Vaughn B (1990) Determination of the Surface and Bed Topography in Central Greenland. Journal of Glaciology 36(122), 17–30 (doi:10.3189/S0022143000005505) as reported in  Firestone J, Waddington ED and Cunningham J (1990) The Potential for Basal Melting Under Summit, Greenland. Journal of Glaciology 36(123), 163–168 (doi:10.3189/S0022143000009400)"

df_summit["method"] = "NA"
df_summit["durationOpen"] = "NA"
df_summit["durationMeasured"] = "NA"
df_summit["error"] = "NA"

df_all = pd.concat((df_all, 
    df_summit[needed_cols],
    ), ignore_index=True,
)

# %% Koch Wegener 1913
print('Loading Koch Wegener 1913')

df = pd.read_excel("Data/Koch Wegener/data.xlsx")

df["depthOfTemperatureObservation"] = 10
df["date"] = pd.to_datetime(df.date,utc=True)
df[
    "reference"
] = "Koch, Johann P., and Alfred Wegener. Wissenschaftliche Ergebnisse Der Dänischen Expedition Nach Dronning Louises-Land Und Quer über Das Inlandeis Von Nordgrönland 1912 - 13 Unter Leitung Von Hauptmann J. P. Koch : 1 (1930). 1930."
df["reference_short"] = "Koch (1913)"
df["site"] = "Koch 1912-13 winter camp"
df["method"] = "electric resistance thermometer"
df["durationOpen"] = "NA"
df["durationMeasured"] = "NA"
df["error"] = 0.2
df_all = pd.concat((df_all, 
    df[needed_cols],
    ), ignore_index=True,
)


# %% Thomsen shallow thermistor
print('Loading Thomsen shallow thermistor')

df = pd.read_excel("Data/Thomsen/data-formatted.xlsx")
# df = df.set_index('d').interpolate(method= 'index')

for date in df.date.unique():
    tmp = df.loc[df.date == date]
    if tmp.temperature.isnull().any():
        df.loc[df.date == date, "temperature"] = interp_pandas(
            tmp.set_index("depth").temperature
        ).values


df["depthOfTemperatureObservation"] = df["depth"]
df["temperatureObserved"] = df["temperature"]
df["note"] = "from unpublished pdf"
df["date"] = pd.to_datetime(df.date)
df[
    "reference"
] = "Thomsen, H. ., Olesen, O. ., Braithwaite, R. . and Bøggild, C. .: Ice drilling and mass balance at Pâkitsoq, Jakobshavn, central West Greenland, Rapp. Grønlands Geol. Undersøgelse, 152, 80–84, doi:10.34194/rapggu.v152.8160, 1991."
df["reference_short"] = "Thomsen et al. (1991)"
df["method"] = "thermistor"
df["durationOpen"] = "NA"
df["durationMeasured"] = "NA"
df["error"] = 0.2
df_all = pd.concat((df_all, 
    df[needed_cols],
    ), ignore_index=True,
)
                             

# %% Historical swc
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

df_all = pd.concat((df_all,  df_swc[needed_cols]), ignore_index=True)

# %% Clement
df = pd.read_excel("Data/Clement/data.xlsx")
df.date= pd.to_datetime(df.date)
df["depthOfTemperatureObservation"] = df.depth
df["temperatureObserved"] = df.temperature

df["reference"] = "Clement, P. “Glaciological Activities in the Johan Dahl Land Area, South Greenland, As a Basis for Mapping Hydropower Potential”. Rapport Grønlands Geologiske Undersøgelse, vol. 120, Dec. 1984, pp. 113-21, doi:10.34194/rapggu.v120.7870."
df["reference_short"] = "Clement (1984)"
df["note"] = "digitized"
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5

df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)


# %% PROMICE
print("Loading PROMICE")
df_promice = pd.read_csv("Data/PROMICE/PROMICE_10m_firn_temperature.csv", sep=";")
df_promice = df_promice.loc[df_promice.temperatureObserved.notnull()]
df_promice = df_promice.loc[df_promice.site != "QAS_A", :]
df_promice.loc[(df_promice.site == "CEN") & (df_promice.temperatureObserved > -18),
               "temperatureObserved"] = np.nan

df_promice["method"] = "RS Components thermistors 151-243"
df_promice["durationOpen"] = 0
df_promice["durationMeasured"] = 30 * 24
df_promice["error"] = 0.2
df_promice["note"] = ""

df_all = pd.concat((df_all, df_promice[needed_cols]), ignore_index=True)

# %% GC-Net
print("Loading GC-Net")
df_GCN = pd.read_csv("Data/GC-Net/10m_firn_temperature.csv")
df_GCN = df_GCN.loc[df_GCN.temperatureObserved.notnull()]
df_GCN["method"] = "thermocouple"
df_GCN["durationOpen"] = 0
df_GCN["durationMeasured"] = 30 * 24
df_GCN["error"] = 0.5
df_all = pd.concat((df_all, df_GCN[needed_cols]), ignore_index=True)


# %% Miege aquifer
print("Loading firn aquifer data")
metadata = np.array([ ["FA-13", 66.181, 39.0435, 1563],
        ["FA-15-1", 66.3622, 39.3119, 1664],
        ["FA-15-2", 66.3548, 39.1788, 1543]])

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
    dates = pd.to_datetime((temp.Year * 1000000 + temp.Month * 10000
            + temp.Day * 100 + temp["Hours (UTC)"]).apply(str),
        format="%Y%m%d%H")
    temp = temp.iloc[:, 4:]

    ellapsed_hours = (dates - dates[0]).dt.total_seconds()/60/60
    accum_depth = ellapsed_hours.values * thickness_accum / 365 / 24
    depth_cor = pd.DataFrame()
    depth_cor = depth.values.reshape((1, -1)).repeat(
        len(dates), axis=0
    ) + accum_depth.reshape((-1, 1)).repeat(len(depth.values), axis=1)

    df_10 = ftl.interpolate_temperature(
        dates, depth_cor, temp.values, title=site + " Miller et al. (2020)"
    )
    for i in range(depth_cor.shape[1]):
        df_10['depth_'+str(i+1)] = depth_cor[:,i]
        df_10['temp_'+str(i+1)] = temp.values[:,i]
    df_10.loc[np.greater(df_10["temperatureObserved"], 0), "temperatureObserved"] = 0
    df_10["site"] = site
    df_10["latitude"] = float(metadata[k, 1])
    df_10["longitude"] = -float(metadata[k, 2])
    df_10["elevation"] = float(metadata[k, 3])
    df_miege = pd.concat((df_miege, df_10), ignore_index=True)
    
df_miege["depthOfTemperatureObservation"] = 10
df_miege["reference"] = "Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W"
df_miege["reference_short"] = "Miller et al. (2020)"
df_miege["note"] = ""
df_miege["method"] = "digital thermarray system from RST©"
df_miege["durationOpen"] = 0
df_miege["durationMeasured"] = 30
df_miege["error"] = 0.07

temp_var = [v for v in df_miege.columns if 'temp_' in v]
depth_var = [v for v in df_miege.columns if 'depth_' in v]

                           
df_all = pd.concat((df_all,  df_miege[needed_cols]), ignore_index=True)

# %% Harper ice temperature
print("Loading Harper ice temperature")
df_harper = pd.read_csv("Data/Harper ice temperature/harper_iceTemperature_2015-2016.csv")

df_1 = df_harper[[v for v in df_harper.columns if not v=='temperature_2016_celsius'] ].rename(columns={'temperature_2015_celsius':'temperatureObserved'})

df_1['date'] = pd.to_datetime("2015-01-01")
df_2 = df_harper[[v for v in df_harper.columns if not v=='temperature_2015_celsius']].rename(columns={'temperature_2016_celsius':'temperatureObserved'})                                                                                                                                      
df_2['date'] = pd.to_datetime("2016-01-01")

df_harper = pd.concat((df_1,df_2),ignore_index=True).dropna(subset=['temperatureObserved'])

df_harper["note"] = ""
df_harper["depth"] = df_harper.depth_m - df_harper.height_m

df_harper = df_harper.set_index(["depth"]).sort_index()
df_harper = df_harper.reset_index()

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
df_harper = df_harper.rename(columns={"depth": "depthOfTemperatureObservation"})
df_harper["method"] = "TMP102 digital temperature sensor"
df_harper["durationOpen"] = 0
df_harper["durationMeasured"] = 30 * 24
df_harper["error"] = 0.1

# for b in df_harper.borehole.unique():
#     plt.figure()
#     plt.title(b)
#     for d in df_harper.loc[df_harper.borehole == b,'date'].unique():
#         df_harper.loc[df_harper.borehole == b].loc[df_harper.date==d].set_index('depth').temperatureObserved.plot(ax=plt.gca())

df_all = pd.concat((df_all, 
    df_harper[needed_cols],
    ), ignore_index=True,
)

# %%  FirnCover
print("Loading FirnCover")

filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

rtd_df = rtd_df.reset_index()
rtd_df = rtd_df.set_index(["sitename", "date"])
df_firncover = pd.DataFrame()
depth_var= ["depth_" + str(i) for i in range(24)]
temp_var = ["rtd" + str(i) for i in range(24)]
for site in sites:
    df_d = rtd_df.xs(site, level="sitename").reset_index()
    # df_d.to_csv('FirnCover_'+site+'.csv')
    df_d['temperatureObserved'] = ftl.interpolate_temperature(
        df_d["date"],
        df_d[depth_var].values,
        df_d[temp_var].values,
        title=site + " FirnCover",
    ).set_index('date')
    df_d["site"] = site
    if site == "Crawford":
        df_d["site"] = "CP1"
    df_d["latitude"] = statmeta_df.loc[site, "latitude"]
    df_d["longitude"] = statmeta_df.loc[site, "longitude"]
    df_d["elevation"] = statmeta_df.loc[site, "elevation"]

    df_firncover = pd.concat((df_firncover, df_d))
df_firncover[
    "reference"
] = "MacFerrin, M. J., Stevens, C. M., Vandecrux, B., Waddington, E. D., and Abdalati, W. (2022) The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) dataset, 2013–2019, Earth Syst. Sci. Data, 14, 955–971, https://doi.org/10.5194/essd-14-955-2022,"
df_firncover["reference_short"] = "MacFerrin et al. (2021, 2022)"
df_firncover["note"] = ""
df_firncover["depthOfTemperatureObservation"] = 10

# Correction of FirnCover bias
p = np.poly1d([1.03093649, -0.49950273])
df_firncover["temperatureObserved"] = p(df_firncover["temperatureObserved"].values)

df_firncover["method"] = "Resistance Temperature Detectors + correction"
df_firncover["durationOpen"] = 0
df_firncover["durationMeasured"] = 30 * 24
df_firncover["error"] = 0.5

df_all = pd.concat((df_all, 
    df_firncover[needed_cols],
    ), ignore_index=True,
)

# %% SPLAZ KAN_U
print("Loading SPLAZ at KAN-U")
site = "KAN_U"
num_therm = [32, 12, 12]
df_splaz = pd.DataFrame()

depth_var = ["depth_" + str(i) for i in range(32)]
temp_var = ["rtd" + str(i) for i in range(32)]
for k, note in enumerate(["SPLAZ_main", "SPLAZ_2", "SPLAZ_3"]):

    ds = xr.open_dataset("Data/SPLAZ/T_firn_KANU_" + note + ".nc")
    df = ds.to_dataframe()    
    df.reset_index(inplace=True)
    df2 = pd.DataFrame()
    df2["date"] = df.loc[df["level"] == 1, "time"]
    for i in range(1, num_therm[k] + 1):
        df2["rtd" + str(i - 1)] = df.loc[df["level"] == i, "Firn temperature"].values
        df2["depth_" + str(i - 1)] = df.loc[df["level"] == i, "depth"].values
    for i in range(num_therm[k] + 1, 32 + 1):
        df2["rtd" + str(i - 1)] = np.nan
        df2["depth_" + str(i - 1)] = np.nan
    df2[df2 == -999] = np.nan
    df2 = df2.set_index(["date"]).resample("H").mean()

    df2['temperatureObserved'] = ftl.interpolate_temperature(
        df2.index, df2[depth_var].values,
        df2[temp_var].values,
        min_diff_to_depth=1.5, kind="linear",  title="KAN_U " + note,
    ).set_index('date')

    df2["note"] = note
    df2["reference_short"] = note+" (Charalampidis et al., 2016; Charalampidis et al., 2022) "
    df2=df2.reset_index()
    df_splaz = pd.concat((df_splaz, df2))
df_splaz["latitude"] = 67.000252
df_splaz["longitude"] = -47.022999
df_splaz["elevation"] = 1840
df_splaz[
    "reference"
] = "Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10."

df_splaz["site"] = site
df_splaz["depthOfTemperatureObservation"] = 10

df_splaz["method"] = "RS 100 kΩ negative-temperature coefficient thermistors"
df_splaz["durationOpen"] = 0
df_splaz["durationMeasured"] = 30 * 24
df_splaz["error"] = 0.2

df_splaz=df_splaz.dropna(subset=temp_var, how='all')

df_all = pd.concat((df_all,  df_splaz[needed_cols]), ignore_index=True)
  

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
    df_site["time"] =[ pd.to_datetime('2007-01-01') + pd.Timedelta(days=d) for d in df_site.iloc[:, 0]]
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
    df_site["surface_height"] = df_hs.iloc[df_hs.index.get_indexer(df_site.index, method="nearest")].values

    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        df_site[depth_label[i]] = (
            depth[i] + df_site["surface_height"].values - df_site["surface_height"].iloc[0]
        )
        if site != "H5":
            df_site[temp_label[i]] = (
                df_site[temp_label[i]].rolling(24 * 3, center=True).mean().values
                )
    df_site.index = df_site.index.rename('date')
    df_site['temperatureObserved'] = ftl.interpolate_temperature(
        df_site.index,
        df_site[depth_label].values,
        df_site[temp_label].values,
        title=site + " Humphrey et al. (2012)",
    ).set_index('date')

    df_site["site"] = site
    df_site["latitude"] = df.loc[df.site == site, "latitude"].values[0]
    df_site["longitude"] = df.loc[df.site == site, "longitude"].values[0]
    df_site["elevation"] = df.loc[df.site == site, "elevation"].values[0]
    df_site = df_site.reset_index()
    df_humphrey = pd.concat((df_humphrey, df_site))
df_humphrey = df_humphrey.reset_index(drop=True)
df_humphrey = df_humphrey.loc[df_humphrey.temperatureObserved.notnull()]
df_humphrey["depthOfTemperatureObservation"] = 10
df_humphrey["reference"] = "Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/"
df_humphrey["reference_short"] = "Humphrey et al. (2012)"
df_humphrey["note"] = "no surface height measurements, using interpolating surface height using CP1 and SwissCamp stations"

df_humphrey["method"] = "sealed 50K ohm thermistors"
df_humphrey["durationOpen"] = 0
df_humphrey["durationMeasured"] = 30 * 24
df_humphrey["error"] = 0.5

df_all = pd.concat((df_all, df_humphrey[needed_cols]), ignore_index=True)

# %% loading Hills
print("Loading Hills")
df_meta = pd.read_csv("Data/Hills/metadata.txt", sep=" ")
df_meta.date_start = pd.to_datetime(df_meta.date_start, format="%m/%d/%y")
df_meta.date_end = pd.to_datetime(df_meta.date_end, format="%m/%d/%y")

df_meteo = pd.read_csv("Data/Hills/Hills_33km_meteorological.txt", sep="\t")
df_meteo["date"] = pd.to_datetime("2014-07-18") + pd.to_timedelta((df_meteo.Time.values - 197) *24*60, unit='minute').round('T')
df_meteo = df_meteo.set_index("date").resample("D").mean()
df_meteo["surface_height"] = df_meteo.DistanceToTarget.iloc[0] - df_meteo.DistanceToTarget

df_hills = pd.DataFrame()
for site in df_meta.site[:-1]:
    print(site)

    df = pd.read_csv("Data/Hills/Hills_" + site + "_IceTemp.txt", sep="\t")
    df["date"] = \
        pd.to_datetime(df_meta.loc[df_meta.site == site, "date_start"].values[0]) \
            + pd.to_timedelta((df.Time.values - df.Time.values[0]) * 24 * 60 * 60,
                              unit="seconds").round('T')
    df = df.set_index("date").resample('H').mean()

    df["surface_height"] = np.nan
    df.loc[[i in df_meteo.index for i in df.index], "surface_height"] = df_meteo.loc[
        [i in df.index for i in df_meteo.index], "surface_height"
    ]
    df["surface_height"] = df.surface_height.interpolate(
        method="linear", limit_direction="both"
    )
    if all(np.isnan(df.surface_height)):
        df["surface_height"] = 0
    plt.figure()
    df.surface_height.plot()
    plt.title(site)

    depth = df.columns[1:-1].str.replace("Depth_", "").values.astype(float)
    print(len(depth))
    temp_label = ["temp_" + str(len(depth) - i) for i in range(len(depth))]
    depth_label = ["depth_" + str(len(depth) - i) for i in range(len(depth))]

    for i in range(len(temp_label)):
        df = df.rename(columns={df.columns[i + 1]: temp_label[i]})
        df.iloc[:14, i + 1] = np.nan
        if site in ["T-14", "T-11b"]:
            df.iloc[:30, i + 1] = np.nan

        df[depth_label[i]] = depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]

    # df = df.resample("D").mean()

    df['temperatureObserved'] = ftl.interpolate_temperature(
        df.index, df[depth_label].values,
        df[temp_label].values, title=site,
    ).set_index('date')

    df["latitude"] = df_meta.latitude[df_meta.site == site].iloc[0]
    df["longitude"] = df_meta.longitude[df_meta.site == site].iloc[0]
    df["elevation"] = df_meta.elevation[df_meta.site == site].iloc[0]
    df["depthOfTemperatureObservation"] = 10
    df["site"] = site
    
    df_hills = pd.concat((df_hills, df.reset_index()), ignore_index=True)

df_hills["note"] = "monthly mean, interpolated at 10 m"
df_hills[
    "reference"
] = "Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418"

df_hills["reference_short"] = "Hills et al. (2018)"
df_hills[
    "method"
] = "digital temperature sensor model DS18B20 from Maxim Integrated Products, Inc."
df_hills["durationOpen"] = 0
df_hills["durationMeasured"] = 30 * 24
df_hills["error"] = 0.0625

df_all = pd.concat((df_all,  df_hills[needed_cols]), ignore_index=True)

# %% Achim Dye-2
from datetime import datetime

print("Loading Achim Dye-2")

# loading temperature data
df = pd.read_csv("Data/Achim/CR1000_PT100.txt", header=None)
df.columns = [ "time_matlab", "temp_1", "temp_2", "temp_3", "temp_4", "temp_5", "temp_6", "temp_7", "temp_8"]
df["time"] = pd.to_datetime([
        datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366) for matlab_datenum in df.time_matlab
    ] )
df = df.set_index("time")
df = df.resample("H").mean().drop(columns="time_matlab")

# loading surface height data
df_surf = pd.read_csv("Data/Achim/CR1000_SR50.txt", header=None)
df_surf.columns = ["time_matlab", "sonic_m", "height_above_upgpr"]
df_surf["time"] = pd.to_datetime([
        datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366) for matlab_datenum in df_surf.time_matlab
    ])
df_surf = df_surf.set_index("time")
df_surf = df_surf.resample("H").nearest().drop(columns=["time_matlab", "height_above_upgpr"])

# loading surface height data from firncover
filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
_, sonic_df, _, _, _ = ftl.load_metadata(filepath, sites)

sonic_df = sonic_df.xs("DYE-2", level="sitename").reset_index()
sonic_df = sonic_df.set_index("date").drop(columns="delta").resample("H").nearest()

sonic_df = pd.concat((sonic_df, df_surf.loc[sonic_df.index[-1] :] - 1.83))

plt.figure()
sonic_df.sonic_m.plot()
df_surf.sonic_m.plot()

sonic_df = sonic_df.resample("H").mean()
df["surface_height"] = -(sonic_df.loc[df.index[0] : df.index[-1]] - sonic_df.loc[df.index[0]]).values

depth = 3.4 - np.array([3, 2, 1, 0, -1, -2, -4, -6])
temp_label = ["temp_" + str(i + 1) for i in range(len(depth))]
depth_label = ["depth_" + str(i + 1) for i in range(len(depth))]

for i in range(len(depth)):
    df[depth_label[i]] = depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]

df.loc["2018-05-18":, "depth_1"] = df.loc["2018-05-18":, "depth_1"].values - 1.5
df.loc["2018-05-18":, "depth_2"] = df.loc["2018-05-18":, "depth_2"].values - 1.84

df['temperatureObserved'] = ftl.interpolate_temperature(
    df.index, df[depth_label].values, df[temp_label].values, title="Dye-2 Achim"
).set_index("date")

df["site"] = "DYE-2"
df["latitude"] = 66.4800
df["longitude"] = -46.2789
df["elevation"] = 2165.0
df["depthOfTemperatureObservation"] = 10
df["note"] = "interpolated at 10 m, monthly mean, using surface height from FirnCover station"
df["reference"] = "Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018. "
df["reference_short"] = "Heilig et al. (2018)"
df["method"] = "thermistors"
df["durationOpen"] = 0
df["durationMeasured"] = 30 * 24
df["error"] = 0.25
df=df.reset_index().rename(columns={'time':'date'})

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)


# %% Camp Century Climate
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

df['temperatureObserved'] = ftl.interpolate_temperature(
    df.index,
    df[depth_label].values,
    df[temp_label].values,
    title="Camp Century Climate (long)",
).set_index('date')
df.loc[df.temperatureObserved>-15, "temperatureObserved"] = np.nan
df["site"] = "CEN"
df["latitude"] = 77.1333
df["longitude"] = -61.0333
df["elevation"] = 1880
df["depthOfTemperatureObservation"] = 10
df["note"] = ""
df["reference"] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
df["reference_short"] = "THM_long, Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df["method"] = "thermistors"
df["durationOpen"] = 0
df["durationMeasured"] = 30 * 24
df["error"] = 0.2
df = df.reset_index()

df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)


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
    df[depth_label[i]] = depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]

df['temperatureObserved'] = ftl.interpolate_temperature(df.index,
    df[depth_label].values, df[temp_label].values,
    title="Camp Century Climate (short)").set_index('date')
df.loc[df.temperatureObserved > -15, "temperatureObserved"] = np.nan
df["site"] = "CEN"
df["latitude"] = 77.1333
df["longitude"] = -61.0333
df["elevation"] = 1880
df["depthOfTemperatureObservation"] = 10
df["note"] = "THM_short, interpolated at 10 m"
df["reference"] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
# df_10.temperatureObserved.plot()
df["reference_short"] = "THM_short, Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df["method"] = "thermistors"
df["durationOpen"] = 0
df["durationMeasured"] = 30 * 24
df["error"] = 0.2
df = df.reset_index()

df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

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
        tmp = df_echel.iloc[0, :].copy()
        tmp['Name'] = df_profiles.loc[(site, date), "site"].iloc[0]
        # tmp['longitude'] = df_profiles.loc[(site, date), "longitude"].iloc[0]
        # tmp['latitude'] = df_profiles.loc[(site, date), "latitude"].iloc[0]
        # tmp['elevation'] = df_profiles.loc[(site, date), "elevation"].iloc[0]
        f = interp1d(
            df_profiles.loc[(site, date), "depth"].values,
            df_profiles.loc[(site, date), "temperature"].values,
            fill_value="extrapolate",
        )
        df_profiles['temperatureObserved'] = f(10)
        tmp['depthOfTemperatureObservation'] = 10
        tmp['note'] = "digitized, interpolated at 10 m"
        df_echel = pd.concat((df_echel, tmp))

df_echel["method"] = "thermistors and thermocouples"
df_echel["durationOpen"] = 0
df_echel["durationMeasured"] = 0
df_echel["error"] = 0.3

df_profiles['reference_short'] = df_echel["reference_short"].iloc[0]
df_profiles['reference'] = df_echel["reference"].iloc[0]

df_all = pd.concat((df_all, df_echel[needed_cols]), ignore_index=True)

# %% Larternser EGIG
df_laternser = pd.read_excel("Data/Laternser 1992/Laternser94.xlsx")

df_laternser["reference"] = "Laternser, M., 1994 Firn temperature measurements and snow pit studies on the EGIG traverse of central Greenland, 1992. Eidgenössische Technische Hochschule.  Versuchsanstalt für Wasserbau  Hydrologie und Glaziologic. (Arbeitsheft 15)."
df_laternser["reference_short"] = "Laternser (1994)"

# interpolating the profiles that do not have 10 m depth
tmp, ind = np.unique(
    [str(x) + str(y) for (x, y) in zip(df_laternser.site, df_laternser.date)],
    return_inverse=True,
)
df_interp = pd.DataFrame()
for i in np.unique(ind):
    core_df = df_laternser.loc[ind == i, :]
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
    tmp.note = "interpolated at 10 m, " + tmp.note
    df_interp = pd.concat((df_interp, tmp))

    plt.figure()
    plt.plot(
        core_df.temperatureObserved, -core_df.depthOfTemperatureObservation, marker="o"
    )
    plt.plot(tmp.temperatureObserved, -tmp.depthOfTemperatureObservation, marker="o")
    plt.title(tmp.site)

df_laternser = pd.concat((df_laternser, df_interp))

df_laternser["method"] = "Fenwal 197-303 KAG-401 thermistors"
df_laternser["durationOpen"] = 0
df_laternser["error"] = 0.02

df_all = pd.concat((df_all,  df_laternser[needed_cols]), ignore_index=True)
# %% Wegener 1929-1930
df1 = pd.read_csv("Data/Wegener 1930/200mRandabst_firtemperature_wegener.csv", sep=";")
df3 = pd.read_csv("Data/Wegener 1930/ReadMe.txt", sep=";")

df1["depth"] = df1.depth / 100
df1 = pd.concat((df1, pd.DataFrame.from_dict({"depth": [10], "Firntemp": [np.nan]})), ignore_index=True)
df1 = interp_pandas(df1.set_index("depth"), kind="linear")

df_wegener = pd.DataFrame.from_dict({
        "date": [df3.date.iloc[0]],
        "temperatureObserved": df1.loc[10].values[0],
        "depthOfTemperatureObservation": 10,
        "latitude": [df3.latitude.iloc[0]],
        "longitude": [df3.longitude.iloc[0]],
        "elevation": [df3.elevation.iloc[0]],
        "reference": [df3.reference.iloc[0]],
        "site": [df3.name.iloc[0]],
    })
df = df1.reset_index().rename(columns={'Firntemp':'temperatureObserved', 'depth':'depthOfTemperatureObservation'})
df["date"] = df3.date.iloc[0]
df["latitude"] = df3.latitude.iloc[0]
df["longitude"] = df3.longitude.iloc[0]
df["elevation"] = df3.elevation.iloc[0]
df["reference"] = df3.reference.iloc[0]
df["reference_short"] = "Wegener (1930)"
df["site"] = df3.name.iloc[0]

df2 = pd.read_csv("Data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv", sep=";")
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
df_new["reference"] = df3.reference.iloc[1].strip()
df_wegener = pd.concat((df_wegener, df_new))
df_wegener["reference_short"] = "Wegener (1930)"
df_wegener["note"] = ""

df_wegener["method"] = "electric resistance thermometer"
df_wegener["durationOpen"] = "NA"
df_wegener["durationMeasured"] = "NA"
df_wegener["error"] = 0.2

df_raw = pd.read_csv("Data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv", sep=";")
temp_var = df_raw.columns[1:].tolist()
depth_var=temp_var
df_raw['date'] = date
for i, d in enumerate(temp_var):
    df_raw['depth_'+str(i)] = pd.to_numeric(d.replace('m',''))
    depth_var[i] = 'depth_'+str(i)
    df_raw = df_raw.rename(columns={temp_var[i]:'temp_'+str(i)})
    temp_var[i] = 'temp_'+str(i)
df_raw['site'] = 'Eismitte'
df_raw["latitude"] = df3.latitude.iloc[1]
df_raw["longitude"] = df3.longitude.iloc[1]
df_raw["elevation"] = df3.elevation.iloc[1]
df_raw["reference"] = df3.reference.iloc[1].strip()
df_raw["reference_short"] = "Sorge (1930)"
    
df_all = pd.concat((df_all,  df_wegener[needed_cols] ), ignore_index=True )


# %% Ambach
meta = pd.read_csv(
    "Data/Ambach1979b/metadata.txt",
    sep="\t", header=None,
    names=["site", "file", "date", "latitude", "longitude", "elevation"],
)
meta.date = pd.to_datetime(meta.date)
df_all = pd.DataFrame()
for file in meta.file:
    df = pd.read_csv("Data/Ambach1979b/" + file + ".txt", header=None, names=["temperature", "depth"])
    df['site']=file.split('_')[0]
    df['latitude'] = meta.loc[meta.file == file,'latitude'].iloc[0]
    df['longitude'] = meta.loc[meta.file == file,'longitude'].iloc[0]
    df['elevation'] = meta.loc[meta.file == file,'elevation'].iloc[0]
    df['date']=pd.to_datetime(file.split('_')[1])
    df_all = pd.concat((df_all,df))
    df = pd.read_csv("Data/Ambach1979b/" + file + ".txt", header=None, names=["temperature", "depth"])
    if df.depth.max() < 7.5:
        meta.loc[meta.file == file, "temperatureObserved"] = df.temperature.iloc[-1]
        meta.loc[meta.file == file, "depthOfTemperatureObservation"] = df.index.values[-1]
    else:
        df.loc[df.shape[0]] = [np.nan, 10]  # adding a row
        df = df.set_index("depth")
        df = interp_pandas(df)
        plt.figure()
        df.plot()
        plt.title(file)
        meta.loc[meta.file == file, "temperatureObserved"] = df.temperature.iloc[-1]
        meta.loc[meta.file == file, "depthOfTemperatureObservation"] = df.index.values[-1]

meta["reference"] = "Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979"
meta["reference_short"] = "Ambach (1979)"
df_all["reference"] = "Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979"
df_all["reference_short"] = "Ambach (1979)"

meta["note"] = "digitized, interpolated at 10 m"
meta["method"] = "NA"
meta["durationOpen"] = "NA"
meta["durationMeasured"] = "NA"
meta["error"] = "NA"

df_all = pd.concat((df_all, meta[needed_cols]), ignore_index=True)

# %% Covi
sites = ["DYE-2", "EKT", "SiteJ"]
filenames = ["AWS_Dye2_20170513_20190904_CORR_daily",
    "AWS_EKT_20170507_20190904_CORR_daily",
    "AWS_SiteJ_20170429_20190904_CORR_daily"]
df_covi = pd.DataFrame()
for site, filename in zip(sites, filenames):
    df = pd.read_csv("Data/Covi/" + filename + ".dat", skiprows=1)
    depth = np.flip(16 - np.concatenate((np.arange(16, 4.5, -0.5), np.arange(4, -1, -1))))
    depth_ini = 1
    depth = depth + depth_ini
    df["date"] = df.TIMESTAMP.astype(str)
    temp_label = ["Tfirn" + str(i) + "(C)" for i in range(1, 29)]

    df.date = pd.to_datetime(df.date)
    df = df.drop(columns=['TIMESTAMP'])

    if site in ["DYE-2", "EKT"]:
        # loading surface height from FirnCover station
        filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
        sites = ["Summit",  "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
        statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
        statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

        df["surface_height"] = np.interp(df.date.values.astype(float), 
                          sonic_df.loc[site].index.values.astype(float), 
                          -sonic_df.loc[site].sonic_m.values)
        
    else:
        df["surface_height"] = df["SR50_corr(m)"]

    df["surface_height"] = df["surface_height"] - df["surface_height"].iloc[0]
    df["surface_height"] = df["surface_height"].interpolate().values
    plt.figure()
    df.set_index('date').surface_height.plot()
    if site in ["DYE-2", "EKT"]: (-sonic_df.loc[site].sonic_m).plot()
    plt.title(site)

    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    for i in range(len(temp_label)):
        df[depth_label[i]] = depth[i] + df["surface_height"].values
    df = df.set_index("date")
    df['temperatureObserved'] = ftl.interpolate_temperature(
        df.index, df[depth_label].values, df[temp_label].values,
        title=site + " Covi et al.",
    ).set_index('date')

    df["site"] = site
    if site == "SiteJ":
        df["latitude"] = 66.864952
        df["longitude"] = -46.265141
        df["elevation"] = 2060
    else:
        df["latitude"] = statmeta_df.loc[site, "latitude"]
        df["longitude"] = statmeta_df.loc[site, "longitude"]
        df["elevation"] = statmeta_df.loc[site, "elevation"]
    df["depthOfTemperatureObservation"] = 10
    df["note"] = ""

    df["reference"] = "Covi, F., Hock, R., and Reijmer, C.: Challenges in modeling the energy balance and melt in the percolation zone of the Greenland ice sheet. Journal of Glaciology, 69(273), 164-178. doi:10.1017/jog.2022.54, 2023. and Covi, F., Hock, R., Rennermalm, A., Leidman S., Miege, C., Kingslake, J., Xiao, J., MacFerrin, M., Tedesco, M.: Meteorological and firn temperature data from three weather stations in the percolation zone of southwest Greenland, 2017 - 2019. Arctic Data Center. doi:10.18739/A2BN9X444, 2022."
    df["reference_short"] = "Covi et al. (2022, 2023)"
    df["note"] = ""
    df["method"] = "thermistor"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.1
    
    df_covi = pd.concat((df_covi,  df.reset_index()), ignore_index=True)
df_all = pd.concat((df_all,  df_covi[needed_cols]), ignore_index=True)

# %% Giese & Hawley
from datetime import datetime, timedelta

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


df_giese["temperatureObserved"] = np.nan

for date in df_giese.index:
    f = interp1d(
        df_giese.loc[(date), ["depth_7", "depth_8"]].values,
        df_giese.loc[(date), ["temp_7", "temp_8"]].values,
        fill_value="extrapolate",
    )
    df_giese.loc[date, "temperatureObserved"] = f(10)

df_giese = df_giese.resample("M").mean().reset_index()
df_giese["date"] = df_giese.time
df_giese["site"] = "Summit"
df_giese["latitude"] = 72 + 35 / 60
df_giese["longitude"] = -38 - 30 / 60
df_giese["elevation"] = 3208
df_giese["depthOfTemperatureObservation"] = 10

df_giese[
    "reference"
] = "Giese AL and Hawley RL (2015) Reconstructing thermal properties of firn at Summit, Greenland, from a temperature profile time series. Journal of Glaciology 61(227), 503–510 (doi:10.3189/2015JoG14J204)"
df_giese["reference_short"] = "Giese and Hawley (2015)"
df_giese["note"] = "digitized and interpolated at 10m"

df_giese["method"] = "thermistors"
df_giese["durationOpen"] = 0
df_giese["durationMeasured"] = 24 * 30
df_giese["error"] = 0.5

df_all = pd.concat((df_all, df_giese[needed_cols]), ignore_index=True)

# %% IMAU
df_meta = pd.read_csv("data/IMAU/meta.csv")
depth_label = ["depth_" + str(i) for i in range(1, 6)]
temp_label = ["temp_" + str(i) for i in range(1, 6)]
df_imau = pd.DataFrame()
for i, site in enumerate(["s5", "s6", "s9"]):
    df = pd.read_csv("data/IMAU/" + site + "_tsub.txt")
    df["date"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
        df["doy"] - 1, unit="d"
    )
    df = df.set_index("date").drop(columns=["year", "doy"]).resample("D").mean()

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
    df_10 = ftl.interpolate_temperature(
        df.index,
        df[depth_label].values,
        df[temp_label].values,
        title=site,
        min_diff_to_depth=min_diff_to_depth,
        kind="slinear",
        surface_height=surface_height,
    )
    df_10 = df_10.set_index("date").resample("M").mean().reset_index()

    df_10["site"] = site
    df_10["note"] = ""
    df_10["latitude"] = df_meta.loc[df_meta.site == site, "latitude"].values[0]
    df_10["longitude"] = df_meta.loc[df_meta.site == site, "longitude"].values[0]
    df_10["elevation"] = df_meta.loc[df_meta.site == site, "elevation"].values[0]
    df_imau = pd.concat((df_imau, df_10))
df_imau[
    "reference"
] = " Paul C. J. P. Smeets, Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954"
df_imau["reference_short"] = "Smeets et al. (2018)"
df_imau["depthOfTemperatureObservation"] = 10
df_imau["method"] = "thermistor"
df_imau["durationOpen"] = 0
df_imau["durationMeasured"] = 24 * 30
df_imau["error"] = 0.2
df_all = pd.concat((df_all, df_imau[needed_cols]), ignore_index=True)

# %% Braithwaite
df = pd.read_excel("Data/Braithwaite/data.xlsx")
df["temperatureObserved"] = df[10].values
df["depthOfTemperatureObservation"] = 10

df["reference"] = "Braithwaite, R. (1993). Firn temperature and meltwater refreezing in the lower accumulation area of the Greenland ice sheet, Pâkitsoq, West Greenland. Rapport Grønlands Geologiske Undersøgelse, 159, 109–114. https://doi.org/10.34194/rapggu.v159.8218"
df["reference_short"] = "Braithwaite (1993)"
df["note"] = "from table"
df["method"] = "thermistor"
df["durationOpen"] = 0
df["durationMeasured"] = 0
df["error"] = 0.5
df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

# %% Schytt
df = pd.read_excel("Data/Schytt Tuto/data.xlsx")
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date == date]
        if df_date.shape[0] < 2:
            continue
        tmp = df_date.iloc[0:1, :].copy()

        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "digitized, interpolated at 10 m"
        df_interp = pd.concat((df_interp, tmp))

        plt.figure()
        plt.plot(df_date.temperatureObserved, -df_date.depth, marker="o")
        plt.plot(
            tmp.temperatureObserved, -tmp.depthOfTemperatureObservation, marker="o"
        )
        plt.title(tmp.site.values[0] + " " + tmp.date.values[0])


df_interp["depthOfTemperatureObservation"] = 10
df_interp["date"] = pd.to_datetime(df_interp.date)
df_interp["reference"] = "Schytt, V. (1955) Glaciological investigations in the Thule Ramp area, U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 28, 88 pp. https://hdl.handle.net/11681/5989"
df_interp["reference_short"] = "Schytt (1955)"
df_interp["note"] = "from table"

df_interp["method"] = "copper-constantan thermocouples"
df_interp["durationOpen"] = 0
df_interp["durationMeasured"] = 0
df_interp["error"] = 0.5

df_all = pd.concat((df_all, 
    df_interp[needed_cols],
    ), ignore_index=True,
)

# %% Griffiths & Schytt
df = pd.read_excel("Data/Griffiths Tuto/data.xlsx")
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date == date]
        if df_date.shape[0] < 2:
            continue
        tmp = df_date.iloc[0:1, :].copy()
        if df_date["depth"].max() < 8:
            continue
        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "digitized, interpolated at 10 m"
        df_interp =pd.concat(( df_interp, tmp))

        plt.figure()
        plt.plot(df_date.temperatureObserved, -df_date.depth, marker="o")
        plt.plot(tmp.temperatureObserved, 
                 -tmp.depthOfTemperatureObservation, 
                 marker="o")
        plt.title(tmp.site.values[0] + " " + tmp.date.values[0])

df_interp["depthOfTemperatureObservation"] = 10
df_interp["date"] = pd.to_datetime(df_interp.date)
df_interp["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
df_interp["reference_short"] = "Griffiths (1960)"
df_interp["note"] = "from table"
df_interp["method"] = "copper-constantan thermocouples"
df_interp["durationOpen"] = 0
df_interp["durationMeasured"] = 0
df_interp["error"] = 0.5

df_all = pd.concat((df_all, df_interp[needed_cols]), ignore_index=True)

# %% Griffiths & Meier
df = pd.read_excel("Data/Griffiths Tuto/data_crevasse3.xlsx")
df_interp = pd.DataFrame()

for site in df.site.unique():
    df_site = df.loc[df.site == site]
    for date in df_site.date.unique():
        df_date = df_site.loc[df_site.date == date]
        if df_date.shape[0] < 2:
            continue
        tmp = df_date.iloc[0:1, :].copy()
        if df_date["depth"].max() < 8:
            continue
        f = interp1d(
            df_date["depth"].values,
            df_date["temperatureObserved"].values,
            fill_value="extrapolate",
        )
        tmp.temperatureObserved = min(f(10), 0)
        tmp.depthOfTemperatureObservation = 10
        tmp.note = "measurement made close to an open crevasse, digitized, interpolated at 10 m"
        df_interp = pd.concat((df_interp, tmp))

        plt.figure()
        plt.plot(df_date.temperatureObserved, -df_date.depth, marker="o")
        plt.plot(
            tmp.temperatureObserved, -tmp.depthOfTemperatureObservation, marker="o"
        )
        plt.title(tmp.site.values[0] + " " + tmp.date.values[0])

df_interp["depthOfTemperatureObservation"] = 10
df_interp["date"] = pd.to_datetime(df_interp.date)
df_interp["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
df_interp["reference_short"] = "Griffiths (1960)"

df_interp.loc[df_interp.date <= '1955-12-31','reference'] = "Meier, M. F., Conel, J. E., Hoerni, J. A., Melbourne, W. G., & Pings, C. J. (1957). Preliminary Study of Crevasse Formation. Blue Ice Valley, Greenland, 1955. OCCIDENTAL COLL LOS ANGELES CALIF. https://hdl.handle.net/11681/6029"
df_interp.loc[df_interp.date <= '1955-12-31',"reference_short"] = "Meier et al. (1957)"

df_interp["latitude"] = 76.43164
df_interp["longitude"] = -67.54949
df_interp["elevation"] = 800
df_interp["method"] = "copper-constantan thermocouples"
df_interp["durationOpen"] = 0
df_interp["durationMeasured"] = 0
df_interp["error"] = 0.5
# only keeping measurements more than 1 m into the crevasse wall
df_interp = df_interp.loc[df_interp["distance from crevasse"] >= 1, :]
df_all = pd.concat((df_all,df_interp[needed_cols]), ignore_index=True)


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
    [str(x) + str(y) for (x, y) in zip(df.site, df.date)], return_inverse=True
)
df_interp = pd.DataFrame()
for i in np.unique(ind):
    core_df = df.loc[ind == i, :]
    if core_df["depthOfTemperatureObservation"].max() < 9:
        continue
    if core_df["depthOfTemperatureObservation"].min() > 11:
        continue
    if len(core_df["depthOfTemperatureObservation"]) < 2:
        continue

    tmp = core_df.iloc[0, :].copy()
    f = interp1d(
        np.log(core_df["depthOfTemperatureObservation"].values),
        core_df["temperatureObserved"].values,
        fill_value="extrapolate",
    )
    tmp.temperatureObserved = f(np.log(10))
    tmp.depthOfTemperatureObservation = 10
    tmp.note = "digitized, interpolated at 10 m"
    df_interp = pd.concat((df_interp, tmp))

    plt.figure()
    plt.plot(core_df.temperatureObserved,
        -(core_df.depthOfTemperatureObservation),
        marker="o", linestyle="None")
    plt.plot(tmp.temperatureObserved, -tmp.depthOfTemperatureObservation, marker="o")
    x = np.arange(0, 20, 0.1)
    plt.plot(f(np.log(x)), -x)
    plt.title(tmp.site)

    # core_df  = interp_pandas(core_df)
df = pd.concat((df, df_interp))

df["reference"] = "van der Veen, C. J., Mosley-Thompson, E., Jezek, K. C., Whillans, I. M., and Bolzan, J. F.: Accumulation rates in South and Central Greenland, Polar Geography, 25, 79–162, https://doi.org/10.1080/10889370109377709, 2001."
df["reference_short"] = "van der Veen et al. (2001)"
df["method"] = "thermistor"
df["durationOpen"] = 8 * 24
df["durationMeasured"] = 0
df["error"] = 0.1
df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)


# %% Checking values
df_all = df_all.loc[~df_all.temperatureObserved.isnull(), :]

df_ambiguous_date = df_all.loc[pd.to_datetime(df_all.date,utc=True, errors="coerce").isnull(), :]
df_bad_long = df_all.loc[df_all.longitude.astype(float) > 0, :]
# df_bad_long.to_csv('bad_lat.csv')
df_no_coord = df_all.loc[np.logical_or(df_all.latitude.isnull(), df_all.latitude.isnull()), :]
df_invalid_depth = df_all.loc[
    pd.to_numeric(df_all.depthOfTemperatureObservation, errors="coerce").isnull(), :
]
df_no_elev = df_all.loc[df_all.elevation.isnull(), :]
if len(df_no_elev)>0:
    df_no_elev.to_csv('missing_elev.csv')

# %% Removing nan and saving
tmp = df_all.loc[np.isnan(df_all.temperatureObserved.astype(float).values)]
df_all = df_all.loc[~np.isnan(df_all.temperatureObserved.astype(float).values)]

df_all.to_csv("output/subsurface_temperature_summary.csv")

# %% averaging to monthly and clipping to contiguous ice sheet
df = pd.read_csv("output/subsurface_temperature_summary.csv", low_memory=False)

df_ambiguous_date = df.loc[pd.to_datetime(df.date.astype(str), utc=True, format='mixed', errors="coerce").isnull(), :]
df = df.loc[~pd.to_datetime(df.date, utc=True, format='mixed', errors="coerce").isnull(), :]

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

df = df.loc[df.depthOfTemperatureObservation == 10, :]
df = df.loc[df.temperatureObserved.notnull(), :]

df["date"] = pd.to_datetime(df.date, utc=True, format='mixed')
df.loc[df.reference_short.isnull(), "reference_short"] = df.loc[
    df.reference_short.isnull(), "reference"
]

df.loc[df.site.isnull(), "site"] = [
    "unnamed " + str(i) for i in df.loc[df.site.isnull()].index
]

df_m = pd.DataFrame()
# generating the monthly file
for ref in df.reference_short.unique():
    for site in df.loc[df.reference_short == ref, "site"].unique():
        df_loc = df.loc[(df.reference_short == ref) & (df.site == site), :].copy()

        if (
            np.sum((df_loc.date.diff() < "2 days") & (df_loc.date.diff() > "0 days"))
            > 5
        ):
            print(ref, site, "... averaging to monthly")
            col_non_num = ["Unnamed: 0", "site", "reference", "reference_short", "note",'error', 'durationOpen', 'durationMeasured', 'method']
            df_loc_first = df_loc.set_index("date")[col_non_num].resample("M").first()

            col_num = ['latitude', 'longitude', 'elevation', 'depthOfTemperatureObservation', 'temperatureObserved']
            df_loc = df_loc.set_index("date")[col_num].resample("M").mean()

            df_loc[col_non_num] = df_loc_first[col_non_num]
            df_loc = df_loc.loc[df_loc.latitude.notnull(), :]
            df_loc['durationMeasured'] = 30
            if any(df_loc.depthOfTemperatureObservation.unique() != 10):
                print("Some non-10 m depth")
                print(df_loc.depthOfTemperatureObservation.unique())
                print(df_loc.loc[df_loc.depthOfTemperatureObservation != 10].head())
                df_loc = df_loc.loc[df_loc.depthOfTemperatureObservation == 10, :]

            df_m = pd.concat((df_m, df_loc.reset_index()[needed_cols]), 
                             ignore_index=True)
        else:
            print(ref, site, "... not averaging")
            df_m = pd.concat((df_m,  df_loc[needed_cols]), ignore_index=True)

df_m.latitude = np.round(df_m.latitude, 5)
df_m.longitude = np.round(df_m.longitude, 5)
df_m.temperatureObserved = np.round(df_m.temperatureObserved, 2)
df_m['date'] = pd.to_datetime(df_m['date'], utc=True).dt.date

# import geopandas as gpd
# ice = gpd.GeoDataFrame.from_file("Data/misc/IcePolygon_3413.shp")
# ice = ice.to_crs("EPSG:3413").to_crs(4326)
    
# gdf = gpd.GeoDataFrame(
#     df_m, geometry=gpd.points_from_xy(df_m.longitude, df_m.latitude), crs="EPSG:4326"
# )
# df_m = df_m.loc[gdf.within(ice.geometry[0]), :]

df_m.to_csv("output/10m_temperature_dataset_monthly.csv", index=False)

#%% Dataset composition plot and table
df_m = pd.read_csv("output/10m_temperature_dataset_monthly.csv", parse_dates=['date'])

df_m['coeff'] = 1
df_summary = (df_m.groupby('reference_short')
              .apply(lambda x: x.coeff.sum())
              .reset_index(name='num_measurements'))
df_summary=df_summary.sort_values('num_measurements')
explode = 0.5*(df_summary.num_measurements.max() - df_summary.num_measurements)/df_summary.num_measurements.max()

fig, ax=plt.subplots(1,1, figsize=(12,8))
plt.subplots_adjust(bottom=0.4)
patches, texts = plt.pie( df_summary.num_measurements,
                         startangle=90,
                         explode=explode)
labels = df_summary.reference_short
sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, df_summary.num_measurements),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='lower left', bbox_to_anchor=(-1, -.8),
           fontsize=8, ncol=3, title='Data origin (listed clock-wise)')
plt.ylabel('')

plt.savefig('figures/dataset_composition.png',dpi=300)


df_summary =(df_m.groupby('reference_short')
                  .apply(lambda x: x.date.min().year)
                  .reset_index(name='start_year'))
df_summary['end_year'] =(df_m.groupby('reference_short')
                    .apply(lambda x: x.date.max().year)
                    .reset_index(name='end_year')).end_year
df_summary['num_measurements'] = (df_m.groupby('reference_short')
                          .apply(lambda x: x.coeff.sum())
                          .reset_index(name='num_measurements')).num_measurements

df_summary.sort_values('start_year').to_csv('output/dataset_composition.csv',sep='|', index=None)
