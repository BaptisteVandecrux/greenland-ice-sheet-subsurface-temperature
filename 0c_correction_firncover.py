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

print("Loading FirnCover")
time.sleep(0.2)

filepath = os.path.join("Data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df = ftl.load_metadata(filepath, sites)
statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

rtd_df = rtd_df.reset_index()
rtd_df = rtd_df.set_index(["sitename", "date"])
site = 'DYE-2'

df_firncover = rtd_df.xs(site, level="sitename")

# Achim Dye-2
print("Loading Achim Dye-2")
time.sleep(0.2)

ds = xr.open_dataset("Data/Achim/T_firn_DYE-2_16.nc")
df = ds.to_dataframe()
df = df.reset_index(1).groupby("level").resample("D").mean()
df.reset_index(0, inplace=True, drop=True)
df.reset_index(inplace=True)

df_achim = pd.DataFrame()
df_achim["date"] = df.loc[df["level"] == 1, "time"]
for i in range(1, 9):
    df_achim["rtd" + str(i)] = df.loc[df["level"] == i, "Firn temperature"].values
for i in range(1, 9):
    df_achim["depth_" + str(i)] = df.loc[df["level"] == i, "depth"].values
df_achim =df_achim.set_index('date')

print('loading Samiras data')
df_samira = pd.read_excel('Data/too shallow/Samira 1.60m deep/pitB_temp.xlsx')
df_samira=df_samira.set_index('date')
df_samira_a = pd.read_excel('Data/too shallow/Samira 1.60m deep/pitA_temp.xlsx')
df_samira_a = df_samira_a.set_index('date')

print('loading GC-Net surface height')
ds = xr.open_dataset("C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/AWS/GC-Net/20190501_jaws/08c.dat_Req1957.nc")
df = ds.to_dataframe()
df = df.reset_index(1)
df = df.loc[df.nbnd == 0][['snh1','snh2']]

df = df.loc['2015-01-01':]
# height_gcnet = df['Depth']-df.Depth.iloc[0]-0.5

plt.figure()
# df.snh1.plot()
(df.snh2-1).plot()
df_firncover["depth_1"].plot()


df_firncover = df_firncover.loc['2016-05-01':'2016-12-31']
df_samira = df_samira.loc['2016-05-01':'2016-12-31']
df_samira_a = df_samira_a.loc['2016-05-01':'2016-12-31']

# %%
plt.close('all')
fig, ax = plt.subplots(1,2)
i=2
df_firncover["depth_"+str(i)].plot(ax=ax[0])
df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
print('firncover',i,df_firncover["depth_"+str(i)].loc['2016-05-11'])

i= 5
df_samira["depth_"+str(i)].plot(ax=ax[0])
df_samira["temp_"+str(i)].plot(ax=ax[1],label='Samira - B')
print('samira-b',i,df_samira["depth_"+str(i)].loc['2016-05-11'].iloc[0])

i= 3
df_samira_a["depth_"+str(i)].plot(ax=ax[0])
df_samira_a["temp_"+str(i)].plot(ax=ax[1],label='Samira - A')
print('samira-a',i,df_samira_a["depth_"+str(i)].loc['2016-05-11'].iloc[0])
plt.legend()
plt.tight_layout()

ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Temperature (degC)')

# %% sensor 3 of firncover matches with sensor 4 in samira-a
fig, ax = plt.subplots(1,2)
i=3
df_firncover["depth_"+str(i)].plot(ax=ax[0])
df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
print('firncover',i,df_firncover["depth_"+str(i)].loc['2016-05-11'])

i= 2
df_achim["depth_"+str(i)].plot(ax=ax[0])
df_achim["rtd"+str(i)].plot(ax=ax[1],label='Achim')
print('achim',i,df_achim["depth_"+str(i)].loc['2016-05-11'])

i= 4
df_samira_a["depth_"+str(i)].plot(ax=ax[0])
df_samira_a["temp_"+str(i)].plot(ax=ax[1],label='Samira - A')
print('samira-a',i,df_samira_a["depth_"+str(i)].loc['2016-05-11'].iloc[0])
plt.legend()
plt.tight_layout()
ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Temperature (degC)')
# %
fig, ax = plt.subplots(1,2)
i=4
df_firncover["depth_"+str(i)].plot(ax=ax[0])
df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
print('firncover',i,df_firncover["depth_"+str(i)].loc['2016-05-11'])

i= 5
df_samira_a["depth_"+str(i)].plot(ax=ax[0])
df_samira_a["temp_"+str(i)].plot(ax=ax[1],label='Samira - A')
print('samira-a',i,df_samira_a["depth_"+str(i)].loc['2016-05-11'].iloc[0])
plt.legend()
plt.tight_layout()
ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Temperature (degC)')
# %
fig, ax = plt.subplots(1,2)
i=5
df_firncover["depth_"+str(i)].plot(ax=ax[0])
df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
print('firncover',i,df_firncover["depth_"+str(i)].loc['2016-05-11'])

i= 3
df_achim["depth_"+str(i)].plot(ax=ax[0])
df_achim["rtd"+str(i)].plot(ax=ax[1],label='Achim')
print('achim',i,df_achim["depth_"+str(i)].loc['2016-05-11'])
plt.tight_layout()
ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Temperature (degC)')
# %
fig, ax = plt.subplots(1,2)
i=7
df_firncover["depth_"+str(i)].plot(ax=ax[0])
df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
print('firncover',i,df_firncover["depth_"+str(i)].loc['2016-05-11'])

i= 4
df_achim["depth_"+str(i)].plot(ax=ax[0])
df_achim["rtd"+str(i)].plot(ax=ax[1],label='Achim')
print('achim',i,df_achim["depth_"+str(i)].loc['2016-05-11'])

plt.tight_layout()
ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Temperature (degC)')

#%%
df_firncover = df_firncover.resample('D').mean().loc['2016-05-11':'2016-09-16']
df_achim=df_achim.resample('D').mean().loc['2016-05-11':'2016-09-16']
df_samira_a=df_samira_a.resample('D').mean().loc['2016-05-11':'2016-09-16']
df_samira=df_samira.resample('D').mean().loc['2016-05-11':'2016-09-16']

X = np.array([])
Y = np.array([])
source = np.array([])
i=2
X = np.append(X, df_firncover["rtd"+str(i)].values)
X = np.append(X, df_firncover["rtd"+str(i)].values)
i= 5
Y = np.append(Y, df_samira["temp_"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+3)
i= 3
Y = np.append(Y,df_samira_a["temp_"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+2)

i=3
X = np.append(X, df_firncover["rtd"+str(i)].values)
X = np.append(X, df_firncover["rtd"+str(i)].values)
i= 2
Y = np.append(Y, df_achim["rtd"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+1)
i= 4
Y = np.append(Y,df_samira_a["temp_"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+2)

i=4
X = np.append(X, df_firncover["rtd"+str(i)].values)
i= 5
Y = np.append(Y,df_samira_a["temp_"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+2)

i=5
X = np.append(X, df_firncover["rtd"+str(i)].values)
i= 3
Y = np.append(Y, df_achim["rtd"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+1)

i=7
X = np.append(X, df_firncover["rtd"+str(i)].values)
i= 4
Y = np.append(Y, df_achim["rtd"+str(i)].values)
source = np.append(source,  df_samira["temp_"+str(i)].values*0+1)

#%%
z = np.polyfit(X, Y, 1)
p = np.poly1d(z)

fig, ax = plt.subplots(1,2)
ax[0].plot(X,Y,'o',linestyle = 'None',color = 'k')
ax[0].plot([-18, 2],[-18, 2],color = 'r',label = '1:1 line')
ax[0].plot(np.arange(-18,2,0.5), p(np.arange(-18,2,0.5)), label = 'correction function')
ax[0].set_xlabel('FirnCover temperatures ($^o$C)')
ax[0].set_ylabel('Reference temperatures ($^o$C)')
ax[0].set_xlim(-18, 2)
ax[0].set_ylim(-18, 2)
ax[0].legend()
ax[0].set_title('before correction')
ax[0].annotate('ME = %0.2f\nRMSE = %0.2f' % (np.mean(X-Y), 
                                             np.sqrt(np.mean((X-Y)**2))),
               (-16,-2.5))

print(z)
# [ 1.03093649 -0.49950273]
# after correction
ax[1].plot(p(X),Y,'o',linestyle = 'None',color = 'k')
ax[1].plot([-18, 2],[-18, 2],color = 'r')
ax[1].set_xlabel('FirnCover temperatures ($^o$C)')
ax[1].set_ylabel('Reference temperatures ($^o$C)')
ax[1].set_xlim(-18, 2)
ax[1].set_ylim(-18, 2)
ax[1].set_title('after correction')
ax[1].annotate('ME = %0.2f\nRMSE = %0.2f' % (np.round(np.mean(p(X)-Y),2), 
                                             np.sqrt(np.mean((p(X)-Y)**2))),
               (-16,-2.5))
plt.tight_layout()
# %%
plt.close('all')

for site in sites:
    df_firncover = rtd_df.xs(site, level="sitename")
    fig, ax = plt.subplots(1,2)
    i=0
    for i in range(23):
        if (df_firncover["depth_"+str(i)].iloc[0] < 0.1) | (np.isnan(df_firncover["depth_"+str(i)].iloc[0])):
            continue
        df_firncover["depth_"+str(i)].plot(ax=ax[0])
        df_firncover["rtd"+str(i)].plot(ax=ax[1],label='FirnCover')
        print(site,i,df_firncover["depth_"+str(i)].iloc[0])
        plt.title(site)
        break