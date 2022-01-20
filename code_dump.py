# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
from sklearn import linear_model

# %% Trend analysis: elevation, latitude, avg_snowfall_mm

X = df[["elevation", "latitude", "avg_snowfall_mm"]]
Y = df["temperatureObserved"]
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("Intercept: \n", regr.intercept_)
print("Coefficients: \n", regr.coef_)

xx, yy, zz = np.meshgrid(
    np.arange(0, 3400, 100), np.arange(60, 83, 1), np.arange(0, 1600, 10)
)
z = regr.intercept_ + regr.coef_[0] * xx + regr.coef_[1] * yy + regr.coef_[2] * zz

# Trend  fit
df["temperature_anomaly"] = (
    df.temperatureObserved
    - regr.intercept_
    - regr.coef_[0] * df.elevation
    - regr.coef_[1] * df.latitude
    - regr.coef_[2] * df.avg_snowfall_mm
)
y_data = df.temperature_anomaly.values[dates.year > 1990]
time_recent = dates[dates.year > 1990]
curve_fit = np.polyfit(pd.to_numeric(time_recent), y_data, 1)
temp_trend = curve_fit[1] + curve_fit[0] * pd.to_numeric(time_recent)

df = df.sort_index()
x = pd.to_numeric(df.index)
y = df.temperature_anomaly

coef = np.polyfit(x, y, 4)
p = np.poly1d(coef)
y_pred = p(x)

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.8, bottom=0.1, left=0.1, right=0.7)
for ref in df.reference_short.unique():
    ax.plot(
        df.loc[df.reference_short == ref, "temperature_anomaly"],
        "o",
        markeredgecolor="gray",
        label=ref,
    )
for txt, x, y in zip(
    df.loc[df.temperature_anomaly.abs() > 5, "site"],
    df.loc[df.temperature_anomaly.abs() > 5].index,
    df.loc[df.temperature_anomaly.abs() > 5, "temperature_anomaly"],
):
    ax.annotate(txt, (x, y))
ax.plot(time_recent, temp_trend, "--", color="red", linewidth=3)
ax.plot(df.index, y_pred, "--", color="orange", linewidth=3)
ax.set_xlabel("Year")
ax.set_ylabel("10 m firn temperature anomaly")
ax.set_title(
    "Elevation - Latitude - Snowfall model\n RMSE = %0.3f"
    % np.sqrt(np.mean(df["temperature_anomaly"] ** 2))
)
lgnd = plt.legend(bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
for i in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[i]._legmarker.set_markersize(10)
fig.savefig("figures/ELS_trend_time.png")


# %% Model lat_elev_poly
df = df_save
site_remove = ["SouthDome", "SDM", "SE-Dome", "H2", "H3", "FA_13", "FA_15_1", "FA_15_2"]
df = df.loc[~np.isin(df.site, site_remove)]
elevation_bins = np.arange(0, 3300, 100)
latitude_bins = np.arange(60, 85, 5)

X = df[["elevation", "latitude"]]
Y = df["temperatureObserved"]


def polyfit2d(x, y, z, order=2):
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z


# Fit a 3rd order, 2d polynomial
m = polyfit2d(df["elevation"], df["latitude"], df["temperatureObserved"], order=5)

xx, yy = np.meshgrid(
    np.arange(0, 3400, 100), np.arange(60, 83, 1)
)  # ,np.arange(0,1600,10))
zz = polyval2d(xx.astype(float), yy.astype(float), m)
zz[zz > 0] = 0
zz[zz < -50] = -50

# Trend  fit
df["temperature_anomaly"] = df.temperatureObserved - polyval2d(
    df.elevation, df.latitude, m
)
df_save["temperature_anomaly"] = df_save.temperatureObserved - polyval2d(
    df_save.elevation, df_save.latitude, m
)

dates = df.index
y_data = df.temperature_anomaly.values[dates.year > 1990]
time_recent = dates[dates.year > 1990]
curve_fit = np.polyfit(pd.to_numeric(time_recent), y_data, 1)
y = curve_fit[1] + curve_fit[0] * pd.to_numeric(time_recent)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(df["elevation"], df["latitude"], df["temperatureObserved"], color="black")
for site in site_remove:
    ax.scatter(
        df_save.loc[df_save.site == site, "elevation"],
        df_save.loc[df_save.site == site, "latitude"],
        df_save.loc[df_save.site == site, "temperatureObserved"],
    )
    # ax.text(df_save.loc[df_save.site==site,'elevation'].mean(),
    #             df_save.loc[df_save.site==site,'latitude'].mean(),
    #             df_save.loc[df_save.site==site,'temperatureObserved'].mean(),
    #             site, size=10, zorder=1, color='k')
ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, alpha=0.5)
ax.set_xlabel("Elevation (m)")
ax.set_ylabel("Latitude ($^o$)")
ax.set_zlabel("10 m firn temperature ($^o$C)")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(df["temperature_anomaly"], ".")
ax2.plot(time_recent, y, "--", linewidth=3)
for site in site_remove:
    ax2.plot(df_save.loc[df_save.site == site, "temperature_anomaly"], "o")
    ax2.annotate(
        site,
        (
            df_save.loc[df_save.site == site, "temperature_anomaly"].index.mean(),
            df_save.loc[df_save.site == site, "temperature_anomaly"].mean(),
        ),
    )
for txt, x, y in zip(
    df.loc[df.temperature_anomaly.abs() > 3, "site"],
    df.loc[df.temperature_anomaly.abs() > 3].index,
    df.loc[df.temperature_anomaly.abs() > 3, "temperature_anomaly"],
):
    ax2.annotate(txt, (x, y))
ax2.set_xlabel("Year")
ax2.set_ylabel("10 m firn temperature anomaly")
ax2.set_title(
    "Elevation - Latitude model\n RMSE = %0.3f"
    % np.sqrt(np.mean(df["temperature_anomaly"] ** 2))
)
# plt.savefig('figures/lat_elev_poly.png')

# %% transect analysis
df_select = df.loc[np.logical_and(df.latitude < 69, df.latitude < 90), :]
# df = df.loc[df.longitude <-36.5,:]

year = df_select.latitude.values * np.nan
for i, x in enumerate(pd.to_datetime(df_select.date)):
    try:
        year[i] = x.year
    except:
        year[i] = x
ind = np.logical_or(
    df_select.site == "Eismitte", df_select.site == "200 km Randabstand"
)


plt.figure()
sc = plt.scatter(
    df_select.elevation, df_select.temperatureObserved, 150, year, cmap="tab20"
)
plt.scatter(
    df_select.elevation.loc[ind],
    df_select.temperatureObserved.loc[ind],
    200,
    year[ind],
    cmap="tab20",
)
for i, txt in enumerate(df_select.site.unique()):
    print(txt, df_select.loc[df_select.site == txt, "elevation"].mean())
    plt.annotate(  # str(txt), #
        txt,
        (
            df_select.loc[df_select.site == txt, "elevation"].mean(),
            df_select.loc[df_select.site == txt, "temperatureObserved"].mean(),
        ),
    )
plt.colorbar(sc, label="Year")

plt.ylabel("10 m firn temperature (deg C)")
plt.xlabel("Elevation (m a.s.l.)")
plt.savefig("figures/EGIG.png")



# %% find unique locations of measurements
df_all= df[['site','latitude','longitude']]
df_gcnet = pd.read_csv('GC-Net_location.csv')[['Name','Northing','Easting']].rename(columns={'Name':'site','Northing':'latitude','Easting':'longitude'})
df_promice = pd.read_csv('PROMICE_coordinates_2015.csv', sep=';')[['Station name', 'Northing', 'Easting']].rename(columns={'Station name':'site','Northing':'latitude','Easting':'longitude'})

df_all = df_all.append(df_gcnet)
df_all = df_all.append(df_promice)
df_all.latitude = round(df_all.latitude,5)
df_all.longitude = round(df_all.longitude,5)
latlon =np.array([str(lat)+str(lon) for lat, lon in zip(df_all.latitude, df_all.longitude)])
uni, ind = np.unique(latlon, return_index = True)
print(latlon[:3])
print(latlon[np.sort(ind)][:3])
df_all = df_all.iloc[np.sort(ind)]
site_uni, ind = np.unique([str(s) for s in df_all.site], return_index = True)
df_all.iloc[np.sort(ind)].to_csv('coords.csv',index=False)
