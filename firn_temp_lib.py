# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:43:45 2020

@author: bav
"""
import numpy as np
import pandas as pd
import tables as tb
import progressbar
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# interpolation function
def interpolate_temperature(
    dates,
    depth_cor,
    temp,
    depth=10,
    min_diff_to_depth=1.5,
    kind="quadratic",
    title="",
    plot=True,
    surface_height=[]
):
    depth_cor = depth_cor.astype(float)
    df_interp = pd.DataFrame()
    df_interp["date"] = dates
    df_interp["temperatureObserved"] = np.nan

    # preprocessing temperatures for small gaps
    tmp = pd.DataFrame(temp)
    tmp["time"] = dates.values
    tmp = tmp.set_index("time")
    tmp = tmp.resample("H").mean()
    # tmp = tmp.interpolate(limit=24*7)
    temp = tmp.loc[dates].values

    for i in progressbar.progressbar(range(len(dates))):
        x = depth_cor[i, :].astype(float)
        y = temp[i, :].astype(float)
        ind_no_nan = ~np.isnan(x + y)
        x = x[ind_no_nan]
        y = y[ind_no_nan]
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        if (len(x) < 2 or np.min(np.abs(x - depth)) > min_diff_to_depth)  :
            continue
        f = interp1d(x, y, kind, fill_value="extrapolate")
        df_interp.iloc[i, 1] = np.min(f(depth), 0)

    if df_interp.iloc[:5, 1].std() > 0.1:
        df_interp.iloc[:5, 1] = np.nan
    # df_interp['temperatureObserved']  = df_interp['temperatureObserved'].interpolate(limit=24*7).values
    if plot:
        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%Y-%m')
        
        for i in range(len(depth_cor[0, :])-1,0,-1):
            if all(np.isnan(depth_cor[:, i])):
                continue
            else:
                break
        if len(surface_height)==0:
            surface_height = depth_cor[:, i]-depth_cor[:, i][np.isfinite(depth_cor[:, i])][0]

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(17,6))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.8)           
        ax1.plot(dates, surface_height, color='black',  linewidth=3)
        for i in range(np.shape(depth_cor)[1]):
            ax1.plot(dates, -depth_cor[:, i] + surface_height)

        ax1.plot(dates, surface_height-10, color="red", linewidth=5)
        ax1.set_ylim(np.nanmin(surface_height)*1.1-10, 
                     np.nanmax(surface_height)*1.1)
        ax1.set_xlim(min(dates), max(dates))
        ax1.set_ylabel("Height (m)")
        ax1.xaxis.set_major_formatter(myFmt)
        ax1.tick_params(axis='x', rotation=45)    
        
        for i in range(np.shape(depth_cor)[1]):
            ax2.plot(dates, temp[:, i])                    
        ax2.plot(dates, df_interp["temperatureObserved"], marker="o",markersize=5, color='red', linestyle=None)
        ax2.set_ylabel("Firn temperature (degC)")
        ax2.set_ylim(np.nanmin(temp)*1.2, min(1,0.8*np.nanmax(temp)))
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.tick_params(axis='x', rotation=45)    
        ax2.axes.grid()
        ax2.set_xlim(min(dates), max(dates))
        
        fig.suptitle(title)  # or plt.suptitle('Main title')
        im = plt.imread('figures/legend_1.png') # insert local path of the image.
        newax = fig.add_axes([0.15,0.8,0.2,0.2], anchor='NW', zorder=0)
        newax.imshow(im)
        newax.axes.xaxis.set_visible(False)
        newax.axes.yaxis.set_visible(False)        
        fig.savefig('figures/string processing/interp_'+title+'.png', dpi=300)
    return df_interp

    
# %%
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
    plt.xlim(0, 60)
    return s


#%% Loading metadata, RTD and sonic ranger
def load_metadata(filepath, sites):
    CVNfile = tb.open_file(filepath, mode="r", driver="H5FD_CORE")
    datatable = CVNfile.root.FirnCover

    statmeta_df = pd.DataFrame.from_records(
        datatable.Station_Metadata[:].tolist(),
        columns=datatable.Station_Metadata.colnames,
    )
    statmeta_df.sitename = statmeta_df.sitename.str.decode("utf-8")
    statmeta_df.iridium_URL = statmeta_df.iridium_URL.str.decode("utf-8")
    statmeta_df["install_date"] = pd.to_datetime(
        statmeta_df.installation_daynumer_YYYYMMDD.values, format="%Y%m%d"
    )
    statmeta_df["rtd_date"] = pd.to_datetime(
        statmeta_df.RTD_installation_daynumber_YYYYMMDD.values, format="%Y%m%d"
    )
    zz = []
    for ii in range(len(statmeta_df.RTD_depths_at_installation_m[0])):
        st = "rtd%s" % ii
        zz.append(st)
    zz = np.flip(zz)

    statmeta_df[zz] = pd.DataFrame(
        statmeta_df.RTD_depths_at_installation_m.values.tolist(),
        index=statmeta_df.index,
    )
    statmeta_df.set_index("sitename", inplace=True)
    statmeta_df.loc["Crawford", "rtd_date"] = statmeta_df.loc[
        "Crawford", "install_date"
    ]
    statmeta_df.loc["NASA-SE", "rtd_date"] = statmeta_df.loc[
        "NASA-SE", "install_date"
    ] - pd.Timedelta(days=1)

    # Meteorological_Daily to pandas
    metdata_df = pd.DataFrame.from_records(datatable.Meteorological_Daily[:])
    metdata_df.sitename = metdata_df.sitename.str.decode("utf-8")
    metdata_df["date"] = pd.to_datetime(
        metdata_df.daynumber_YYYYMMDD.values, format="%Y%m%d"
    )

    for site in sites:
        msk = (metdata_df["sitename"] == site) & (
            metdata_df["date"] < statmeta_df.loc[site, "rtd_date"]
        )
        metdata_df.drop(metdata_df[msk].index, inplace=True)
        if site == "NASA-SE":
            # NASA-SE had a new tower section in 5/17; distance raised is ??, use 1.7 m for now.
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-10")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
            #     metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 1.7
            # )
            m3 = (
                (metdata_df["sitename"] == site)
                & (metdata_df["date"] > "2017-02-12")
                & (metdata_df["date"] < "2017-04-12")
            )
            metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
        # elif site == "Crawford":
            # Crawford has bad sonic data for 11/3/17 to 2/16/18
            # m2 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2017-11-03")
            #     & (metdata_df["date"] < "2018-02-16")
            # )
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = np.nan
        if site == "EKT":
            # EKT had a new tower section in 5/17; distance raised is 0.86 m.
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-05")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
            #     metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.86
            # )
            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2018-05-15")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.5
            )
        # elif site == "Saddle":
        #     # Saddle had a new tower section in 5/17; distance raised is 1.715 m.
        #     m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-07")
            # metdata_df.loc[m2,'sonic_range_dist_corrected_m']=metdata_df.loc[m2,'sonic_range_dist_corrected_m']-1.715
        # elif site == "EastGrip":
            # Eastgrip has bad sonic data for 11/7/17 onward
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-11-17")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = np.nan
            # m3 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2015-10-01")
            #     & (metdata_df["date"] < "2016-04-01")
            # )
            # metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
            # m4 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2016-12-07")
            #     & (metdata_df["date"] < "2017-03-01")
            # )
            # metdata_df.loc[m4, "sonic_range_dist_corrected_m"] = np.nan
        if site == "DYE-2":
            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2016-04-29")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.3
            )
            # m3 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2015-12-24")
            #     & (metdata_df["date"] < "2016-05-01")
            # )
            # metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
    #         m4 = (metdata_df['sitename']==site)&(metdata_df['date']>'2016-12-07')&(metdata_df['date']<'2017-03-01')
    #         metdata_df.loc[m4,'sonic_range_dist_corrected_m']=np.nan

    metdata_df.reset_index(drop=True)
    # metdata_df.set_index(['sitename','date'],inplace=True)

    sonic_df = metdata_df[
        ["sitename", "date", "sonic_range_dist_corrected_m"]
    ].set_index(["sitename", "date"])
    sonic_df.columns = ["sonic_m"]
    sonic_df.sonic_m[sonic_df.sonic_m < -100] = np.nan
    sonic_df.loc["Saddle", "2015-05-16"] = sonic_df.loc["Saddle", "2015-05-17"]

    # filtering
    gradthresh = 0.1

    for site in sites:
        if site in ["Summit", "NASA-SE"]:
            tmp = 0
        else:
            # applying gradient filter on KAN-U, Crawford, EwastGRIP, EKT, Saddle and Dye-2
            vals = sonic_df.loc[site, "sonic_m"].values
            vals[np.isnan(vals)] = -9999
            msk = np.where(np.abs(np.gradient(vals)) >= gradthresh)[0]
            vals[msk] = np.nan
            vals[msk - 1] = np.nan
            vals[msk + 1] = np.nan
            vals[vals == -9999] = np.nan
            sonic_df.loc[site, "sonic_m"] = vals
        sonic_df.loc[site, "sonic_m"] = (
            sonic_df.loc[site].interpolate(method="linear").values
        )
        sonic_df.loc[site, "sonic_m"] = smooth(sonic_df.loc[site, "sonic_m"].values)

    for site in sonic_df.index.unique(level="sitename"):
        dd = statmeta_df.loc[site]["rtd_date"]
        if site == "Saddle":
            dd = dd + pd.Timedelta("1D")
        sonic_df.loc[site, "delta"] = (
            sonic_df.loc[[site]].sonic_m - sonic_df.loc[(site, dd)].sonic_m
        )

    rtd_depth_df = statmeta_df[zz].copy()
    zz2 = [
        "depth_0",
        "depth_1",
        "depth_2",
        "depth_3",
        "depth_4",
        "depth_5",
        "depth_6",
        "depth_7",
        "depth_8",
        "depth_9",
        "depth_10",
        "depth_11",
        "depth_12",
        "depth_13",
        "depth_14",
        "depth_15",
        "depth_16",
        "depth_17",
        "depth_18",
        "depth_19",
        "depth_20",
        "depth_21",
        "depth_22",
        "depth_23",
    ]
    zz2 = np.flip(zz2)
    rtd_depth_df.columns = zz2

    xx = statmeta_df.RTD_top_usable_RTD_num
    for site in sites:
        vv = rtd_depth_df.loc[site].values
        ri = np.arange(xx.loc[site], 24)
        vv[ri] = np.nan
        rtd_depth_df.loc[site] = vv
    rtd_d = sonic_df.join(rtd_depth_df, how="inner")
    rtd_dc = rtd_d.copy()
    rtd_dep = rtd_dc[zz2].add(-rtd_dc["delta"], axis="rows")

    rtd_df = pd.DataFrame.from_records(
        datatable.Firn_Temp_Daily[:].tolist(),
        columns=datatable.Firn_Temp_Daily.colnames,
    )
    rtd_df.sitename = rtd_df.sitename.str.decode("utf-8")
    rtd_df["date"] = pd.to_datetime(rtd_df.daynumber_YYYYMMDD.values, format="%Y%m%d")
    rtd_df = rtd_df.set_index(["sitename", "date"])

    rtd_df[zz] = pd.DataFrame(
        rtd_df.RTD_temp_avg_corrected_C.values.tolist(), index=rtd_df.index
    )

    # filtering
    rtd_df.replace(-100.0, np.nan, inplace=True)

    for i in range(0, 4):
        vals = rtd_df.loc["Crawford", zz[i]].values
        vals[vals > -1] = np.nan
        rtd_df.loc["Crawford", zz[i]] = vals
    rtd_df = rtd_df.join(rtd_dep, how="inner").sort_index(axis=0)
    for site in sites:
        rtd_df.loc[site, zz][:14] = np.nan
    return statmeta_df, sonic_df, rtd_df, rtd_dep, metdata_df


#%%
def smooth(x, window_len=14, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int(window_len / 2 - 1) : -int(window_len / 2)]


#%%
def interp_gap(data, gap_size):
    mask = data.copy()
    grp = (mask.notnull() != mask.shift().notnull()).cumsum()
    grp["ones"] = 1
    for i in list("abcdefgh"):
        mask[i] = (grp.groupby(i)["ones"].transform("count") < gap_size) | data[
            i
        ].notnull()
    return mask


#%%
def hampel(vals_orig, k=7, t0=3):
    """
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    """
    # Make copy so original not edited
    vals = vals_orig.copy()
    # Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median - vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return vals
