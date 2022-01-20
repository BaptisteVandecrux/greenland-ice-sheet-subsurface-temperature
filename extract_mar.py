# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

for year in range(1950,2021):
    print(year)
    try:
        ds = xr.open_dataset(
            "I:\Baptiste\Data\RCM\MAR\MARv3.12_fixed\MARv3.12-10km-daily-ERA5-"
            + str(year)
            + ".nc"
        )
    except:
        print("Cannot read ", year)

    if year == 1950:
        tmp = xr.Dataset(dict(TI1 = ds.TI1[dict(OUTLAY=15)].drop_vars('OUTLAY'),
                           TT = ds.TTZ[dict(ZTQLEV=0)].drop_vars('ZTQLEV')))
    else:
        tmp = xr.concat((tmp,xr.Dataset(dict(TI1 = ds.TI1[dict(OUTLAY=15)].drop_vars('OUTLAY'),
                               TT = ds.TTZ[dict(ZTQLEV=0)].drop_vars('ZTQLEV')))), dim='TIME')

tmp.to_netcdf('MAR_all_year.nc')
