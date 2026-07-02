import datetime as dt
import os
import sys

import numpy as np
import xarray as xr
from onset_utils import onset_period_1d

print("imports done")
max_periods = 2
outdir = (
    "/home/users/franmorr/hk26/hackathon-monsoons/onset_identification/onset_dates/"
)


outfile = f"{outdir}/chirps_2020_dwtps.nc"
if os.path.exists(outfile):
    print(f"{outfile} exists, skipping...")
pr_latlon = xr.open_dataset(
    "/home/users/franmorr/firstrains/OnsetPeriod/data/CHIRPS/CHIRPS_precip.day.total_1981-2024.025dg.aw.nc"
).total_precipitation.sel(time=slice("2020-02-01", "2021-03-31"))
pr_latlon = pr_latlon.rename({"latitude": "lat", "longitude": "lon"})

print("calculating onset")
first_days, last_days = xr.apply_ufunc(
    onset_period_1d,
    pp_latlon,
    input_core_dims=[["time"]],
    output_core_dims=[["period"], ["period"]],
    kwargs={
        "max_periods": max_periods,
        "max_dry_frac_rainfall": 0.1,
        "refine": True,
        "deltat": int(pp_latlon.time.dt.dayofyear[0].item()),
    },
    vectorize=True,
    dask="parallelized",
    # output_dtypes=[float, float],
    dask_gufunc_kwargs={
        "output_sizes": {"period": max_periods},
        "meta": (np.array((), dtype=float), np.array((), dtype=float)),
    },
)

first_days = first_days.assign_coords(period=np.arange(max_periods))
last_days = last_days.assign_coords(period=np.arange(max_periods))
first_days = first_days.rename("first_day_of_period")
last_days = last_days.rename("last_day_of_period")
dwtps = xr.merge([first_days, last_days])

print(f"saving to {outfile}")
dwtps.to_netcdf(outfile)