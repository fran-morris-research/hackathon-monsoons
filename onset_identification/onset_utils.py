import os
import sys
import datetime as dt
import cmocean as cmo

import cartopy.crs as ccrs
import easygems.healpix as egh
import intake
import matplotlib.pyplot as plt
import numpy as np

# import healpy as hp
import xarray as xr

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Filter out annoying warning.
import warnings

from utils import hp_mods, get_nn_lon_lat_index

warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

import OnsetPeriod_toolbox as optb


def onset_period(
    pp,
    deltat=None,
    max_dry_frac_rainfall=0.1,
    refine=False,
    precip_threshold=None,
    intensity_threshold=None,
    fwin=60,
    minlen=30
):
    if deltat == None:
        deltat = int(pp.time.dt.dayofyear[0].item())
    # step 1
    flt = optb.butterworth(pp, fwin)

    # step 2
    grd = np.gradient(flt)

    # step 3
    rny = xr.where(((flt > 0) & (grd > 0)), 1, 0)
    fd, lns = optb.FindOnsetPeriods(rny, minlen)
    # print("original onsets at",fd)
    # print("original lengths",lns)
    # secret step 3.5 - if there aren't any onset periods, just return nans.
    if fd==[]:
        # print("no onsets found")
        return np.array([np.nan]), np.array([np.nan])
        
    # step 4: refine
    if refine:
        for ons_ix in range(len(fd)):
            this_pp = pp[fd[ons_ix] : fd[ons_ix] + lns[ons_ix]]
            this_flt = flt[fd[ons_ix] : fd[ons_ix] + lns[ons_ix]]
            new_fd, new_ln=optb.RefineOns(
                this_pp, this_flt.max() * max_dry_frac_rainfall
            )
            fd[ons_ix]+=new_fd
            lns[ons_ix]=new_ln

    # print("refined onsets at",fd)
    # print("refined lengths",lns)
    ## on occasion this will return nans, so remove them
    new_fd=[d for d in fd if not np.isnan(d)]
    new_lns=[ln for ln in lns if not np.isnan(ln)]
    fd=new_fd
    lns=new_lns
    
    # secret step 5
    # remove first onset period if it starts at the very beginning of the sim
    if fd and fd[0] < 20:
        fd = fd[1:]
        lns = lns[1:]

    # print("double refined onsets at",fd)
    # print("double refined lengths",lns)
    
    if precip_threshold:
        if pp.mean(dim="time") < precip_threshold:
            # print("pixel does not pass precip test")
            return np.array([np.nan]), np.array([np.nan])

    if intensity_threshold:
        if "%" in str(intensity_threshold):
            percentile = float(intensity_threshold.split("%")[0])
            intensity_threshold=np.percentile(flt,percentile)
        elif type(intensity_threshold)==str:
            raise NotImplementedError("Only absolute numerical values and percentile strings (\"{X}%\" format) have been implemented as intensity threshold quantities. Please use one of these formats.") 
        new_lns = [lns[ons_ix] for ons_ix in range(len(lns)) if (flt[fd[ons_ix] : fd[ons_ix] + lns[ons_ix]]).mean() > intensity_threshold]
        new_fd = [fd[ons_ix] for ons_ix in range(len(fd)) if (flt[fd[ons_ix] : fd[ons_ix] + lns[ons_ix]]).mean() > intensity_threshold]
        # ons_ix=0
        # while ons_ix in range(len(fd)):
        #     this_flt_mean = 
        #     if this_flt_mean < intensity_threshold:
        #         fd.remove(fd[ons_ix])
        #         lns.remove(lns[ons_ix])
        #     else:
        #         ons_ix+=1
        lns=new_lns
        fd=new_fd
        if fd==[]:
            # print("onset periods do not pass intensity test")
            return np.array([np.nan]), np.array([np.nan])

    # print("final onsets",fd,lns)
    # shift back to day of year rather than day of simulation
    first_days = np.array(fd) + deltat
    last_days = np.array(fd) + np.array(lns) + deltat

    return first_days, last_days

def onset_period_1d(
    pp_1d,
    time,
    max_periods=5,
    max_dry_frac_rainfall=0.3,
    refine=False,
    precip_threshold=None,
    intensity_threshold=None,
    fwin=60,
    minlen=30
):
    """
    Apply onset_period to one grid-cell time series.

    Returns fixed-length arrays so xarray can combine outputs.
    """
    pp = xr.DataArray(
        pp_1d,
        dims=["time"],
        coords={"time": time},
    )

    deltat = int(pp.time.dt.dayofyear[0].item())

    first_days, last_days = onset_period(
        pp,
        deltat=deltat,
        max_dry_frac_rainfall=max_dry_frac_rainfall,
        refine=refine,
        precip_threshold=precip_threshold,
        intensity_threshold=intensity_threshold,
        fwin=fwin,
        minlen=minlen,
    )

    
    first_out = np.full(max_periods, np.nan)
    last_out = np.full(max_periods, np.nan)

    n = min(len(first_days), max_periods)
    if n>0:
        first_out[:n] = first_days[:n]
        last_out[:n] = last_days[:n]

    return first_out, last_out

def seasonality_index(pp):
    rbar = pp.sel(time=slice("2020-03", "2021-02")).mean(dim="time")
    x_m = pp.sel(time=slice("2020-03", "2021-02")).resample(time="1ME").mean()
    si = (1 / rbar) * (abs((x_m - (rbar)))).mean(dim="time")
    return si