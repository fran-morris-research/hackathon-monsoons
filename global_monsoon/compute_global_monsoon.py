import datetime as dt
import os
import sys
import warnings

import easygems.healpix as egh
import healpy as hp
import intake
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)
from utils import get_nn_lon_lat_index, haversine, hp_mods, hp_to_latlon

url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
cat = intake.open_catalog(url)["UK"]

sims = [
    # "um_glm_n2560_RAL3p3_tuned_hk26",
    # "ifs_tco3999-ng5_rcbmf_cf",
    # "icon_d3hp003",
    "casesm2_10km_nocumulus",
    "nicam_gl11",
    "arp-gem-2p6km",
]
zoom_prime = 7


def get_proportion(pr, NDJFM=None, MJJAS=None, proportion=0.55, localsummer="lat"):
    if NDJFM == None:
        NDJFM = pr.sel(time=slice("2020-11-01", "2021-03-31")).mean(dim="time")
    if MJJAS == None:
        MJJAS = pr.sel(time=slice("2020-05-01", "2020-09-30")).mean(dim="time")

    if localsummer == "lat":
        mjjas_summer = MJJAS.lat > 0
        ndjfm_summer = NDJFM.lat < 0
    elif localsummer == "rain":
        mjjas_summer = (MJJAS - NDJFM) > 0
        ndjfm_summer = (NDJFM - MJJAS) > 0
    annual_sum_ndjfm_summer = pr[:, ndjfm_summer.values].sum(dim="time")
    ndjfm_sum_ndjfm_summer = pr.sel(time=slice("2020-11-01", "2021-03-31")).sum(
        dim="time"
    )
    annual_sum_mjjas_summer = pr[:, mjjas_summer.values].sum(dim="time")
    mjjas_sum_mjjas_summer = pr.sel(time=slice("2020-05-01", "2020-09-30")).sum(
        dim="time"
    )
    annual_sum = pr.sum(dim="time")
    all_sum_local_summer = ndjfm_sum_ndjfm_summer + mjjas_sum_mjjas_summer
    proportion_local_summer = all_sum_local_summer / annual_sum
    return proportion_local_summer


def restrict_proportion(pr, NDJFM=None, MJJAS=None, proportion=0.55):
    proportion_local_summer = get_proportion(
        pr, NDJFM=NDJFM, MJJAS=MJJAS, proportion=proportion
    )
    restricted_by_proportion = proportion_local_summer > proportion
    return restricted_by_proportion


def get_diff(pr, NDJFM=None, MJJAS=None, lim=2):
    if NDJFM == None:
        NDJFM = pr.sel(time=slice("2020-11-01", "2021-03-31")).mean(dim="time")
    if MJJAS == None:
        MJJAS = pr.sel(time=slice("2020-05-01", "2020-09-30")).mean(dim="time")
    diff = abs(NDJFM - MJJAS)
    return diff


def restrict_diff(pr, NDJFM=None, MJJAS=None, lim=2):
    diff = get_diff(pr, NDJFM=NDJFM, MJJAS=MJJAS, lim=lim)
    return diff > lim


for sim_ix, sim in enumerate(sims):
    sim_cat = cat[sim]
    if "IMERG" in sim:
        zooms = [9]
    elif "ifs" in sim:
        zooms = [7]
    elif "arp-gem-2p6km" in sim:
        zooms = [8]
    else:
        zooms = [zoom_prime]
    for zoom_ix, zoom in enumerate(zooms):
        outdir = f"masks/{sim}_zoom_{zoom}.zarr"
        if os.path.exists(outdir):
            print(f"{outdir} exists - skipping...")
        else:
            print(f"computing monsoon domain for {sim} @ zoom {zoom}")
            if "hk26" in sim:
                ds = sim_cat(zoom=zoom, time="PT1H").to_dask().pipe(hp_mods).pr
            else:
                if "icon" in sim or "nicam" in sim:
                    ds = sim_cat(zoom=zoom).to_dask().pipe(egh.attach_coords).pr
                elif "IMERG" in sim:
                    ds = sim_cat(zoom=9).to_dask().pipe(egh.attach_coords).precipitation
                elif "arp" in sim:
                    ds = sim_cat(zoom=8).to_dask().pipe(egh.attach_coords).pr
                else:
                    ds = (
                        sim_cat(zoom=zoom, time="PT1H")
                        .to_dask()
                        .pipe(egh.attach_coords)
                        .pr
                    )
            # ds_latlon = hp_to_latlon(ds, zoom)
            pr = (
                ds.resample(time="1D")
                .mean()
                .sel(time=slice("2020-03-01", "2021-02-28"))
            )

            pr *= 3600 * 24
            pr["units"] = "mm day-1"
            plot = restrict_proportion(pr, proportion=0.55) & restrict_diff(pr, lim=2)
            plot.to_zarr(outdir)