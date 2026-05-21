import datetime as dt
import os
import sys

import cartopy.crs as ccrs
import cmocean as cmo
import easygems.healpix as egh
import healpy as hp
import intake
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Filter out annoying warning.
import warnings

from utils import get_nn_lon_lat_index, haversine, hp_mods, hp_to_latlon

warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

import OnsetPeriod_toolbox as optb
from onset_utils import onset_period, onset_period_1d, seasonality_index

print("imports done")
max_periods = 2
outdir = "onset_dates/"
# zooms = [3, 4, 5, 6]
zooms = [7, 8,9,]
# zooms = [3,4]
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)"]
projection = ccrs.PlateCarree()

for sim in ["um_glm_n1280_GAL9_v2_hk26", "um_glm_n2560_RAL3p3_tuned_hk26"]:
    sim_cat = cat[sim]
    fig, axes = plt.subplots(
        len(zooms),
        2,
        # height_ratios=[1] * len(zooms) + [0.1],
        # figsize=(6, 10),
        subplot_kw={"projection": projection},
        layout="constrained",
    )
    label_ix = 0
    for zoom_ix, zoom in enumerate(zooms):
        print(sim,zoom)
        ds = sim_cat(zoom=zoom, time="PT1H").to_dask().pipe(hp_mods)
        ds_latlon = hp_to_latlon(ds)
        pp_latlon = ds_latlon.pr.resample(time="1D").mean().chunk(dict(time=-1))
        pp_latlon*=3600
        pp_latlon.units="mm h-1"
        first_days, last_days = xr.apply_ufunc(
            onset_period_1d,
            pp_latlon,
            pp_latlon["time"],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["period"], ["period"]],
            kwargs={
                "max_periods": max_periods,
                "max_dry_frac_rainfall": 0.1,
                "refine": False,
                "precip_threshold": 0.02,
                "intensity_threshold": None,
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
            dask_gufunc_kwargs={
                "output_sizes": {"period": max_periods},
            },
        )

        first_days = first_days.assign_coords(period=np.arange(max_periods))
        last_days = last_days.assign_coords(period=np.arange(max_periods))

        for ix in range(max_periods):
            label = labels[label_ix]
            ax = axes[zoom_ix, ix]
            ax.set_global()
            ax.coastlines()
            m = (first_days.sel(period=ix) % 365).plot(
                ax=ax, vmin=0, vmax=365, cmap=cmo.cm.phase, add_colorbar=False
            )

            ax.set_title(f"{label} zoom={zoom}")

            first_days.to_netcdf(f"{outdir}/{sim}_zoom_{zoom}_first_days.nc")
            last_days.to_netcdf(f"{outdir}/{sim}_zoom_{zoom}_last_days.nc")
            label_ix += 1

    fig.suptitle(sim)
    fig.colorbar(
        m,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.07,
        shrink=0.7,
        label="day of year",
    )

    plt.savefig(f"{sim}_zoom_{str(zooms)}.png")