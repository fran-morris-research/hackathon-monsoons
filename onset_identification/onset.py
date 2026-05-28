import datetime as dt
import os
import sys

# project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# Filter out annoying warning.
import warnings

import cartopy.crs as ccrs
import cmocean as cmo
import easygems.healpix as egh
import healpy as hp
import intake
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import hp_mods, hp_to_latlon

warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

from onset_utils import onset_period_1d

print("imports done")
max_periods = 2
outdir = (
    "/home/users/franmorr/hk26/hackathon-monsoons/onset_identification/onset_dates/"
)
plot = False

if sys.argv[1]!="None":
    zooms = [int(sys.argv[1])]
else:
    zooms = [
        3,
        4,
        5,
        6,
        # 7,
        # 8,
        # 9,
    ]


labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)"]
projection = ccrs.PlateCarree()

# Open catalog.
url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
cat = intake.open_catalog(url)["UK"]

for sim in [
    "um_glm_n1280_GAL9_v2_hk26",
    "um_glm_n2560_RAL3p3_tuned_hk26",
    "um_glm_n2560_CoMA9_hk26",
    "um_glm_n1280_CoMA9_hk26",
]:
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
        print(sim, zoom)
        outfile = f"{outdir}/{sim}_zoom_{zoom}_dwtps.nc"
        if os.path.exists(outfile):
            print(f"{outfile} exists, skipping...")
            if plot:
                dwtps = xr.open_dataset(outfile)
            pass
        else:
            ds = sim_cat(zoom=zoom, time="PT1H").to_dask().pipe(hp_mods)
            ds_latlon = hp_to_latlon(ds, zoom)
            pp_latlon = ds_latlon.pr.resample(time="1D").mean().chunk(dict(time=-1))
            pp_latlon *= 3600
            pp_latlon["units"] = "mm h-1"
            first_days, last_days = xr.apply_ufunc(
                onset_period_1d,
                pp_latlon,
                pp_latlon["time"],
                input_core_dims=[["time"], ["time"]],
                output_core_dims=[["period"], ["period"]],
                kwargs={
                    "max_periods": max_periods,
                    "max_dry_frac_rainfall": 0.1,
                    "refine": True,
                    "precip_threshold": 0.05,
                    "intensity_threshold": "60%",
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
            first_days = first_days.rename("first_day_of_period")
            last_days = last_days.rename("last_day_of_period")
            dwtps = xr.merge([first_days, last_days])
            dwtps.to_netcdf(outfile)

        if plot:
            for ix in range(max_periods):
                label = labels[label_ix]
                ax = axes[zoom_ix, ix]
                ax.set_global()
                ax.coastlines()
                m = (dwtps.first_day_of_period.sel(period=ix) % 365).plot(
                    ax=ax, vmin=0, vmax=365, cmap=cmo.cm.phase, add_colorbar=False
                )

                ax.set_title(f"{label} zoom={zoom}")

                label_ix += 1
    if plot:
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

        plt.savefig(f"images/{sim}_zooms_{''.join([str(zoom) for zoom in zooms])}.png")