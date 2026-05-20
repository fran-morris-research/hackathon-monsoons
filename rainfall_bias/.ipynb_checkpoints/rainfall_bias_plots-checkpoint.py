# Load in all the files

import cartopy.crs as ccrs
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easygems.healpix as egh
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import matplotlib.ticker as ticker
import os
import glob
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import SymLogNorm

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

utils_path = Path("../utils.py").resolve()

spec = spec_from_file_location("utils", utils_path)
utils = module_from_spec(spec)
spec.loader.exec_module(utils)

hp_mods = utils.hp_mods
plot_all_fields = utils.plot_all_fields

# Load in GAL and IMERG global data
folder = "//work/scratch-pw4/hjwood"
fname_GAL = "GAL_seasonal_mean_global.nc"
file_GAL = os.path.join(folder, fname_GAL)
ds_GAL = xr.open_dataset(file_GAL)

fname_IMERG = "IMERG_seasonal_mean_global.nc"
file_IMERG = os.path.join(folder, fname_IMERG)
ds_IMERG = xr.open_dataset(file_IMERG)


# Plot the global GAL data
pr_GAL = ds_GAL.pr *3600

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=True,
)

vmax = float(pr_GAL.max())

for ax, s in zip(axes.flat, pr_GAL.season.values):
    ax.set_global()
    plot = egh.healpix_show(
        pr_GAL.sel(season=s),
        ax=ax,
        cmap="viridis",
        norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
    )
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(str(s), fontsize=25)
    ax.coastlines(color="white")

cbar = fig.colorbar(plot, ax=axes, orientation="horizontal")
cbar.set_label("Precipitation [mm/hr]", fontsize=20)
cbar.ax.tick_params(labelsize=20)
fig.suptitle("GAL", fontsize=30)
plt.savefig("global_GAL.png")
plt.show()

# Plot the global IMERG data
pr_IMERG = ds_IMERG.precipitation

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=True,
)

vmax = float(pr_IMERG.max())

for ax, s in zip(axes.flat, pr_IMERG.season.values):
    ax.set_global()
    plot = egh.healpix_show(
        pr_IMERG.sel(season=s),
        ax=ax,
        cmap="viridis",
        norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
    )
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(str(s), fontsize=25)
    ax.coastlines(color="white")

cbar = fig.colorbar(plot, ax=axes, orientation="horizontal")
cbar.set_label("Precipitation", fontsize=20)
cbar.ax.tick_params(labelsize=20)
fig.suptitle("IMERG", fontsize=30)
plt.savefig("global_IMERG.png")
plt.show()

# Plot global differences
diff = pr_GAL - pr_IMERG

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 8),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=True,
)

limit = np.abs(diff).max()

norm = SymLogNorm(
    linthresh=0.1,   # linear region around zero
    linscale=1,
    vmin=-limit,
    vmax=limit,
    base=10
)

for ax, s in zip(axes.flat, diff.season.values):
    ax.set_global()
    plot = egh.healpix_show(
        diff.sel(season=s),
        ax=ax,
        cmap="coolwarm",
        norm=norm,)
    ax.set_title(str(s), fontsize=25)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.coastlines(color="black")
    ax.set_extent([-180,180,-50,50])

cbar = fig.colorbar(plot, ax=axes, orientation="horizontal")
cbar.set_label("Precipitation [mm/hr]", fontsize=20)
cbar.ax.tick_params(labelsize=20)
fig.suptitle("GAL - IMERG", fontsize=30)
plt.savefig("tropical_pr_bias.png")
plt.show()

# Plot regional 
regions = [
    {"lat_min": 0, "lat_max": 25, "lon_min": -25, "lon_max": 20, "name": "West Africa"},
    {"lat_min": 0, "lat_max": 35, "lon_min": 50, "lon_max": 110, "name": "South Asia"},
    {"lat_min": 15, "lat_max": 50, "lon_min": 95, "lon_max": 150, "name": "East Asia"},
    {"lat_min": 0, "lat_max": 35, "lon_min": -120, "lon_max": -60, "name": "North America"},
    {"lat_min": -35, "lat_max": -5, "lon_min": 5, "lon_max": 50, "name": "Southern Africa"},
    {"lat_min": -30, "lat_max": 10, "lon_min": 100, "lon_max": 160, "name": "Australia"},
    {"lat_min": -35, "lat_max": 10, "lon_min": -90, "lon_max": -30, "name": "South America"},    
]

for r in regions:

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    
    limit = np.abs(diff).max()
    
    norm = SymLogNorm(
        linthresh=0.1,   # linear region around zero
        linscale=1,
        vmin=-limit,
        vmax=limit,
        base=10
    )
    
    for ax, s in zip(axes.flat, diff.season.values):
        ax.set_global()
        plot = egh.healpix_show(
            diff.sel(season=s),
            ax=ax,
            cmap="coolwarm",
            norm=norm,)
        ax.set_title(str(s), fontsize=25)
        ax.coastlines(color="black")
        ax.set_extent([r["lon_min"],r["lon_max"],r["lat_min"],r["lat_max"]])
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    
    cbar = fig.colorbar(plot, ax=axes, orientation="horizontal")
    cbar.set_label("Precipitation [mm/hr]", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    name = r["name"]
    fig.suptitle(f"GAL - IMERG for {name}", fontsize=30)
    plt.savefig(f"{name}_pr_bias.png")
    plt.show()