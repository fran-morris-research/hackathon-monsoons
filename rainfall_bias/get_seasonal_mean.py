import cartopy.crs as ccrs
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easygems.healpix as egh
from utils import hp_mods, plot_all_fields
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import matplotlib.ticker as ticker
import os
import glob
import xarray as xr


warnings.filterwarnings("ignore", message=".*The return type of `Dataset.dims` will be changed.*", category=FutureWarning)

# Scratch output directory
SCRATCH = "//work/scratch-pw4/hjwood"

# Open catalog and print hk26 available models.
url = 'https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml'
cat = intake.open_catalog(url)['UK']

# Load in the data
GALsim = 'um_glm_n1280_GAL9_v2_hk26'
GALsim_cat = cat[GALsim]
GAL1h = GALsim_cat(zoom=9, time='PT1H').to_dask().chunk({"time": 720}).pipe(hp_mods)
ds_imerg = cat['IR_IMERG'](zoom=9).to_dask().chunk({"time": 1440}).pipe(egh.attach_coords)

# Southern Africa Mask
def saf_mask(ds):
    mask = (ds.lon > 0) & (ds.lon < 55) & (ds.lat > -45) & (ds.lat < 5)
    return mask

GAL_OUT = f"{SCRATCH}/GAL_seasonal_mean.nc"
IMERG_OUT = f"{SCRATCH}/IMERG_seasonal_mean.nc"
DIFF_OUT = f"{SCRATCH}/diff_GAL_IMERG.nc"

GAL_mask = GAL1h.where(saf_mask(GAL1h), drop=True)
IMERG_mask = ds_imerg.where(saf_mask(ds_imerg), drop=True)

GAL_m = GAL_mask.pr.groupby("time.season").mean("time")
IMERG_m = IMERG_mask["precipitation"].groupby("time.season").mean("time")

GAL_m.to_netcdf(GAL_OUT)
IMERG_m.to_netcdf(IMERG_OUT)

diff = GAL_m - IMERG_m
diff.to_netcdf(DIFF_OUT)

print("Saved:", GAL_OUT)
print("Saved:", IMERG_OUT)
print("Saved:", DIFF_OUT)