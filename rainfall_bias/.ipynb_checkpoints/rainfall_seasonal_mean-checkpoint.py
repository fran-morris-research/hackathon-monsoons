import intake
import warnings
import xarray as xr
import easygems.healpix as egh

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

utils_path = Path("../utils.py").resolve()

spec = spec_from_file_location("utils", utils_path)
utils = module_from_spec(spec)
spec.loader.exec_module(utils)

hp_mods = utils.hp_mods
plot_all_fields = utils.plot_all_fields


warnings.filterwarnings("ignore", message=".*The return type of `Dataset.dims` will be changed.*", category=FutureWarning)

# Scratch output directory
SCRATCH = "//work/scratch-pw5/hjwood/bias"

# Open catalog and print hk26 available models.
url = 'https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml'
cat = intake.open_catalog(url)['UK']

# Load in the data
GALsim = 'um_glm_n1280_GAL9_v2_hk26'
GALsim_cat = cat[GALsim]
GAL1h = GALsim_cat(zoom=9, time='PT1H').to_dask().chunk({"time": 240}).pipe(hp_mods)

RALsim = 'um_glm_n2560_RAL3p3_tuned_hk26'
RALsim_cat = cat[RALsim]
RAL1h = RALsim_cat(zoom=9, time='PT1H').to_dask().pipe(hp_mods)

COMA1sim = 'um_glm_n1280_CoMA9_hk26'
COMA1sim_cat = cat[COMA1sim]
COMA1_1h = COMA1sim_cat(zoom=9, time='PT1H').to_dask().pipe(hp_mods)

COMA2sim = 'um_glm_n2560_CoMA9_hk26'
COMA2sim_cat = cat[COMA2sim]
COMA2_1h = COMA2sim_cat(zoom=9, time='PT1H').to_dask().pipe(hp_mods)

ds_imerg = cat['IR_IMERG'](zoom=9).to_dask().chunk({"time": 240}).pipe(egh.attach_coords)


GAL_OUT = f"{SCRATCH}/GAL_seasonal_mean_global.nc"
RAL_OUT = f"{SCRATCH}/RAL_seasonal_mean_global.nc"
COMA1_OUT = f"{SCRATCH}/COMA1_seasonal_mean_global.nc"
COMA2_OUT = f"{SCRATCH}/COMA2_seasonal_mean_global.nc"
IMERG_OUT = f"{SCRATCH}/IMERG_seasonal_mean_global.nc"

GAL_mask = GAL1h.sel(time=slice("2020-01-20", "2021-03-31"))
RAL_mask = RAL1h.sel(time=slice("2020-01-20", "2021-03-31"))
COMA1_mask = COMA1_1h.sel(time=slice("2020-01-20", "2021-03-31"))
COMA2_mask = COMA2_1h.sel(time=slice("2020-01-20", "2021-03-31"))
IMERG_mask = ds_imerg.sel(time=slice("2020-01-20", "2021-03-31"))

GAL_m = GAL_mask.pr.groupby("time.season").mean("time")
RAL_m = RAL_mask.pr.groupby("time.season").mean("time")
COMA1_m = COMA1_mask.pr.groupby("time.season").mean("time")
COMA2_m = COMA2_mask.pr.groupby("time.season").mean("time")
IMERG_m = IMERG_mask["precipitation"].groupby("time.season").mean("time")

GAL_m.to_netcdf(GAL_OUT)
RAL_m.to_netcdf(RAL_OUT)
COMA1_m.to_netcdf(COMA1_OUT)
COMA2_m.to_netcdf(COMA2_OUT)
IMERG_m.to_netcdf(IMERG_OUT)

print("Saved:", GAL_OUT)
print("Saved:", RAL_OUT)
print("Saved:", COMA1_OUT)
print("Saved:", COMA2_OUT)
print("Saved:", IMERG_OUT)