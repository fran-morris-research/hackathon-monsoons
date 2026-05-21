import os
import glob
import pandas as pd
import xarray as xr

# Scratch output directory
SCRATCH = "//work/scratch-pw5/hjwood/p_budget_terms"

# Open daily and mean velocity files
ufile = f"{SCRATCH}/x_MAM_saf.nc"
vfile = f"{SCRATCH}/y_MAM_saf.nc"

uds = xr.open_dataset(ufile)
vds = xr.open_dataset(vfile)

av_ufile = f"{SCRATCH}/x_MAM_mean_saf.nc"
av_vfile = f"{SCRATCH}/y_MAM_mean_saf.nc"

u_m_ds = xr.open_dataset(av_ufile)
v_m_ds = xr.open_dataset(av_vfile)

if len(uds.time) != len(vds.time):
    raise ValueError("daily u and v files must have the same length")

# Extract the velocities from the netcdf files
u = uds["x_wind"]
v = vds["y_wind"]
u_m = u_m_ds["x_wind"]
v_m = v_m_ds["y_wind"]

# Calculate the eddy terms
u_eddy = (u - u_m)
v_eddy = (v - v_m)
u_sq_eddy = u_eddy**2
v_sq_eddy = v_eddy**2
uv_eddy = u_eddy * v_eddy

# Average over the season
u_eddy_m = u_eddy.mean("time")
v_eddy_m = v_eddy.mean("time")
u_sq_eddy_m = u_sq_eddy.mean("time")
v_sq_eddy_m = v_sq_eddy.mean("time")
uv_eddy_m = uv_eddy.mean("time")

u_eddy_m.to_dataset(name="u_prime").to_netcdf(os.path.join(SCRATCH, "u_prime_MAM_saf.nc"))
v_eddy_m.to_dataset(name="v_prime").to_netcdf(os.path.join(SCRATCH, "v_prime_MAM_saf.nc"))
u_sq_eddy_m.to_dataset(name="u_sq_prime").to_netcdf(os.path.join(SCRATCH, "u_sq_prime_MAM_saf.nc"))
v_sq_eddy_m.to_dataset(name="v_sq_prime").to_netcdf(os.path.join(SCRATCH, "v_sq_prime_MAM_saf.nc"))
uv_eddy_m.to_dataset(name="uv_prime").to_netcdf(os.path.join(SCRATCH, "uv_prime_MAM_saf.nc"))