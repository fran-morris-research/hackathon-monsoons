import os
import glob
import pandas as pd
import xarray as xr

# Scratch output directory
SCRATCH = "//work/scratch-pw5/hjwood/p_budget_terms"

# Open daily and mean velocity files
ufile = f"{SCRATCH}/x_MAM_saf.nc"
vfile = f"{SCRATCH}/y_MAM_saf.nc"
wfile = f"{SCRATCH}/z_MAM_saf.nc"

uds = xr.open_dataset(ufile)
vds = xr.open_dataset(vfile)
wds = xr.open_dataset(wfile)

av_ufile = f"{SCRATCH}/x_MAM_mean_saf.nc"
av_vfile = f"{SCRATCH}/y_MAM_mean_saf.nc"
av_wfile = f"{SCRATCH}/z_MAM_mean_saf.nc"

u_m_ds = xr.open_dataset(av_ufile)
v_m_ds = xr.open_dataset(av_vfile)
w_m_ds = xr.open_dataset(av_wfile)

if len(uds.time) != len(vds.time):
    raise ValueError("daily u and v files must have the same length")

if len(uds.time) != len(wds.time):
    raise ValueError("daily u and w files must have the same length")

if len(vds.time) != len(wds.time):
    raise ValueError("daily v and w files must have the same length")

# Extract the velocities from the netcdf files
u = uds["x_wind"]
v = vds["y_wind"]
w = wds["upward_air_velocity"]
u_m = u_m_ds["x_wind"]
v_m = v_m_ds["y_wind"]
w_m = w_m_ds["upward_air_velocity"]

# Calculate the eddy terms
w_eddy = (w - w_m)
u_eddy = (u - u_m)
v_eddy = (v - v_m)
w_sq_eddy = w_eddy**2
uw_eddy = u_eddy * w_eddy
vw_eddy = v_eddy * w_eddy

# Average over the season
w_eddy_m = w_eddy.mean("time")
w_sq_eddy_m = w_sq_eddy.mean("time")
uw_eddy_m = uw_eddy.mean("time")
vw_eddy_m = vw_eddy.mean("time")

w_eddy_m.to_dataset(name="w_prime").to_netcdf(os.path.join(SCRATCH, "w_prime_MAM_saf.nc"))
w_sq_eddy_m.to_dataset(name="w_sq_prime").to_netcdf(os.path.join(SCRATCH, "w_sq_prime_MAM_saf.nc"))
uw_eddy_m.to_dataset(name="uw_prime").to_netcdf(os.path.join(SCRATCH, "uw_prime_MAM_saf.nc"))
vw_eddy_m.to_dataset(name="vw_prime").to_netcdf(os.path.join(SCRATCH, "vw_prime_MAM_saf.nc"))