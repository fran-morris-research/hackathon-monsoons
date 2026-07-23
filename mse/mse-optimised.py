import os
import sys
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cmocean as cmo
import easygems.healpix as egh
import intake
import matplotlib.pyplot as plt
import metpy.constants as mpconst
import numpy as np
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(1, "../onset_identification")
from utils import (
    domains,
    get_nn_lon_lat_index,
    haversine,
    hp_mods,
    hp_to_latlon,
    ifs_hp_mods,
    relon,
    sim_colors,
    sim_labels,
    sims,
    sims_new,
    um_sims,
)

print("imports done")


def calculate_mse(
    temperature,
    geopotential,
    specific_humidity,
):
    cp = mpconst.dry_air_spec_heat_press.magnitude
    Lv = mpconst.water_heat_vaporization.magnitude

    return cp * temperature + geopotential + Lv * specific_humidity


def mass_weighted_column_integral(da, sfc_p):
    if da.pressure.units == "hPa" and sfc_p.units == "Pa":
        sfc_p /= 100
        sfc_p.attrs["units"] = "hPa"
    if da.pressure.units == "Pa" and sfc_p.units == "hPa":
        p /= 100
        p.attrs["units"] = "hPa"

    da = da.sortby("pressure")
    g = mpconst.g.magnitude
    p = da["pressure"].values.astype(float)  # 1D mid-level pressures (Pa)
    # build interface/bound values (half-levels)
    mid = 0.5 * (p[1:] + p[:-1])
    bounds = np.concatenate(
        ([p[0] - 0.5 * (p[1] - p[0])], mid, [p[-1] + 0.5 * (p[-1] - p[-2])])
    )

    # finding the higher p_hi and lower p_lo pressure of the bounds
    p_hi = np.maximum(bounds[:-1], bounds[1:])
    p_lo = np.minimum(bounds[:-1], bounds[1:])

    # create 1d das for broadcasting
    p_hi_da = xr.DataArray(p_hi, coords={"pressure": p}, dims=["pressure"])
    p_lo_da = xr.DataArray(p_lo, coords={"pressure": p}, dims=["pressure"])

    # compute effective dp for each layer at every horizontal grid point:
    #             dp_eff = max(0, min(p_hi, p_sfc) - p_lo)
    # -> for each level, if the higher (below) pressure bound is smaller
    #    than (above) the surface pressure, take p_hi-p_lo the normal bound
    # -> if the higher (below) pressure bound is larger than (below) surface
    #    pressure, take p_sfc-p_lo as the band
    # -> if that gives a negative value, p_sfc is smaller than p_lo, meaning
    #    that the entire band is below the surface so we should set it to 0.

    # p_top_da has dim 'pressure' and sfc_p_da has e.g. ('lat','lon')
    dp_eff = (np.minimum(p_hi_da, sfc_p) - p_lo_da).clip(min=0)

    int_da = (da * dp_eff).sum(dim="pressure") / g

    return int_da


def calc_meridional_energy_flux(vH):
    R_e = mpconst.earth_avg_radius.magnitude
    vH_rad = vH.assign_coords({"longitude": np.deg2rad(vH["longitude"])})
    vH_int = vH_rad.integrate(coord="longitude")
    f_total = vH_int * R_e * np.cos(np.deg2rad(vH["latitude"]))
    f_total.name = "meridional_energy_flux"
    f_total.attrs["units"] = "W"
    return f_total


def main():
    era5 = True
    dyamond = False
    if era5:
        era5_path = (
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        )
        full_era5 = xr.open_zarr(
            era5_path, chunks="auto", storage_options=dict(token="anon")
        )
        era5_dyperiod = full_era5.sel(time=slice("2020-03-01", "2021-02-28")).rename(
            {"level": "pressure"}
        )
        # era5_dyperiod=full_era5.sel(time=slice("2020-03-01", "2020-03-08")).rename({"level":"pressure"})
        print("extracting variables")
        va = era5_dyperiod.v_component_of_wind
        ta = era5_dyperiod.temperature
        hus = era5_dyperiod.specific_humidity
        zg = era5_dyperiod.geopotential
        ps = era5_dyperiod.surface_pressure

        del era5_dyperiod

        h = calculate_mse(ta, zg, hus) 
        H = mass_weighted_column_integral(h, ps)
        monthly_H = H.resample(time="1ME").mean()
        print("calculating vH")
        vH = mass_weighted_column_integral(va * h, ps)
        print("calculating energy flux")
        monthly_energy_flux = (
            calc_meridional_energy_flux(vH).resample(time="1ME").mean()
        )

        with xr.set_options(use_flox=False):
            monthly_H = H.resample(time="1ME").mean().rename("column_integrated_mse")
            monthly_energy_flux = (
                calc_meridional_energy_flux(vH).resample(time="1ME").mean()
            ).rename("meridional_mse_flux")

        print("Executing Dask pipeline and writing to disk...")
        xr.save_mfdataset(
            [monthly_H.to_dataset(), monthly_energy_flux.to_dataset()],
            ["mse/era5_mse.nc", "mef/era5_mef.nc"],
        )
    if dyamond:
        url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
        cat = intake.open_catalog(url)["online"]
        zoom = 5
        g = mpconst.g.magnitude

        for sim in sims_new:
            sim_cat = cat[sim]
            print(f"\nProcessing simulation: {sim}")

            H_outfile = f"mse/monthly_{sim}_MSE_zoom_{zoom}.nc"
            MEF_outfile = f"mse/monthly_{sim}_MEF_zoom_{zoom}.nc"

            if os.path.exists(H_outfile) and os.path.exists(MEF_outfile):
                print(f"Outputs for {sim} already exist, skipping...")
                continue

            print("Setting up lazy data pipelines...")

            # --- 1. DATA LOADING BRANCHES ---
            if "hk26" in sim:
                # Select ONLY required variables before applying grid transforms
                raw_3d = sim_cat(zoom=zoom, time="PT3H").to_dask().pipe(hp_mods)
                raw_3d = raw_3d[["va", "ta", "hus", "zg"]].sel(
                    time=slice("2020-03-01", "2021-02-28")
                )

                ds = hp_to_latlon(raw_3d, zoom)
                if max(ds.longitude) > 180:
                    ds = relon(ds).sortby("longitude")

                va, ta, hus, zg = ds.va, ds.ta, ds.hus, ds.zg * g

                raw_ps = (
                    sim_cat(zoom=zoom, time="PT1H")
                    .to_dask()
                    .pipe(hp_mods)["ps"]
                    .sel(time=slice("2020-03-01", "2021-02-28"))
                )
                ps = hp_to_latlon(raw_ps, zoom)
                if max(ps.longitude) > 180:
                    ps = relon(ps).sortby("longitude")

            elif "icon" in sim:
                raw_ds = (
                    sim_cat(zoom=zoom, time="P1D", time_method="mean")
                    .to_dask()
                    .pipe(egh.attach_coords)
                )
                raw_ds = raw_ds[["ps", "va", "ta", "hus", "zg"]].sel(
                    time=slice("2020-03-01", "2021-02-28")
                )

                ds = hp_to_latlon(raw_ds, zoom)
                if max(ds.longitude) > 180:
                    ds = relon(ds).sortby("longitude")

                ps, va, ta, hus, zg = ds.ps, ds.va, ds.ta, ds.hus, ds.zg * g

            elif "nicam" in sim or "cas" in sim:
                raw_3d = (
                    sim_cat(zoom=zoom, time="PT6H").to_dask().pipe(egh.attach_coords)
                )
                raw_3d = raw_3d[["va", "ta", "hus", "zg"]].sel(
                    time=slice("2020-03-01", "2021-02-28")
                )

                ds = hp_to_latlon(raw_3d, zoom)
                if max(ds.longitude) > 180:
                    ds = relon(ds).sortby("longitude")
                ds = ds.rename({"lev": "pressure"})

                va, ta, hus, zg = ds.va, ds.ta, ds.hus, ds.zg * g

                raw_ps = (
                    sim_cat(zoom=zoom, time="PT3H")
                    .to_dask()
                    .pipe(egh.attach_coords)["ps"]
                    .sel(time=slice("2020-03-01", "2021-02-28"))
                )
                if "nicam" in sim:
                    raw_ps = raw_ps.interp_like(ds)
                ps = hp_to_latlon(raw_ps, zoom)
                if max(ps.longitude) > 180:
                    ps = relon(ps).sortby("longitude")

            elif "ifs" in sim:
                zoom = 7
                raw_ds = sim_cat(zoom=zoom, dim="3D").to_dask().pipe(ifs_hp_mods)
                raw_ds = raw_ds[
                    [
                        "v",
                        "t",
                        "q",
                        "z",
                    ]
                ].sel(time=slice("2020-03-01", "2021-02-28"))

                ds = hp_to_latlon(raw_ds, zoom)
                if max(ds.longitude) > 180:
                    ds = relon(ds).sortby("longitude")
                ps = hp_to_latlon(
                    sim_cat(zoom=zoom, dim="2D")
                    .to_dask()
                    .pipe(ifs_hp_mods)
                    .sp.rename("ps"),
                    zoom,
                )

                if max(ps.longitude) > 180:
                    ps = relon(ps).sortby("longitude")
                ds = ds.rename(
                    {
                        "v": "va",
                        "q": "hus",
                        "t": "ta",
                        "z": "zg",
                        "level": "pressure",
                    }
                )
                ps, va, ta, hus, zg = ps, ds.va, ds.ta, ds.hus, ds.zg
                ds.pressure.attrs["units"] = "hPa"

            else:
                raw_ds = sim_cat(zoom=zoom).to_dask().pipe(egh.attach_coords)  # Default
                raw_ds = raw_ds[["ps", "va", "ta", "hus", "zg"]].sel(
                    time=slice("2020-03-01", "2021-02-28")
                )

                ds = hp_to_latlon(raw_ds, zoom)
                if max(ds.longitude) > 180:
                    ds = relon(ds).sortby("longitude")

                ps, va, ta, hus, zg = ds.ps, ds.va, ds.ta, ds.hus, ds.zg * g

            print("Building lazy computation graph...")

            # Local Moist Static Energy (3D)
            h = calculate_mse(ta, zg, hus)

            # Column Integrals (2D)
            H_2d = mass_weighted_column_integral(h, ps)
            vH_2d = mass_weighted_column_integral(va * h, ps)

            with xr.set_options(use_flox=False):
                monthly_H = (
                    H_2d.resample(time="1ME").mean().rename("column_integrated_mse")
                )
                monthly_energy_flux = (
                    calc_meridional_energy_flux(vH_2d).resample(time="1ME").mean()
                ).rename("meridional_mse_flux")

            print("Executing Dask pipeline and writing to disk...")
            xr.save_mfdataset(
                [monthly_H.to_dataset(), monthly_energy_flux.to_dataset()],
                [H_outfile, MEF_outfile],
            )

            print(f"Done processing {sim}!")


if __name__ == "__main__":
    main()