"""utils.py

Code extracted from https://github.com/BMaybee/East_Atlantic_vortices/blob/b885038517f5e0698fdcdef68790b6539093423c/get_mean_state.py
and https://github.com/BMaybee/East_Atlantic_vortices/blob/b885038517f5e0698fdcdef68790b6539093423c/composite_sampling.py
and https://github.com/digital-earths-UK-hackathon/hk26/blob/main/notebooks/utils.py

Contains useful scripts for handling kscale data.
"""

import math as maths

import cartopy.crs as ccrs
import easygems.healpix as egh
import healpy as hp
import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

domains = {
    "global": {"lonmin": -180, "lonmax": 180, "latmin": -90, "latmax": 90},
    "SAm": {"lonmin": -80, "lonmax": -30, "latmin": -40, "latmax": 10},
    "SAf": {"lonmin": 10, "lonmax": 55, "latmin": -40, "latmax": 0},
    "WAf": {"lonmin": -30, "lonmax": 30, "latmin": -5, "latmax": 25},
    "SAs": {"lonmin": 60, "lonmax": 100, "latmin": 0, "latmax": 35},
    "Ind": {"lonmin": 69, "lonmax": 89, "latmin": 8, "latmax": 29},
    "EAs": {"lonmin": 90, "lonmax": 140, "latmin": 0, "latmax": 50},
    "Aus": {"lonmin": 110, "lonmax": 160, "latmin": -30, "latmax": 0},
}

sims = [
    "um_glm_n2560_RAL3p3_tuned_hk26",
    "icon_d3hp003",
    "casesm2_10km_nocumulus",
    "nicam_gl11",
    "ifs_tco3999-ng5_rcbmf_cf",
]
sims_new = [
    "um_glm_n2560_RAL3p3_tuned_hk26",
    "icon_d3hp003",
    "casesm2_10km_nocumulus",
    "nicam_gl11",
    "ifs_tco3999-ng5_rcbmf",
]
um_sims = [
    "um_glm_n2560_RAL3p3_tuned_hk26",
    "um_glm_n1280_GAL9_v2_hk26",
    "um_glm_n2560_CoMA9_hk26",
    "um_glm_n1280_CoMA9_hk26",
]
rainfall_obs = ["IMERG_IR", "CHIRPS"]
sim_labels = {
    "um_glm_n2560_RAL3p3_tuned_hk26": "UM-RAL3",
    "icon_d3hp003": "ICON",
    "casesm2_10km_nocumulus": "CAS-ESMv2",
    "nicam_gl11": "NICAM",
    "ifs_tco3999-ng5_rcbmf_cf": "IFS",
}

sim_specific_labels = {
    "um_glm_n2560_RAL3p3_tuned_hk26": "UM-RAL3-5km",
    "um_glm_n1280_GAL9_v2_hk26": "UM-GAL9-10km",
    "um_glm_n2560_CoMA9_hk26": "UM-CoMA9-5km",
    "um_glm_n1280_CoMA9_hk26": "UM-CoMA9-10km",
    "icon_d3hp003": "ICON-atmos only",
    "icon_ngc4008": "ICON-nextGEMS",
    "casesm2_10km_nocumulus": "CAS-ESMv2",
    "nicam_gl11": "NICAM",
    "ifs_tco3999-ng5_rcbmf_cf": "IFS-rbcmf",
    "ifs_tco3999-ng5_rcbmf": "IFS-rbcmf",
    "ifs_tco3999-ng5_deepoff": "IFS-deepoff",
}
sim_colors = {
    "um_glm_n2560_RAL3p3_tuned_hk26": "xkcd:light olive green",
    "um_glm_n1280_GAL9_v2_hk26": "xkcd:olive green",
    "um_glm_n2560_CoMA9_hk26": "xkcd:dark sea green",
    "um_glm_n1280_CoMA9_hk26": "xkcd:dark green",
    "icon_d3hp003": "xkcd:dark cyan",
    "icon_ngc4008": "xkcd:dark aqua",
    "casesm2_10km_nocumulus": "xkcd:terra cotta",
    "nicam_gl11": "xkcd:medium purple",
    "ifs_tco3999-ng5_rcbmf_cf": "xkcd:cerulean",
    "ifs_tco3999-ng5_rcbmf": "xkcd:cerulean",
    "ifs_tco3999-ng5_deepoff": "xkcd:royal blue",
}


def relon(ds):
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))


def hp_to_latlon(ds, zoom, regional=False):
    if regional:
        import ast

        domain_bounds = ast.literal_eval(ds.attrs.get("regional_bounds"))
        lon1 = domain_bounds["lower_left_lon"]
        lon2 = domain_bounds["upper_right_lon"]
        lat1 = domain_bounds["lower_left_lat"]
        lat2 = domain_bounds["upper_right_lat"]
    else:
        lon1 = 0
        lon2 = 360
        lat1 = -90
        lat2 = 90

    # Call function to compute nlon, nlat based on zoom level
    nlon, nlat = healpix_zoom_to_grid_area_match(zoom)  # , lat1, lat2, lon1, lon2)
    lons = np.linspace(lon1, lon2, nlon)
    lats = np.linspace(lat1, lat2, nlat)

    # Get healpix_index coord corresponding to lat/lon mesh
    idx = get_nn_lon_lat_index(2**zoom, lons, lats)

    return ds.sel(cell=idx)


def healpix_zoom_to_grid_area_match(zoom):
    nside = 2.0**zoom

    # Healpix pixel area
    pixel_area = 4 * np.pi / (12 * nside**2)

    # Angular resolution
    theta = np.sqrt(pixel_area)

    # Compute nlon, nlat
    nlon = int(round(2 * np.pi / theta))
    nlat = int(round(np.pi / theta))

    return nlon, nlat


def get_nn_lon_lat_index(nside, lons, lats):
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("latitude", lats), ("longitude", lons)],
    )


def hp_mods(ds):
    """Convert from CF-compliant to be compatible with egh, and attach lat/lon coords"""
    return ds.rename({"healpix_index": "cell"}).pipe(egh.attach_coords)


def ifs_hp_mods(ds):
    return (
        ds.assign_coords({"cell": ds.value})
        .rename({"value": "cell"})
        .set_xindex("cell")
        .pipe(egh.attach_coords)
    )


def plot_all_fields(ds_plot):
    """Plot all fields for a given dataset. Assumes that each field is 2D - i.e. sel(time=..., [pressure=...]) has been applied"""
    zoom = ds_plot.crs.attrs["refinement_level"]
    projection = ccrs.Robinson(central_longitude=0)
    # Do not plot orog, land surf.
    plot_vars = [
        (name, da)
        for name, da in ds_plot.data_vars.items()
        if name not in {"orog", "sftlf", "weights"}
    ]
    rows = maths.ceil(len(plot_vars) / 5)
    fig, axes = plt.subplots(
        rows,
        5,
        figsize=(30, rows * 20 / 6),
        subplot_kw={"projection": projection},
        layout="constrained",
    )
    if "pressure" in ds_plot.coords:
        plt.suptitle(f"{ds_plot.simulation} z{zoom} @{float(ds_plot.pressure)}hPa")
    else:
        plt.suptitle(f"{ds_plot.simulation} z{zoom}")

    for ax, (name, da) in zip(axes.flatten(), plot_vars):
        if name == "mrsol":
            da = da.isel(depth=0)
            name = "mrsol@depth=0"
        time = pd.Timestamp(ds_plot.time.values.item())

        if abs(da.max() + da.min()) / (da.max() - da.min()) < 0.5:
            # data looks like it needs a diverging cmap.
            # figure out some nice bounds.
            pl, pu = np.percentile(da.values[~np.isnan(da.values)], [2, 98])
            vmax = np.abs([pl, pu]).max()
            kwargs = dict(
                cmap="bwr",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            kwargs = {}
        ax.set_title(f"time: {time} - {name}")
        ax.set_global()
        im = egh.healpix_show(da, ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f"{long_name} ({da.attrs.get('units', '-')})")
        ax.coastlines()


def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# def haversine(lon1, lat1, lon2, lat2):
#     """
#     Calculate the great circle distance in kilometers between two points
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.asin(np.sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
#     return c * r


# function to convert TOA OLR to empirically corrected brightness temperature (Yang and Slingo, 2001)
def tb_from_olr(OLR):
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8  # W m^-2 K^-4
    tf = (OLR / sigma) ** 0.25
    Tb = (-a + np.sqrt(a**2 + 4 * b * tf)) / (2 * b)
    return Tb


###################
# Function to load any native grid DY3 UM pp file.
#    - date = any format datesamp object. Pd is easiest
#    - var_name = STASH diagnostic name as listed in iris variable list
#    - sim = one of the available simulations. See list in function, can be extended easily (manual)
#    - relon = native grid longitude is 0-360, which is naff for East Atlantic. If True, relabel as -180-180
#    - stream = which pp file from a given day to load. There are 4 main options, a--d, with c/d holding pressure-level data.
#    - p = pressure range. If loading a/b streams don't need to do anything, applied only if stream=c/d
#    - minlon/maxlon/minlat/maxlat - variables to define cropping to a particular region
#    - coarsen - if false, do not interpolate to a coarser grid; if not false, can be set to the degree resolution to coarsen to
###################
def load_file(
    date,
    var_name,
    sim="n1280_GAL9",
    relon=True,
    stream="c",
    p=slice(None, None),
    minlon=-180,
    maxlon=180,
    minlat=-90,
    maxlat=90,
    coarsen_grid=False,
    coarsen_timestep=1,
):
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)
    # data in 12 hour chunks, labelled 0 or 12, so get appropriate value.
    tidx = int(date.hour / 12) * 12

    if sim == "CTC_km4p4_RAL3P3":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_data/10km-GAL9-nest/CTC_km4p4_RAL3P3/field.pp/apver{}.pp/CTC_km4p4_RAL3P3.n1280_GAL9_nest.apver{}_{}T{:02d}.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[-1]
        ).sel(latitude=slice(minlat, maxlat))
    elif sim == "n2560_RAL3p3":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3.apver{}_{}T{:02d}.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[-1]
        ).sel(latitude=slice(minlat, maxlat))
    elif sim == "n2560_RAL3p3_tuned":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/5km-RAL3p3-tuned/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3_tuned.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[0]
        ).sel(latitude=slice(minlat, maxlat))
    elif sim == "n1280_10km-CoMA9":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-CoMA9/glm/field.pp/apver{}.pp/glm.n1280_CoMA9_v2.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[0]
        ).sel(latitude=slice(minlat, maxlat))
    elif sim == "n1280_GAL9":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-GAL9/glm/field.pp/apver{}.pp/glm.n1280_GAL9_v2.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[-1]
        ).sel(latitude=slice(minlat, maxlat))

    if stream == "c" or stream == "d":
        ds = ds.sel(pressure=p)

    if relon:
        ds = (
            ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            .sortby("longitude")
            .sel(longitude=slice(minlon, maxlon))
        )
    if coarsen_grid != False:
        # optional coarsening step; currently specified just for vertical profiles
        # spatial coarsening defined by degree spacing specified by coarsen_grid
        # temporal coarsening defined by hour spacing specified by coarsen_timestep
        # TODO: change this to be area-weighted regridding probably
        lons, lats = (
            np.arange(minlon, maxlon, coarsen),
            np.arange(minlat, maxlat, coarsen_grid),
        )
        if stream == "c" or stream == "d":
            ds = (
                ds[ds.time.dt.hour.isin(np.arange(0, 24, coarsen_timestep))]
                .interp(longitude=lons)
                .interp(latitude=lats)
            )
        else:
            ds = ds.interp(longitude=lons).interp(latitude=lats)
    return ds