""" utils.py

Code extracted from https://github.com/BMaybee/East_Atlantic_vortices/blob/b885038517f5e0698fdcdef68790b6539093423c/get_mean_state.py
and https://github.com/BMaybee/East_Atlantic_vortices/blob/b885038517f5e0698fdcdef68790b6539093423c/composite_sampling.py

Contains useful scripts for handling kscale data.
"""

import numpy as np
import xarray as xr

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.asin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


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
# IMPORTANT: the function crops data to an Atlantic domain internally. Modify this if working elsewhere.
###################
def load_file(
    date, var_name, sim="n1280_GAL9", relon=True, stream="c", p=slice(None, None)
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
        ).sel(latitude=slice(-5, 45))
    elif sim == "n2560_RAL3p3":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3.apver{}_{}T{:02d}.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[-1]
        ).sel(latitude=slice(-5, 45))
    elif sim == "n2560_RAL3p3_tuned":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/5km-RAL3p3-tuned/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3_tuned.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[0]
        ).sel(latitude=slice(-5, 45))
    elif sim == "n1280_10km-CoMA9":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-CoMA9/glm/field.pp/apver{}.pp/glm.n1280_CoMA9_v2.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[0]
        ).sel(latitude=slice(-5, 45))
    elif sim == "n1280_GAL9":
        ds = xr.DataArray.from_iris(
            iris.load(
                "/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-GAL9/glm/field.pp/apver{}.pp/glm.n1280_GAL9_v2.apver{}_{}T{:02d}00Z.pp".format(
                    stream, stream, date_str, tidx
                ),
                var_name,
            )[-1]
        ).sel(latitude=slice(-5, 45))

    if stream == "c" or stream == "d":
        ds = ds.sel(pressure=p)

    if relon:
        ds = (
            ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            .sortby("longitude")
            .sel(longitude=slice(-85, 55))
        )
    if (
        args.coarsen is not None
    ):  # GLOBAL variable. Use for mean state to speed things up significantly
        # For vertical profile data - coarsen to 0.5 degrees and restrict to 6 hourly timesteps (same as TRACK)
        lons, lats = np.arange(-85, 55.5, 0.5), np.arange(-5, 45.5, 0.5)
        if stream == "c" or stream == "d":
            ds = (
                ds[ds.time.dt.hour.isin(np.arange(0, 24, 6))]
                .interp(longitude=lons)
                .interp(latitude=lats)
            )
        else:
            ds = ds.interp(longitude=lons).interp(latitude=lats)
    return ds