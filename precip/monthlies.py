import os
import warnings

import easygems.healpix as egh
import intake
import xarray as xr
from regional_timeseries import get_rainfall, get_winds
from utils import domains, hp_mods, hp_to_latlon, ifs_hp_mods, relon, sims_new, um_sims

zoom = 5
print("python file open")


def main():
    print("running")
    warnings.filterwarnings(
        "ignore",
        message=".*The return type of `Dataset.dims` will be changed.*",
        category=FutureWarning,
    )

    region = "global"
    lonmin = domains[region]["lonmin"]
    lonmax = domains[region]["lonmax"]
    latmin = domains[region]["latmin"]
    latmax = domains[region]["latmax"]
    for sim in sims_new:
        print(sim)
        outdir = f"precip/monthlymeans/winds/{sim}_{region}_zoom_{zoom}.nc"
        if os.path.exists(outdir):
            print(f"{outdir} exists; skipping...")
        else:
            ds = get_winds(sim, zoom).resample(time="1ME").mean()
            ds.attrs.pop("hiopy::enable", None)
            for var in ds:
                ds[var].attrs.pop("hiopy::enable", None)
            ds.to_netcdf(outdir)


if __name__ == "__main__":
    main()