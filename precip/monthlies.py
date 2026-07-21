import warnings
import os 
import easygems.healpix as egh
import intake
import xarray as xr
from regional_timeseries import get_rainfall
from utils import domains, hp_mods, hp_to_latlon, ifs_hp_mods, relon, sims_new, um_sims

zoom = 7
print("python file open")


def main():
    print("running")
    warnings.filterwarnings(
        "ignore",
        message=".*The return type of `Dataset.dims` will be changed.*",
        category=FutureWarning,
    )

    region="global"
    lonmin = domains[region]["lonmin"]
    lonmax = domains[region]["lonmax"]
    latmin = domains[region]["latmin"]
    latmax = domains[region]["latmax"]
    for sim in um_sims:
        print(sim)
        outdir = f"precip/monthlymeans/{sim}_{region}_zoom_{zoom}.nc"
        if os.path.exists(outdir):
            print(f"{outdir} exists; skipping...")
        else:
            pr = get_rainfall(sim, zoom).resample(time="1ME").mean()
            pr.attrs.pop('hiopy::enable', None)
    
            pr.to_netcdf(outdir)


if __name__ == "__main__":
    main()