import warnings

import easygems.healpix as egh
import intake
import xarray as xr
from utils import hp_mods, hp_to_latlon, ifs_hp_mods


def get_rainfall(sim, zoom, region=None):
    url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
    cat = intake.open_catalog(url)["online"]
    sim_cat = cat[sim]
    if "hk26" in sim:
        ds = sim_cat(zoom=zoom, time="PT1H").to_dask().pipe(hp_mods).pr
    else:
        if "icon" in sim or "nicam" in sim:
            ds = sim_cat(zoom=zoom).to_dask().pipe(egh.attach_coords).pr
        elif "IMERG" in sim:
            zoom = 9
            ds = sim_cat(zoom=zoom).to_dask().pipe(egh.attach_coords).precipitation
        elif "arp" in sim:
            zoom = 8
            ds = sim_cat(zoom=zoom).to_dask().pipe(egh.attach_coords).pr
        elif "ifs" in sim:
            zoom = 7
            ds = sim_cat(zoom=zoom).to_dask().pipe(ifs_hp_mods).tp
        else:
            ds = sim_cat(zoom=zoom, time="PT1H").to_dask().pipe(egh.attach_coords).pr
    ds = hp_to_latlon(ds, zoom)
    if max(ds.longitude) > 180:
        ds = relon(ds).sortby("longitude")
    return ds


def relon(ds):
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))


domains = {
    "SAm": {"lonmin": -80, "lonmax": -30, "latmin": -40, "latmax": 10},
    "SAf": {"lonmin": 10, "lonmax": 55, "latmin": -40, "latmax": 0},
    "WAf": {"lonmin": -30, "lonmax": 30, "latmin": -5, "latmax": 25},
    "SAs": {"lonmin": 60, "lonmax": 100, "latmin": 0, "latmax": 35},
    "Ind": {"lonmin": 69, "lonmax": 89, "latmin": 8, "latmax": 29},
    "EAs": {"lonmin": 90, "lonmax": 140, "latmin": 0, "latmax": 50},
    "Aus": {"lonmin": 110, "lonmax": 160, "latmin": -30, "latmax": 0},
}

sims = [
    # "um_glm_n1280_GAL9_v2_hk26",
    # "um_glm_n2560_CoMA9_hk26",
    # "um_glm_n1280_CoMA9_hk26",
    "um_glm_n2560_RAL3p3_tuned_hk26",
    "ifs_tco3999-ng5_rcbmf",
    "icon_d3hp003",
    "casesm2_10km_nocumulus",
    "nicam_gl11",
    "arp-gem-2p6km",
    # "IR_IMERG",
]
zoom = 7
print("python file open")


def main():
    print("running")
    warnings.filterwarnings(
        "ignore",
        message=".*The return type of `Dataset.dims` will be changed.*",
        category=FutureWarning,
    )

    for region in domains:
        lonmin = domains[region]["lonmin"]
        lonmax = domains[region]["lonmax"]
        latmin = domains[region]["latmin"]
        latmax = domains[region]["latmax"]
        for sim in sims:
            print(sim)
            outdir = f"precip/{sim}_{region}_zoom_{zoom}.nc"
            pr = (
                get_rainfall(sim, zoom)
                .sel(longitude=slice(lonmin, lonmax), latitude=slice(latmin, latmax))
                .mean(dim=("longitude", "latitude"))
            )
            pr.to_netcdf(outdir)


if __name__ == "__main__":
    main()