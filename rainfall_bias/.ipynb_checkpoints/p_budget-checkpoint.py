from collections import defaultdict
from pathlib import Path
import cartopy.crs as ccrs
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights

# Scratch output directory
SCRATCH = "//work/scratch-pw5/hjwood/p_budget_terms"
Y_OUT = f"{SCRATCH}/z_MAM_mean_saf.nc"

# This is the main location for the simulation output.
dy3dir = Path('/gws/nopw/j04/kscale/DYAMOND3_reruns')

# We ran out of space, so some output is located under scratch.
dy3dir_scratch = Path('/work/scratch-pw5/rwjones/kscale/DYAMOND3_reruns')

def find_sims(dirs):
    """Returns a dict of sims, where values are another dict containing basedir, is_global and on_scratch"""
    sims = {}
    for d in dirs:
        for first_ppa in sorted(d.glob('*/*/field.pp/apvera.pp/*.apvera_20200120T0000Z.pp')):
            simkey = '.'.join(first_ppa.name.split('.')[:-2])
            basedir = first_ppa.parent.parent.parent
            is_global = 'glm' in simkey
            on_scratch = 'gws' not in str(first_ppa)
            sims[simkey] = {'basedir': basedir, 'is_global': is_global, 'on_scratch': on_scratch}
    return sims


sims = find_sims([dy3dir, dy3dir_scratch])

for simkey in sorted(sims):
    siminfo = sims[simkey]
    print(f'{simkey}:')
    for k, v in siminfo.items():
        print(f'  {k:<11} = {v}')

def _parse_date_from_pp_path(path):
    datestr = path.stem.split('.')[-1].split('_')[1]
    if datestr[-1] == 'Z':
        return pd.to_datetime(datestr, format="%Y%m%dT%H%MZ")
    else:
        return pd.to_datetime(datestr, format="%Y%m%dT%H")


def find_dyamond3_pp_dates_to_paths(basedir):
    """Search for pp_paths with a specific date (N.B. filename sensitive)."""
    pp_paths = sorted(basedir.glob('field.pp/apve*/**/*.pp'))
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(dict)
    for path in pp_paths:
        if 'apvere' in path.stem:
            continue
        stream = path.name.split('.')[-2][5]
        dates_to_paths[_parse_date_from_pp_path(path)][stream] = path
    # Only keep completed downloads.
    dates_to_paths = {k: v for k, v in dates_to_paths.items() if len(v) == 4}
    return dates_to_paths

def _parse_date_from_pp_path(path):
    datestr = path.stem.split('.')[-1].split('_')[1]
    if datestr[-1] == 'Z':
        return pd.to_datetime(datestr, format="%Y%m%dT%H%MZ")
    else:
        return pd.to_datetime(datestr, format="%Y%m%dT%H")


def find_dyamond3_pp_dates_to_paths(basedir):
    """Search for pp_paths with a specific date (N.B. filename sensitive)."""
    pp_paths = sorted(basedir.glob('field.pp/apve*/**/*.pp'))
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(dict)
    for path in pp_paths:
        if 'apvere' in path.stem:
            continue
        stream = path.name.split('.')[-2][5]
        dates_to_paths[_parse_date_from_pp_path(path)][stream] = path
    # Only keep completed downloads.
    dates_to_paths = {k: v for k, v in dates_to_paths.items() if len(v) == 4}
    return dates_to_paths

siminfo = sims['glm.n2560_RAL3p3_tuned']
dates_to_paths = find_dyamond3_pp_dates_to_paths(siminfo['basedir'])

start = pd.Timestamp("2020-03-01")
end = pd.Timestamp("2020-05-31")

cubes = []

for dt, paths in sorted(dates_to_paths.items()):
    if start <= dt < end:
        cube = iris.load_cube(str(paths['c']), 'upward_air_velocity')
        cube_crop = cube.extract(
            iris.Constraint(
            pressure=lambda p: 100 <= p <= 1000,
            latitude=lambda lat: -35 <= lat <= -5,
            longitude=lambda lon: 5 <= lon <= 50))
        cubes.append(cube_crop)

combined = iris.cube.CubeList(cubes).concatenate_cube()

for coord_name in ['latitude', 'longitude']:
    coord = combined.coord(coord_name)
    if not coord.has_bounds():
        coord.guess_bounds()
    
time_mean = combined.collapsed('time', iris.analysis.MEAN)

x_da = xr.DataArray.from_iris(time_mean)
x_da.to_netcdf(Y_OUT)
print("Saved:", Y_OUT)