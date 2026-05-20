import cartopy.crs as ccrs
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import easygems.healpix as egh
degree_sign= u'\N{DEGREE SIGN}'
import datetime
from matplotlib.colors import BoundaryNorm

from utils import hp_mods, plot_all_fields

# Filter out annoying warning.
import warnings
warnings.filterwarnings("ignore", message=".*The return type of `Dataset.dims` will be changed.*", category=FutureWarning)


import pickle
import sys
import os.path
import datetime
import numpy as np
from pathlib import Path
import pandas as pd
import shutil

def format_x_and_y_ticks(ax: plt.axes, xlocs: list, ylocs: list):
    """Format geoaxes ticks.

    Args:
        ax (plt.axes): axes
        xlocs (list): x locations
        ylocs (list): y locations

    Returns:
        ax: axes
    """

    # Make nicely formatted x- and y-ticks
    xlabels = []
    for j in range(len(xlocs)):
        if xlocs[j] > 0 :
            xlabels.append(f"{abs(xlocs[j]):.1f}{degree_sign}E")
        elif xlocs[j] == 0:
            xlabels.append(f"{abs(xlocs[j]):.0f}{degree_sign}")
        else:
            xlabels.append(f"{abs(xlocs[j]):.1f}{degree_sign}W")
            
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabels)

    ylabels = []
    for j in range(len(ylocs)):
        if ylocs[j] < 0 :
            ylabels.append(f"{abs(ylocs[j]):.1f}{degree_sign}S")
        elif ylocs[j] == 0:
            ylabels.append(f"{abs(ylocs[j]):.0f}{degree_sign}")
        else:
            ylabels.append(f"{ylocs[j]:.1f}{degree_sign}N")
    ax.set_yticks(ylocs)
    ax.set_yticklabels(ylabels)
    ax.grid(linewidth=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax

variable_names = {"uwnd": "eastward wind",
                  "vwnd": "northward wind"}

# Headers for use in calls to print
h1a='<<<========================================================\n'
h1b='========================================================>>>\n'
h2a='<<<---------------------------\n'
h2b='--------------------------->>>\n'

class PlotERA5bias:
    """
    Convert pickle files to csv files for storm tracks.
        
    Attributes
    ---------
        self.X : str
            X, "x"
    """

    def __init__(self, **kwargs):
        
        self.__dict__.update(kwargs)
        self.verbose = True

        if variable not in ["uwnd", "vwnd"]:
            raise UserWarning
        
        self.plotBias(self.region, self.model_id)

    def __repr__(self):
        return f"PlotERA5bias({self.__dict__})"

    def __str__(self):
        
        if self.verbose == True:
            readable_string = h1a + "Current variables stored in PlotERA5bias() instance:\n\n"
            for key in self.__dict__:
                readable_string += (f"{key} = {self.__dict__[key]}\n")
            readable_string += h1b
            return readable_string
        else:
            return self.__repr__()
        
    def find_regions(self, region):

        regions = { "West Africa": {"lat_min": 0, "lat_max": 25, "lon_min": -25, "lon_max": 20}, 
            "South Asia": {"lat_min": 0, "lat_max": 35, "lon_min": 50, "lon_max": 110},
            "East Asia": {"lat_min": 15, "lat_max": 50, "lon_min": 95, "lon_max": 150},
            "North America": {"lat_min": 0, "lat_max": 35, "lon_min": -120, "lon_max": -60},
            "Southern Africa": {"lat_min": -35, "lat_max": -5, "lon_min": 5, "lon_max": 50},
            "Maritime Continent": {"lat_min": -30, "lat_max": 20, "lon_min": 100, "lon_max": 160},
            "South America": {"lat_min": -35, "lat_max": 10, "lon_min": -90, "lon_max": -30}}
        
        return regions[region]


    def removeAndMakePath(self, path: str):
        """
        If path exists in current directory,
        remove it and make an empty input_files,
        if it does not exist, make the path.

        Arguments
        ---------
            path : str
                absolute path to directory
        """

        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

    def get_region_mask(self, lat1, lat2, lon1, lon2, ds3h_plot):

        if lon1 < 0:
            region_mask = (ds3h_plot.lat >= lat1) & (ds3h_plot.lat <= lat2) & ((ds3h_plot.lon >= lon1+360) &  (ds3h_plot.lon <= lon2+360) |(ds3h_plot.lon >= lon1) &  (ds3h_plot.lon <= lon2)  )
        else:
            region_mask = (ds3h_plot.lat >= lat1) & (ds3h_plot.lat <= lat2) & (ds3h_plot.lon >= lon1) &  (ds3h_plot.lon <= lon2)

        return region_mask

    def plotBias(self, region: str, model_id: str):
        """

        Arguments
        ---------
            start_date : datetime.datetime
                start date

        """

        era5_var = {"uwnd": "u", 
                    "vwnd": "v"}
        model_var = {"uwnd": "ua",
                     "vwnd": "va"}

        # Open catalog and print hk26 available models.
        url = 'https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml'
        cat = intake.open_catalog(url)['UK']

        # Load specific model
        sim_cat = cat[model_id]

        # Monthly means
        ds3h = sim_cat(zoom=8, time='PT3H').to_dask().pipe(hp_mods)
        ds3h_plot = ds3h.sel(pressure=self.level).resample(time="1D").mean().resample(time="1ME").mean()
        ds3h_plot = ds3h_plot[model_var["uwnd"]]

        ds_era5 = cat['ERA5'](zoom=8).to_dask().pipe(egh.attach_coords)
        ds_era5_monthly_mean = ds_era5[era5_var["uwnd"]].sel(level=self.level, time=slice(datetime.datetime(2020,1,1), datetime.datetime(2021,3,2)))

        region_bounds = self.find_regions(self.region)
        lat1, lat2, lon1, lon2 = region_bounds["lat_min"], region_bounds["lat_max"], region_bounds["lon_min"],  region_bounds["lon_max"]
         
        projection = ccrs.PlateCarree(central_longitude=0)

        fig, axes = plt.subplots(3,5, figsize=(20,15), subplot_kw={'projection': projection})
           
        cmap = plt.get_cmap('PuOr_r')
        v = np.linspace(-14,14, 8)
        norm = BoundaryNorm(v, ncolors=cmap.N, extend="both")
        
        for i, ax in enumerate(axes.ravel()):
            ax.set_global()
            
            monthly_bias = ds3h_plot.isel(time=i) - ds_era5_monthly_mean.isel(time=i)
            region_mask = self.get_region_mask(lat1, lat2, lon1, lon2, monthly_bias)
            
            img = egh.healpix_show(monthly_bias.where(region_mask), ax=ax, cmap=cmap, norm=norm)
            ax.set_title(pd.to_datetime(ds_era5_monthly_mean.isel(time=i).time.values).strftime("%Y-%m"))
        
            ax.set_extent([lon1, lon2, lat1, lat2])
            ax.coastlines()

            xlocs = range(lon1, lon2+1, 15)
            ylocs = range(lat1, lat2+1, 15)
            format_x_and_y_ticks(ax, xlocs, ylocs)
            if i%5 !=0:
                ax.axes.get_yaxis().set_ticklabels([])

        cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.02])   # [left, bottom, width, height]
            
        cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(self.label)

        plt.subplots_adjust(hspace=-0.3)

        plt.suptitle(self.model_id, y=0.85)
            
        plt.savefig(f"figures/era5_bias_{self.region.replace(' ', '_')}_{self.variable}_{self.level}_{self.model_id}.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":

    # models = ["um_glm_n1280_CoMA9_hk26", "um_glm_n1280_GAL9_v2_hk26", "um_glm_n2560_CoMA9_hk26",
    #         "um_glm_n2560_RAL3p3_tuned_hk26"]
    # regions_list = ["West Africa", "South Asia", "East Asia", "North America", "Southern Africa", "Maritime Continent", "South America"]

    import json
    with open(os.path.join(os.path.dirname(__file__), "input_files", "input_file" + sys.argv[1] + ".json")) as json_file: 
        input_data = json.load(json_file) 

    # Get all relevant information from input file
    variable = input_data["variable"]
    region = input_data["region"]
    level = float(input_data["level"])
    label = f"{variable_names[variable].title()} bias to ERA5 at {int(level)} hPa (m/s)"
    model_id = input_data["model"]

    dictionary = {  "label": label,
                    "variable": variable,
                    "region": region,
                    "model_id": model_id,
                    "level": level}

    aa = PlotERA5bias(**dictionary)
