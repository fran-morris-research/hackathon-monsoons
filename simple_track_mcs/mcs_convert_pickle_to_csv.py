import pickle
import sys
import os.path
import datetime
import numpy as np
from pathlib import Path
import pandas as pd
import shutil

# Headers for use in calls to print
h1a='<<<========================================================\n'
h1b='========================================================>>>\n'
h2a='<<<---------------------------\n'
h2b='--------------------------->>>\n'

class ConvertPickleToCSV:
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

        self.convertFiles(self.start_date, self.end_date, self.model_id)

    def __repr__(self):
        return f"ConvertPickleToCSV({self.__dict__})"

    def __str__(self):
        
        if self.verbose == True:
            readable_string = h1a + "Current variables stored in ConvertPickleToCSV() instance:\n\n"
            for key in self.__dict__:
                readable_string += (f"{key} = {self.__dict__[key]}\n")
            readable_string += h1b
            return readable_string
        else:
            return self.__repr__()

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

    def convertFiles(self, start_date: datetime.datetime, end_date, model_id: str):
        """

        Arguments
        ---------
            start_date : datetime.datetime
                start date

        """

        dates = pd.date_range(start=start_date, end=end_date, freq="1ME")
       
        for date in dates:
            print(h1a)
            datestamp = date.strftime("%Y%m")
            print(datestamp)

            fname = Path("/gws/ssde/j25b/kscale/USERS/cscullio/DYAMOND3/data_reruns/simpleTrack") / model_id/ f"tracks_{datestamp}.p"

            with open(fname, "rb") as f:
                data_full = pickle.load(f)

            lon1, lon2 = -180, 180
            lat1, lat2 = -50, 50
            
            region = {"lons": (lon1, lon2), "lats": (lat1, lat2)}
            
            dfs = []
            
            for idx, data in enumerate(data_full):
            
                if data.is_any_in_region(region):
            
                    try:
                        area = data.get_area()              # array
                        times = data.get_times()            # array
                        lats = data.get_centroidLat()       # array
                        lons = data.get_centroidLon()       # array
                        u = data.get_u()                    # array
                        v = data.get_v()                    # array
                        meanBT = data.get_mean_Tbs()        # array
                        minBT = data.get_min_Tbs()          # array
                
                        pr_max = data.get_pr_max()          # array
                        pr_mean = data.get_pr_mean()        # array
                        pr_perc90 = data.get_pr_90perc()        # array
                
                        life = data.get_lifetime()          # 1 value
                        life = [life for i in times]
                
                        storm_id = data.ID                  # 1 value
                        storm_id = [storm_id for i in times]
                
                        start_time = data.get_start_time()  # 1 value
                        start_time = [start_time for i in times]
                
                        end_time = data.get_end_time()      # 1 value
                        end_time = [end_time for i in times]
                
                        df = pd.DataFrame({"storm_id": storm_id,
                                    "life": life,
                                    "area": area,
                                    "Initial_UTC": start_time,
                                    "Final_UTC": end_time,
                                    "lon": lons,
                                    "lat": lats,
                                    "u": u,
                                    "v": v,
                                    "pr_mean": pr_mean,
                                    "pr_max": pr_max,
                                    "pr_perc90": pr_perc90,
                                    "min_BT": minBT,
                                    "mean_BT": meanBT,
                                    "timestamp": times})
                        dfs.append(df)
                    except:
                        pass

            df_combined = pd.concat(dfs)
            # save_file = Path("/gws/ssde/j25a/ncas_climate/vol1/users/ekarl20/data/hackathon_monsoon_data/") / model_id / f"tracks_{datestamp}.csv"
            save_file = Path("/work/scratch-nopw2/franmorr/hk/mcs/") / model_id / f"tracks_{datestamp}.csv"
            df_combined.to_csv(save_file, index=False)
            
            print("Saved to file:")
            print(save_file)
            print(h1b)


if __name__ == "__main__":

    models = ["n1280_CoMA9", "n1280_GAL9_v2", "n2560_CoMA9_hier_v2", "n2560_RAL3p3_tuned"]

    start_date = datetime.datetime(2020,2,1)
    end_date = datetime.datetime(2021,2,2)
    model_id = models[-1]

    for model_id in models[:3]:
        dictionary = {  "start_date": start_date,
                        "end_date": end_date,
                        "model_id": model_id}
    
        aa = ConvertPickleToCSV(**dictionary)
