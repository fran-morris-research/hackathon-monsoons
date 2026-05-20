import json
import os.path
from glob import glob
import shutil
import numpy as np
import datetime
import shutil
import json

# Headers for use in calls to print
h1a='<<<========================================================\n'
h1b='========================================================>>>\n'
h2a='<<<---------------------------\n'
h2b='--------------------------->>>\n'

def removeAndMakePath(path: str):
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

def countFiles(dir_path: str) -> int:
    """
    Counts how many files are in directory.

    Arguments
    ---------
        dir_path : str
            absolute path to directory

    Returns:
        sum : int
            sum of files in directory
    """

    return sum([len(files) for r, d, files in os.walk(dir_path)])

class GenerateSubmissionFiles:
    """Generate .json files to be used by wind_bias.py

    Attributes
    ----------
    """

    def __init__(self, **kwargs):
        
        self.__dict__.update(kwargs)

        # Make input_files directory
        self.input_file_dir_name = 'input_files'
        self.submission_file_dir_name = 'submission_files'
        self.input_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.input_file_dir_name)
        self.submission_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.submission_file_dir_name)
        
        removeAndMakePath(self.input_file_dir)
        removeAndMakePath(self.submission_file_dir)
    
    def __repr__(self):
        return f"GenerateInputFiles({self.__dict__})"

    def __str__(self):
        
        if self.verbose == True:
            readable_string = h1a + "Current variables stored in GenerateInputFiles() instance:\n\n"
            for key in self.__dict__:
                readable_string += (f"{key} = {self.__dict__[key]}\n")
            readable_string += h1b
            return readable_string
        else:
            return self.__repr__()

    def writeInputFiles(self):
        """
        Checks if data exists and loops over variables and dates
        to write input .json files to be used by calculate_daily_mean.py.
        """

        count = 1

        for v in range(len(self.variables)):
            for i in range(len(self.levels)):
                for m in range(len(self.models)):
                    for r in range(len(self.regions)): 

                        dictionary = { 
                                        "level": self.levels[i],
                                        "region": self.regions[r],
                                        "model": self.models[m],
                                        "variable": self.variables[v]
                                    }
                            
                        with open(os.path.join(self.input_file_dir, f'input_file{count}.json'), 'w') as fp:
                            json.dump(dictionary, fp)
                            print(f"Writing input file for {self.variables[v]} at level {self.levels[i]} for model {self.models[m]} and region {self.regions[r]}.")
                            count = count + 1


    def writeSubmissionFiles(self):
        """
        Counts number of files in input_files/ and writes submission
        file to SLURM. 
        """
        
        no_of_files = countFiles(os.path.join(os.path.dirname(__file__), self.input_file_dir_name))
             
        count = 0
        print("\n")
        print(h1a)

        # Create submission scripts for array files
        # with X files at a time

        sub_string = f"1-{int(no_of_files)}%{int(self.no_files_in_batch)}"
        print(f"Array {sub_string} to be submitted to SLURM.")
        self.submissionFile(sub_string)
        print(h1b)

    def submissionFile(self, sub_string):
        """Writes .sbatch submission file to SLURM on JASMIN.

        Arguments
        -----------
            sub_string : str
                e.g. "1-1851" telling how many input files is to be submitted
        """

        filename = os.path.join(self.submission_file_dir, f"slurm_submission_{sub_string}.sbatch")
        f = open(filename, "w")
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --partition={self.partition}\n")
        f.write(f"#SBATCH --qos={self.qos}\n")
        f.write(f"#SBATCH --job-name={self.job_name}\n")
        f.write(f"#SBATCH --account=ncas_climate\n")
        f.write("#SBATCH -o submission_files/j%A_%a.out\n")
        f.write("#SBATCH -e submission_files/j%A_%a.err\n")
        f.write(f"#SBATCH --array={sub_string}\n")
        f.write(f"#SBATCH --time={self.time_out}\n")
        f.write(f"#SBATCH --mem={self.memory}\n")
        f.write("#SBATCH --mail-user=e.karlowska@reading.ac.uk\n")
        f.write("#SBATCH --mail-type=FAIL\n")
        f.write("#SBATCH --requeue\n")
        f.write("\n")
        f.write("source activate hk26_env")
        f.write("\n")
        f.write("export DIR='/home/users/ekarl20/ncas_postdoc/hackathon-monsoons/era5_bias'\n")
        f.write("echo $DIR\n")
        f.write("\n")
        f.write("python ${DIR}/wind_bias.py ${SLURM_ARRAY_TASK_ID}")
        f.close()




if __name__ == "__main__":

    job_name = "era5bias"
    partition = "standard"
    qos = "short"
    memory = "30000"                     # 1 GB
    time_out = "01:00:00"
    no_files_in_batch = 122
   
    variables = ["vwnd", "uwnd"]
    levels = ["850", "200"]
    regions = ["West Africa", "South Asia", "East Asia", "North America", "Southern Africa", "Maritime Continent", "South America"]
    models =  ["um_glm_n1280_CoMA9_hk26", "um_glm_n1280_GAL9_v2_hk26", "um_glm_n2560_CoMA9_hk26",
            "um_glm_n2560_RAL3p3_tuned_hk26"]

    #-------------------------------------------#

    dictionary = {}
    dictionary["variables"] = variables
    dictionary["levels"] = levels
    dictionary["regions"] = regions
    dictionary["models"] = models
    dictionary["partition"] = partition
    dictionary["memory"] = memory
    dictionary["verbose"] = True
    dictionary["job_name"] = job_name
    dictionary["time_out"] = time_out
    dictionary["no_files_in_batch"] = no_files_in_batch
    dictionary["qos"] = qos

    pressure_extract = []
    for i in range(len(levels)):
        try:
            pressure_extract.append(float(levels[i]))
        except:
            pressure_extract.append(None)
    dictionary["pressure_extract"] = pressure_extract

    sub_files = GenerateSubmissionFiles(**dictionary)
    sub_files.writeInputFiles()
    sub_files.writeSubmissionFiles()

