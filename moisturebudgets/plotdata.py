import iris
import iris.coord_categorisation as icc
import numpy as np

fpath=sys.argv[1]
regionname=sys.argv[2]

def pentad(doy,value):
    return (value-1-(value-1)%5)/5
def process_pentad(field):
    icc.add_day_of_year(field,'time')
    icc.add_categorised_coord(field,'pentad','day_of_year',pentad)
    output=field.aggregated_by('pentad',iris.analysis.MEAN)
    return output

[pr,ev,mw,me,ms,mn]=[
    iris.load_cube(fpath+varname+regionanme+'.nc')
    for varname in ['pr','ev','mw','me','ms','mn']
    ]
