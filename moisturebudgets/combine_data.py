import iris
from iris.util import equalise_attributes as ea
import sys

fpath=sys.argv[1]
regionname=sys.argv[2]
pcptype=sys.argv[3]

def process(inname,varname,outname):
    field=iris.load(inname,varname)
    ea(field)
    field=field.concatenate_cube()
    # Convert to mm/day equivalent.
    if (field.units==None): 
        field=field*86400
    else:
        field.convert_units('kg m-2 day-1')
    iris.save(field,outname)

if (pcptype=='stratrain'):
    process(fpath+'*apverb*','stratiform_rainfall_flux',fpath+'pr'+regionname+'.nc')
else:
    process(fpath+'*apverb*','precipitation_flux',fpath+'pr'+regionname+'.nc')
process(fpath+'*apvera*',None,fpath+'ev'+regionname+'.nc')
process(fpath+'*apverb*','easterly_flux',fpath+'me'+regionname+'.nc')
process(fpath+'*apverb*','westerly_flux',fpath+'mw'+regionname+'.nc')
process(fpath+'*apverb*','northerly_flux',fpath+'mn'+regionname+'.nc')
process(fpath+'*apverb*','southerly_flux',fpath+'ms'+regionname+'.nc')
