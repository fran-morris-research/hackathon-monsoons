import sys
import iris
import numpy as np

# This code takes an input file and boundaries of a box and calculates the
# moisture budget terms for that file and box and outputs them into a file
# with the same name in a different directory.

#latmin=8.;latmax=29.;lonmin=69.;lonmax=89.
# Maybe there's a neater way of reading all these in.
fstemin=sys.argv[1]
fin=sys.argv[2]
fout=fin[:-3]+'.nc'
fstemout=sys.argv[3]
pcptype=sys.argv[4]
latmin=float(sys.argv[5])
latmax=float(sys.argv[6])
lonmin=float(sys.argv[7])
lonmax=float(sys.argv[8])

# Some constants
rEarth=6.378e6
d2r=np.pi/180
boxarea=rEarth**2*(
    np.sin(latmax*d2r)-np.sin(latmin*d2r)
)*(lonmax*d2r-lonmin*d2r)

# This is used to pick out the mean stratiform rain rather than the
# instantaneous field.
def cube_is_mean(cube):
   return any(cm.method=='mean' for cm in cube.cell_methods)
 
def addbounds(field):
   for lc in 'latitude','longitude':
      if (field.coord(lc).bounds is None):
         field.coord(lc).guess_bounds()

# This takes an average of a field over the region of interest.
def takemean(infield):
   addbounds(infield)
   outfield=infield.intersection(
      longitude=(lonmin,lonmax),latitude=(latmin,latmax)
   )
   outfield=outfield.collapsed(
      ('latitude','longitude'),
      iris.analysis.MEAN,
      weights=iris.analysis.cartography.area_weights(outfield)
   )
   return outfield

# This calculates the average of a value along a longitude line.
# The value is divided by the area of the region of interest to give
# a quantity equivalent to kg/m2/s of precipitation.
def lonline(infield,longitude):
   outfield=infield.intersection(
      latitude=(latmin,latmax)
   )
   outfield=outfield.interpolate(
      [('longitude',longitude)],iris.analysis.Linear()
   )
   outfield=outfield.collapsed(
      'latitude',iris.analysis.MEAN
   )*rEarth*(latmax*d2r-latmin*d2r)
   return outfield/boxarea

# Same as above for a latitude line.
def latline(infield,latitude):
   outfield=infield.intersection(
      longitude=(lonmin,lonmax)
   )
   outfield=outfield.interpolate(
      [('latitude',latitude)],iris.analysis.Linear()
   )
   outfield=outfield.collapsed(
      'longitude',iris.analysis.MEAN
   )*rEarth*np.sin(latitude*d2r)*(lonmax*d2r-lonmin*d2r)
   return outfield/boxarea

# Evaporation term is in the "a" stream.
if 'apvera' in fstemin:
   
   infield=iris.load_cube(fstemin+fin,'surface_upward_latent_heat_flux')
   Lvap_water=iris.cube.Cube(2.257e6,units='J kg-1')
   outfield=takemean(infield/Lvap_water)
   iris.save(outfield,fstemout+fout)
   
# Other fields are in the "b" stream. Note that fluxes are defined as
# into the box in all directions.
elif 'apverb' in fstemin:

   if (pcptype=='stratrain'):
      inpr=iris.load(fstemin+fin,'stratiform_rainfall_flux')
   else:
      inpr=iris.load(fstemin+fin,'precipitation_flux')
   inpr=inpr.extract_cube(iris.Constraint(cube_func=cube_is_mean))
   outpr=takemean(inpr)

   inuq=iris.load_cube(fstemin+fin,'m01s30i462')
   outme=lonline(-inuq,lonmax)
   outme.rename('easterly_flux')
   outmw=lonline(inuq,lonmin)
   outmw.rename('westerly_flux')
   
   invq=iris.load_cube(fstemin+fin,'m01s30i463')
   outmn=latline(-invq,latmax)
   outmn.rename('northerly_flux')
   outms=latline(invq,latmin)
   outms.rename('southerly_flux')

   iris.save([outpr,outme,outmw,outmn,outms],fstemout+fout)
