""" get_mean_state.py
Taken from code by Ben Maybee - April 2026
https://github.com/BMaybee/East_Atlantic_vortices/blob/b885038517f5e0698fdcdef68790b6539093423c/get_mean_state.py

"""
import iris
import numpy as np
import xarray as xr
import pandas as pd
import glob
import os
import glob
import warnings
import argparse
import itertools
import calendar
from multiprocessing import Pool
import metpy.calc as mpcalc
from metpy.units import units
from dynamics import *
warnings.filterwarnings("ignore")

# NOTE - this file is designed to use multiprocessing and thus must be run on a server which supports parallel processing.
# Fundamental to efficient calculation of mean state.


# function to convert TOA OLR to empirically corrected brightness temperature (Yang and Slingo, 2001)
def tb_from_olr(OLR):
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb
    
###################
# Function to load any native grid DY3 UM pp file. 
#    - date = any format datesamp object. Pd is easiest
#    - var_name = STASH diagnostic name as listed in iris variable list
#    - sim = one of the available simulations. See list in function, can be extended easily (manual)
#    - relon = native grid longitude is 0-360, which is naff for East Atlantic. If True, relabel as -180-180
#    - stream = which pp file from a given day to load. There are 4 main options, a--d, with c/d holding pressure-level data.
#    - p = pressure range. If loading a/b streams don't need to do anything, applied only if stream=c/d
# IMPORTANT: the function crops data to an Atlantic domain internally. Modify this if working elsewhere.
###################
def load_file(date,var_name,sim="n1280_GAL9",relon=True,stream="c",p=slice(None,None)):
    date_str="%04d%02d%02d"%(date.year,date.month,date.day)
    # data in 12 hour chunks, labelled 0 or 12, so get appropriate value.
    tidx=int(date.hour/12)*12

    if sim=="CTC_km4p4_RAL3P3":
        ds = xr.DataArray.from_iris(iris.load("/gws/nopw/j04/kscale/DYAMOND3_data/10km-GAL9-nest/CTC_km4p4_RAL3P3/field.pp/apver{}.pp/CTC_km4p4_RAL3P3.n1280_GAL9_nest.apver{}_{}T{:02d}.pp".format(
                            stream,stream,date_str,tidx), var_name)[-1]).sel(latitude=slice(-5,45))
    elif sim=="n2560_RAL3p3":
        ds = xr.DataArray.from_iris(iris.load("/gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3.apver{}_{}T{:02d}.pp".format(
                            stream,stream,date_str,tidx), var_name)[-1]).sel(latitude=slice(-5,45))
    elif sim=="n2560_RAL3p3_tuned":
        ds = xr.DataArray.from_iris(iris.load("/gws/nopw/j04/kscale/DYAMOND3_reruns/5km-RAL3p3-tuned/glm/field.pp/apver{}.pp/glm.n2560_RAL3p3_tuned.apver{}_{}T{:02d}00Z.pp".format(
                            stream,stream,date_str,tidx), var_name)[0]).sel(latitude=slice(-5,45))
    elif sim=="n1280_10km-CoMA9":
        ds = xr.DataArray.from_iris(iris.load("/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-CoMA9/glm/field.pp/apver{}.pp/glm.n1280_CoMA9_v2.apver{}_{}T{:02d}00Z.pp".format(
                            stream,stream,date_str,tidx), var_name)[0]).sel(latitude=slice(-5,45))
    elif sim=="n1280_GAL9":
        ds = xr.DataArray.from_iris(iris.load("/gws/nopw/j04/kscale/DYAMOND3_reruns/10km-GAL9/glm/field.pp/apver{}.pp/glm.n1280_GAL9_v2.apver{}_{}T{:02d}00Z.pp".format(
                            stream,stream,date_str,tidx), var_name)[-1]).sel(latitude=slice(-5,45))

    if stream=="c" or stream=="d":
        ds=ds.sel(pressure=p)
        
    if relon:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude").sel(longitude=slice(-85,55))
    if args.coarsen is not None: # GLOBAL variable. Use for mean state to speed things up significantly
        # For vertical profile data - coarsen to 0.5 degrees and restrict to 6 hourly timesteps (same as TRACK)
        lons,lats=np.arange(-85,55.5,0.5),np.arange(-5,45.5,0.5)
        if stream=="c" or stream=="d":
            ds=ds[ds.time.dt.hour.isin(np.arange(0,24,6))].interp(longitude=lons).interp(latitude=lats)
        else:
            ds=ds.interp(longitude=lons).interp(latitude=lats)
    return ds



# Used when just calculating for single hour: split threads over segments of full period
def parallelise(idx):
    period_seg=period[idx*fact:(idx+1)*fact]

    j=0
    for date in period_seg:
        print(date)
        day_arrs=[]
        day_counts=[]

        if args.hour is None:
            hrs=[0,12]
        else:
            hrs=[12*int(args.hour/12)]
            
        for hr in hrs:
            # first go through all the options where we need to do some calculations from available diagnostics
            if var=="equivalent_potential_temperature":
                q=load_file(date+pd.Timedelta(hr,"h"),"specific_humidity",p=plev,sim=sim,stream="c").compute()
                mr=mpcalc.mixing_ratio_from_specific_humidity(q)
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",p=plev,sim=sim,stream="d").compute()
    
                pvals=np.ones(q.shape)
                for i,p in enumerate(q.pressure.values):
                    pvals[:,i,:,:]=p*pvals[:,i,:,:]
                Td=mpcalc.dewpoint_from_specific_humidity(pvals * units("hPa"), T, q)
                ffile=mpcalc.equivalent_potential_temperature(pvals * units("hPa"), T, Td).metpy.dequantify()

            elif var=="potential_temperature":
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",p=plev,sim=sim,stream="d").compute()
    
                pvals=np.ones(T.shape)
                for i,p in enumerate(T.pressure.values):
                    pvals[:,i,:,:]=p*pvals[:,i,:,:]
                ffile=mpcalc.potential_temperature(pvals * units("hPa"), T).metpy.dequantify()

            elif var=="potential_vorticity":
                u=load_file(date+pd.Timedelta(hr,"h"),"x_wind",p=plev,sim=sim,stream="c").compute()
                v=load_file(date+pd.Timedelta(hr,"h"),"y_wind",p=plev,sim=sim,stream="c").compute()  
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",p=plev,sim=sim,stream="d").compute() # don't need to regrid as using args.coarsen=True
    
                pvals=np.ones(u.shape)
                for i,p in enumerate(u.pressure.values):
                    pvals[:,i,:,:]=p*pvals[:,i,:,:]
                theta=mpcalc.potential_temperature(pvals * units("hPa"), T)
                ffile = mpcalc.potential_vorticity_baroclinic(theta, pvals * units("hPa"), u, v).metpy.dequantify() 
                
            elif var=="moist_static_energy":
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",p=plev,sim=sim,stream="d").compute()
                z=load_file(date+pd.Timedelta(hr,"h"),"geopotential_height",p=plev,sim=sim,stream="d").compute()
                q=load_file(date+pd.Timedelta(hr,"h"),"specific_humidity",p=plev,sim=sim,stream="c").compute()

                cp,g,Lc=1.005,9.81,2.26e6
                ffile=cp*T + g*z + Lc*q

            elif var=="mass_flux":
                w=load_file(date+pd.Timedelta(hr,"h"),"upward_air_velocity",sim=sim,p=plev,stream="c").compute()
                q=load_file(date+pd.Timedelta(hr,"h"),"specific_humidity",sim=sim,p=plev,stream="c").compute()
                mr=mpcalc.mixing_ratio_from_specific_humidity(q)
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",sim=sim,stream="d",p=plev).compute()
                ffile=w*mpcalc.density(w.pressure.values[:,None,None] * units("hPa"), T, mr).metpy.dequantify()

            elif var=="saturation_deficit":
                q=load_file(date+pd.Timedelta(hr,"h"),"specific_humidity",sim=sim,p=plev,stream="c").compute()
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",sim=sim,p=plev,stream="d").compute()
                sat_mr=mpcalc.saturation_mixing_ratio(T.pressure.values[:,None,None] * units("hPa"), T)
                q_sat=mpcalc.specific_humidity_from_mixing_ratio(sat_mr).metpy.dequantify()

                ffile = q-q_sat

            elif var=="ltrh":
                q=load_file(date+pd.Timedelta(hr,"h"),"specific_humidity",sim=sim,p=slice(600,1000),stream="c").compute()
                T=load_file(date+pd.Timedelta(hr,"h"),"air_temperature",sim=sim,stream="d",p=slice(600,1000)).compute()
                sat_mr=mpcalc.saturation_mixing_ratio(T.pressure.values[None,:,None,None] * units("hPa"), T)
                q_sat = mpcalc.specific_humidity_from_mixing_ratio(sat_mr).metpy.dequantify()

                ffile = q.integrate("pressure") / q_sat.integrate("pressure")
                
            else:
                ffile=load_file(date+pd.Timedelta(hr,"h"),var,sim=sim,p=plev,stream=file_list)

            if len(hrs)==1:
                print(ffile)
                ffile=ffile.sel(time=date+pd.Timedelta(args.hour,"h"))
                print(ffile)
            elif hr==0:
                ffile=ffile[ffile.time.dt.hour!=0]

            day_arrs.append(ffile.values)
            day_counts.append((~np.isnan(ffile.values)).astype(int))

        # rather than just stick files in a big list and calculate a mean at the end (VERY SLOW),
        # here sum the data and keep track (index level) of how many items contribute to the sum.
        # Returning the two arrays at end of loop allows exact reconstruction of linear mean from
        # multiple parallelisation loops. This is fast.
        ffile=np.vstack(day_arrs)
        countf=np.vstack(day_counts)
        if j==0:
            ds=ffile
            count=countf
            j+=1
        else:
            ds=np.nansum(np.stack([ds,ffile]),axis=0)
            count=np.nansum(np.stack([count,countf]),axis=0)

    return ds,count

#######################
# sim = simulation name string. Must correspond to one with a path specified in load_file
# var = STASH variable name (not code) present in iris list of a native grid UM file
# filelist = the "stream" in which variable is stored. One of [a,b,c,d]
# month = numerical value of month we wish to calculate mean for
# hour = option to specify mean state at a given hour. If using coarsen (recommended), redundant as code fast enough to do them all efficiently
# coarsen = coarsen from native grid to 0p5deg, 6 hourly frequency. Use this for background climatologies; don't if want to show native grid detail.

#########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=False, default="n1280_GAL9")
parser.add_argument("-v", "--var", required=True)
parser.add_argument("-f", "--filelist", required=True)
parser.add_argument("-m", "--month", required=True,type=int)
parser.add_argument("-hr", "--hour", required=False, type=int)
parser.add_argument("-c", "--coarsen", required=False)
args = parser.parse_args()
##########################################################################

sim=args.sim
var=args.var
file_list=args.filelist

# NOTE RANGE CROP - only care about summer
period=pd.date_range("2020-05-01","2020-10-31")
period=period[period.month==args.month]

try:
    plev=int(var[-3:])
    var=var[:-4]
except:
    if "plevs" in var:
        plev=slice(100,None)
        var=var[:-6]
    else:
        plev=900

# some funny variable naming quirks in the pp files
if var=="tcw":
    var="m01s30i461"
elif var=="surface_net_downward_shortwave_flux":
    var="m01s01i202"
elif var=="precip" and sim not in ["n1280_GAL9","n1280_10km-CoMA9"]:
    var="stratiform_rainfall_flux"
elif var=="precip" and sim in ["n1280_GAL9","n1280_10km-CoMA9"]:
    var="precipitation_flux"

psize=10
p=Pool(psize)
fact=int(len(period)/psize)
out=p.map(parallelise,np.arange(psize))
# stack the sums
sums=np.stack([part[0] for part in out])
Sum=np.nansum(sums,axis=0)
# stack the counts
counts=np.stack([part[1] for part in out])
Count=np.nansum(counts,axis=0)

if file_list=="c" or file_list=="d":
    ref=load_file(pd.Timestamp("2020-%02d-01 12:00"%args.month),"x_wind",sim=sim,p=plev,stream="c").isel(time=0)
    if args.coarsen is None:
        hr_coords=[hr%24 for hr in np.arange(3,25,3)]
    else:
        hr_coords=[6,12,18,0]
else:
    ref=load_file(pd.Timestamp("2020-%02d-01 12:00"%args.month),var,sim=sim,stream=file_list).isel(time=0)
    hr_coords=[hr%24 for hr in np.arange(1,25)]

# calculate means and turn into xarray objects
if args.hour is None:    
    out=xr.DataArray(Sum/Count,dims=["hour"]+list(ref.dims),name=var)
    for dim in ref.dims:
        out=out.assign_coords({dim:ref[dim].values})
    try:
        out=out.assign_coords(hour=hr_coords)
    except:
        out=out.assign_coords(hour=hr_coords[:-1])

    month=calendar.month_abbr[args.month]
    if args.coarsen is None:
        out.to_netcdf(f"/gws/nopw/j04/kscale/USERS/bmaybee/DYAMOND3/mean_states/{sim}_mean_{month}_{args.var}.nc")
    else:
        out.to_netcdf(f"/gws/nopw/j04/kscale/USERS/bmaybee/DYAMOND3/mean_states/{sim}_mean_{month}_{args.var}_0p5deg.nc")
        
else:
    out=xr.DataArray(Sum/Count,coords=ref.coords,name=var)
    month=calendar.month_abbr[args.month]
    if args.coarsen is None:
        out.to_netcdf(f"/gws/nopw/j04/kscale/USERS/bmaybee/DYAMOND3/mean_states/{sim}_mean_{month}_{args.hour}Z_{args.var}.nc")
    else:
        out.to_netcdf(f"/gws/nopw/j04/kscale/USERS/bmaybee/DYAMOND3/mean_states/{sim}_mean_{month}_{args.hour}Z_{args.var}_0p5deg.nc")