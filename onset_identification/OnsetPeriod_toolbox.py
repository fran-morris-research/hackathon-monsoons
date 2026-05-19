import sys
import numpy as np
import xarray as xr
import netCDF4 as nc
import itertools as it
import datetime
import cftime
import requests
import scipy.interpolate as sci

#############################
#Main functions related to Onset period analysys
##############################

def IDOnsetPeriods(pp,fpp,dtime,unts,cal,minlen,mxrainy):
    '''
        Main Function to identify the onset periods per pixel
        in observation or model data (GCM, RCM, CPM).

        INPUT:
            pp - precipitation time series
            fpp - filtered precipitation time series
            dtime - time array
            unts - units of the time array
            cal - calendar of the time array
            minlen - minimum length of the onset period
            mxrainy - maximum number of rainy seasons allowed

        RETURNS:
            grad - mean gradient of the filtered precipitation
            onsrainy - first day and duration of the climatological onset periods
            onsperiod - dictionary with the first day, duration, mean gradient, 
                        and total precipitation of the onset periods
            otheronset - dictionary with the first day, duration, mean gradient,
                        and total precipitation of the onset periods that 
                        do not match the climatological onset periods
    '''

    nyrs=len(np.unique(dtime[:,0])); yr0=int(dtime[0,0])
    ndays=360 if cal=='360_day' else 365

    #Identifying periods with increasing precipitation
    grad=np.gradient(fpp)
    rny=np.where(grad > np.quantile(np.abs(grad),0.25),1,0)

    firstday,lenseason=FindOnsetPeriods(rny,minlen)
    #first day and duration of all periods with increasing precipitation

    #Checking the number of rainy seasons
    frst,lnt,w0=FindNrRainy(firstday,lenseason,nyrs,ndays,mxnr=mxrainy)

    #returns the first day and length of the periods with largest frequency 
    #and the day with the minimum frequency of onsets (w0), which will 
    #be used as the beginning of the water year 

    #if w0=200 - the rainy season on the year can start at any day after 200 
    # but before day 200 of the next year. 
    # That way, a season at day 400 in the year 1998 indicate that the season 
    # started on feb of 1999. 
    # The water year can be one year less than yr0 to accound for the days
    # between 01/jan/yr0 and w0
    nrrainy=len(frst)
    onsrainy=np.vstack((frst,lnt)).T

    #It is possible to have a pixel with no defined onset on 
    #regions with too constant or too little precipitation
    if(len(frst) > 0):
        candonset=dict((yr,[]) for yr in range(yr0-1,yr0+nyrs,1))  #all candidates
        onsperiod=dict((yr,[]) for yr in range(yr0-1,yr0+nyrs,1))  #selected onset(s)
        otheronset=dict((yr,[]) for yr in range(yr0-1,yr0+nyrs,1))  
        #onset of a rainy period outside the expected period

        for ii in range(len(firstday)):
            grd=np.nanmean(grad[firstday[ii]:firstday[ii]+lenseason[ii]])
            tot=np.nansum(pp[firstday[ii]:firstday[ii]+lenseason[ii]])
            if(grd < 0): print (ii)
            yr=int(dtime[firstday[ii],0]); dy=0
            if(nc.num2date(firstday[ii],units=unts,calendar=cal).dayofyr-1 < w0): yr-=1;dy=1
            candonset[yr].append((firstday[ii],lenseason[ii],grd,tot,dy))

        #Selecting the onsets in each year
        for yr in candonset.keys():
            onsperiod[yr],otheronset[yr]=SelectOnsets(candonset[yr],onsrainy,ndays,unts,cal)


        #Quality control
        onsperiod,otheronset=QC_Onset(nrrainy,onsperiod,otheronset,onsrainy,w0,ndays,unts,cal)
                    
        #Classifying other onsets: false onset (same side of the minimum related to the dry season);
        #                          wet spell (outside the wet season)
        #                          second onset (happend after the onset)
        #                          other (other situations to be checked)
        keys={'OCM':'false onset','MOC':'false onset',
            'OMC':'wet spell','CMO':'wet spell',
            'MCO':'second onset','COM':'second onset'}

        grd=np.asarray([np.nanmean(grad[ii::ndays]) for ii in range(ndays)])

        globalmin=np.asarray([ii for ii in range(1,ndays-1) \
                            if ((grd[ii-1] < 0) & (grd[ii] > 0))])
        if(len(globalmin)==0): globalmin=np.asarray([0])

        if(len(globalmin) > nrrainy): #as many mins as onset periods
            ps=[np.argmin(abs(globalmin-onsrainy[ii,0])) for ii in range(int(nrrainy))]
            globalmin=globalmin[np.asarray(ps)]

        for yr in otheronset.keys():
            if(len(otheronset[yr]) > 0):
                mins=list(nc.date2num(nc.num2date(globalmin,units='days since '+str(yr)+'-01-01',\
                                            calendar=cal),units=unts,calendar=cal))
                otheronset[yr]=ClassifyOthers(otheronset[yr],onsperiod[yr],mins,keys)
    else:
        onsrainy=np.empty(0)
        onsperiod=[]; otheronset=[]

    return grad,onsrainy,onsperiod,otheronset


def IDOnsPerEMSea(pp,grd,minlen):
    '''
        Function to identify all onset period candidates and calculate their
        main statistics (total precipitation, mean gradient, duration, etc.)
        Used only in Seasonal and Subseasonal forecast and hindcast

        INPUT:
            pp - precipitation time series
            grd - gradient of the filtered precipitation
            minlen - minimum length of the onset period
        
        RETURNS:
            ons - array with the first day, duration, mean gradient, total
                  precipitation, wet and dry spells of all onset period candidates
    '''

    ndays=pp.shape[0]
    rny=np.where(grd > np.quantile(np.abs(grd),0.25),1,0)

    firstday,lenseason=FindOnsetPeriods(rny,minlen)

    #Extra test because the time series does not have an entire cycle
    #Remove onset candidates starting during the first month of the simulation
    if(len(firstday)>0):
        rmv=[]
        for ii in range(len(firstday)):
            if(firstday[ii] <= minlen): rmv.append(ii)

        firstday,lenseason=DelCandidate(rmv,firstday,lenseason)

    #first day, duration, mean gradient and tot precip of all periods with increasing precipitation
    ons=np.zeros((1,len(firstday),12))
    ons[0,:,0]=firstday; ons[0,:,1]=lenseason

    for ii in range(len(firstday)):
        ons[0,ii,2]=np.nanmean(grd[int(ons[0,ii,0]):int(ons[0,ii,0]+ons[0,ii,1])])
        ons[0,ii,3]=np.nansum(pp[int(ons[0,ii,0]):int(ons[0,ii,0]+ons[0,ii,1])])

    #Id wet and dry spells for in all candidates
    tmpmask=np.zeros((ndays))
    for ii in range(ons.shape[1]):
        ps=list(range(int(ons[0,ii,0]),int(ons[0,ii,0]+ons[0,ii,1]),1))
        tmpmask[ps]=1
    
    ons[0,:,4:]=WetDrySpells(tmpmask,pp,ons.shape[1],ons)

    return ons

def IDOnsPerClimatolEM(ons,nens,nyr,ndays):
    '''
        Function to identify the climatological onset periods based on all 
        onset period candidates identified.
        Used only in seasonal and subseasonal forecast and hindcast

        INPUT:
            ons - array with the first day, duration, mean gradient, total
                  precipitation, wet and dry spells of all onset period candidates
            nens - number of ensemble members
            nyr - number of years in the dataset
            ndays - number of days per year
        
        RETURNS:
            frst - list with the first day of each climatological onset period
            lnt - durantion of each climatological onset period
    '''

    onsmask=np.zeros((nyr,nens,ndays))
    for yr in range(nyr):
        for em in range(nens):
            nv=len(ons[yr,em,np.isfinite(ons[yr,em,:,0]),0])
            for ii in range(nv):
                ps=list(range(int(ons[yr,em,ii,0]),int(ons[yr,em,ii,0])+int(ons[yr,em,ii,1])))
                onsmask[yr,em,ps]=1
        
    onsmask=np.reshape(onsmask,(nyr*nens,ndays))

    frst,lnt,w0=FindNrRainy([],[],nyr,ndays,tmpmask=onsmask,hcast=True)

    return frst,lnt

def ClassOnsetEM(yr,ns,fdlen,onsrainy,ndays,unts,cal):
    '''
        Function to classify the onset periods
        Used only in the seasonal and subseasonal
        
        INPUT:
            yr - year of the dataset
            ns - number of climatological onset periods
            fdlen - array with the first day, duration and other statistics 
                    of the onset periods
            onsrainy - first day and duration of the climatological onset periods
            ndays - number of days per year
            unts - units of the time array
            cal - calendar of the time array
        
        RETURNS:
            onsdat - array with the first day, duration, mean gradient, total
                     precipitation, wet and dry spells of the selected onset periods
            othrdat - array with the first day, duration, mean gradient, total
                      precipitation, wet and dry spells of the onset periods that
                      do not match the climatological onset periods
            onsmask - mask with the onset periods
        '''

    nv=len(fdlen[np.isfinite(fdlen[:,0]),0])
    dtime=np.asarray(range(ndays))

    if(nv > 0):
        candonset={yr:[]}; onsperiod={yr:[]}; otheronset={yr:[]}
        for ii in range(nv):
            candonset[yr].append(tuple(fdlen[ii,:2].astype(int))+tuple(fdlen[ii,2:4])+(0,)) 
        
        #Selecting the onsets in each year
        onsperiod[yr],otheronset[yr]=SelectOnsets(candonset[yr],onsrainy,\
                                                  ndays,unts,cal)

        #Classifying other onsets: false onset (same side of the minimum related to the dry season);
        #                          wet spell (outside the wet season)
        #                          second onset (happend after the onset)
        #                          other (other situations to be checked)
        if(len(otheronset[yr]) > 0): #ALL AS OTHER FOR NOW
            for ii in range(len(otheronset[yr])):
                otheronset[yr][ii]=otheronset[yr][ii]+('other',)

        #Spatialising the results
        #Creating a mask of zeros and ones to mark the onset periods in each year
        onsdat,othrdat,onsmask=SpatialOnsMsk(ns,onsperiod,otheronset,dtime)

    else:
        onsdat=np.empty(0); othrdat=np.empty(0); onsmask=np.empty(0)
    
    return onsdat,othrdat,onsmask

def FFT_Precip(indat,axis,minp,maxp,frq=False,keepLTM=True):
    '''
        filters the precipitaiton time series using discrete FT 

        INPUT:
            data - input data
            axis - axis over which the filter will be applied
            minp - minimum period retained
            maxp - maximum period retained
            frq [True/False] - returns the period of the retained harmonics
            keepLTM [True/False] - keep the zero harmonic (equivalent to 
                                the LTM of the time series) on the filtered results

        RETURNS:
            avefft - filtered precipitation                            
    '''

    ndays=indat.shape[axis]

    ppfft=np.fft.rfftn(indat,axes=(axis,))

    #Retaining oscillations with periods closest to minp and maxp
    nri=np.argmin(abs((1/np.fft.rfftfreq(ndays)[1:])-maxp))+1
    nrf=np.argmin(abs((1/np.fft.rfftfreq(ndays)[1:])-minp))+1
    #+1: frequency of harmonic 0 = Inf (LTM)

    retain=[0,]+list(range(nri,nrf+1)) if keepLTM else list(range(nri,nrf+1))

    freq=1/np.fft.rfftfreq(ndays)[retain]

    avefft=np.zeros_like(ppfft)
    avefft[retain,...]=ppfft[retain,...]

    avefft=np.fft.irfftn(avefft,axes=(axis,))

    if frq: return avefft,freq
    else: return avefft


def FindOnsetPeriods(indat,minlen):
    '''
        Given an time series of 0's and 1's, this functions locates the 
        position and length of sequences of 1's with minimum number of entries
        defined by minlen. 
        
        INPUT:
            indat - time series of 0's and 1's
            minlen - minimum number of days in a sequence of 1's
        
        RETURNS:
            firstday - list with the position of the first entry in a 
                       sequence of 1's longer than minlen
            lenseason - number of conseqcutive entries equal 1.
    '''

    ps=0; lenseason=[]; firstday=[]
    for key,grp in it.groupby(indat):
        tmp=len(list(grp))
        if key == 0:
            pass
        else:
            if(tmp >= minlen):
                lenseason.append(tmp)
                firstday.append(ps)  
        ps=ps+tmp              

    return firstday,lenseason


def OnsetClim(fd,ln,ave,grd):
    '''
        DEPRECATED after adopting the frequency of days with 
        sustained precipitation increase. 
        For that, use the ''FindNrRainy'' function
        -------
        Identifies the climatological onset periods.
        
        INPUT:
            fd - list with the first day of each candidate onset period
            ln - list with the duration of each candidate onset period
            rny - array with 0s and 1s marking the postion of the candidate onsets
            ave - filtered precipitaiton (using fft)
            grd - gradient of the filtered precipitation

        RETURNS:
            fd - list with the first day of each climatological onset period
            ln - list with the duration of each climatological onset period
    '''

    ndays=len(ave)

    #removing onsets happening on the same ascending part of the curve
    #there must have a point with grad == 0 between two onsets
    if(len(fd) > 1):
        qq=np.nanquantile(np.abs(grd),0.025)
        mins=[ii for ii in range(1,ndays-1) \
                    if ((ave[ii] < 0.75*ave.max()) & \
                        ((grd[ii-1] < 0) & (grd[ii+1] > qq)))]
        #ii+1--> julian day where the local minimum happened
        if(len(mins) == 0): mins=[0]
        rmv=[]

        seq=fd+mins  #day of the local minima and onset
        tag=np.hstack((np.repeat('O',len(fd)),np.repeat('M',len(mins))))
        #tag indicating 'M' as a local minimum and 'O' as onset. 
        #Same order as seq array
        pst=np.argsort(seq) #sorting sequence of dates

        chk=CheckPattern(seq,tag,'OO')
        #Position of the first entry in the 'firstday' list 
        #followed by another 'firstday' without a minimum between them
        if(len(chk) > 0):
            for ii in pst[chk]:
                grd0=np.abs(grd[fd[ii]:fd[ii]+ln[ii]]).mean()
                grd1=np.abs(grd[fd[ii+1]:fd[ii+1]+ln[ii+1]]).mean()
                #marking to remove candidate with smaller grd
                rmv.append(ii if grd1 > grd0 else ii+1) 

        fd,ln=DelCandidate(rmv,fd,ln)

    #Second test: grad[final]-grad[initial] within the onset period should be at least 10% of the max(grad)
    if(len(fd) > 1):
        rmv=[ii for ii in range(len(fd)) \
                if grd[fd[ii]+ln[ii]-1]-grd[fd[ii]] < 0.1*grd.max()]

        fd,ln=DelCandidate(rmv,fd,ln)

    #Third test: if the total filtered precipitation during the onset period is too small
    if(len(fd) > 1):
        tot=ave.sum()
        rmv=[]
        for ii in range(len(fd)):
            accum=ave[fd[ii]:fd[ii]+ln[ii]].sum()
            if(accum/tot < 0.025): rmv.append(ii)

        fd,ln=DelCandidate(rmv,fd,ln)
    
    #For now limiting the number of onset periods to two. 
    #So removing the shortest onset period if > 2 onset periods
    if(len(fd) > 2):
        ps=np.argsort(ln)
        rmv=list(ps[:-2])  
        #keeping the last two ps which correspond to the longer periods
        fd,ln=DelCandidate(rmv,fd,ln)


    return fd,ln


def FindNrRainy(fd,ln,nyrs,ndays,minlen=25,mxnr=False,tmpmask=False,hcast=False):
    '''
        Identifies the number of rainy seasons and the expected 
        window that it should occur. The identification is based on
        the frequency (over the years) with each each day is classified
        as having a sustainded precipitation increase (positive gradient
        of the filtered precipitation) considering all years in the dataset.

        INPUT:
            fd     - list with the first day of each period with
                    sustained precipitation increase
            ln     - duration of the period 
            nyrs   - number of years in the dataset
            ndays  - number of days per year (UM model has 360 days)
            minlen - minimum number of days in a period for it to
                    be considered a onset period
            mxnr   - maximum number of rainy seasons allowed

        RETURN:
            frst  - list with the first day of each climatological onset period
            lnt   - durantion of each climatological onset period
            w0    - day of the year considered as the beginning of the water year.
                    Defined as the day with the minimum freqeuncy of days with
                    sustained precipitation increase
            pdfNS - the frequency of years with sustained precipitation increase
                    (per day of the year), after nomalisation (values range from 0
                    to 1) and smoothing (15-day running mean)
    '''

    #mask with periods in onset 
    if(isinstance(tmpmask,bool)):
        tmpmask=np.zeros((nyrs*ndays))
        for ii in range(len(fd)):
            ps=list(range(fd[ii],fd[ii]+ln[ii]))
            tmpmask[ps]=1
        
        tmpmask=np.reshape(tmpmask,(nyrs,ndays))


    #Number of times a day is classified as onset over all years
    if (len(tmpmask.shape) > 1): 
        pdf=np.nansum(tmpmask,0)
    else: pdf=tmpmask

    pdfN=(pdf-pdf.min())/(pdf.max()-pdf.min()) 
    #normalizing such that values range from 0 to 1. Note that 1 does not mean that 
    #the days was in an onset period over all years, but that that was the day classified 
    #as onset the largest number of years

    #locating the minimum and shifting the time series to start at this point
    #Not done for seasonal hindcast because it does not cover one entire year
    if(not hcast):
        w0=np.argmin(pdfN)
        if(w0 >0): pdfN=np.hstack((pdfN[w0:],pdfN[:w0]))
    else: w0=0

    #smoothing using a running mean
    ws=14
    padded = np.pad(pdfN, (ws//2,ws-ws//2-1), mode='edge')
    pdfNS = np.convolve(padded, np.ones((ws,))/ws, mode='valid')

    #Locating periods with largest frequence of onset (above 0.5)
    #(not really 50% since it is normalized by the maximum)
    pps=[ps for ps in range(ndays-1) if ((pdfNS[ps] < .5) & (pdfNS[ps+1]>=.5))]
    nps=[ps for ps in range(ndays-1) if ((pdfNS[ps] >= .5) & (pdfNS[ps+1]<.5))]

    if(len(pps) == 0): 
        if(pdfNS.max() < 0.5):
            pps=[ps for ps in range(ndays-1) if ((pdfNS[ps] < 0.5*pdfNS.max()) &\
                                                (pdfNS[ps+1]>=0.5*pdfNS.max()))]
            nps=[ps for ps in range(ndays-1) if ((pdfNS[ps] >= 0.5*pdfNS.max()) & \
                                                (pdfNS[ps+1]<0.5*pdfNS.max()))]
        if(pdfNS.min() > 0.5):
            thr=0.5*(pdfNS.max()-pdfNS.min())+pdfNS.min()
            pps=[ps for ps in range(ndays-1) if ((pdfNS[ps] < thr) &\
                                                 (pdfNS[ps+1]>=thr))]
            nps=[ps for ps in range(ndays-1) if ((pdfNS[ps] >= thr) & \
                                                 (pdfNS[ps+1]<thr))]


    #Sanity check point: since the time series was shifted to start at the 
    #minimum value, the curve should cross the y=0.5 an even number of 
    #time (one going up and one going dow)
    if(len(nps) != len(pps)):
        if(len(nps) < len(pps)):
            if(pdfNS[-1] >= .5): nps.append(ndays-1)
            else: 
                if(pdfNS[-1] >= 0.5*pdfNS.max()): nps.append(ndays-1)
                else: print ('!!!!!CHECK!!!!')
        else:
            if(pdfNS[0] >= .5): pps.insert(0,0)
            else: 
                if(pdfNS[0] >= 0.5*pdfNS.max()): pps.insert(0,0)
                else: print ('!!!!!CHECK!!!!')
    else:
        if(pps[0] > nps[0]): pps.insert(0,0)
        if(pps[-1] > nps[-1]): nps.append(ndays-1)

            
    #checking for breaks in the periods (value below 0.5 for a few days)
    mrg=[]
    for ii in range(len(pps)-1):
        if(pps[ii+1]-nps[ii] <= 30): #Arbitrary, check if it is a good size
            if (np.min(pdfNS[nps[ii]:pps[ii+1]+1]) > 0.3): mrg.append(ii)
            #in this case, the minimum value between the two periods is 
            #too small, so it is likely a break in the season. Merge both seasons
                
    if(len(mrg) >0):
        mrg.sort()
        for ii in mrg[::-1]: nps.pop(ii); pps.pop(ii+1)

    #01/03/25 - lower thrshold for seasons not present all years (pdfNS.max < 1)
    if((len(pps) > 1) | (len(nps) > 1)):
        ps=(np.asarray([pdfNS[pps[ii]:nps[ii]].max() \
                        for ii in range(len(pps))]) < pdfNS.max()).nonzero()[0]
        if(len(ps) > 0):
            for jj in ps:
                tmp1=[ii for ii in range(ndays-1) \
                            if ((pdfNS[ii] < .5*(pdfNS[pps[jj]:nps[jj]].max())) &\
                                (pdfNS[ii+1] >= .5*(pdfNS[pps[jj]:nps[jj]].max())))]
                tmp2=[ii for ii in range(ndays-1) \
                            if ((pdfNS[ii] >= .5*(pdfNS[pps[jj]:nps[jj]].max())) &\
                                (pdfNS[ii+1] < .5*(pdfNS[pps[jj]:nps[jj]].max())))]
                if(len(tmp2) < len(tmp1)):tmp2.append(ndays-1)
                if(len(tmp1) < len(tmp2)):tmp1.insert(0,0)
                if((len(tmp1) > 0) | (len(tmp2) > 0)):
                    if(tmp1[0] > tmp2[0]): tmp1.insert(0,0)
                    if(tmp1[-1] > tmp2[-1]): tmp2.append(ndays-1)

                    #locating the correct original interval - new first day that is closest
                    #to the original first day
                    kk=np.argmin(abs(pps[jj]-np.asarray(tmp1)))
                    tmp1=tmp1[kk]; tmp2=tmp2[kk]
                    #Check if new period is separate from other candidates
                    keep=True
                    for ii in range(len(pps)):
                        if (ii != jj):
                            #Checking if new envelope overlaps with other periods
                            if(tmp1 in range(pps[ii],nps[ii])): keep=False
                            if(tmp2 in range(pps[ii],nps[ii])): keep=False
                            #Check if new envelope contains other periods
                            if(pps[ii] in range(tmp1,tmp2)): keep=False
                            if(nps[ii] in range(tmp1,tmp2)): keep=False
                    if(keep): pps[jj]=tmp1; nps[jj]=tmp2

    #eliminating periods with less than minlen days 
    ln=np.asarray([nps[ii]-pps[ii]+1 for ii in range(len(pps))])
    if(np.all(ln < minlen)):  #In case all periods are too short, check if 
                              #the longest one is at least 75% of the minlen (19 days)
        if(ln.max() > .75*minlen):
            ps=np.argmax(ln)
            pps=[pps[ps]]; nps=[nps[ps]]
    else:
        rmv=(ln < minlen).nonzero()[0]
        if(len(rmv) > 0):
            rmv.sort()
            for ii in rmv[::-1]: pps.pop(ii); nps.pop(ii)

    #If there is a limit to the number of rainy seasons to be considered
    if(not isinstance(mxnr,bool)):  
        while(len(pps) > mxnr):   #For now curbing the nr of rainy seasons to a max
            #Eliminating the seasons with smaller pdfNS
            mx=np.asarray([pdfNS[pps[ii]:nps[ii]].max() for ii in range(len(pps))])
            ii=np.argmin(mx)
            pps.pop(ii); nps.pop(ii)


    #Returning first day of the onset periods, their length, and 
    #the position of the minimum - this is the day with the least 
    #number of onset candidates
    lnt=np.asarray(nps)-np.asarray(pps)
    frst=np.asarray(pps)+w0
    frst=np.where(frst >=ndays,frst-ndays,frst)
    #sorting according to the first day
    lnt=lnt[np.argsort(frst)]
    frst=frst[np.argsort(frst)]
    if(not hcast):
        pdfNS=np.hstack((pdfNS[-w0:],pdfNS[:(ndays-w0)]))

    return frst,lnt,w0


def SelectOnsets(cand,onsrainy,ndays,unts,cal):
    '''
        Select the onset candiditates in the cand list that are 
        closer to the climatological onsets in onsrainy

        INPUT:
            cand - list with the first day, duration, mean gradient,
                   and total precipitation of the onset periods
            onsrainy - first day and duration of the climatological onset periods
            ndays - number of days per year
            unts - units of the time array
            cal - calendar of the time array

        RETURNS:    
            onset - selected onset(s)
            otheronset - onset of a rainy period outside the expected period
        '''

    onset=[]        #selected onset(s)
    otheronset=[]   #onset of a rainy period outside the expected period
    #For each onset period: tuples with the day when the onset period 
    #starts (counting from the first day of simulation. Can be 
    #transformed into julian day using the dtime array), its duration, 
    #the mean filtered precipitation gradient, and the total precipitation
    #during the period
      
    ns=len((onsrainy[:,0] >= 0).nonzero()[0])  #number of climatological rainy seasons

    toOther=list(range(len(cand)))
    #list tagging all candidates. Candidates still in this 
    #list at the end of the iteration are added to the 
    #otheronset list and will later be classified as 
    #false onsets or not

    #Matching the onset candidates to the climatological rainy seasons
    for ss in range(ns):

        clm0=nc.num2date(onsrainy[ss,0],units=unts,calendar=cal).dayofyr-1 #(0 to ndays-1)
        clmons=np.arange(clm0,clm0+onsrainy[ss,1])
        
        if(len(toOther) > 0):
            cnd=[]
            for ii in toOther:
                sea0=nc.num2date(cand[ii][0],units=unts,calendar=cal).dayofyr-1 #(0 to ndays-1)
                sea=np.arange(sea0,sea0+cand[ii][1])
                inters=Intersect(sea,clmons,ndays)
                if len(inters) > 0: 
                    cnd.append((ii,len(inters)))

            if(len(cnd) > 0):        #At least one candidate at the same period as climatological onset
                if(len(cnd) == 1):   #Only one candidate at the same period as climatological onset
                    onset.append(cand[cnd[0][0]])
                    toOther.remove(cnd[0][0])
                else:                #More than one candidate at the same period as climatological onset
                    #Selecting candidate with largest gradient (intensity of the onset) 
                    #(3th data in the tupples in the cand list))
                    ps=cnd[np.argmax([cand[jj[0]][2] for jj in cnd])][0]
                    onset.append(cand[ps])
                    toOther.remove(ps)
            else:  #No candidate at the same period as climatological onset
                #Looking for the closest period
                dst=[]
                for ii in toOther:
                    sea0=nc.num2date(cand[ii][0],units=unts,calendar=cal).dayofyr-1 #(0 to ndays-1)
                    sea1=sea0+cand[ii][1]
                    
                    #circular distance between end of candidate period and beginning of clim onset
                    tmpx=np.cos(abs(sea1-clmons[0])*(np.pi/(ndays/2+1)))
                    tmpy=np.sin(abs(sea1-clmons[0])*(np.pi/(ndays/2+1)))
                    dst1=abs(np.arctan2(tmpy,tmpx)/(np.pi/(ndays/2+1)))

                    #circular distance between end of clim onset and beggining of candidate period
                    tmpx=np.cos(abs(sea0-clmons[-1])*(np.pi/(ndays/2+1)))
                    tmpy=np.sin(abs(sea0-clmons[-1])*(np.pi/(ndays/2+1)))
                    dst2=abs(np.arctan2(tmpy,tmpx)/(np.pi/(ndays/2+1)))

                    dst.append(np.min([dst1,dst2]))

                if(np.min(dst) < ndays/(3.*ns)): 
                    # not considering events that happen in a different part of the water year
                    onset.append(cand[toOther[np.argmin(dst)]])
                    toOther.pop(np.argmin(dst))
                else: onset.append((np.nan,np.nan,np.nan,np.nan,np.nan))
        
        else: onset.append((np.nan,np.nan,np.nan,np.nan,np.nan))
 
    #move unused candidates to otheronset list
    if(len(toOther) > 0): 
        for ii in toOther: otheronset.append(cand[ii])

    return onset,otheronset


def QC_Onset(ns,onsperiod,otheronset,onsrainy,w0,ndays,unts,cal):
    '''
        Quality control function. Compares the onset periods with the
        climatological onset periods and the median onset periods.
        
        INPUT:
            ns - number of climatological onset periods
            onsperiod - dictionary with the first day, duration, mean gradient,
                        and total precipitation of the onset periods
            otheronset - dictionary with the first day, duration, mean gradient,
                         and total precipitation of the onset periods that
                         do not match the climatological onset periods
            onsrainy - first day and duration of the climatological onset periods
            w0 - day of the year considered as the beginning of the water year.
            ndays - number of days per year
            unts - units of the time array
            cal - calendar of the time array
        
        RETURNS:
            onsperiod - dictionary with the first day, duration, mean gradient,
                        and total precipitation of the onset periods
            otheronset - dictionary with the first day, duration, mean gradient,
                         and total precipitation of the onset periods that
                         do not match the climatological onset periods
    '''

    yyrs=np.asarray(list(onsperiod.keys())); yyrs.sort()

    for ss in range(ns):

        for tt in range(3): #tt = 0 - checking against the climatological start of the onset period
                            #tt = 1 - checking against median
                            #tt = 2 - checking against median and removing outliers (thrs > 3)

            #Quality control - checking for outliers
            #Here the onset period is defined in days of the year (0-ndays-1)
            indat=np.asarray([onsperiod[yr][ss][0] for yr in yyrs])
            nans=(np.isnan(indat)).nonzero()[0]; indat[nans]=0.
            indat=np.asarray([float(nc.num2date(ii,units=unts,calendar=cal).dayofyr)-1\
                               for ii in indat])
            indat[nans]=np.nan

            #Shifting dates around w0 - easier to estimate the quantiles
            indat=np.where(indat < w0,indat+ndays,indat)
            thrs=2 if (tt in [0,1]) else 2.5

            target=onsrainy[ss,0] if (tt==0) else np.nanquantile(indat,0.5)
            if(target < w0): target=target+ndays #Shifting dates around w0

            
            outl,iqr=CheckOutliers(indat,ndays,thrs=thrs,target=target)

            if((tt < 2) & (len(outl) > 0)):  #Trying to find a better fit
                for yr in outl:
                    yr1=yyrs[0]+yr
                    #Check if other onsets would be a better fit
                    if(len(otheronset[yr1]) > 0):
                        goodfits=[]
                        for kk in range(len(otheronset[yr1])):
                            indat2=indat.copy()
                            indat2[yr]=nc.num2date(otheronset[yr1][kk][0],\
                                                   units=unts,calendar=cal).dayofyr-1 #(0 to ndays-1)

                            outl2,iqr2=CheckOutliers(indat2,ndays,thrs=thrs,target=target)

                            if(np.isin(yr,outl2)): pass
                            else: goodfits.append(kk)
                        
                        if(len(goodfits) > 0):  
                            tot1=onsperiod[yr1][ss][3]
                            tot2=[otheronset[yr1][goodfits[jj]][3] for jj in range(len(goodfits))]

                            ps=np.argmax([tot1]+tot2)
                            #ps==0 - the current onset is the best fit, so keep it
                            #ps > 0 - change the onset period
                            if(ps > 0): #Change to new onset period
                                toOther=onsperiod[yr1][ss]
                                onsperiod[yr1][ss]=otheronset[yr1].pop(goodfits[ps-1])
                                otheronset[yr1].append(toOther)
                    
            if((tt == 2) & (len(outl) > 0)):  #Move outliers to otheronset
                for yr1 in yyrs[outl]:
                    clmons=np.arange(onsrainy[ss,0],onsrainy[ss,0]+onsrainy[ss,1])

                    #Check if the period overlaps with climatological onset
                    sea0=nc.num2date(onsperiod[yr1][ss][0],units=unts,calendar=cal).dayofyr-1 #(0 to ndays-1)
                    sea=np.arange(sea0,sea0+onsperiod[yr1][ss][1])
                    inters=Intersect(sea,clmons,ndays)
                    if len(inters) > 0: 
                        pass
                    else:
                        toOther=onsperiod[yr1][ss]
                        onsperiod[yr1][ss]=(np.nan,np.nan,np.nan,np.nan)
                        otheronset[yr1].append(toOther)

    return onsperiod,otheronset


def ClassifyOthers(other,onset,mins,keys):
    '''
        This function classifies the other onsets based on their position 
        relative to the onset periods and the day with minimum precipitation
        (indicative of the dry season).

        INPUT:
            other - list of other onsets to be classified (tagged as 'O')
            onset - list of onset periods (tagged as 'C')
            mins - position of the minimum filtred precipitation (tagged as 'M')
            keys - keys indicating the type of other onset:
                'OCM' or 'MOC' :'false onset'
                'OMC' or 'CMO':'wet spell',
                'MCO' or 'COM':'second onset'
                none of the above: 'other'
        
        RETURNS:
            other - list of the classified other onsets 
    '''

    for ii in range(len(other)):

        li=len(other[ii])

        dd=[other[ii][0]]                           
        ons=[onset[jj][0] for jj in range(len(onset))]

        seq=dd+ons+mins
        tag=np.hstack((np.repeat('O',len(dd)),np.repeat('C',len(ons)),\
                        np.repeat('M',len(mins))))
        #tag indicating 'M' as a local minimum and 'O' as 
        #the other onset and C for the main onset. 
        #Same order as seq array
        cnd=[]
        for kk in keys.keys():
            chk=CheckPattern(seq,tag,kk)
            if(len(chk) > 0): cnd.append(kk)
        if(len(cnd)==0):
            other[ii]=other[ii]+('other',)
        elif (len(cnd)==1):
            other[ii]=other[ii]+(keys[cnd[0]],)
        else:  
            ps=[jj for jj in range(len(cnd)) if cnd[jj][1]=='O']
            other[ii]=other[ii]+(keys[cnd[ps[0]]],) \
                      if (len(ps) > 0) else other[ii]+(keys[cnd[0]],)


    return other


def SpatialOnsMsk(ns,onsperiod,otheronset,dtime):
    '''
        Function to spatialise the onset periods and other onsets
        in the time series, creating a mask with the days in the onset periods  
        (1) and other onsets (2,3,4,5) in each year.

        INPUT:
            ns - number of climatological onset periods
            onsperiod - dictionary with the first day, duration, mean gradient,
                        and total precipitation and other statistics of the 
                        onset periods
            otheronset - dictionary with the first day, duration, mean gradient,
                        and total precipitation and other statistics of the 
                        onset periods that do not match the climatological onset periods
            dtime - time array

        RETURNS:
            onsdat - array with the first day, duration, mean gradient, total
                    precipitation, wet and dry spells of the selected onset periods
            othrdat - array with the first day, duration, mean gradient, total
                      precipitation, wet and dry spells of the onset periods that
                      do not match the climatological onset periods
            onsmask - mask with the onset periods   
    '''

    nyrs=len(onsperiod.keys()); yr0=np.min(list(onsperiod.keys()))
    onsdat=np.zeros((nyrs,ns,4)); onsdat[:]=np.nan
    othrdat=np.zeros((nyrs,5,6)); othrdat[:]=np.nan

    tmpmask=np.zeros((dtime.shape[0]))
    for yr in onsperiod.keys():
        #Onset periods - 1
        for ss in range(ns):
            if(not np.isnan(onsperiod[yr][ss][0])):
                ps=np.arange(onsperiod[yr][ss][0],onsperiod[yr][ss][0]+onsperiod[yr][ss][1])
                tmpmask[ps]=1
                
                #onset day and length for each year
                onsdat[yr-yr0,ss,:]=onsperiod[yr][ss][:4]
        #Other onsets
        if(len(otheronset[yr]) > 0):
            ss=0
            for itt,types in enumerate(['false onset','second onset','wet spell','other']):
                vld=(np.asarray([otheronset[yr][jj][5] for jj in range(len(otheronset[yr]))]) == types).nonzero()[0]
                if(len(vld) > 0):
                    for jj in vld:
                        ps=np.arange(otheronset[yr][jj][0],otheronset[yr][jj][0]+otheronset[yr][jj][1])
                        tmpmask[ps]=itt+2

                        othrdat[yr-yr0,ss,:]=otheronset[yr][jj][:5]+(itt+2,)
                        ss+=1

    return onsdat,othrdat,tmpmask


def WetDrySpells(msk,pp,ns,ons):
    '''
        Function to estimate wet and dry spell statistics during onset periods:
            PrecipRate: Total precipitation divided by the length of the onset period
            PrecipInt: Total precipitation divided by the number of wet days in the onset period
            FrcWetDays: Number of wet days divided by the length of the onset period
            LengthWetSpell: number of wet days (PP >= 1) divided by the number of wet spells (sequence of wet days)
            MaxWetSpell: longest wet spell
            FrcDryDays: Number of dry days divided by the length of the onset period
            LengthDrySpell: number of dry days (PP < 1) divided by the number of dry spells (sequence of dry days)
            MaxDrySpell: longest dry spell
        
        INPUT:
            msk - mask with the onset periods
            pp - precipitation time series
            ns - number of onset periods
            ons - onset periods
        
        RETURNS:
            onsStats - array with the statistics of the onset periods
    '''

    ntot=len(pp)
    nyrs=ons.shape[0]

    onsStats=np.zeros((nyrs,ns,8)); onsStats[:]=np.nan

    yrmsk=np.where(msk == 1,1,np.nan)
    vldpp=pp*yrmsk
    
    vldpp[vldpp < 1]=0.

    wet=np.where(vldpp > 0,1.,0)
    dd0=(np.diff(wet)==1).nonzero()[0]+1; 
    ddf=(np.diff(wet)==-1).nonzero()[0]
    #note that ddf does not have the +1 because diff[i]=x[i+1]-x[i].
    if(wet[0]==1):dd0=np.insert(dd0,0,0)
    if(wet[-1]==1):ddf=np.insert(ddf,len(ddf),ntot-1)
    wetlen=[(ii,jj,vldpp[int(ii):int(jj)+1]) for ii,jj in zip(dd0,ddf)]

    #dry days
    dry=np.where(vldpp == 0,1.,0)
    dd0=(np.diff(dry)==1).nonzero()[0]+1; 
    ddf=(np.diff(dry)==-1).nonzero()[0]
    #note that ddf does not have the +1 because diff[i]=x[i+1]-x[i].
    if(dry[0]==1):dd0=np.insert(dd0,0,0)
    if(dry[-1]==1):ddf=np.insert(ddf,len(ddf),ntot-1)
    drylen=[(ii,jj,vldpp[int(ii):int(jj)+1]) for ii,jj in zip(dd0,ddf)]

    #Matching the characteristics to the periods
    for yr in range(nyrs):
        for ss in range(int(ns)):
            if(np.isfinite(ons[yr,ss,0])):
                #onset period
                di=ons[yr,ss,0]
                df=di+ons[yr,ss,1]                       
                
                #wet spell stats
                vld=[ii for ii in range(len(wetlen)) if ((wetlen[ii][0] >= di) & (wetlen[ii][1] <= df))] 

                if(len(vld) > 0):
                    #total precipitation during the onset period
                    totpp=sum([sum(wetlen[ii][2]) for ii in vld])
                    #note that totpp may differ from the total 
                    #precipitation calculated previously because it 
                    #does not include values below 1mm/day

                    #number of wet days
                    nrwet=sum([len(wetlen[ii][2]) for ii in vld])
                    #Precipitation rate: total precip during wet days 
                    #divided by duration of the onset period
                    onsStats[yr,ss,0]=totpp/ons[yr,ss,1]

                    #Precipitation intensity: total precipitation 
                    #during wet days divided by the number of wet days 
                    #during the onset period
                    onsStats[yr,ss,1]=totpp/float(nrwet)

                    #Fraction of wet days: number of wet days divided 
                    #by the duration of the onset period
                    onsStats[yr,ss,2]=100*(nrwet/ons[yr,ss,1])

                    #Mean duration of the wet spells = number of wet 
                    #days divided by the number of wet spells
                    onsStats[yr,ss,3]=nrwet/float(len(vld))
                    
                    #Longest wet spell in the onset period
                    onsStats[yr,ss,4]=max([len(wetlen[ii][2]) for ii in vld])

                else: onsStats[yr,ss,:5]=0.

                #Dry spell stats
                vld=[ii for ii in range(len(drylen)) if ((drylen[ii][0] >= di) & (drylen[ii][1] <= df))] 

                if(len(vld) > 0):
                    #number of dry days
                    nrdry=sum([len(drylen[ii][2]) for ii in vld])
                    
                    #Fraction of dry days: number of dry days divided 
                    #by the duration of the onset period
                    onsStats[yr,ss,5]=100*(nrdry/ons[yr,ss,1])
                    
                    #Mean duration of the dry spells = number of dry
                    #days divided by the number of dry spells
                    onsStats[yr,ss,6]=nrdry/float(len(vld))
                    #Longest dry spell in the onset period
                    onsStats[yr,ss,7]=max([len(drylen[ii][2]) for ii in vld])
                else: onsStats[yr,ss,5:]=0.
    
    return onsStats


############################
#Functions related to onset date analysis based on Bombardi et al 2019 (B19)
############################

def AnnualCycleB19(pp,jday,mask,hcast=False):

    if hcast:
        cycle=pp
    else:
        cycle=np.asarray([np.nanmean(pp[jday == tt,:,:]*mask,0) for tt in np.unique(jday)])

    #FFT for annual cycle
    yrfft=np.fft.rfftn(cycle*mask,axes=(0,))
    ampl=2*np.abs(yrfft/float(cycle.shape[0]))**2 #Amplitude of the harmonics
    Zampl=ampl/np.sum(ampl,0) #Normalizing the amplitude. 

    #explained variance of each harmonic
    svar=np.nanvar(cycle,0,ddof=1) #Variance of the data
    frcharm=ampl/svar
    
    #Removing regionswhere the explained variance of the second and 
    #third harmonics are larger than the first harmonic one
    mask=np.where(frcharm[2,...] >= frcharm[1,...],np.nan,mask)
    mask=np.where(frcharm[3,...] >= frcharm[1,...],np.nan,mask)

    # Calculating the day [day of year] that will be used as starting point (t0) for
    # the calculation of the rainy and dry seasons characteristics
    #Using only the first harmonic
    harmonic1=yrfft.copy()
    harmonic1[2:,...]=0. #First harmonic
    harmonic1=np.fft.irfftn(harmonic1,axes=(0,))

    w0=(np.argmin(harmonic1,0)+1)*mask

    return cycle,w0,mask,Zampl

def SeasonOnsetB19(ndays,jday,sday,data,time,dryseason=False,hcast=False):
    
    nds=ndays if hcast else ndays/2
    if(dryseason):
        tmpjday=jday[::-1]
        tmptime=time[::-1,:]
        data=data[::-1]
    else:
        tmpjday=jday
        tmptime=time
     
    vld=(tmpjday == sday).nonzero()[0] #Equivalent to position on time array
    
    mtot=len(data)
    curve=np.zeros((len(vld),int(nds)))
    sjday=np.zeros((len(vld)))
    sdate=np.zeros((len(vld),3)) #year, month and day for each year

    for yt,tt in enumerate(vld):
        # Starting the calculation of accumulated anomalies in the rainy season
        if tt < (mtot-15):         # -15 to avoid calcualtion with short time series for last year
            ned=tt+int(nds)
            if ned > mtot: ned=mtot-1 #Beyond the end of the timeseries
            curve[yt,0:ned-tt]=np.cumsum(data[tt:ned])

            # Calculating onset of the season
            if not np.all(np.isnan(curve[yt,:])):
                ons=np.nanargmin(curve[yt,:])+tt   #position in time array
                if(ons == mtot): ons-=1

                if(hcast): #Using hindcast data (less than 365 days)
                    if sday<184: #within the forecast window
                        if yt < (len(vld)-1):
                            if tmpjday[ons]>1 and tmpjday[ons]<185:
                                sjday[yt]=tmpjday[ons]
                                sdate[yt,:]=tmptime[ons,:]
                            else:
                                sjday[yt]=np.nan
                                sdate[yt,:]=np.nan
                        else:
                            if tmpjday[ons]>1:
                                sjday[yt]=tmpjday[ons]
                                sdate[yt,:]=tmptime[ons,:]
                            else:
                                sjday[yt]=np.nan
                                sdate[yt,:]=np.nan                    
                    else:
                        sjday[yt]=np.nan
                        sdate[yt,:]=np.nan
            
                else:
                    sjday[yt]=ons #testing using days since yr0 instead of julian day
                    #sjday[yt]=tmpjday[ons]
                    sdate[yt,:]=tmptime[ons,:]
            else:
                sjday[yt]=np.nan
                sdate[yt,:]=np.nan
                
    if(dryseason):
        if(hcast):
            sjday=sjday[::-1]
        else:
            sjday=mtot-sjday[::-1]-1

        sdate=sdate[::-1,:]
        curve=curve[::-1,:]
        if(np.isnan(sjday[0])):  #Case of a missing value at the end of the ts (control has one day less in yrf)
            sjday=np.roll(sjday,len(sjday)-1)
            sdate=np.roll(sdate,len(sjday)-1,axis=0)
            curve=np.roll(curve,len(sjday)-1,axis=0)

    if(hcast):
        return sjday[-1], sjday[:-1], sdate[:-1,...], curve[:-1,...]
    else:
        return sjday, sdate, curve

def SeasonOnsetB17(ytot,jday,sday,data,time,outl,npass=50,dryseason=False):
    
    mtot=len(data)
    sjday=np.zeros((len(outl)))  #Only for outliers
    sdate=np.zeros((len(outl),3)) #year, month and day for each year

    if(dryseason):
        tmpjday=jday[::-1]
        tmptime=time[::-1,:]
        data=data[::-1]
    else:
        tmpjday=jday
        tmptime=time

    vld=(tmpjday == sday).nonzero()[0]
    #idx=vld[-(outl+1)] if dryseason else vld[outl] #Flipping the order of the years if dry season. 
    idx=vld[outl]
    for yt,tt in enumerate(idx): #executing only for years considered outliers
        # Starting the calculation of accumulated anomalies in the rainy season
        if tt < mtot-15:         # -5 to avoid calcualtion with short time series for last year
            ned=tt+ytot
            if ned > mtot-1: ned=mtot-1

            sseries=np.cumsum(data[tt:ned])
            if ned == mtot-1:  #NEW
                tmp=sseries.copy()
                if 2*sseries.shape[0] > ytot:  #If the series just need a small increment
                    sseries=np.hstack((sseries,tmp[::-1][0:ytot-sseries.shape[0]]))
                else: #If the series is smaller than 1/4 year 
                    sseries=np.hstack((sseries,tmp[::-1]))

            #------Starting Bombardi and Carvalho (2009) adaptation
            # Smoothing the time series of accumulated anomalies
            ssmooth=butterworth(sseries,18) 
            #using a cutoff of 15 I obtained similar results as using a 1-2-1 filter passed 50 times, as in Bombardi

            # Calculating the first derivative of sseries
            dsdt=np.gradient(ssmooth)
        
            # Calculating onset and demise of the rainy season
            stp=2  #size of the window: negative stp times before and positive stp times after the candidate onset
            beg=[ii for ii in range(stp,dsdt.shape[0]-stp) \
                    if ((np.all(dsdt[ii-stp:ii] < 0)) & \
                        (np.all(dsdt[ii+1:ii+stp+1] > 0) & (dsdt[ii] <= 0)))]

            if(len(beg) > 0):
                beg=beg[0]+tt
                if(beg>= mtot): beg=mtot-1
                sjday[yt]=beg #testing using days since yr0 instead of julian day
                # sjday[yt]=tmpjday[beg]
                sdate[yt]=tmptime[beg,:]

    if(dryseason):
        sjday=mtot-sjday[::-1]-1
        sdate=sdate[::-1,:]
        
        
    return sjday, sdate

def QCB19(indat,sday,ndays,thrs=1.5):

    #shifting all values to be in the same water year
    data=np.where(indat < sday,indat+ndays,indat)  
    
    target=np.nanquantile(data,0.5)  #Median onset date
    if(target < sday): target=target+ndays #Shifting dates around w0

    outl,iqr=CheckOutliers(data,ndays,thrs=thrs,target=target)

    return outl,iqr


def LenghtTotPPB19(wjd,wdate,djd,ddate,pp,ntot,unts,cal):
    
    ndays=360 if cal == 'noleap' else 365
    nyrs=len(wjd)

    dwet=np.zeros((nyrs+1)); twet=np.zeros((nyrs+1))
    ddry=np.zeros((nyrs+1)); tdry=np.zeros((nyrs+1))

    von=(wjd >= 0).nonzero()[0]
    vdm=(djd >= 0).nonzero()[0]

    ons=[ii for ii in wjd[von]]; dms=[ii for ii in djd[vdm]]
    # if(sys.version_info[0] == 2):
    #     #In python 2.7, the cftime.datetime function is not calendar-aware. Thus, a date
    #     #such as 2000/02/30 won't raise an error. In python 3 and above, this is only
    #     #accepted if calendar='360_days'.
    #     ons=[nc.date2num(cftime.datetime(int(wdate[yy,0]),int(wdate[yy,1]),int(wdate[yy,2])),\
    #                     units=unts,calendar=cal) for yy in von]

    #     dms=[nc.date2num(cftime.datetime(int(ddate[yy,0]),int(ddate[yy,1]),int(ddate[yy,2])),\
    #                     units=unts,calendar=cal) for yy in vdm]
    # else:
    #     ons=[nc.date2num(cftime.datetime(int(wdate[yy,0]),int(wdate[yy,1]),int(wdate[yy,2]),calendar=cal),\
    #                     units=unts,calendar=cal) for yy in von]

    #     dms=[nc.date2num(cftime.datetime(int(ddate[yy,0]),int(ddate[yy,1]),int(ddate[yy,2]),calendar=cal),\
    #                     units=unts,calendar=cal) for yy in vdm]


    #Removing instances with onset and demise at the same day/year
    bth=np.isin(np.asarray(ons),np.asarray(dms)).nonzero()[0]
    if(len(bth) > 0):
        #bth refers to positions in ons list. Now finding these values on dms list
        btd=np.isin(np.asarray(dms),np.asarray(ons)[bth]).nonzero()[0]
        #removing entries from each list:
        tmp=[ons.remove(ii) for ii in np.asarray(ons)[bth]]
        tmp=[dms.remove(ii) for ii in np.asarray(dms)[btd]]


    #Merging, sorting and adding 0 to the start and ntot to the end
    if((len(ons) > 0) & (len(dms) > 0)):
        dts=ons+dms+[ntot]; dts.sort()           
        if(dts[0] > 0): dts=[0]+dts
        length=np.asarray(dts)[1:]-np.asarray(dts[:-1])

        flgsea=np.zeros((len(dts)-1),dtype='str')
        flgsea[np.isin(np.asarray(dts),np.asarray(ons)).nonzero()[0]]='w'
        flgsea[np.isin(np.asarray(dts),np.asarray(dms)).nonzero()[0]]='d'
        flgsea[0] = 'w' if flgsea[1] == 'd' else 'd'

        dy=0; wy=0
        for yy,fl in enumerate(flgsea):
            #extent and total precipitation during the season
            #Check if it is a full season (discard begining of first and end of last year) or if 
            #an onset/demise was not captured (very long season)
            if((length[yy] > .2*ndays) & (length[yy] < .9*ndays)): 
                # Since considering only one rainy season, the season should be longer than .33 of the year. 
                # This should be changed when considering two rainy seasons
                accum=np.nansum(pp[int(dts[yy]):int(dts[yy+1])])
                #Id the season
                if(fl == 'w'):
                    #print yy,wy,fl,length[yy],accum
                    dwet[wy]=length[yy]
                    twet[wy]=accum
                else:
                    #print yy,dy,fl,length[yy],accum
                    ddry[dy]=length[yy]
                    tdry[dy]=accum
            if(fl == 'w'): wy+=1
            else: dy+=1

    #Removing missing
    twet[dwet == 0]=np.nan
    tdry[ddry == 0]=np.nan
    dwet[dwet == 0]=np.nan
    ddry[ddry == 0]=np.nan

    return dwet,twet,ddry,tdry


############################
#Loading and saving onset data
############################

def loadonset(infile,ppfile):
    '''
        Loads the variables related to the onset periods

        INPUT:
            infile - name of the netcdf file with the onset periods
            ppfile - name of the netcdf file with the filtered precipitation

        RETURNS:
        yr0,lon,lat - first year of data, longitude and latidute arrays
        nrrainy - array with the number of onset seaons per pixel
        onsrainy - climatological period with larger probability of onset
                   per pixel and onset period
        aveonset - first day and duration of each onset period per year
                    pixel and onset period
        onsmask - array flagging the onset periods classifed as 1 - 5.
        fltpp - filtered precipitation
    '''


    inp=nc.Dataset(infile,'r')

    yr0=int(inp.variables['time'].units.split(' ')[2].split('-')[0])+1  
    #First water year starts on the year -1 

    lon=inp.variables['lon'][:]; lat=inp.variables['lat'][:]

    #Climatology data
    nrrainy=inp.groups['climatol'].variables['NrRainy'][:]
    onsrainy=inp.groups['climatol'].variables['Onset'][:]

    #Annual data
    aveonset=inp.groups['onset'].variables['AveOnset'][:]
    onsmask=inp.groups['onset'].variables['OnsetMask'][:]

    onsmask=np.moveaxis(onsmask,0,-2)

    inp.close()

    #Filtered precipitation
    inp=nc.Dataset(ppfile,'r')
    fltpp=inp.variables['PP'][:]
    inp.close()

    return yr0,lon,lat,nrrainy,onsrainy,aveonset,onsmask,fltpp


def savenc(indat,outfile,fftpp=False,dfftpp=False):
    '''
        Function to save the onset period data in a netcdf file
        Used for observational datasets and models (GCM, RCM, CPM)

        INPUT:
            indat - dictionary with the onset period data
            outfile - name of the netcdf file to be created
            fftpp - dictionary with info related to filtered precipitation. 
                    If included, it is saved in its onw netcdf file
            dfftpp - dictionary with info related to the first derivative 
                     of the filtered precipitation. If included, it is saved 
                     in its own netcdf file
    '''

    lons=indat['lon']; nlon=len(lons)
    lats=indat['lat']; nlat=len(lats)
    ndays=indat['ndays']

    print("Saving onset period file")

    outnc = nc.Dataset(outfile, "w", format="NETCDF4")

    outnc.createDimension("lon", nlon)
    outnc.createDimension("lat", nlat)
    outnc.createDimension('jday',ndays)
    outnc.createDimension("time",None)

    #Creating coordinates
    time = outnc.createVariable("time","i4","time")
    lat = outnc.createVariable("lat","f4","lat")
    lon = outnc.createVariable("lon","f4","lon")
    jday = outnc.createVariable('jday','i4','jday')

    # Filling coordinates
    lon[:]=lons
    lon.units='degrees_east'; lon.long_name='longitude'
    lat[:]=lats
    lat.units='degrees_north'; lat.long_name='latitude'
    jday[:]=range(1,ndays+1)
    jday.units='days'; jday.long_name='Julian days (1-'+str(ndays+1)+')'

    time.long_name='Time'
    time.units='common_years since '+str(indat['yr0']-1)+'-01-01 00:00'
    time.calendar=indat['cal']
    time[:]=list(range(0,indat['nyrs']+1,1))

    #--------Climatological data group
    climatol=outnc.createGroup('climatol')

    climatol.createDimension('sea',indat['mxrainy'])
    climatol.createDimension('stats',2)
    climatol.createVariable('stats',str,'stats')
    climatol.variables['stats'][:]=np.asarray(['FirstDay','Length'])

    #Nr of rainy seasons per grid point
    climatol.createVariable("NrRainy","f4", ("lat","lon",),zlib=True)
    climatol.variables['NrRainy'].long_name = 'Number of rainy seasons'
    climatol.variables['NrRainy'][:] = indat['nrrainy']
    climatol.variables['NrRainy'].units = ''
    climatol.variables['NrRainy'].description='Number of rainy seasons in each grid point'

    #onset period - first day and length per rainy season
    climatol.createVariable("Onset","f4", ("lat","lon","sea","stats",),zlib=True)
    climatol.variables['Onset'].long_name = 'First day (julian day) and length of each climatological onset period'
    climatol.variables['Onset'][:] = indat['onsrainy']
    climatol.variables['Onset'].description='First day (julian day) and length of each onset period per grid point (First guess)'

    #--------Annual analysis group
    onset=outnc.createGroup('onset')

    onset.createDimension('stats1',len(indat['nstats']))
    onset.createDimension('stats2',6)
    onset.createDimension('qq',5)
    onset.createDimension('sea',indat['mxrainy'])
    onset.createDimension('sea2',6)
    onset.createDimension("hrtime",indat['htime'].shape[0])

    onset.createVariable('stats1',str,'stats1')
    onset.variables['stats1'][:]=np.asarray(indat['nstats'])

    onset.createVariable('stats2',str,'stats2')
    onset.variables['stats2'][:]=np.asarray(['first day','length','intensity','tot.precip','year','type'])

    onset.createVariable('qq','f4','qq')
    onset.variables['qq'][:]=np.asarray([0.05,0.25,0.50,0.75,0.95])

    onset.createVariable("hrtime","f4","hrtime")
    onset.variables['hrtime'].long_name='time'
    onset.variables['hrtime'].units=indat['unts']
    onset.variables['hrtime'].calendar=indat['cal']
    onset.variables['hrtime'][:]=indat['htime']
    onset.variables['hrtime'].description='Time array to transform the first day of the onset period into dates'

    #onset period per year - first day and length
    onsetdata=np.moveaxis(indat['onsetdata'],2,0)

    onset.createVariable("Onset","f4", ("time","lat","lon","sea","stats1",),zlib=True)
    onset.variables['Onset'].long_name = 'Statistics for each onset period'
    onset.variables['Onset'][:] = onsetdata
    onset.variables['Onset'].description = 'Statistics for each rainy season onset per grid point and year as in stats1'

    #Other onset periods - first day, length and type
    otherdata=np.moveaxis(indat['otherdata'],2,0)

    onset.createVariable("OtherOns","f4", ("time","lat","lon","sea2","stats2",),zlib=True)
    onset.variables['OtherOns'].long_name = 'First day (julian day), length, intensity, year (-1,0,1), and type of other onset periods'
    onset.variables['OtherOns'][:] = otherdata
    onset.variables['OtherOns'].description = "First day (julian day), length, intensity, year, and type of other onset periods "\
                                            "per grid point and year. Types are: false onset (2), second onset (3), "\
                                            "wet spell (4), other (5)"

    #Quantiles (5th, 25th, 50th, 75th, 95th) of onset day and length
    onset.createVariable('AveOnset','f4',("lat","lon","sea","qq","stats1",),zlib=True)
    onset.variables['AveOnset'].long_name = 'Quantiles for statistics during onset periods'
    onset.variables['AveOnset'][:] = indat['aveonset']
    onset.variables['AveOnset'].description = "5th, 25th, 50th, 75th, and 95th percentiles for statistics in each "\
                                              "onset period per grid point."

    #Mask indicating the periods of onset in each year and grid point
    onsmask=np.moveaxis(indat['onsmask'],2,0)
    onset.createVariable('OnsetMask','i4',("time","lat","lon","jday",),zlib=True)
    onset.variables['OnsetMask'].long_name = 'Mask with onset periods'
    onset.variables['OnsetMask'][:] = onsmask
    onset.variables['OnsetMask'].description = "Mask with 1 indicating each onset period, 2 indicating "\
                                            "false onset, 3 indicading second onset, 4 indicating wet spells, "\
                                            "and 5 indicating other onsets per year in each grid point"

    #Global attributes
    outnc.description = indat['description']

    outnc.history=datetime.date.today().strftime("%d/%m/%y")

    outnc.close()

    #########
    #Saving filtered precipitation and its derivatives

    if(not isinstance(fftpp,bool)):

        print("Saving filtered precipitation data")

        outnc = nc.Dataset(fftpp['outfile'], "w", format="NETCDF4")

        # Creating dimensions
        outnc.createDimension("lon", nlon)
        outnc.createDimension("lat", nlat)
        outnc.createDimension("time",None)

        #Creating coordinates
        time = outnc.createVariable("time","i4","time")
        lat = outnc.createVariable("lat","f4","lat")
        lon = outnc.createVariable("lon","f4","lon")
        pp = outnc.createVariable("PP","f4",("time","lat","lon"),zlib=True)

        # Filling coordinates
        lon[:]=lons
        lon.units='degrees_east'; lon.long_name='longitude'
        lat[:]=lats
        lat.units='degrees_north'; lat.long_name='latitude'

        time.long_name='Time'
        time.units=indat['unts']; time.calendar=indat['cal']
        time[:]=indat['htime']

        #FFT precip
        pp.long_name='FFT Daily Precipitation'
        pp.units='mm.day^{-1}'
        pp.description='FFT daily precipitation (between 75 and 400 days)'
        pp[:]=fftpp['fftpp']

        #Global attributes
        outnc.description = fftpp['description']
        outnc.history=datetime.date.today().strftime("%d/%m/%y")

        outnc.close()

    ####First derivative
    if(not isinstance(dfftpp,bool)):

        print("Saving first derivative of the filtered precipitation data")

        outnc = nc.Dataset(dfftpp['outfile'], "w", format="NETCDF4")

        # Creating dimensions
        outnc.createDimension("lon", nlon)
        outnc.createDimension("lat", nlat)
        outnc.createDimension("time",None)

        #Creating coordinates
        time = outnc.createVariable("time","i4","time")
        lat = outnc.createVariable("lat","f4","lat")
        lon = outnc.createVariable("lon","f4","lon")
        dpdt = outnc.createVariable("PP","f4",("time","lat","lon"),zlib=True)

        # Filling coordinates
        lon[:]=lons
        lon.units='degrees_east'; lon.long_name='longitude'
        lat[:]=lats
        lat.units='degrees_north'; lat.long_name='latitude'

        time.long_name='Time'
        time.units=indat['unts']; time.calendar=indat['cal']
        time[:]=indat['htime']

        #FFT precip first derivative
        dpdt.long_name='First derivative of the FFT Daily Precipitation'
        dpdt.units='mm.day^{-1}'
        dpdt.description='First derivative of the FFT daily precipitation (between 75 and 400 days)'
        dpdt[:]=dfftpp['dfftpp']

        #Global attributes
        outnc.description = dfftpp['description']

        outnc.history=datetime.date.today().strftime("%d/%m/%y")

        outnc.close()

def savencEM(indat,outfile):
    '''
        Function to save a variable in a netcdf file
        Variable should have a time and an ensemble dimension
        Used in Seasonal and Subseasonal forecasts and hindcasts only

        INPUT:
            indat - dictionary with the variable to be saved
            outfile - name of the netcdf file to be created
    '''

    lons=indat['lon']; nlon=len(lons)
    lats=indat['lat']; nlat=len(lats)
    ndays=indat['ndays']; nens=indat['nens']
    varn=indat['varn']

    outnc = nc.Dataset(outfile, "w", format="NETCDF4")

    # Creating dimensions
    outnc.createDimension("lon", nlon)
    outnc.createDimension("lat", nlat)
    outnc.createDimension("time",ndays)
    outnc.createDimension("number",nens)

    #Creating coordinates
    number = outnc.createVariable("number","i4","number")
    time = outnc.createVariable("time","i4","time")
    lat = outnc.createVariable("lat","f4","lat")
    lon = outnc.createVariable("lon","f4","lon")
        
    # Filling coordinates
    lon[:]=lons
    lon.units='degrees_east'; lon.long_name='longitude'
    lat[:]=lats
    lat.units='degrees_north'; lat.long_name='latitude'

    time.long_name='Time'
    time.units=indat['unts']; time.calendar=indat['cal']
    time[:]=indat['dtime']

    number.long_name='Ensemble members'
    number[:]=np.asarray(range(nens))

    #Main variable    
    outnc.createVariable(varn,"f4",("time","number","lat","lon"),zlib=True)

    outnc.variables[varn].long_name=indat['vlong']
    outnc.variables[varn].units=indat['vunit']
    outnc.variables[varn][:]=indat['indat']

    #Global attributes
    outnc.description = indat['description']
    outnc.history=datetime.date.today().strftime("%d/%m/%y")

    outnc.close()

def saveOnsStatsEM(indat,outfile,fileform):
    '''
        Function to save the statistics for all onset period candidates
        in a netcdf file
        Used only in Seasonal and Subseasonal forecasts and hindcasts

        INPUT:
            indat - dictionary with the statistics to be saved
            outfile - name of the netcdf file to be created
            fileform - format of the data to be saved. Can be:
                AllCamndidates, from OnsetPeriod_IDCandidates.py
                Climatol, from OnsetPeriod_Climatology.py
    '''

    lons=indat['lon']; nlon=len(lons)
    lats=indat['lat']; nlat=len(lats)

    outnc = nc.Dataset(outfile, "w", format="NETCDF4")

    # Creating dimensions
    outnc.createDimension("lon", nlon)
    outnc.createDimension("lat", nlat)

    #Creating coordinates
    lat = outnc.createVariable("lat","f4","lat")
    lon = outnc.createVariable("lon","f4","lon")
        
    # Filling coordinates
    lon[:]=lons
    lon.units='degrees_east'; lon.long_name='longitude'
    lat[:]=lats
    lat.units='degrees_north'; lat.long_name='latitude'

    if fileform == 'AllCandidates':

        #Extra dimensions
        nens=indat['nens']; nstats=len(indat['nstats'])
        outnc.createDimension("stats",nstats)
        outnc.createDimension("sea",6)
        outnc.createDimension("number",nens)

        #Creating coordinates
        number = outnc.createVariable("number","i4","number")
        stats = outnc.createVariable("stats",str,"stats")

        # Filling coordinates
        stats.long_name='Stats per onset period candidate'
        stats[:]=np.asarray(indat['nstats'])

        number.long_name='Ensemble members'
        number[:]=np.asarray(range(nens))

        #Variables
        ons=outnc.createVariable('ons',"f4",("lat","lon","number","sea","stats"),zlib=True)

        ons.long_name='Statistics for each onset period candidates'
        ons[:]=indat['indat']

    elif fileform == 'Climatol':
            
        #Extra dimensions
        outnc.createDimension("sea",indat['mxrainy'])
        outnc.createDimension("stats",2)

        #Creating coordinates
        stats = outnc.createVariable("stats",str,"stats")
        stats[:]=np.asarray(['FirstDay','Length'])

        #Variables
        nrrainy=outnc.createVariable("NrRainy","f4",("lat","lon",),zlib=True)
        nrrainy.long_name = 'Number of rainy seasons'
        nrrainy.units = ''
        nrrainy.description='Number of rainy seasons in each grid point'
        nrrainy[:]=indat['nrrainy']

        onsrainy=outnc.createVariable("Onset","f4",("lat","lon","sea","stats",),zlib=True)
        onsrainy.long_name = 'First day (julian day) and length of each climatological onset period'
        onsrainy[:]=indat['onsrainy']

    elif fileform == 'PerYear':

        #Extra dimensions
        nens=indat['nens']; nstats=len(indat['nstats'])
        mxrainy=indat['mxrainy']; ndays=indat['ndays']

        outnc.createDimension("stats",nstats)
        outnc.createDimension("sea",mxrainy)
        outnc.createDimension("number",nens)
        outnc.createDimension("jday",ndays)
        outnc.createDimension("stats2",5)
        outnc.createDimension("sea2",6)

        #Creating coordinates
        stats=outnc.createVariable('stats',str,'stats')
        stats[:]=np.asarray(indat['nstats'])

        stats2=outnc.createVariable('stats2',str,'stats2')
        stats2[:]=np.asarray(['first day','length','intensity','tot.precip','type'])

        number=outnc.createVariable('number','i4','number')
        number[:]=np.asarray(range(nens))

        #Variables
        outnc.createVariable("Onset","f4", ("number","lat","lon","sea","stats",),zlib=True)
        outnc.variables['Onset'].long_name = 'Statistics for each onset period'
        outnc.variables['Onset'][:] = indat['onsetdata']
        outnc.variables['Onset'].description = 'Statistics for each rainy season onset per grid point and year as in stats'

        outnc.createVariable("OtherOns","f4", ("number","lat","lon","sea2","stats2",),zlib=True)
        outnc.variables['OtherOns'].long_name = 'First day (julian day), length, intensity, year (-1,0,1), and type of other onset periods'
        outnc.variables['OtherOns'][:] = indat['otherdata']
        outnc.variables['OtherOns'].description = "First day (julian day), length, intensity, year, and type of other onset periods "\
                                                "per grid point and year. Types are: false onset (2), second onset (3), "\
                                                "wet spell (4), other (5)"

        #Mask indicating the periods of onset in each year and grid point
        outnc.createVariable('OnsetMask','i4',("number","lat","lon","jday",),zlib=True)
        outnc.variables['OnsetMask'].long_name = 'Mask with onset periods'
        outnc.variables['OnsetMask'][:] = indat['onsmask'],
        outnc.variables['OnsetMask'].description = "Mask with 1 indicating each onset period, 2 indicating "\
                                                "false onset, 3 indicading second onset, 4 indicating wet spells, "\
                                                "and 5 indicating other onsets per year in each grid point"

    else: print('Format not recognized')

    #Global attributes
    outnc.description = indat['description']
    outnc.history=datetime.date.today().strftime("%d/%m/%y")

    outnc.close()


def loadOnsDateB19(infile):

    inp=nc.Dataset(infile,'r')

    lon=inp.variables['lon'][:]; lat=inp.variables['lat'][:]
    cal=inp.variables['time'].calendar

    w0=inp.groups['onset'].variables['w0'][:]
    onset_jday=inp.groups['onset'].variables['DOY'][:].data
    onset_jday[onset_jday > 400]=np.nan
    demise_jday=inp.groups['demise'].variables['DOY'][:].data
    demise_jday[demise_jday > 400]=np.nan

    durwet=inp.groups['wetseason'].variables['Duration'][:]
    totwet=inp.groups['wetseason'].variables['TotPrecip'][:]
    durdry=inp.groups['dryseason'].variables['Duration'][:]
    totdry=inp.groups['dryseason'].variables['TotPrecip'][:]
    hvar=inp.groups['fft'].variables['Amplitude'][:]

    inp.close()

    return lon,lat,cal,w0,onset_jday,demise_jday,durwet,totwet,durdry,totdry,hvar

def saveOnsDateB19(indat,outfile):

    outnc = nc.Dataset(outfile, "w", format="NETCDF4")

    lons=indat['lon']; nlon=len(lons)
    lats=indat['lat']; nlat=len(lats)
    yr0=indat['yr0']; nyrs=len(indat['onset_jday'])

    # Creating dimensions
    outnc.createDimension("lon", nlon)
    outnc.createDimension("lat", nlat)
    outnc.createDimension('date',3)
    outnc.createDimension("time",None)

    #Creating coordinates
    time = outnc.createVariable("time","i4","time")
    lat = outnc.createVariable("lat","f4","lat")
    lon = outnc.createVariable("lon","f4","lon")
    date = outnc.createVariable('date',str,'date')

    # Filling coordinates
    lon[:]=lons; lat[:]=lats
    date[:]=np.asarray(['YYYY','MM','DD'])
    lon.units='degrees_east'; lon.long_name='longitude'
    lat.units='degrees_north'; lat.long_name='latitude'
    date.units=''; date.long_name='Date (YYYY MM DD)'

    time.long_name='Time'
    time.units='common_years since '+str(yr0)+'-01-01 00:00'
    time.calendar=indat['cal']
    time[:]=range(0,nyrs+1,1)

    #Onset info - group
    onset=outnc.createGroup('onset')

    #Start of the water year
    onset.createVariable("w0","f4", ("lat","lon",),zlib=True)
    onset.variables['w0'].long_name = 'Start of the water year'
    onset.variables['w0'][:] = indat['startwet']

    
    #onset day of the year
    onset.createVariable("DOY","f4", ("time","lat","lon",),zlib=True)
    onset.variables['DOY'].long_name = 'Wet season onset [Day of Year]'
    onset.variables['DOY'][:] = indat['onset_jday']

    #onset date
    onset.createVariable("date","f4", ("time","lat","lon","date"),zlib=True)
    onset.variables['date'].long_name = 'Wet season onset [YYYY MM DD]'
    onset.variables['date'][:] = indat['onset_date']

    #Demise info - group
    demise=outnc.createGroup('demise')

    #onset day of the year
    demise.createVariable("DOY","f4", ("time","lat","lon",),zlib=True)
    demise.variables['DOY'].long_name = 'Wet season demise [Day of Year]'
    demise.variables['DOY'][:] = indat['demise_jday']

    demise.createVariable("date","f4", ("time","lat","lon","date"),zlib=True)
    demise.variables['date'].long_name = 'Wet season demise [YYYY MM DD]'
    demise.variables['date'][:] = indat['demise_date']

    #wet season duration and accum precip - group
    wetseason=outnc.createGroup('wetseason')

    #duration
    wetseason.createVariable('Duration','f4',("time","lat","lon",),zlib=True)
    wetseason.variables['Duration'].long_name = 'Duration of the wet season [days]'
    wetseason.variables['Duration'][:] = indat['durwet']

    #total precipitation
    wetseason.createVariable('TotPrecip','f4',("time","lat","lon",),zlib=True)
    wetseason.variables['TotPrecip'].long_name = 'Total Precipitation during the wet season [mm]'
    wetseason.variables['TotPrecip'][:] = indat['totwet']

    #dry season duration and accum precip - group
    dryseason=outnc.createGroup('dryseason')

    #duration
    dryseason.createVariable('Duration','f4',("time","lat","lon",),zlib=True)
    dryseason.variables['Duration'].long_name = 'Duration of the dry season [days]'
    dryseason.variables['Duration'][:] = indat['durdry']

    #total precipitation
    dryseason.createVariable('TotPrecip','f4',("time","lat","lon",),zlib=True)
    dryseason.variables['TotPrecip'].long_name = 'Total Precipitation during the dry season [mm]'
    dryseason.variables['TotPrecip'][:] = indat['totdry']

    #Harmonics and coeff group
    fft=outnc.createGroup('fft')

    fft.createDimension('harmonic',3)
    fft.createVariable('harmonic','i4','harmonic')
    fft.variables['harmonic'][:]=range(3)

    fft.createVariable("Amplitude","f4", ("harmonic","lat","lon",),zlib=True)
    fft.variables['Amplitude'].long_name = 'Normalized amplitude of each harmonic'
    fft.variables['Amplitude'][:] = indat['harmonics'][:3,...]

    #Global attributes
    outnc.description = indat['description']

    outnc.history=datetime.date.today().strftime("%d/%m/%y")

    outnc.close()


def saveOnsDateB19_V2(indat,outfile):

    outnc = nc.Dataset(outfile, "w", format="NETCDF4")

    lons=indat['lon']; nlon=len(lons)
    lats=indat['lat']; nlat=len(lats)
    yr0=indat['yr0']; nyrs=indat['onset_jday'].shape[0]

    # Creating dimensions
    outnc.createDimension("lon", nlon)
    outnc.createDimension("lat", nlat)
    outnc.createDimension("years",None)

    #Creating coordinates
    years = outnc.createVariable("years","i4","years")
    lat = outnc.createVariable("lat","f4","lat")
    lon = outnc.createVariable("lon","f4","lon")

    # Filling coordinates
    lon[:]=lons; lat[:]=lats
    lon.units='degrees_east'; lon.long_name='longitude'
    lat.units='degrees_north'; lat.long_name='latitude'

    years.long_name='Years'
    years[:]=list(range(indat['yr0'],indat['yr0']+nyrs,1))

    onset=outnc.createVariable('DOY','f4',("years","lat","lon",),zlib=True)
    onset.long_name = 'Wet season onset [Day of Year]'
    onset[:] = indat['onset_jday']

    #Global attributes
    outnc.description = indat['description']

    outnc.history=datetime.date.today().strftime("%d/%m/%y")

    outnc.close()


########################
#Auxiliary functions
###########################

def Intersect(clm,indat,ndays):
    '''
        Auxiliary function to check if two time series intersect each other

        INPUT:
            clm - time series representing the climatological onset period
            indat - time series representing the onset period to be checked
            ndays - number of days in the year

        RETURNS:
            inters - indices of the intersecting days
    '''

    inters=np.isin(indat,clm).nonzero()[0]
    if len(inters) > 0: 
        pass
    else:  #Checking if onset is in the beggining of year 
        clm=np.where(clm >= ndays,clm-ndays,clm)
        indat=np.where(indat >= ndays,indat-ndays,indat)
        inters=np.isin(indat,clm).nonzero()[0]

    return inters


def CheckOutliers(indat,ndays,thrs=1.5,target=False):
    '''
        Auxiliary function to identify outliers in the first day of onset
        It estimates the interquartile ranges and indentifies values outside it

        INPUT:
            indat - array with the first day of onset periods
            ndays - number of days in the year
            thrs - threshold to define outliers
            target - target value for the onset day (either the median 
                     of all years or the climatological first day)
        
        RETURNS:
            outl - indices of the outliers
            iqr - interquartile range

    '''
    qqs=np.nanquantile(indat-target,(.25,.75))
    iqr=qqs[1]-qqs[0]
    
    qq25=target-(thrs*iqr)
    qq75=target+(thrs*iqr)
    
    outl=((indat < qq25) | (indat > qq75)).nonzero()[0]

    return outl,iqr


def CheckPattern(seq,tag,key):
    '''
        Auxiliary function to identify the presence of the sequence in 'key' 
        in string representing a sequnce of diferent events

        INPUT:
            seq - is a list with the day of events
            tag - identifies the type of events (characters representing each type of event)
            key - the sequence to be identified in tag

        RETURNS:
            chk - indices of the sequence in the tag array
    '''

    pst=np.argsort(seq) #sorting sequence of dates
    test=''.join(tag[pst]) #temporal sequence of tags
    chk=[i for i in range(len(seq)-(len(key)-1)) if (test[i:i+len(key)] == key)]          

    return chk


def DelCandidate(rmv,firstday,lenseason):
    '''
        Auxiliary function to remove the items listed in rmv from the 
        firstday and lenseason lists and returns updated lists 

        INPUT:
            rmv - list with the indices to be removed
            firstday - list with the first day of onset periods
            lenseason - list with the length of onset periods

        RETURNS:
            firstday - updated list with the first day of onset periods
            lenseason - updated list with the length of onset periods
    '''

    fd,ln=firstday.copy(),lenseason.copy()
    if(len(rmv) > 0):
        rmv.sort()
        for ii in rmv[::-1]:
            ln.pop(ii)
            fd.pop(ii)
    return fd,ln


def CheckLeap(indat,dtime,calendar):
    '''
        Checks if the input dataset has leap days and
        removes it from the dataset and the time array

        INPUT:
            indat - input dataset
            dtime - time array
            calendar - calendar type
        
        RETURNS:
            indat - updated dataset
            dtime - updated time array
            ntot - number of days in the dataset
            cal - calendar type
            unts - time units
    '''
       
    #Removing leap days, if exists
    if(calendar in ['standard','gregorian']):
        leap=((dtime[:,1]==2) & (dtime[:,2]==29)).nonzero()[0]
        indat[leap-1,...]=(indat[leap,...]+indat[leap-1,...])/2.
        #Removing data from indat array
        indat=np.delete(indat,leap,axis=0)
        #Removing days from the time array
        dtime=np.delete(dtime,leap,axis=0)
        cal='noleap'
    else: 
        cal=calendar

    #Array with day of the year
    #From date to days since 1970 (for convention)
    ntot=dtime.shape[0]
    unts='days since '+str(int(dtime[0,0]))+'-1-1'

    return indat,dtime,ntot,cal,unts

def removeLeapEMSea(dtime,data,unt=False,cal=False):
    '''
        Checks if the input dataset has leap days and
        removes the leap day in the seasonal averaged dataset.
        
        Used only in Seasonal and Subseasonal forecasts and hindcasts
    '''


    flag=False
    if(len(dtime.shape) == 1):
        flag=True
        tmp=nc.num2date(dtime,units=unt,calendar=cal)
        dtime=np.asarray([[tmp[ii].year,tmp[ii].month,tmp[ii].day] \
                          for ii in range(len(tmp))])

    leap=((dtime[:,1]==2) & (dtime[:,2]==29)).nonzero()[0]
    if(len(leap) > 0):
        if (len(data.shape) > 1):
            data[leap-1,...]=(data[leap,...]+data[leap-1,...])/2.
            #Removing data from precip array
            data=np.delete(data,leap,axis=0)
        else:
            data[leap-1]=(data[leap]+data[leap-1])/2.
            #Removing data from precip array
            data=np.delete(data,leap,axis=0)

        #Removing days from the time array
        dtime=np.delete(dtime,leap,axis=0)

    #Reverting back to julian days (if input data is julian day)
    if(flag):
        tmp=np.asarray([nc.date2num(cftime.datetime(dtime[ii,0],dtime[ii,1],dtime[ii,2]),\
                                    units=unt,calendar='noleap') \
                       for ii in range(dtime.shape[0])])
        dtime=tmp

    return data,dtime

def PatchFFT(flt,dtime):
    '''
        Check if the filtered precipitation has the same length as the 
        dtime array. If not, patches the precip using the last two values
        in the series. returns the patched filtered precip

        INPUT:
            flt - filtered precipitation
            dtime - time array
        
        RETURNS:
            flt - patched filtered precipitation
    '''

    if(flt.shape[0] < dtime.shape[0]): 
        df=flt[-1,...]-flt[-2,...]
        xtr=np.where(df >=0,flt[-1:,...]+df,flt[-1:,...]+df)
        flt=np.vstack((flt,xtr))

    return flt

def NearXY(ll,lt,lon,lat,nrrainy):
    '''
        Locates the nearest valid coordinate given a input lon and lat

        INPUT:
            ll - longitude of the point
            lt - latitude of the point  
            lon - longitude array
            lat - latitude array
            nrrainy - array with the number of onset seasons per pixel
        
        RETURNS:
            xx - index of the nearest longitude
            yy - index of the nearest latitude
    '''

    stp=(lon[1]-lon[0])/2.
    xx=((lon >= ll-stp) & (lon <= ll+stp)).nonzero()[0][0]
    yy=((lat >= lt-stp) & (lat <= lt+stp)).nonzero()[0][0]

    if(np.isnan(nrrainy[yy,xx])):
        ii=0
        while(np.all(np.isnan(nrrainy[yy-ii:yy+ii+1,xx-ii:xx+ii+1]))):ii+=1
        vld=(nrrainy[yy-ii:yy+ii+1,xx-ii:xx+ii+1] > 0).nonzero()
        yy=yy-ii+vld[0][0]; xx=xx-ii+vld[1][0]
    

    return xx,yy


def butterworth(indat,cutoff,ftype='lowpass'):
    '''
        lowpass is the default, but can be also 'highpass' , 'bandpass' or 'bandstop'
    '''
    from scipy.signal import butter, filtfilt, sosfilt
 
    padd1 = indat[:int(2*cutoff)][::-1]
    padd2 = indat[-int(2*cutoff):][::-1]
    paddannual = np.hstack((padd1,indat,padd2))
    b, a = butter(2, 1/float(cutoff),btype=ftype)
    y = filtfilt(b, a, paddannual[~np.isnan(paddannual)])

    outdat=np.zeros(paddannual.shape); outdat[:]=np.nan
    outdat[~np.isnan(paddannual)]=y
    
    return outdat[int(2*cutoff):-int(2*cutoff)]


def CropInterp(input1,input2,sim,lon1,lat1,lon2,lat2):

    #If GCM - interpolating obs to model's resolution
    if(sim in ['UKMO-n216-H','UKMO-n96-H','UKMO-n216-AO','UKMO-n96-AO']):
        input1,lonf,latf=SimpleInterp(input1,lon1,lat1,lon2,lat2)
        if(len(input1.shape) > 2): input1=np.moveaxis(input1,0,-1)
    else: lonf,latf=lon1,lat1

    #Cropping to same area:
    lat3=np.max([np.min(lat1),np.min(lat2)])
    lat4=np.min([np.max(lat1),np.max(lat2)])
    lon3=np.max([np.min(lon1),np.min(lon2)])
    lon4=np.min([np.max(lon1),np.max(lon2)])

    #Cropping Input1
    yyi,yyf=((latf >= lat3) & (latf <= lat4)).nonzero()[0][[0,-1]]; yyf+=1
    xxi,xxf=((lonf >= lon3) & (lonf <= lon4)).nonzero()[0][[0,-1]]; xxf+=1
    input1=input1[yyi:yyf,xxi:xxf,...]

    #Cropping input2
    yyi,yyf=((lat2 >= lat3) & (lat2 <= lat4)).nonzero()[0][[0,-1]]; yyf+=1
    xxi,xxf=((lon2 >= lon3) & (lon2 <= lon4)).nonzero()[0][[0,-1]]; xxf+=1
    input2=input2[yyi:yyf,xxi:xxf,...]
    #coordinates:
    latf=lat2[yyi:yyf]; lonf=lon2[xxi:xxf]

    return input1,input2,latf,lonf

def SimpleInterp(data,loni,lati,lonf,latf,method='linear'):
    '''
        Function to interpolate dataset from (loni,lati) to (lonf,latf)
    '''

    Xi, Yi = np.meshgrid(loni,lati)
    Xf, Yf = np.meshgrid(lonf,latf)

    if(data.ndim > 2):
        shp=np.asarray(data.shape)
        nt=shp[((shp != len(loni)) & (shp != len(lati))).nonzero()[0][0]]
        if(shp[0] != nt):  #Changing array order to put time as first dimension
            data=np.moveaxis(data,(shp == nt).nonzero()[0][0],0)
    else: 
        nt=1
        data=data[np.newaxis,...]

    outd=np.zeros((nt,len(latf),len(lonf)))

    for i in range(nt):
        print (i+1,' of ',nt)
        outd[i,:,:]=sci.griddata((Xi.flatten(),Yi.flatten()),data[i,:,:].flatten(),\
                                   (Xf,Yf),method=method)
                    
    return np.squeeze(outd),lonf,latf


def DiffDate(obs,sim,od,sd):

    #Transforming to 365 days
    if(od != sd):
        dd=np.max([od,sd])
        obs=obs*dd/float(od)
        sim=sim*dd/float(sd)
    else: dd=od

    diff1=(sim-obs)
    diff2=(sim-(obs+dd))
    diff3=((sim+dd)-obs)

    ddiff=np.zeros(obs.shape[:2])    
    # Iterate over each element's position
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            # Find the minimum value and its corresponding array
            min_value = min(abs(diff1[i, j]), abs(diff2[i, j]), abs(diff3[i, j]))
            if min_value == abs(diff1[i, j]):
                ddiff[i,j]=diff1[i,j]
            elif min_value == abs(diff2[i, j]):
                ddiff[i,j]=diff2[i,j] 
            else:
                ddiff[i,j]=diff3[i,j]
    
    return ddiff

def hrtime2jday(indat,itime,iunt,ical):

    tjday=[nc.num2date(ii,units=iunt,calendar=ical).dayofyr for ii in itime]
    tjday=np.hstack((tjday,[400]))  #adding 400 to flag missing
    tmp=np.ravel(indat); tmp[np.isnan(tmp)]=-1
    tmp2=np.asarray([float(tjday[int(ii)]) for ii in tmp]) 
    indat=np.reshape(tmp2,indat.shape) 
    indat[indat==400]=np.nan

    return indat

def spatialAverage(data,lon,lat):

    if(lon.ndim < 2): lon,lat=np.meshgrid(lon,lat)

    latgrid=np.cos(np.deg2rad(lat))

    if(~np.ma.isMaskedArray(data)): 
        data=np.ma.array(data,mask=np.isnan(data))
    
    latgrid=np.where(data.mask,np.nan,latgrid)
    weight=latgrid/np.nansum(latgrid)
    spave=(data*weight).sum()

    return spave


def FillingMissing(indat,sm,soth,ws,extra=False):
    '''
        This function fills in missing values in mean season (sm) using
        suitable candidates from other seasons (soth). 
        
        INPUT
            indat - xarray dataset with the onset periods
            sm - season to be filled
            soth - seasons to be used as candidates
            ws - number of pixels around the target pixel used in the analysis. 
                 For smaller domains, 5 is enough. For global dataset, use large 
                 number (15, for example). Should always be odd.
            extra - extra xarrays to be organized. In this case, 
                    all entries in "extra" will be organized based on "indat"
        
        RETURNS
            indat - updated xarray dataset
            extra - updated xarray dataset
    '''
    
    dp=int((ws-1)/2)

    #Pixels with missing data in the main season
    mainmsk=indat.sel(sea=sm,stats='FirstDay').isnull().stack(z=('lat','lon'))
    #Pixels with valid data in the other seasons
    msk=indat.sel(sea=soth,stats='FirstDay').notnull().sum('sea').stack(z=('lat','lon'))
    
    pxl=msk.where(mainmsk & (msk > 0),drop=True).coords['z'].values  #lat and lon of missing data
    #Pixels with missing data in the main season byt valid data in other seasons

    if(len(pxl) > 0):
        for yy,xx in pxl:
            iy=np.where(indat.lat.values==yy)[0][0]
            dym=iy-dp if (iy-dp) >= 0 else 0; dyx=iy+dp if (iy+dp) < len(indat.lat) else len(indat.lat)
            ix=np.where(indat.lon.values==xx)[0][0]
            dxm=ix-dp if (ix-dp) >= 0 else 0; dxx=ix+dp if (ix+dp) < len(indat.lon) else len(indat.lon)

            #Check if the candidate value within the [min,max] values of 
            #the first day in a window +/- 2 lat/lon around the missing value
            mn=indat.sel(sea=sm,stats='FirstDay').isel(lat=slice(dym,dyx),\
                         lon=slice(dxm,dxx)).min().item()
            mx=indat.sel(sea=sm,stats='FirstDay').isel(lat=slice(dym,dyx),\
                         lon=slice(dxm,dxx)).max().item()

            #Checking if there is a valid value in another season:
            cnd=indat.sel(sea=soth,stats='FirstDay',lat=yy,lon=xx).values

            if(np.any(np.isfinite(cnd))):
                
                pst=[ii for ii,vv in enumerate(cnd) \
                            if ((np.isfinite(vv)) & \
                                ((vv >= mn) & (vv <= mx)))]
                
                if(len(pst) > 0):
                    if len(pst)==1: pst=soth[pst[0]]
                    else:
                        #select candidate with value closest to the median over the window
                        med=indat.sel(sea=sm,stats='FirstDay').isel(lat=slice(dym,dyx),\
                                      lon=slice(dxm,dxx)).median(skipna=True).item()
                        pst=soth[abs(cnd[pst]-med).argmin()]

                    indat.loc[{'lat':yy,'lon':xx,'sea':sm}]=indat.sel(lat=yy,lon=xx,sea=pst).values
                    indat.loc[{'lat':yy,'lon':xx,'sea':pst}]=np.nan

                    if(not isinstance(extra,bool)):
                        extra.loc[{'lat':yy,'lon':xx,'sea':sm}]=extra.sel(lat=yy,lon=xx,sea=pst).values
                        extra.loc[{'lat':yy,'lon':xx,'sea':pst}]=np.nan

    if(not isinstance(extra,bool)):return indat,extra
    else: return indat


def CheckSeasons(indat,sm,soth,ws,extra=False):
    '''
        Function to check if onset periods in other seasons are a better 
        match to the main onset season.

        INPUT
            indat - xarray dataset with the onset periods
            sm - season to be checked
            soth - seasons to be used as candidates
            ws - number of pixels around the target pixel used in the analysis. 
                 For smaller domains, 5 is enough. For global dataset, use large 
                 number (15, for example). Should always be odd.
            extra - extra xarrays to be organized. In this case, 
                    all entries in "extra" will be organized based on "indat"
            
        RETURNS
            indat - updated xarray dataset
            extra - updated xarray dataset
    '''

    dp=int((ws-1)/2)

    #Pixels with valid data in the main season
    mainmsk=indat.sel(sea=sm,stats='FirstDay').notnull().stack(z=('lat','lon'))
    #Pixels with valid data in the other seasons
    msk=indat.sel(sea=soth,stats='FirstDay').notnull().sum('sea').stack(z=('lat','lon'))
    #Pixels with valid data in the main season and other seasons
    pxl=msk.where((msk > 0) & mainmsk,drop=True).coords['z'].values  #lat and lon of valid data

    #check if values in other seasons could be a better fit in the main season
    for yy,xx in pxl:
        iy=np.where(indat.lat.values==yy)[0][0]
        dym=iy-dp if (iy-dp) >= 0 else 0; dyx=iy+dp if (iy+dp) < len(indat.lat) else len(indat.lat)
        ix=np.where(indat.lon.values==xx)[0][0]
        dxm=ix-dp if (ix-dp) >= 0 else 0; dxx=ix+dp if (ix+dp) < len(indat.lon) else len(indat.lon)

        med=indat.sel(sea=sm,stats='FirstDay').isel(lat=slice(dym,dyx),\
                      lon=slice(dxm,dxx)).median(skipna=True)
        diff=abs(indat.sel(lon=xx,lat=yy,stats='FirstDay')-med)
        pst=diff.argmin().item()
        if (pst != sm):
            #target value from other season is closer to those 
            #in the main season than the value in the main season

            tmp=indat.sel(lat=yy,lon=xx).copy()
            tmp.loc[{'sea':sm}]=indat.sel(lat=yy,lon=xx,sea=pst)
            tmp.loc[{'sea':pst}]=indat.sel(lat=yy,lon=xx,sea=sm)
            indat.loc[{'lat':yy,'lon':xx}]=tmp

            del tmp
 
            if(not isinstance(extra,bool)):

                tmp=extra.sel(lat=yy,lon=xx).copy()
                tmp.loc[{'sea':sm}]=extra.sel(lat=yy,lon=xx,sea=pst)
                tmp.loc[{'sea':pst}]=extra.sel(lat=yy,lon=xx,sea=sm)
                extra.loc[{'lat':yy,'lon':xx}]=tmp
                
                del tmp

    if(not isinstance(extra,bool)):return indat,extra
    else: return indat


def ENSO_Year(yr1,yr2):
    def candidate_year(cnd):
        ps=0; firstday=[]; length=[]
        for key,grp in it.groupby(cnd):
            tmp=len(list(grp))
            if key == 0: pass
            else:
                if(tmp <= 5): cnd[ps:ps+tmp]=0
            ps=ps+tmp              
        return cnd

    oniurl='https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
    response = requests.get(oniurl)

    response=response.text.split('\n')
    dummy=response.pop(0)  #removing header line
    dummy=response.pop(-1)  #removing last empty line

    yr0=int(response[0].split(' ')[3])
    yrf=int(response[-1].split(' ')[3])

    oni=xr.DataArray(coords=[np.arange(yr0,yrf+1),\
                             ['JFM','FMA','MAM','AMJ','MJJ','JJA',\
                              'JAS','ASO','SON','OND','NDJ','DJF']],\
                     dims=['year','seas'])

    for txt in response:
        tmp=txt.split(' ')
        seas=tmp[2]
        yr=int(tmp[3])
        if(seas == 'DJF'): yr-=1
        val=float(tmp[-1])
        if(yr < yr0): continue
        oni.loc[{ 'year': yr, 'seas': seas }] = val
    
    #EN
    cnd=xr.where(oni >= 0.5,1,0).values.flatten()
    cnd=candidate_year(cnd) 
    cnd=np.reshape(cnd,oni.shape)
    en=oni.year[cnd[:,-1] == 1].values

    #LN
    cnd=xr.where(oni <= -0.5,1,0).values.flatten()
    cnd=candidate_year(cnd) 
    cnd=np.reshape(cnd,oni.shape)
    ln=oni.year[cnd[:,-1] == 1].values
    
    tmp1=np.isin(np.arange(yr1,yr2+1),en,invert=True) #Non-EN years
    tmp2=np.isin(np.arange(yr1,yr2+1),ln,invert=True) #Non-LN years
    neutral=(tmp1 & tmp2).nonzero()[0]+yr1 #Neutral years


    ENSOyr={
        'EN':en[(en >= yr1) & (en <= yr2)],
        'LN':ln[(ln >= yr1) & (ln <= yr2)],
        'neutral':neutral}

    return ENSOyr


def CircularQuantile(indat,ndays,qq,dim=['years']):
    '''
    this code was adapted from a ChatGPT suggestion
    '''
    def circquantile(indat,ndays,qq):
        '''
        Here indat is a 1D np array
        '''

        indat=np.ravel(indat)  # Flatten the input array

        if np.any(np.isfinite(indat)):

            tmp=np.sort(indat)

            best_span=[np.ptp((tmp-tmp[jj])%ndays) for jj in range(len(tmp))]
            best_span=np.where(np.isnan(best_span),ndays+1,best_span) # to avoid NaN in the argmin
            pst=np.argmin(best_span)
            linear=(tmp-tmp[pst])%ndays
            
            qtllinear = np.nanquantile(linear,qq)
            qtl =(qtllinear+tmp[pst])%ndays

        else:
            qtl = np.full_like(qq, np.nan, dtype=float)

        return qtl
    
    outdat=xr.apply_ufunc(
        circquantile,
        indat,
        input_core_dims=[dim],
        output_core_dims=[["quantile"]],
        kwargs={"ndays": ndays, "qq": qq},
        vectorize=True,
        output_dtypes=[float],
    )
    outdat=outdat.assign_coords(quantile=qq)

    return outdat

def CircularMedian(indat,ndays,dim=['years']):

    out=CircularQuantile(indat,ndays,[0.5],dim=dim)

    out=out.sel(quantile=0.5)  #Selecting the median

    return out


def CircularMean(indat,ndays,dim=['years']):
    '''
    this code was adapted from a ChatGPT suggestion
    '''

    angle=(indat/ndays)*2*np.pi
    # Convert to Cartesian coordinates
    angx=np.cos(angle); angy=np.sin(angle)
    aveang=np.arctan2(angy.mean(dim=dim,skipna=True),\
                      angx.mean(dim=dim,skipna=True))
    aved=(aveang/(2*np.pi))*ndays
    aved=np.round(aved % ndays)
    aved=aved.where(aved > 0,1)  # Avoid zero value for first day of onset

    #Mask all missing
    aved=aved.where(indat.notnull().any(dim=dim),np.nan)

    return aved


def CircularMedian_wrong(indat,ndays,dim=['years']):
    '''
    this code was adapted from a ChatGPT suggestion
    '''

    angle=(indat/ndays)*2*np.pi
    # Convert to Cartesian coordinates
    angx=np.cos(angle); angy=np.sin(angle)
    aveang=np.arctan2(angy.median(dim=dim,skipna=True),\
                      angx.median(dim=dim,skipna=True))
    aved=(aveang/(2*np.pi))*ndays
    aved=np.round(aved % ndays)
    aved=aved.where(aved > 0,1)  # Avoid zero value for first day of onset

    #Mask all missing
    aved=aved.where(indat.notnull().any(dim=dim),np.nan)

    return aved

def CircularVar(indat,ndays,dim=['years']):
    '''
    this code was adapted from a ChatGPT suggestion
    '''
    angle=(indat/ndays)*2*np.pi   
    # Convert to Cartesian coordinates
    angx=np.cos(angle); angy=np.sin(angle)
    # Compute mean x and y along the "time" axis
    mean_x = angx.mean(dim=dim,skipna=True)
    mean_y = angy.mean(dim=dim,skipna=True)    
    # Compute resultant vector length R
    R=np.sqrt(mean_x**2 + mean_y**2)
    
    circstd=np.sqrt(-2*np.log(R))*(ndays/(2*np.pi))  # Standard deviation in days
    circvar=circstd**2  # Squared to get variance    # Compute circular variance

    return circvar
 

def PolyAreas(sub):

    match sub:
        case 'SAm':
            areas={'SEBr':[[302,312,321,302,302], [-28,-28,-20,-20,-28]],
                   'EBr':[[310,315,320,320,315,310,310],[-18,-20,-20,-15,-10,-10,-18]],
                   'EAmz':[[299,305,305,294,294,299],[-15,-15,-7,-7,-12,-15]],
                   'WAmz':[[284,293,297,287,284],[-6,-14,-7,0,-6]],
                   'NAmz':[[290,296,296,302,306,298,290,290],[5,5,0,0,6,12,12,5]], 
                   }
        case 'SAfr':
            areas={'ZBW':[[21,28,28,21,21],[-16,-16,-12,-12,-16]],
                   'ZBE':[[28,33,33,28,28],[-15,-15,-11,-11,-15]],
                   'ZBS':[[22,32,32,22,22],[-18,-18,-12,-12,-18]],
                   'SAf':[[25,31,31,25,25],[-29,-29,-25,-25,-29]],
                  }
        case _:
            print('Unknown subregion: '+sub)
            areas={}
    
    return areas

def poly2mask(indat,poly):

    from matplotlib.path import Path
    
    poly=np.asarray(poly).T  #poly is the list of coordinates from previous function
    msk=indat.where(((indat.lon >= poly[:,0].min()) & (indat.lon <= poly[:,0].max())) &\
                     ((indat.lat >= poly[:,1].min()) & (indat.lat <= poly[:,1].max())))
    tmp=msk.stack(z=('lon','lat'))
    pts=tmp.where(tmp.notnull(),drop=True).coords['z'].values
    pts=np.asarray([[ii,jj] for ii,jj in pts]) #converting from tuple to array

    path=Path(poly)

    if(len(pts) > 0):
        pvld=path.contains_points(pts)
        rmv=pts[pvld == False]  #points outside the polygon
        if(len(rmv) > 0):
            for xx,yy in rmv:
                #Remove points outside the polygon
                msk.loc[{'lon':xx,'lat':yy}]=np.nan

    return msk
