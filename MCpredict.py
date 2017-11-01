# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:29:08 2017

@author: hazel.bain

 Chen_MC_Prediction.pro

 This module contains functions to read in either real time or historical 
 solar wind data and determine geoeffective and non geoeffective "events" 
 present in the magnetic field data. An event is defined to be > 120 minutes 
 long and with start and end times determined when Bz component of the magentic
 field changes sign. 
 
 A sinusoid is fitted to the Bz component to predict the event Bz maximum 
 (or minimum) value and the event duration, therefore giving some predictive 
 diagnostic of whether of not we expect the event to be geoeffective.
 
 When applied to historical data, classification of the geoeffectiveness of the events is 
 determined by comparison with the Dst value during that event. If Dst < -80 
 the event is considered to be geoeffective. If the Dst is > -80 then the event
 is considered to be non geoeffective. Events occuring in the wake of geoeffective
 events, where the Dst is still recovering are classed as ambigous. 
 
 The top level function Chen_MC_Prediction is called to run the model either to
 real-time data or to a historical dataset e.g.
 
 data, events, events_frac = Chen_MC_Prediction(start_date, end_date, dst_data, kp_data, pdf)


 The original version of this code is from Jim Chen and Nick Arge
 and is called DOACE_hr.pro. This version is adapted from the
 IDL code written by Michele Cash of DOACE.pro modifed to 
 run in near real-time, to read in data from the SWPC database,
 and to make the code more readable by removing goto statements
 and 5 minute data averages.

 Reference Papers: Chen et al. 1996, 1997, 2012
                   Arge et al. 2002



"""

from read_database import get_data

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

def Chen_MC_Prediction(sdate, edate, dst_data, kp_data, pdf, predict = 0,\
                       smooth_num = 25, resultsdir='', \
                       real_time = 0, spacecraft = 'ace',\
                       csv = 1, livedb = 0,\
                       plotting = 1, plt_outfile = 'mcpredict.pdf',\
                       plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots/',\
                       line = [], dst_thresh = -80, kp_thresh = 6):

    """
     This function reads in either real time or historical 
     solar wind data and determine geoeffective and non geoeffective "events" 
     present in the magnetic field data. An event is defined to be > 120 minutes 
     long and with start and end times determined when Bz component of the magentic
     field changes sign. Classification of the geoeffectiveness of the events is 
     determined by comparison with the Dst value during that event. If Dst < -80 
     the event is considered to be geoeffective. If the Dst is > -80 then the event
     is considered to be non geoeffective. Events occuring in the wake of geoeffective
     events, where the Dst is still recovering are classed as ambigous. 
    
     inputs
     ------
     
     sdate - string 
         start time, format "%Y-%m-%d"
     edate - string 
         start time, format "%Y-%m-%d"
     dst_data - dataframe
         dst hourly data
     kp_data - dataframe
         kp 3 hourly data
     pdf - data array
         PDF relating Bzm and tau to Bzm' and tau' - see MC_predict_pdfs.py
     predict - int
         keyword to predict the geoeffectiveness of all events
     smooth_num - int
         temporal smoothing parameter, default = 25
     resultsdir - string
         path to results directory
     real_time - int
         set to 1 to use real time data
     spacecraft - string
         which spacecraft to get the data from "ace" or "dscvr"
     csv - int
         save the data to local csv file
     livedb - int
         read directly from the database
     plotting - int
         set to 1 to output plots
     plt_outfile - string
         plot file name default mcpredict.pdf
     plt_outpath - string
         path to plot file directory 
     line - array of datatime objects
         times of vertical lines to overplot on data
     dst_thresh - int 
         threshold of dst to define geoeffective event
     kp_thresh - int 
         threshold of kp to define geoeffective event


    outputs
    -------
    
    data - data array
        contains smoothed solar wind and plasma data from sdate to edate
        (note this is not split into individual events)
    events - pandas data frame
        each row contains the characteristics for a single event
    events_frac - pandas dataframe
        each row contains the characteristics for a fraction of each event


    """   
    #running in real_time mode
    if real_time == 1:
        print("todo: real-time data required - determine dates for last 24 hours")
    
    print("Start date: " + sdate )
    print("End date  : " + edate + "\n")
    
    #min_duration of at least 120 minutes in order to be considered as an event
    min_duration=120.
    

    #read in mag and solar wind data
    if spacecraft == 'ace':
        
        #read in ace_mag_1m data
        mag_data = get_data(sdate, edate, view = 'ace_mag_1m', \
                            csv = csv, livedb = livedb)

        sw_data = get_data(sdate, edate, view = 'tb_ace_sw_1m', \
                           csv = csv, livedb = livedb)

        #convert to pandas DataFrame
        #MAYBE MOVE THIS STEP INTO THE GET DATA FUNCTION!!!!
        mag = pd.DataFrame(data = mag_data, columns = mag_data.dtype.names)
        sw = pd.DataFrame(data = sw_data, columns = sw_data.dtype.names)  
        sw.rename(columns={'dens': 'n', 'speed': 'v', 'temperature':'t'}, inplace=True)
               
    elif spacecraft == 'dscovr':
        print("todo: dscovr data read functions still todo")
    
    #clean data
    mag_clean, sw_clean = clean_data(mag, sw)
    
    #pd.set_option('display.max_rows',100)
    #print(mag_clean.gsm_lat.iloc[0:100])  
    #pd.reset_option('display.max_rows')
    
    #Create stucture to hold smoothed data
    col_names = ['date', 'bx', 'by', 'bz', 'bt', 'theta_z', 'theta_y']        
    data = pd.concat([mag_clean['date'], \
            pd.Series(mag_clean['gsm_bx']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['gsm_by']).rolling(window = smooth_num).mean(),\
            pd.Series(mag_clean['gsm_bz']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['bt']).rolling(window = smooth_num).mean(),\
            pd.Series(mag_clean['gsm_lat']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['gsm_lon']).rolling(window = smooth_num).mean()], axis=1, keys = col_names)
          
    #data['theta_z'] = pd.Series(180.*np.arcsin(mag_clean['gsm_bz']/mag_clean['bt'])/np.pi)\
    #                    .rolling(window = smooth_num).mean()   #in degrees
    #data['theta_y'] = pd.Series(180.*np.arcsin(mag_clean['gsm_by']/mag_clean['bt'])/np.pi)\
    #                    .rolling(window = smooth_num).mean()   #in degrees
                        
    #data['theta_z'] = data['theta_z'].interpolate()      
    #data['theta_y'] = data['theta_y'].interpolate()                    

    data['sw_v'] = pd.Series(sw_clean['v']).rolling(window = smooth_num).mean()
    data['sw_n'] = pd.Series(sw_clean['n']).rolling(window = smooth_num).mean()    

    #add empty columns for the predicted data values at each step in time
    data['istart_bz'] = 0
    data['iend_bz'] = 0
    data['tau_predicted'] = 0
    data['tau_actual'] = 0
    data['frac_est'] = 0
    data['bzm_predicted'] = 0
    data['bzm_actual'] = 0
    data['i_bzmax'] = 0
    data['theta_z_max'] = 0
    data['dtheta_z'] = 0
    
    data['istart_by'] = 0
    data['iend_by'] = 0
    data['tau_predicted_y'] = 0
    data['tau_actual_y'] = 0
    data['bym_predicted'] = 0
    data['bym_actual'] = 0
    data['i_bymax'] = 0
    data['theta_y_max'] = 0
    data['dtheta_y'] = 0
    
    data['lambda'] = 0
    data['chi'] = 0
    
    
#==============================================================================
#     #drop the first 98 rows due to smoothing
#     data.drop(data.index[[0,98]])
#     
#     pd.set_option('display.max_rows',100)
#     print(data.bz.iloc[0:100])  
#     pd.reset_option('display.max_rows')
#==============================================================================
    
    #Incrementally step through the data and look for mc events.
    #An event is defined as the time bounded by sign changes of Bz.
    #An event needs to have a min_durtaion of min_duration.
    #Once a valid event is found, predict it's duration 

    ##check to make sure starting Bz data isn't NaN
    first_good_bz_data = np.min(np.where(np.isnan(data['bz']) == False)) + 1
    
    iend = first_good_bz_data
    iStartNext = first_good_bz_data
    
    for i in np.arange(first_good_bz_data, len(data['date'])):
        istart = iStartNext
        
        #check for bz sign change to signal end of an event, if not
        #move on to next data step
                
        if not event_end(data, i):
            continue 
        
        #print("----Event found----")
        
        iend = i-1
        iStartNext = i
        
        #now we have an event, check that is meets the min_duration
        #if not move to the next data step and look for new event
        if not long_duration(istart, iend, min_duration):
            continue
        
        #ignore the first input data step
        if istart == first_good_bz_data:
            continue

        #print("----Event of correct duration found----")
        
        #now try and predict the duration of the bz component
        predict_duration(data, istart, iend)

        #find the corresponding By event start and end
        #print("istart %i iend %i" % (istart, iend))
        istart_y, iend_y = By_start(data, istart, iend)
        #print(istart_y, iend_y)
        
#==============================================================================
#         if istart_y != None:
#             
#             #predict the by event duration
#             predict_duration(data, istart_y, iend_y, component = 'y')
#             lambda_chi_calc(data)
#==============================================================================
       

        if icme_event(istart, iend, len(data['date'])):
            validation_stats, data, resultsdir, istart, iend
       
    
    #create new dataframe to record event characteristics
    events, events_frac, events_time_frac = create_event_dataframe(data, dst_data, kp_data, pdf, dst_thresh=dst_thresh, predict = predict)   
    
    #plot some stuff   
    if plotting == 1:
        
        evt_times = events[['start','end']].values
        mcpredict_plot(data, events_frac, dst_data, kp_data, line=line, bars = evt_times, plt_outfile = plt_outfile, plt_outpath = plt_outpath)
    
    return data, events, events_frac, events_time_frac



def create_event_dataframe(data, dst_data, kp_data, pdf, dst_thresh = -80, kp_thresh = 6, t_frac = 5, predict = 0):

    """
    Create two dataframes containing the characteristics for 
    each solar wind magnetic field event
    
    inputs
    ------
    
    data - data array
        contains solar wind and plasma data for time period under study
        (note this is not split into events)
    dst_data - dataframe
        contains hourly dst data to use as a classifier to determine whether the
        event is geoeffective or non geoeffective and recored the max/min value
    kp_data - dataframe
        contains 3 hourly kp data to use as a classifier to determine whether the
        event is geoeffective or non geoeffective and recored the max/min value        
    pdf - data array
        [bzm, tau, bzm_p, tau_p, frac] P(bzm, tau | (bzm_p, tau_p), f)
    t_frac - int
        number of fractions to split an event into - using 5 for development 
        purposes but should be larger
        

    outputs
    -------
    
    events - pandas data frame
        each row contains the characteristics for a single event
    events_frac - pandas dataframe
        each row contains the characteristics for a fraction of each event

    """

    #start and end times for each event
    #evt_times, evt_indices = find_event_times(data)
    evt_indices = np.transpose(np.array([data['istart_bz'].drop_duplicates().values[1::], \
                                         data['iend_bz'].drop_duplicates().values[1::]]))
    
    evt_indices_by = np.transpose(np.array([data['istart_by'].drop_duplicates().values[1::], \
                                         data['iend_by'].drop_duplicates().values[1::]]))

    
    #start data frame to record each event's characteristics    
    evt_col_names = ['start', 'bzm', 'tau', 'istart_bz', 'iend_bz','theta_z_max','dtheta_z',\
                     'bym', 'tau_by', 'istart_by', 'iend_by', 'lambda', 'chi']        
    events = pd.concat([data['date'][evt_indices[:,0]],\
                    data['bzm_actual'][evt_indices[:,0]],\
                    data['tau_actual'][evt_indices[:,0]],\
                    data['istart_bz'][evt_indices[:,0]],\
                    data['iend_bz'][evt_indices[:,0]] ,\
                    data['theta_z_max'][evt_indices[:,0]],\
                    data['dtheta_z'][evt_indices[:,0]], \
                    
                    data['bym_actual'][evt_indices[:,0]],\
                    data['tau_actual_y'][evt_indices[:,0]],\
                    data['istart_by'][evt_indices[:,0]],\
                    data['iend_by'][evt_indices[:,0]],\
                    
                    data['lambda'][evt_indices[:,0]],\
                    data['chi'][evt_indices[:,0]] ],\
                    axis=1, keys = evt_col_names)
    
    events['end'] =  data['date'][evt_indices[:,1]].values  #needs to be added separately due to different index

    #record the By event start and end times
    by_start_indx = data['istart_by'][evt_indices[:,0]]
    by_end_indx = data['iend_by'][evt_indices[:,0]]
    events['start_by'] = data['date'].iloc[by_start_indx].values
    events['end_by'] = data['date'].iloc[by_end_indx].values
    

    #get min dst and geoeffective flags
    events = dst_geo_tag(events, dst_data, dst_thresh = dst_thresh, dst_dur_thresh = 2.0)

    #get max kp and kp geoeffective flags
    events = kp_geo_tag(events, kp_data, kp_thresh = kp_thresh)

    #split the event into fractions for bayesian stats
    events_frac, events_time_frac = create_event_frac_dataframe(data, events, evt_indices, frac_type = 'time', t_frac = t_frac)
    
    if predict == 1:
        
        #predict geoeffectivenes
        events_frac = predict_geoeff(events_frac, pdf)

    return events, events_frac, events_time_frac
    

def create_event_frac_dataframe(data, events, evt_indices, frac_type = 'frac', t_frac = 5):
    
    ###create a dataframe containing info for every frac of an event
    repeat = t_frac+1    
    events_frac = events.loc[np.repeat(events.index.values, repeat)]
    events_frac.reset_index(inplace=True)

    #remame the column headers to keep track of things
    events_frac.rename(columns={'level_0':'evt_index', 'index':'data_index'}, inplace=True)
    
    frac = pd.DataFrame({'frac':np.tile(np.arange(t_frac+1)*(100/t_frac/100), len(events)),\
                            'frac_start':0.0,\
                            'frac_end':0.0,\
                            'frac_est':0.0,\
                            'bzm_predicted':0.0,\
                            'tau_predicted':0.0,\
                            'bym_predicted':0.0,\
                            'tau_predicted_y':0.0,\
                            'i_bzmax':0})
    
    events_frac = pd.concat([events_frac, frac], axis = 1) 
    
    ##bzm at each fraction of an event    
    for i in range(len(evt_indices)):
        
        frac_ind = evt_indices[i,0] + (np.arange(t_frac+1)*(100/t_frac/100) * \
                    float(evt_indices[i,1]-evt_indices[i,0])).astype(int)
        dfrac = frac_ind[1]-frac_ind[0]
        
        #start and end times of each fraction of an event
        events_frac['frac_start'].iloc[np.where(events_frac['evt_index'] == i)] = data['date'].iloc[frac_ind].values
        frac_ind_end = frac_ind + dfrac
        if (frac_ind[-1]+dfrac) >= len(data['date']):           
            frac_ind_end[-1] = len(data['date'])-1
        events_frac['frac_end'].iloc[np.where(events_frac['evt_index'] == i)] = data['date'].iloc[frac_ind_end].values
        
        events_frac['frac_est'].iloc[np.where(events_frac['evt_index'] == i)] = data['frac_est'].iloc[frac_ind].values

        #predicted bzm, bym ,tau_y and tau_z
        events_frac['bzm_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['bzm_predicted'].iloc[frac_ind].values
        events_frac['tau_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['tau_predicted'].iloc[frac_ind].values
        events_frac['i_bzmax'].iloc[np.where(events_frac['evt_index'] == i)] = data['i_bzmax'].iloc[frac_ind].values
        
        events_frac['bym_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['bym_predicted'].iloc[frac_ind].values
        events_frac['tau_predicted_y'].iloc[np.where(events_frac['evt_index'] == i)] = data['tau_predicted_y'].iloc[frac_ind].values
    
#==============================================================================
#     ##bym at each fraction of the Bz event    
#     for i in range(len(evt_indices_by)):
#         
#         #determine the indices in data for each fraction of an event
#         frac_ind = evt_indices[i,0] + (np.arange(t_frac+1)*(100/t_frac/100) * \
#                     float(evt_indices[i,1]-evt_indices[i,0])).astype(int)
#         
#         events_frac['bym_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['bym_predicted'].iloc[frac_ind].values
#         events_frac['tau_predicted_y'].iloc[np.where(events_frac['evt_index'] == i)] = data['tau_predicted_y'].iloc[frac_ind].values
#==============================================================================
    


    #create dataframe for every 60 mins through an event
    repeat = np.ceil((evt_indices[:,1]-evt_indices[:,0])/60.).astype(int)
    events_time_frac = events.loc[np.repeat(events.index.values, repeat)]
    events_time_frac.reset_index(inplace=True)
    
    #remame the column headers to keep track of things
    events_time_frac.rename(columns={'level_0':'evt_index', 'index':'data_index'}, inplace=True)
    
    time_frac = pd.DataFrame({'frac':0.0,\
                            'frac_start':0.0,\
                            'frac_end':0.0,\
                            'frac_est':0.0,\
                            'bzm_predicted':0.0,\
                            'tau_predicted':0.0,\
                            'bym_predicted':0.0,\
                            'tau_predicted_y':0.0,\
                            'i_bzmax':0}, index=[0])
    
    events_time_frac = pd.concat([events_time_frac, time_frac], axis = 1) 
    
    ##bzm at each fraction of an event    
    for i in range(len(evt_indices)):
        
        time_frac_ind = evt_indices[i,0] + (np.arange((evt_indices[i,1]-evt_indices[i,0])/60.) * 60.)
        dfrac = 60.

        #start and end times of each fraction of an event
        events_time_frac['frac_start'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['date'].iloc[time_frac_ind].values
        time_frac_ind_end = time_frac_ind + dfrac
        if (time_frac_ind[-1]+dfrac) >= len(data['date']):           
            time_frac_ind_end[-1] = len(data['date'])-1
        events_time_frac['frac_end'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['date'].iloc[time_frac_ind_end].values
        
        events_time_frac['frac_est'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['frac_est'].iloc[time_frac_ind].values

        #predicted bzm, bym ,tau_y and tau_z
        events_time_frac['bzm_predicted'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['bzm_predicted'].iloc[time_frac_ind].values
        events_time_frac['tau_predicted'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['tau_predicted'].iloc[time_frac_ind].values
        events_time_frac['i_bzmax'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['i_bzmax'].iloc[time_frac_ind].values
        
        events_time_frac['bym_predicted'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['bym_predicted'].iloc[time_frac_ind].values
        events_time_frac['tau_predicted_y'].iloc[np.where(events_time_frac['evt_index'] == i)] = data['tau_predicted_y'].iloc[time_frac_ind].values
    
#==============================================================================
#     ##bym at each fraction of the Bz event    
#     for i in range(len(evt_indices_by)):
#         
#         #determine the indices in data for each fraction of an event
#         frac_ind = evt_indices[i,0] + (np.arange(t_frac+1)*(100/t_frac/100) * \
#                     float(evt_indices[i,1]-evt_indices[i,0])).astype(int)
#         
#         events_frac['bym_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['bym_predicted'].iloc[frac_ind].values
#         events_frac['tau_predicted_y'].iloc[np.where(events_frac['evt_index'] == i)] = data['tau_predicted_y'].iloc[frac_ind].values
#==============================================================================

    return events_frac, events_time_frac
    
def mcpredict_plot(data, events_frac, dst_data, kp_data, line= [], bars = [], plot_fit = 1, dst_thresh = -80, kp_thresh = 6, \
            plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2/',\
            plt_outfile = 'mcpredict.pdf'):
    
    """
    Plot the ACE_MAG_1m and ACE_SWEPAM_1M data

    Parameters
    ----------
    tstart : string, required 
        Start time for the database query.
    tend: string, required 
        End time for the database query.
    server: string, optional
        default server is swds-st
    database: string, optional
        default database is RA
    view: string, optional
        default view is ace_mag_1m
    csv: int, optional
        output csv file keyword. default is 0
    outpath: string, optional
        csv file path

    Returns
    -------
    None
    
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties
    from matplotlib.dates import DayLocator
    from matplotlib.dates import HourLocator
    from matplotlib.dates import DateFormatter
            
    #start and end times for plot to make sure all plots are consistent
    #st = datetime.strptime(data['date'][0]), "%Y-%m-%d")
    #et = datetime.strptime(data['date'][-1], "%Y-%m-%d")

    st = data['date'][0]
    et = data['date'].iloc[-1]
    
    #read in the dst data
    #dst_data = dst.read_dst(str(st), str(et))

    #plot the ace data
    f, (ax0, ax1, ax1b, ax1c, ax2, ax3, ax3b, ax4, ax5, ax7, ax8) = plt.subplots(11, figsize=(11,15))
 
    plt.subplots_adjust(hspace = .1)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%d-%b')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    color = {0.0:'green', 1.0:'red', 2.0:'grey',3.0:'orange'}
    fitcolor = {0.2:'purple', 0.4:'blue', events_frac.frac.iloc[3]:'green',0.8:'orange', 1.0:'red'}
  
    #----By
    ax0.plot(data['date'], data['by'], label='By (nT)')
    ax0.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    ax0.set_xlim([st, et])
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax0.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)        
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)

    #plot the position of max bz
    for i in np.arange(5, len(events_frac), 6):
        if (events_frac['geoeff'].iloc[i] == 1.0):
            wmax_by = np.where( data['by'].iloc[events_frac['istart_by'].iloc[i] : events_frac['iend_by'].iloc[i]] == events_frac['bym'].iloc[i])[0]

            ax0.axvline(x=data['date'].iloc[events_frac['istart_by'].iloc[i] + wmax_by].values[0], \
                     linewidth=1, linestyle='--', color='grey')

    #max bz line
    for b in range(len(bars)):
        if events_frac['geoeff'].iloc[b*6] == 1.0:
            ax0.hlines(events_frac['bym'].iloc[b*6], bars[b,0], bars[b,1], linestyle='-',color='grey')

    
    #plot the fitted profile at certain intervals through the event  
    if plot_fit == 1:
        for i in range(len(events_frac)):
            
            #only plot the fits for the geoeffective events
            if (events_frac['geoeff'].iloc[i] == 1.0) & (events_frac['frac'].iloc[i] >0.1) & (events_frac['evt_index'].iloc[i] == 5):
                 
                #for each fraction of an event, determine the current fit to the profile up to this point
                pred_dur = events_frac['tau_predicted_y'].iloc[i] * 60.
                fit_times = [ events_frac['start_by'].iloc[i] + timedelta(seconds = j*60) for j in np.arange(pred_dur)]
                fit_profile = events_frac['bym_predicted'].iloc[i] * np.sin(np.pi*np.arange(0,1,1./(pred_dur)) )          
                
                ax0.plot(fit_times, fit_profile, color=fitcolor[events_frac['frac'].iloc[i]])
        
    #----Bz
    ax1.plot(data['date'], data['bz'], label='Bz (nT)')
    ax1.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    ax1.set_xlim([st, et])
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax1.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #plot the position of max bz
    for i in np.arange(5, len(events_frac), 6):
        if (events_frac['geoeff'].iloc[i] == 1.0):
            wmax_bz = np.where( data['bz'].iloc[events_frac['istart_bz'].iloc[i] : events_frac['iend_bz'].iloc[i]] == events_frac['bzm'].iloc[i])[0]

            ax1.axvline(x=data['date'].iloc[events_frac['istart_bz'].iloc[i] + wmax_bz].values[0], \
                     linewidth=1, linestyle='--', color='grey')

    #max bz line
    for b in range(len(bars)):
        if events_frac['geoeff'].iloc[b*6] == 1.0:
            ax1.hlines(events_frac['bzm'].iloc[b*6], bars[b,0], bars[b,1], linestyle='-',color='grey')

    #plot the fitted profile at certain intervals through the event  
    if plot_fit == 1:
        for i in range(len(events_frac)): 
            #only plot the fits for the geoeffective events
            if (events_frac['geoeff'].iloc[i] == 1.0) & (events_frac['frac'].iloc[i] >0.1):
                 
                #for each fraction of an event, determine the current fit to the profile up to this point
                pred_dur = events_frac['tau_predicted'].iloc[i] * 60.
                fit_times = [ events_frac['start'].iloc[i] + timedelta(seconds = j*60) for j in np.arange(pred_dur)]
                fit_profile = events_frac['bzm_predicted'].iloc[i] * np.sin(np.pi*np.arange(0,1,1./(pred_dur)) )          
                
                ax1.plot(fit_times, fit_profile, color=fitcolor[events_frac['frac'].iloc[i]])    


    #----theta_y
    ax1b.plot(data['date'], data['theta_y'], label='theta_y')
    ax1b.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1b.set_xticklabels(' ')
    ax1b.xaxis.set_major_locator(daysLoc)
    ax1b.xaxis.set_minor_locator(hoursLoc)
    ax1b.set_xlim([st, et])
    for l in line:
        ax1b.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax1b.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax1b.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)


    #----theta_z
    ax1c.plot(data['date'], data['theta_z'], label='theta_z')
    ax1c.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1c.set_xticklabels(' ')
    ax1c.xaxis.set_major_locator(daysLoc)
    ax1c.xaxis.set_minor_locator(hoursLoc)
    ax1c.set_xlim([st, et])
    for l in line:
        ax1c.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax1c.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax1c.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #plot the position of max theta
    for i in np.arange(5, len(events_frac), 6):
        if (events_frac['geoeff'].iloc[i] == 1.0):
            
            wmax_th = np.where( data['theta_z'].iloc[events_frac['istart_bz'].iloc[i] : events_frac['iend_bz'].iloc[i]] == events_frac['theta_z_max'].iloc[i])[0]
            
            ax1c.axvline(x=data['date'].iloc[events_frac['istart_bz'].iloc[i] + wmax_th].values[0], \
                     linewidth=1, linestyle='--', color='grey')
    


    #dtheta_z        
    ax2.plot(data['date'], data['dtheta_z'], label='dtheta_z deg/min')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    ax2.set_xlim([st, et])
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax2.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #lambda        
    ax3.plot(data['date'], data['lambda'], label='lambda')
    ax3.hlines(1.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax3.hlines(-1.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax3.set_xticklabels(' ')
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlim([st, et])
    ax3.set_ylim(-3,3)
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax3.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #frac_est        
    ax3b.plot(data['date'], data['frac_est'], label='frac_est')
    ax3b.set_xticklabels(' ')
    ax3b.xaxis.set_major_locator(daysLoc)
    ax3b.xaxis.set_minor_locator(hoursLoc)
    ax3b.set_xlim([st, et])
    ax3b.set_ylim(0,1.5)
    for l in line:
        ax3b.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax3b.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax3b.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
                  
    #----density
#==============================================================================
#     ax2.plot(data['date'], data['sw_n'], label='n ($\mathrm{cm^-3}$)')
#     ax2.set_xticklabels(' ')
#     ax2.xaxis.set_major_locator(daysLoc)
#     ax2.xaxis.set_minor_locator(hoursLoc)
#     ax2.set_xlim([st, et])
#     for l in line:
#         ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
#     for b in range(len(bars)):
#         ax2.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
#     leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
#     leg.get_frame().set_alpha(0.5)
#==============================================================================
    
#==============================================================================
#     #----velocity
#     maxv = max(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) + 50
#     minv =  min(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) - 50
#     ax3.plot(data['date'], data['sw_v'], label='v ($\mathrm{km s^-1}$)')
#     ax3.set_ylim(top = maxv, bottom = minv)
#     ax3.set_xticklabels(' ')
#     ax3.xaxis.set_major_locator(daysLoc)
#     ax3.xaxis.set_minor_locator(hoursLoc)
#     ax3.set_xlim([st, et])
#     for l in line:
#         ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
#     for b in range(len(bars)):
#         ax3.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)       
#     leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
#     leg.get_frame().set_alpha(0.5)
#==============================================================================
    
    #----predicted and actual duration
    ax4.plot(data['date'], data['tau_predicted'], label='$\mathrm{\tau predicted (hr)}$', ls='solid',c='b')
    ax4.plot(data['date'], data['tau_actual'], label='$\mathrm{\tau actual (hr)}$', ls='dotted', c='r')
    ax4.set_xticklabels(' ')
    ax4.xaxis.set_major_locator(daysLoc)
    ax4.xaxis.set_minor_locator(hoursLoc)
    ax4.set_xlim([st, et])
    for l in line:
        ax4.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax4.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax4.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
        
    #----Bz max predicted and actual
    ax5.plot(data['date'], data['bzm_predicted'], label='Bzm predict (nT)', ls='solid', c='b')
    ax5.plot(data['date'], data['bzm_actual'], label='Bzm actual (nT)', ls='dotted', c='r')
    #ax3.hlines(0.0, data['date'][0], data['date'][-1], linestyle='--',color='grey')
    ax5.set_xticklabels(' ')
    ax5.xaxis.set_major_locator(daysLoc)
    ax5.xaxis.set_minor_locator(hoursLoc)
    ax5.set_xlim([st, et])
    for l in line:
        ax5.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax5.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax5.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----P1    
#==============================================================================
#     ax6.plot(events_frac['frac_start'], events_frac['P1_scaled'], linestyle = ' ')
#     ax6.set_xticklabels(' ')
#     ax6.xaxis.set_major_locator(daysLoc)
#     ax6.xaxis.set_minor_locator(hoursLoc)
#     ax6.set_xlim([st, et])
#     for l in line:
#         ax6.axvline(x=l, linewidth=2, linestyle='--', color='black') 
#     for b in range(len(bars)):
#         ax6.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
#     leg = ax6.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
#     leg.get_frame().set_alpha(0.5)
#     ylim = ax6.get_ylim()
#     for i in range(len(events_frac)):
#         x0 = mdates.date2num(events_frac['frac_start'].iloc[i])
#         x1 = mdates.date2num(events_frac['frac_end'].iloc[i]) 
#         width = (x0-x1)
#         y1 = events_frac['P1_scaled'].iloc[i]/ylim[1]
#         if events_frac['P1_scaled'].iloc[i] > 0.2:
#             barcolor = 'red'
#         else:
#             barcolor = 'green'
#         rect = Rectangle((x0 - (width/2.0), 0), width, events_frac['P1_scaled'].iloc[i], color=barcolor)
#         ax6.add_patch(rect)
#         #df = (events_frac['frac_end'].iloc[i] - events_frac['frac_start'].iloc[i]) / 2.
#         #ax6.hlines(events_frac['P1_scaled'].iloc[i], events_frac['frac_start'].iloc[i]-df, events_frac['frac_end'].iloc[i]-df)
#==============================================================================
        
    #----dst
    ax7.plot(dst_data[st:et].index, dst_data[st:et]['dst'], label='Dst')
    ax7.hlines(dst_thresh, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax7.set_xticklabels(' ')
    #ax7.xaxis.set_major_formatter(dateFmt)
    #ax7.xaxis.set_major_locator(daysLoc)
    #ax7.xaxis.set_minor_locator(hoursLoc)
    ax7.set_xlim([st, et])
    for l in line:
        ax7.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax7.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['dstgeoeff'].iloc[b*6]], alpha=0.15) 
    #ax7.set_xlabel("Start Time "+ str(st)+" (UTC)")
    leg = ax7.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #ax8.plot(kp_data[st:et].index, kp_data[st:et]['kp'], label='Kp')
    x0 = mdates.date2num(kp_data.index[0])
    x1 = mdates.date2num(kp_data.index[1])
    y=kp_data.kp.iloc[0]
    w=x1-x0 
    if y < 4.0:
        barcolor = 'green'
    elif y >= 4.0 and y < 5.0:
        barcolor = 'orange'
    elif y >= 5.0:
        barcolor = 'red'
    ax8.bar(x0, y, width = w, color = barcolor, label='Kp')

    for i in range(len(kp_data)-1):
        x0 = mdates.date2num(kp_data.index[i])
        x1 = mdates.date2num(kp_data.index[i+1])
        y=kp_data.kp.iloc[i]
        w=x1-x0 
        if y < 4.0:
            barcolor = 'green'
        elif y >= 4.0 and y < 5.0:
            barcolor = 'orange'
        elif y >= 5.0:
            barcolor = 'red'
        ax8.bar(x0, y, width = w, color = barcolor)

    ax8.hlines(kp_thresh, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax8.set_xticklabels(' ')
    ax8.xaxis.set_major_formatter(dateFmt)
    ax8.xaxis.set_major_locator(daysLoc)
    ax8.xaxis.set_minor_locator(hoursLoc)
    ax8.set_xlim([st, et])
    ax8.set_ylim(0,10)
    for l in line:
        ax8.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax8.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    ax8.set_xlabel("Start Time "+ str(st)+" (UTC)")
    leg = ax8.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)        
        
        
        #x0 = mdates.date2num(kp_data.index[i])
#==============================================================================
#         x1 = mdates.date2num(kp_data.index[i+1]) 
#         width = (x1-x0)
#         y = kp_data.kp.iloc[i]
#         
#         #print(kp_data.index.iloc[i])
#         #print(x0,x1,y)
#         
#         if y < 4.0:
#             barcolor = 'green'
#         elif y >= 4.0 and y < 5.0:
#             barcolor = 'orange'
#         else:
#             barcolor = 'red'
#         rect = Rectangle((x0, 0), width, y, color=barcolor)
#         ax8.add_patch(rect)
#==============================================================================
    
    #plt.show()

    plt.savefig(plt_outpath + plt_outfile, format='pdf')

    plt.close()          
          
    return None
 
    
#==============================================================================
# def find_event_times(data):
#     
#     """
#     Find the start and end time of the events
#     """
# 
#     import copy
#     
#     w = copy.deepcopy(data['bzm_predicted'])
#     w.loc[np.where(data['bzm_predicted'] !=0.)] = 1
#     w = w - np.roll(w,1)
#           
#     st = np.where(w == 1)[0]
#     et = np.where(w == -1)[0]-1
# 
#     evt_indices = np.transpose([st,et])
# 
#     evt_times = np.transpose(np.array([data['date'][st], data['date'][et]]))
#    
#     return evt_times, evt_indices
#==============================================================================
    
def event_end(data, i):
    
    """ 
    An event is defined as the times bounded by B_z passing through 0.
    
    This function determines the end of an event by checking whether Bz
    is changing sign.
    
    Parameters
    ----------
    data : pandas dataframe, required
        Mag and sw data 
    i: int, required
         current array index 
        
    Returns
    -------
    event_end_yes: int
        flag to signal event end
    
    """
    
    if data['bz'][i] * data['bz'][i-1] >= 0:
        event_end_yes = 0
    
    if data['bz'][i] * data['bz'][i-1] < 0:
        event_end_yes = 1
        
    if i == len(data['date'])-1:
        event_end_yes = 1
        
    return event_end_yes
    
    
    
def long_duration(istart, iend, min_duration):

    """
    Check to see if the event last at least min_duration
    
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indexes
    min_duration: int, required 
        min_duration required for event
        
    Returns
    -------
    long_duration: int
        flag indicating a long_duration event
    
    """  
    
    if (iend - istart) <= min_duration:
        long_duration_yes = 0
        
    if (iend - istart) > min_duration:
        long_duration_yes = 1
        
    return long_duration_yes
    
def By_start(data, istart, iend):

    """
    Find the start of the preceeding By rotation event
    
    """     
    
    
    tmp = np.asarray([x*y for x,y in zip(data.by,data.by[1:])])
    tmpneg = np.where(tmp < 0.0)[0]
    
    #print(istart)
    #print(data['date'][istart])
    #print(tmpneg)
    #print(data['date'][tmpneg])    
    #print(data.by[tmpneg])
    
    #By event starting before Bz event
    if len(np.where(tmpneg <= istart)[0]) > 0:
        idx = np.max(np.where(tmpneg <= istart)[0])
        istart_y = tmpneg[idx]
        if (idx + 1) <= (len(tmpneg)-1):
            iend_y = tmpneg[idx + 1]
        else:
            iend_y = iend
    else:
        iend_y = None
        istart_y = None
    
#==============================================================================
#     if len(np.where((tmpneg >= istart) & (tmpneg <= iend))[0]) > 0:
#         #by event starting after bz event
#         after_by0 = tmpneg[np.min(np.where((tmpneg >= istart) & (tmpneg <= iend))[0])]
#         istart_y = after_by0
#     else:
#         istart_y = None
#==============================================================================

    #TODO:
    
#==============================================================================
#     if (istart - prior_by0) <= (after_by0 - istart):
#         istart_y = prior_by0
#     else:
#         istart_y = after_by0
#==============================================================================


    return istart_y, iend_y


def lambda_chi_calc(data):
    
    """
    Define the varibles lambda and chi which are used in the Chen model
    framework to determine the relationship between the predicted By and Bz 
    components and infer the orientation of the magnetic field
    
    lambda = By_predicted / Bz_predicted
    
    chi = {1, |lambda| > 1,
           0, |lambda| < 1}
    """
    
    data['lambda'] = data['bym_predicted'] / data['bzm_predicted']
    data['chi'] = data.apply(chi_define, axis = 1)
    

def chi_define(data):
    
    """
    Quick function to appply to data dataframe and assing chi based on
    
    chi = {1, |lambda| > 1,
           0, |lambda| < 1}
    """
    
    if abs(data['lambda']) > 1:
        chi = 1
    else:
        chi = 0
        
    return chi   

def icme_event(istart, iend, npts):
    
    """
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indices
    npts: int, required
        length of data array
        
    Returns
    -------
    main_event: int
        flag indicating XXX
    """
    
    if (istart < npts/2.) & (iend > npts/2.):
        main_event = 1
    else:
        main_event = 0
        
    return main_event

    

def predict_duration(data, istart, iend, component = 'z'):
    
    """
    The original version of this code is from Jim Chen and Nick Arge
    and is called predall.pro. This version has been modifed to 
    make the code more readable by removing goto statements
    and 5 minute data averages.
    
    Reference Papers: Chen et al. 1996, 1997, 2012
                      Arge et al. 2002
    
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indices

    Returns
    -------
    None
    
    """

    #Extract data from structure needed for prediction routine
    if component == 'z':
        b = data['bz'].values             #in nT
        theta = data['theta_z'].values     #in degrees
    else:
        b = data['by'].values
        theta = data['theta_y'].values

    theta_start = theta[istart]

    increasing = 0

    step = 20
    for i in np.arange(istart+step, iend, step):
        
        #current values
        b_current = b[i]
        theta_current = theta[i]

        #max bz and theta at max bz,  up until current time
        b_max = np.max(abs(b[istart:i]))
        index_b_max = np.where(abs(b[istart:i]) == b_max)[0][0]         
        
        b_max = b[istart + index_b_max]      #to account for sign of B component
        theta_b_max = theta[istart + index_b_max]

        #max theta
        theta_max = np.max(abs(theta[istart:i]))
        index_theta_max = np.where(abs(theta[istart:i]) == theta_max)[0][0]   
        theta_max = theta[istart + index_theta_max]
        
        #indices of the max b component and theta
        i_bmax = istart + index_b_max
        #i_thetamax = istart + index_theta_max


#==============================================================================
#         if ((istart >= 1484) & (iend  <= 2024)):
#             print("\n time: "+ str(data.date.iloc[i]) +" theta_max: " +str(theta_max))
#==============================================================================

        #determine the rotation of theta so far
        if value_increasing(theta_current, theta_b_max):

            ##first time that theta has been increasing 
            if increasing == 0:
                dtheta = (theta_max - theta_start)
                dth = 180.0 
                increasing = 1
            else:
                dtheta = (theta_max - theta_start) + (theta_max - theta_current)
                dth = 2.0 * theta_max
            
#==============================================================================
#             if ((istart >= 1484) & (iend  <= 2024)):
#                 print("increasing")
#                 print("dtheta: " +str(dtheta))
#==============================================================================
        else:

            dtheta = (theta_max - theta_start) + (theta_max - theta_current)
            dth = 2.0 * theta_max
            
#==============================================================================
#             if ((istart >= 1484) & (iend  <= 2024)):
#                 print("decreasing")
#                 print("dtheta: " +str(dtheta))
#==============================================================================
 
        #determine the predicted duration and rate of rotation of field    
        dduration = i - istart
        rate_of_rotation = dtheta/dduration  #in degrees/minutes

        predicted_duration = abs(dth/rate_of_rotation)/60.           #in hours
        
        frac_est = dduration/(predicted_duration*60.)
        
#==============================================================================
#         if ((istart >= 4256) & (iend  <= 4748)):
#             print("dur: " + str(dduration) + ", rate_rot: " + str(rate_of_rotation) + ", pred_dur: " + str(predicted_duration),\
#                   "actual dur: " + str((iend-istart)/60.)) 
#   
#==============================================================================


        #now try and predict B component max
        if value_increasing(b_current, b_max):
            
            form_function = np.sin(np.pi*((i_bmax - istart)/60.)/predicted_duration) #Sin function in radians
            predicted_bmax = b_max/form_function
            
            #if (form_function < 0) & (i >= step-1):
            #    predicted_bzmax = data['bzm_predicted'][i-step-1]
        else:
            predicted_bmax = b_max
            
        #print(b_max, predicted_bmax, dduration/60., predicted_duration, frac_est)    
                           
        if np.abs(predicted_bmax) > 30.:
            predicted_bmax = b_max 
        
        if component == 'z':
            
            data.loc[i-step:i, 'istart_bz'] = istart
            data.loc[i-step:i, 'iend_bz'] = iend   
            data.loc[i-step:i, 'tau_predicted'] = predicted_duration    #[0][0]
            data.loc[i-step:i, 'tau_actual'] = (iend-istart)/60.
            data.loc[i-step:i, 'frac_est'] = frac_est
            data.loc[i-step:i, 'bzm_predicted'] = predicted_bmax
    
            #index of max bz up to the current time - used for fitting bz profile
            data.loc[i-step:i, 'i_bzmax'] = i_bmax
    
            #record theta characteristics
            theta_max_val = np.max(abs(theta[istart:iend]))
            index_theta_max_val = np.where(abs(theta[istart:iend]) == theta_max_val)[0][0]
            data.loc[i-step:i, 'theta_z_max'] = theta[istart + index_theta_max_val]   
            data.loc[i-step:i, 'dtheta_z'] = dtheta
    
            #max value of Bz with sign
            b_max_val = np.max(abs(b[istart:iend]))
            index_b_max_val = np.where(abs(b[istart:iend]) == b_max_val)[0][0]
            data.loc[i-step:i, 'bzm_actual'] = b[istart + index_b_max_val]         
        
            #fill in rest of data record for remaining portion if what is left is less
            #than one step size
            if (iend-i) <= step:                
                data.loc[i:iend, 'istart_bz'] = istart
                data.loc[i:iend, 'iend_bz'] = iend         
                data.loc[i:iend, 'tau_predicted'] = predicted_duration    #[0][0]
                data.loc[i:iend, 'tau_actual'] = (iend-istart)/60.
                data.loc[i:iend, 'bzm_predicted'] = predicted_bmax
                data.loc[i:iend, 'bzm_actual'] = b[istart + index_b_max_val]  
                data.loc[i:iend, 'i_bzmax'] = i_bmax
        else:
            
            data.loc[i-step:i, 'istart_by'] = istart
            data.loc[i-step:i, 'iend_by'] = iend   
            data.loc[i-step:i, 'tau_predicted_y'] = predicted_duration    #[0][0]
            data.loc[i-step:i, 'tau_actual_y'] = (iend-istart)/60.
            data.loc[i-step:i, 'bym_predicted'] = predicted_bmax
    
            #index of max bz up to the current time - used for fitting bz profile
            data.loc[i-step:i, 'i_bymax'] = i_bmax
    
            #record theta characteristics
            theta_max_val = np.max(abs(theta[istart:iend]))
            index_theta_max_val = np.where(abs(theta[istart:iend]) == theta_max_val)[0][0]
            data.loc[i-step:i, 'theta_y_max'] = theta[istart + index_theta_max_val]   
            data.loc[i-step:i, 'dtheta_y'] = dtheta
    
            #max value of Bz with sign
            b_max_val = np.max(abs(b[istart:iend]))
            index_b_max_val = np.where(abs(b[istart:iend]) == b_max_val)[0][0]
            data.loc[i-step:i, 'bym_actual'] = b[istart + index_b_max_val]         
        
            #fill in rest of data record for remaining portion if what is left is less
            #than one step size
            if (iend-i) <= step:
                data.loc[i:iend, 'istart_y'] = istart
                data.loc[i:iend, 'iend_y'] = iend         
                data.loc[i:iend, 'tau_predicted_y'] = predicted_duration    #[0][0]
                data.loc[i:iend, 'tau_actual_y'] = (iend-istart)/60.
                data.loc[i:iend, 'bym_predicted'] = predicted_bmax
                data.loc[i:iend, 'bym_actual'] = b[istart + index_b_max_val]  


def dst_geo_tag(events, dst_data, dst_thresh = -80, dst_dur_thresh = 2.0, geoeff_only = 0):
    
    """"
    Add tag/column to events dataframes to indicate the geoeffectiveness of the
    events
    
    inputs
    ------
    
    events - pandas data frame
        each row contains the characteristics for a single event
    dst_data - dataframe
        contains hourly dst data to use as a classifier to determine whether the
        event is geoeffective or non geoeffective and recored the max/min value
    dst_thresh - int 
         threshold of dst to define geoeffective event        
    dst_dur_thresh - float 
         event is considered geoeffictive if dst < dst_thresh for more than 
         dst_dur_thresh hours 
         
    outputs
    -------
    events - pandas data frame
        each row contains the characteristics for a single event
    
    """
    
    #add min Dst value and geoeffective tag for each event
    dstmin = pd.DataFrame({'dst':[]})
    dstdur = pd.DataFrame({'dstdur':[]})
    dstgeoeff = pd.DataFrame({'dstgeoeff':[]})

    prev_time = events.start.iloc[-2]

    for j in range(len(events)):
        
        #if events is dataframe with events by frac then don't need to calc
        #dst for each fraction
       
        if events.start.iloc[j] != prev_time:
            
            #dst values for event time period
            dst_evt = dst_data[events['start'].iloc[j] : events['end'].iloc[j]]
    
            #if there are no dst data values then quit and move onto the next event interval
            if len(dst_evt) == len(dst_evt.iloc[np.where(dst_evt['dst'] == False)]):
                dstgeoeff.loc[j] = 2           #unknown geoeff tag  
                dstdur.loc[j] = 0.0
                dstmin.loc[j] = 999
                continue
    
            # the min dst value regardless of duration
            dstmin.loc[j] = dst_evt['dst'].min()
            #print("indx %i, j: %i, dstmin %i, " % (events.index[j], j, dstmin.iloc[j]))
            
            
            #determine periods where dst is continuously below threshold dst < -80            
            dst_evt['tag'] = dst_evt['dst'] <= dst_thresh
    
            fst = dst_evt.index[dst_evt['tag'] & ~ dst_evt['tag'].shift(1).fillna(False)]
            lst = dst_evt.index[dst_evt['tag'] & ~ dst_evt['tag'].shift(-1).fillna(False)]
            pr = np.asarray([[i, j] for i, j in zip(fst, lst) if j > i])
            
            #if the event never reaches dst < -80 then it's not geoeffective
            time_below_thresh = []
            if len(pr) == 0:
                dstgeoeff.loc[j] = 0  
                dstdur.loc[j] = 0.0
            else:                               #at some point during event, dst < -80
                #find the range of times that dst is below the thresh for the longest
                for t in pr:
                    time_below_thresh.append((t[1] - t[0] + timedelta(seconds = 3600)).seconds/60./60.)
                np.asarray(time_below_thresh)    

                #event is considered geoeffictive if dst < -80 for more than 2 hours 
                if np.max(time_below_thresh) >= dst_dur_thresh:
                        
                    #now question if the dst is just recovering from previous event being geoeffective
                    if (dst_evt['dst'].iloc[-1] > dst_evt['dst'].iloc[0] + 2):
    
                        # if there the previous event interval also decreases then it could be recovering from that
                        #if j > 0 & geoeff.loc[j-1] == 1:
                        dstgeoeff.loc[j] = 3                       #dst still rising from previous event -> ambiguous
                    else:
                        dstgeoeff.loc[j] = 1
    
                else: 
                    dstgeoeff.loc[j] = 0       # not below dst threshhold for long enough -> it's not geoeffective

                dstdur.loc[j] = np.max(time_below_thresh)                    
                #print("geoeff: %i, dstdur %i, " % (geoeff.iloc[j], dstdur.iloc[j]))
        else:
            dstgeoeff.loc[j] = dstgeoeff.loc[j-1]
            dstmin.loc[j] = dstmin.loc[j-1]
            dstdur.loc[j] = dstdur.loc[j-1]
            
            #print(geoeff.iloc[j-1])
            #print(geoeff.iloc[j])
        #if geoeff.iloc[j].values == 1:
        #    print("geoeff %i, dstmin %i, dstdur %i" % (geoeff.iloc[j], dstmin.iloc[j], dstdur.iloc[j]) )
        
        
        #update prev_time
        prev_time = events.start.iloc[j]
    
    #if events is events by frac of an event then only need geoeff
    if geoeff_only == 0:
        events = events.reset_index()
        events = pd.concat([events, dstmin, dstdur, dstgeoeff], axis = 1) 

        return events
    else:
        return dstgeoeff, dstmin, dstdur
    
def kp_geo_tag(events, kp_data, kp_thresh = 6, kp_dur_thresh = 3, geoeff_only = 0):
    
    """"
    Add tag/column to events dataframes to indicate the geoeffectiveness of the
    events
    
    inputs
    ------
    
    events - pandas data frame
        each row contains the characteristics for a single event
    kp_data - dataframe
        contains 3 hourly kp data to use as a classifier to determine whether the
        event is geoeffective or non geoeffective 
    kp_thresh - int 
         threshold of kp to define geoeffective event
    kp_dur_thresh - int
        time above the threshold required to count as geoeffective
         
    outputs
    -------
    events - pandas data frame
        each row contains the characteristics for a single event
    
    """
    
    #add min Dst value and geoeffective tag for each event
    kpmax = pd.DataFrame({'kp':[]})
    kpdur = pd.DataFrame({'kpdur':[]})
    geoeff = pd.DataFrame({'geoeff':[]})

    prev_time = events.start.iloc[-2]

    for j in range(len(events)):
        
        #if events is dataframe with events by frac then don't need to calc
        #kp for each fraction
       
        if events.start.iloc[j] != prev_time:
            
            #kp values for event time period
            evt_stime = events['start'].iloc[j].replace(minute=0, second=0)         #to make sure to find kp interval
            kp_evt = kp_data[evt_stime : events['end'].iloc[j]]

            #kp value immediately prior to the event            
            prev_interval_num = kp_data.index.get_loc(kp_evt.index[0]) - 1
            kp_prev_interval = kp_data.kp.iloc[prev_interval_num]
            
            #if there are no kp data values then quit and move onto the next event interval
            if len(kp_evt) == len(kp_evt.iloc[np.where(kp_evt['kp'] == False)]):
                geoeff.loc[j] = 2           #unknown geoeff tag  
                kpdur.loc[j] = 0.0
                kpmax.loc[j] = -999
                continue
    
            # the max kp value regardless of duration
            kpmax.loc[j] = kp_evt['kp'].max()
            
            #determine periods where kp is continuously above the threshold of kp = 6           
            kp_evt['tag'] = kp_evt['kp'] >= kp_thresh
    
            fst = kp_evt.index[kp_evt['tag'] & ~ kp_evt['tag'].shift(1).fillna(False)]
            lst = kp_evt.index[kp_evt['tag'] & ~ kp_evt['tag'].shift(-1).fillna(False)]
            pr = np.asarray([[i, j] for i, j in zip(fst, lst) if j >= i])
            
            #if the event never reaches kp > 6 then it's not geoeffective
            time_above_thresh = []
            if len(pr) == 0:
                geoeff.loc[j] = 0  
                kpdur.loc[j] = 0.0
            else:                               #at some point during event, kp > 6
                #find the range of times that kp is above the thresh for the longest
                for t in pr:
                    #time_above_thresh.append((t[1] - t[0] + timedelta(seconds = 3600)).seconds/60./60.)
                    time_above_thresh.append((((t[1] - t[0])*3) + timedelta(seconds = (3600*3))).seconds/60./60.)
                np.asarray(time_above_thresh)    

                #event is considered geoeffictive if kp > 6 for more than 2 hours 
                if np.max(time_above_thresh) >= kp_dur_thresh:
                        
                    #now question if the kp is just recovering from previous event being geoeffective
                    #Was the Kp immediately prior to the event above threshold and is the first kp
                    #value of the event already above threshold.
                    if np.logical_and((kp_prev_interval > kp_thresh),(kp_evt.kp.iloc[0] > kp_thresh)):    
    
                        # if there the kp immediately prior to the event is above threshold then ambigous
                        geoeff.loc[j] = 3   

                    else:
                        geoeff.loc[j] = 1   #kp peaks during the interval
    
                else: 
                    geoeff.loc[j] = 0       # not above kp threshhold for long enough -> it's not geoeffective

                kpdur.loc[j] = np.max(time_above_thresh)                    
                #print("geoeff: %i, dstdur %i, " % (geoeff.iloc[j], dstdur.iloc[j]))
        else:
            geoeff.loc[j] = geoeff.loc[j-1]
            kpmax.loc[j] = kpmax.loc[j-1]
            kpdur.loc[j] = kpdur.loc[j-1]
            
        #update prev_time
        prev_time = events.start.iloc[j]
    
    #if events is events by frac of an event then only need geoeff
    if geoeff_only == 0:
        events = events.reset_index()
        events = pd.concat([events, kpmax, kpdur, geoeff], axis = 1) 
        events.drop(['level_0'],axis=1, inplace = True)

        return events
    else:
        return geoeff, kpmax, kpdur
    

def predict_geoeff(events_frac, pdf):
        
    """
   
    Parameters
    ----------
    data: dataframe, required 
        
    pdf: n x n x n x n x f matrix, required

    Returns
    -------
    None
    
    """   
    
    import scipy.integrate as integrate
    

    #Using Bayesian statistics laid out in Chen papers, determine the probability 
    #of a geoeffective event given the estimated Bzm and tau
    
    #print("predict00")

    
    #predict = pd.DataFrame({'bzmp_ind':[], 'taup_ind':[], 'P1':[], 'P1':[], \
    #                        'bzm_most_prob':[], 'tau_most_prob':[], 'P3':[]})
    
    #cols = ['bzmp_ind', 'taup_ind', 'P1', 'P2' ,'bzm_most_prob', 'tau_most_prob', 'P3'] 
    #predict = pd.DataFrame(0, index = np.arange(len(events_frac.start)), columns = cols)
        
    #print("predict000")
    
    nevents = len(events_frac.start)
    bzmp_ind = np.zeros((nevents),dtype=int)
    taup_ind = np.zeros((nevents),dtype=int)
    P1 = np.zeros((nevents),dtype=float)
    P1_scaled = np.zeros((nevents),dtype=float)
    P2 = np.zeros((nevents),dtype=float)
    bzm_most_prob = np.zeros((nevents),dtype=float)
    tau_most_prob = np.zeros((nevents),dtype=float)
    P3 = np.zeros((nevents),dtype=float)

    for i in range(len(events_frac.start)):
                
        if events_frac.frac_est.iloc[i] < 0.2:
            continue
        
        if events_frac.tau_predicted.iloc[i] > 250:
            continue
        
        #print(i)
        #print(events_frac.frac.iloc[i])
        #print(events_frac.iloc[i])
        
        #find the plane of probabilities for estimates bzmp and taup
        bzmp_ind[i] = np.max(np.where(pdf['axis_vals'][0] < events_frac.bzm_predicted.iloc[i])[0])
        taup_ind[i] = np.min(np.where(pdf['axis_vals'][1] > events_frac.tau_predicted.iloc[i])[0])
        
        #print("bzm_pred, axis vals")
        #print(bzmp_ind[i], taup_ind[i])
        #print(events_frac.bzm_predicted.iloc[i], pdf["axis_vals"][0][bzmp_ind[i]-1], \
        #      pdf["axis_vals"][0][bzmp_ind[i]], pdf["axis_vals"][0][bzmp_ind[i]+1])
        
        
        #print("here")
        #print("max pdf %f", np.max(pdf['P_bzm_tau_e_bzmp_taup'][:,:,:,:,5]))
    

        #the probability of the event being geoeffective with any value of bzm and tau
        #predict.P1.iloc[i] = pdf['P1_map'][bzmp_ind, taup_ind, events_frac.iloc[i]*5]
        
        #print(np.max(pdf['P_bzm_tau_e_bzmp_taup'][:,:,:,:, int(events_frac.frac.iloc[i] * 5)]))
        
        if events_frac.frac_est.iloc[i] < 0.2:
            pdf_frac = 0
        else:
            pdf_frac = 1
        
        
        P1[i] = integrate.simps(integrate.simps(pdf['P_bzm_tau_e_bzmp_taup']\
                       [:,:,bzmp_ind[i], taup_ind[i], pdf_frac], \
                       pdf['axis_vals'][1]),\
                       pdf['axis_vals'][0])
        
        P1_scaled[i] = integrate.simps(integrate.simps((pdf['P_bzm_tau_e_bzmp_taup']\
                       [:,:,bzmp_ind[i], taup_ind[i], pdf_frac]\
                       * (1/pdf["P1_map"][:,:,pdf_frac].max())),\
                       pdf['axis_vals'][1]),\
                       pdf['axis_vals'][0])
        
        #print("P1 ")
        #print(P1[i])
    
        #the probability of the event have actual values bzmp +/- 5 nT and taup +/- 4 hours
        bzmp_ind_low = np.max(np.where(pdf['axis_vals'][0] < (events_frac.bzm_predicted.iloc[i] - 6.12))[0])
        bzmp_ind_high = np.max(np.where(pdf['axis_vals'][0] < (events_frac.bzm_predicted.iloc[i] + 6.12))[0])
        
        taup_ind_low = np.min(np.where(pdf['axis_vals'][1] > (events_frac.tau_predicted.iloc[i] - 5.0))[0])
        taup_ind_high = np.min(np.where(pdf['axis_vals'][1] > (events_frac.tau_predicted.iloc[i] + 5.0))[0])
        
        #print("P2")
        #print(bzmp_ind_low, bzmp_ind_high, taup_ind_low, taup_ind_high)

        P2[i] = integrate.simps(integrate.simps(pdf['P_bzm_tau_e_bzmp_taup']\
                       [bzmp_ind_low:bzmp_ind_high+1, taup_ind_low:taup_ind_high+1, bzmp_ind[i], taup_ind[i], pdf_frac],\
                       pdf['axis_vals'][1][taup_ind_low:taup_ind_high+1]),\
                       pdf['axis_vals'][0][bzmp_ind_low:bzmp_ind_high+1])
        
        #print(P2[i])
        #print("predict2")    

        #The most probable values of bzm and tau based on bzmp and taup  
        prob_max_ind = np.where(pdf['P_bzm_tau_e_bzmp_taup'][:,:,bzmp_ind[i], taup_ind[i], pdf_frac] == 
                 np.max(pdf['P_bzm_tau_e_bzmp_taup'][:,:,bzmp_ind[i], taup_ind[i], pdf_frac]) ) 
        
        #print(len(prob_max_ind[0]))
        
        bzm_most_prob[i] = pdf["axis_vals"][0, prob_max_ind[0]]
        tau_most_prob[i] = pdf["axis_vals"][1, prob_max_ind[1]]
        
        bzm_prob_max_ind_low = np.max(np.where(pdf['axis_vals'][0] < bzm_most_prob[i] - 6.12)[0])
        bzm_prob_max_ind_high = np.max(np.where(pdf['axis_vals'][0] < bzm_most_prob[i] + 6.12)[0])
        
        tau_prob_max_ind_low = np.max(np.where(pdf['axis_vals'][1] < pdf["axis_vals"][1, tau_most_prob[i]] - 5.0)[0])
        tau_prob_max_ind_high = np.max(np.where(pdf['axis_vals'][1] < pdf["axis_vals"][1, tau_most_prob[i]] + 5.0)[0])
        
        P3[i] = integrate.simps(integrate.simps(pdf['P_bzm_tau_e_bzmp_taup']\
                       [bzm_prob_max_ind_low:bzm_prob_max_ind_high+1, tau_prob_max_ind_low:tau_prob_max_ind_high+1, bzmp_ind[i], taup_ind[i], pdf_frac],\
                       pdf['axis_vals'][1][bzm_prob_max_ind_low:bzm_prob_max_ind_high+1]),\
                       pdf['axis_vals'][0][tau_prob_max_ind_low:tau_prob_max_ind_high+1])

        #print("predict3")

    #add new predictions to events_frac datafram
    events_frac["bzmp_ind"] = bzmp_ind
    events_frac["taup_ind"] = taup_ind
    events_frac["P1"] = P1
    events_frac["P1_scaled"] = P1_scaled
    events_frac["P2"] = P2
    events_frac["bzm_most_prob"] = bzm_most_prob
    events_frac["tau_most_prob"] = tau_most_prob
    events_frac["P3"] = P3
    
    return events_frac

def value_increasing(value_current, value_max):
    """
    function determines if the input value is 
    increasing or not
    
    Parameters
    ----------
    value_current, value_max : float, required 
        current and maximum value

    Returns
    -------
    value_increasing: int
        flag determining whether value is increasing
    
    """ 
    
    
    if abs(value_current) < 0.8*abs(value_max):
        value_increasing = 0 
      
    if abs(value_current) > 0.8*abs(value_max): 
        value_increasing = 1
      
    return value_increasing
    
    
    
def validation_stats(data, istart, iend, outdir=''):
    
    """
    print out some validation stats --- not sure this works yet!!
    
    """
    
    duration = (iend-istart)
    
    #compute the unsigned fractional diviation between the predicted and observed Bzm and tau
    fraction_of_event = np.arange(9)/10.
    index_of_event = np.floor((duration*fraction_of_event)+istart)
    print, istart, iend
    print, index_of_event
    
    Bzm_fractional_deviation = abs((data['bzm_predicted'][index_of_event] -data['bzm_actual'][istart]) \
                                  /data['bzm_actual'][istart])
    Tau_fractional_deviation = abs(data['tau_predicted'][index_of_event] - data['tau_actual'][istart]) \
                                  /data['tau_actual'][istart]

    start_date = datetime.strftime(data['date'][istart], "%Y-%m-%d %H:%M")
    
    print, start_date
    print, fraction_of_event
    print, Bzm_fractional_deviation
    print, Tau_fractional_deviation
    
    #fname1='Prediction_Results_Duration.txt'
    #OPENW, dunit, outdir + fname1, /GET_LUN, /APPEND
    #PRINTF, dunit, format='(a-25,10(2x,f5.2))', start_date, Tau_fractional_deviation
    #FREE_LUN, dunit
    
    #fname2='Prediction_Results_BzMax.txt'
    #OPENW, bunit, outdir + fname2, /GET_LUN, /APPEND
    #PRINTF, bunit, format='(a-25,10(2x,f5.2))', start_date, Bzm_fractional_deviation
    #FREE_LUN, bunit
    
    
def clean_data(mag, sw):
    
    """
    Clean solar wind plasma and mag data to remove bad data values and fix 
    data gaps (Michele Cash version translated to python)
    
    Parameters
    ----------
    mag : data array, required 
    sw: data, required

    Returns
    -------
    None
    
    """

    #print(mag['gsm_bz'])
    
    nevents_sw = len(sw['v'])
    nevents_mag = len(mag['gsm_bz'])
    
    
    #print(nevents_mag)
    
    #---check them magnetic field data
    bad_mag = np.where((abs(mag['gsm_bx']) > 90.) & (abs(mag['gsm_by']) > 90.) & (abs(mag['gsm_bz']) > 90.))
    nbad_mag = len(bad_mag[0])   
    
    #no good mag data
    if nevents_mag - nbad_mag < 2:
        print("******* No valid magnetic field data found *******")
        return None

        
    #if there is some bad data, set to NaN and interpolate
    if nbad_mag > 0:   
        mag['gsm_bx'][bad_mag] = np.nan
        mag['gsm_by'][bad_mag] = np.nan
        mag['gsm_bz'][bad_mag] = np.nan
        mag['bt'][bad_mag]     = np.na
        mag['gsm_lat'][bad_mag] = np.nan
        mag['gsm_lon'][bad_mag] = np.nan

    mag['gsm_bx'] = mag['gsm_bx'].interpolate()
    mag['gsm_by'] = mag['gsm_by'].interpolate()
    mag['gsm_bz'] = mag['gsm_bz'].interpolate()
    mag['bt']     = mag['bt'].interpolate()
    mag['gsm_lat'] = mag['gsm_lat'].interpolate()
    mag['gsm_lon'] = mag['gsm_lon'].interpolate()
    
    #---check solar wind velocity
    badsw_v = np.where((sw['v'] < 0.) & (sw['v'] > 3000.))
    nbadsw_v = len(badsw_v[0])
    
    #no valid sw data
    if nevents_sw - nbadsw_v < 2:
        print("******* No valid solar wind plasma data found *******")
        return None

    #if there are some bad sw velcotiy data values, set to NaN and interpolate
    if nbadsw_v > 0:
        print('******* Some bad SWE velocity data *******')
        sw['v'][badsw_v[0]] = np.nan
        
    sw['v'] = sw['v'].interpolate()

    #---check solar wind density which can be good even where the velocity was good
    badsw_n = np.where((sw['n'] < 0.) & (sw['n'] > 300.))
    nbadsw_n = len(badsw_n[0])
    
    if nbadsw_n > 0:
        print('******* Some bad SWE density data *******')
        
        #if there are no good density values, set all density to 4.0
        if nevents_sw - nbadsw_n == 0:
            sw['n'][:] = 4.0
        else:
            sw['n'][badsw_n[0]] = np.nan
    
    sw['n'] = sw['n'].interpolate()
            
    #---check solar wind temperature which can be good even where the velocity was good
    badsw_t = np.where(sw['t'] < 0.) 
    nbadsw_t = len(badsw_t[0])
    
    if nbadsw_t > 0:
        print('******* Some bad SWE temperature data *******')
        
        #if there are no good density values, set all temperature to 0.0
        if nevents_sw - nbadsw_t == 0:
            sw['t'][:] = 0.0
        else:
            sw['t'][badsw_n[0]] = np.nan

    sw['t'] = sw['t'].interpolate()
        
    #print(len(sw))        

    #---interpolate the solar wind velocity to the mag time
    #SWVel=INTERPOL(swe.Speed,swe.jdate,mag.jdate)
    #Np=INTERPOL(swe.Np,swe.jdate,mag.jdate)
    
    #return SWVel, Np  -----QUERY????

    
    return mag, sw
    

    
    
    
    
      
    


        
    