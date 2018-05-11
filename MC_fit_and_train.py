#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:35:14 2018

@author: hazelbain
"""


import MCpredict as MC
import read_dst as dst
import read_kp as kp
import Richardson_ICME_list as icme

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import calendar
import pickle as pickle
import platform

#import MC_predict_pdfs as mcp
#import MC_predict_plots as mcplt
#from MCpredict import predict_geoeff, dst_geo_tag


def fit_training_events(fname = '', dst_thresh = -80, kp_thresh = 6, livedb = 0, csv = 0,
                        predict = 0, plotting = 1):

    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    else:
        proj_dir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/'
    
    
    #read in the dst data
    dst_data = dst.read_dst_df()

    #read in the kp data
    kp_data = kp.read_kp()

    #read in Richardson and Cane ICME list 
    icme_list = icme.read_richardson_icme_list()
    
    
    #### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword
    #t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003', '1-jan-2004','1-jan-2005',\
    #      '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
    #      '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
    
    #t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003', '31-dec-2004','31-dec-2005',\
    #      '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013', \
    #      '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']
    
    t1 = '16-feb-1998'
    t2 = '31-may-2017'
    
    #t1 = '16-feb-1998'
    #t2 = '9-apr-1998'
    
    #convert t1 and t2 to list of datetime
    t1  = datetime.strptime(t1, "%d-%b-%Y") 
    t2  = datetime.strptime(t2, "%d-%b-%Y")
    
    
    #get list of week start and end dates - with overlap of one day
    month_list = [datetime((t1 + timedelta(days=i)).year, (t1 + timedelta(days=i)).month,1) for i in range((t2-t1).days)]
    month_list = list(set(month_list))
    month_list.sort()
    
    date_list = []
    cal = calendar.Calendar()
    for m in month_list:
        for x in cal.monthdatescalendar(m.year, m.month):
            date_list.append([x[0], x[0]+timedelta(days = 8)])
    date_list = np.asarray(date_list)
    
    #add seconds and minutes  
    date_list[:,0] = [datetime.combine(x, datetime.min.time())  for x in date_list[:,0]]
    date_list[:,1] = [datetime.combine(x, datetime.min.time())  for x in date_list[:,1]]    
    
    #set up the empty dataframes to record event data
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_time_frac = pd.DataFrame()        #predicted events characteristics split into time increments
    
    first_index = 0
    errfit = []
    #loop through the weeks
    for i in range(len(date_list[:,0])):
        
        if date_list[i,0] < datetime(1998,2,16):
            first_index = i
            continue

        #just pass the current year of dst_data
        dst_data_week = dst_data.query('index >   "'+datetime.strftime(date_list[i,0] - timedelta(1), "%Y-%m-%d %H:%M:%S") + \
               '"and index < "'+datetime.strftime(date_list[i,1] + timedelta(1), "%Y-%m-%d %H:%M:%S")+'"')
        
        #just pass the current year of kp_data
        kp_data_week = kp_data.query('index >   "'+datetime.strftime(date_list[i,0]- timedelta(1), "%Y-%m-%d %H:%M:%S") + \
               '"and index < "'+datetime.strftime(date_list[i,1] + timedelta(1), "%Y-%m-%d %H:%M:%S")+'"')
        
        
        #just pass the current year of icme_list + a couple of days to make sure there's no overlapping ICME that is missed
        icme_list_week = icme_list.query('year == "'+datetime.strftime(date_list[i,0], "%Y")+'"')

                
        try:   
            
            #format time strings
            stf = datetime.strftime(date_list[i,0], "%Y-%m-%d")
            etf = datetime.strftime(date_list[i,1], "%Y-%m-%d")

            #start and end of ICME
            line1 = list(icme_list_week.query('plasma_start >="'+stf+'" and plasma_start <= "'+ datetime.strftime(date_list[i,1] + timedelta(days = 1), "%Y-%m-%d") +'"')\
                    [['plasma_start','plasma_end']].values.flatten())

            #start and end of MC
            line2 = list(icme_list_week.query('mc_start >="'+stf+'" and mc_start <= "'+ datetime.strftime(date_list[i,1] + timedelta(days = 1), "%Y-%m-%d") +'"')\
                    [['mc_start','mc_end']].values.flatten())

            data, events_tmp, events_time_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
                    dst_data_week, kp_data_week, line = line1, line2=line2,\
                    csv = csv, livedb = livedb, predict = predict,\
                    smooth_num = 100, dst_thresh = dst_thresh, kp_thresh = kp_thresh, plotting = plotting,\
                    plt_outfile = 'mcpredict_'+ datetime.strftime(date_list[i][0], "%Y-%m-%d_%H%M") + '.pdf' ,\
                    plt_outpath = proj_dir + 'longterm_'+fname[0:-1]+'/')
            
            if i == first_index + 1: 
                events = events_tmp
                events_time_frac = events_time_frac_tmp
            elif i > first_index + 1:
                #append
                events = events.append(events_tmp)
                events_time_frac_tmp.evt_index = events_time_frac_tmp.evt_index + (events_time_frac.evt_index.iloc[-1] + 1)  
                events_time_frac = events_time_frac.append(events_time_frac_tmp)

        except:
            
            print("*** Error running Chen MC Prediction ***")
            errfit.append(date_list[i,:])
            #print("Unexpected error:", sys.exc_info()[0])
            #raise
        
    
    #drop duplicates
    events.drop_duplicates(['start'], inplace=True, keep='last')  
    events_time_frac.drop_duplicates(['start','bzm_predicted','tau_predicted'], inplace=True, keep='last')    
                   
    #reset the index 
    events.reset_index(drop=True, inplace = True) 
    events_time_frac.reset_index(drop=True, inplace = True)
    
    #events.to_csv("train/events_"+fname+"train_kp"+str(abs(kp_thresh))+".csv", sep='\t', encoding='utf-8') 
    #events_frac.to_csv("train/events_frac_"+fname+"train_kp"+str(abs(kp_thresh))+".csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_time_frac,open("train/events_time_frac_"+fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh))+".p", "wb"))
    pickle.dump(events,open("train/events_"+fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh))+".p", "wb"))
        
    #mcplt.plot_obs_bz_tau(events_uniq, dd = "train/plots/", fname = fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh)))
    #mcplt.plot_obs_vs_predict(events_frac_uniq, dd = "train/plots/", fname = fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh)))
    #mcplt.plot_theta(events_frac_uniq, dd = "train/plots/", fname = fname+"train_kp"+str(abs(kp_thresh)))

    return events, events_time_frac, errfit

def clean_events():

    #restore
    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    else:
        proj_dir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/'
    
    
    ## read in fit all events files
    events_time_frac = pickle.load(open("train/events_time_frac_fitall3_train_dst80_kp6.p","rb"))
    events = pickle.load(open("train/events_fitall3_train_dst80_kp6.p","rb"))
    
    
    ## correct evt_index for events_time_frac2
    for i in range(0,len(events)):
        stime = events.start.iloc[i]
        idx = events_time_frac.query('start == "'+datetime.strftime(stime, "%d-%b-%Y %H:%M:%S")+'"').index 
        events_time_frac.evt_index.iloc[idx] = i
    
    
    #rename the geoeff column defined by kp to kpgeoeff
    events.rename(index=str, columns={"geoeff": "kpgeoeff"}, inplace = True)
    events_time_frac.rename(index=str, columns={"geoeff": "kpgeoeff"}, inplace = True)
    
    #add new column for manual inspection results of geoeffective events
    events['geoeff'] = pd.Series(np.zeros(len(events)), index=events.index)
    events_time_frac['geoeff'] = pd.Series(np.zeros(len(events_time_frac)), index=events_time_frac.index)

    good_mc = pd.DataFrame()
    boundaries = pd.DataFrame()
    icmesheath = pd.DataFrame()
    reject = pd.DataFrame()
    
#==============================================================================
#     good_mc = good_mc.append(events.iloc[0])        #17-feb-1998
#     good_mc = good_mc.append(events.iloc[529])      #1998-10-18 22:14:00
#     good_mc = good_mc.append(events.iloc[577])      #1998-11-07 22:19:00
#     good_mc = good_mc.append(events.iloc[579])      #1998-11-09 03:56:00
#     good_mc = good_mc.append(events.iloc[586])      #1998-11-13 05:01:00
#     good_mc = good_mc.append(events.iloc[963])      #1999-04-16 16:37:00
#     good_mc = good_mc.append(events.iloc[1963])     #2000-07-15 15:50:00
#     good_mc = good_mc.append(events.iloc[2006])     #2000-08-12 00:19:00
#     good_mc = good_mc.append(events.iloc[2114])     #2000-10-13 16:50:00
#     good_mc = good_mc.append(events.iloc[2141])     #2000-10-28 21:22:00
#     good_mc = good_mc.append(events.iloc[2158])     #2000-11-06 22:25:00
#     good_mc = good_mc.append(events.iloc[2191])     #2000-11-28 18:23:00 (2)
#     good_mc = good_mc.append(events.iloc[2403])     #2001-03-20 02:43:00 (2)
#     good_mc = good_mc.append(events.iloc[2421])     #2001-03-27 19:18:00 (2)
#     good_mc = good_mc.append(events.iloc[2791])     #2001-10-03 06:49:00
#     good_mc = good_mc.append(events.iloc[2844])     #2001-10-31 18:33:00
#     good_mc = good_mc.append(events.iloc[2852])     #2001-11-05 18:00:00
#     good_mc = good_mc.append(events.iloc[2854])     #2001-11-06 13:47:00
#     good_mc = good_mc.append(events.iloc[3398])     #2002-08-20 18:19:00 (2)
#     good_mc = good_mc.append(events.iloc[3899])     #2003-05-09 18:34:00
#     good_mc = good_mc.append(events.iloc[4010])     #2003-06-18 04:44:00
#     good_mc = good_mc.append(events.iloc[4240])     #2003-08-18 01:32:00
#     good_mc = good_mc.append(events.iloc[4848])     #2004-07-24 08:36:00
#     good_mc = good_mc.append(events.iloc[4853])     #2004-07-27 03:52:00
#     good_mc = good_mc.append(events.iloc[5071])     #2004-11-07 20:38:00
#     good_mc = good_mc.append(events.iloc[5076])     #2004-11-10 01:52:00
#     good_mc = good_mc.append(events.iloc[5229])     #2005-01-07 20:59:00
#     good_mc = good_mc.append(events.iloc[5525])     #2005-05-20 03:02:00
#     good_mc = good_mc.append(events.iloc[5662])     #2005-07-10 10:40:00
#     good_mc = good_mc.append(events.iloc[6957])     #2006-12-14 23:17:00
#     good_mc = good_mc.append(events.iloc[9307])     #2009-07-21 08:47:00
#     good_mc = good_mc.append(events.iloc[11314])    #2011-10-24 21:17:00
#     good_mc = good_mc.append(events.iloc[11618])    #2012-03-09 05:48:00
#     good_mc = good_mc.append(events.iloc[11729])    #2012-04-23 16:40:00
#     good_mc = good_mc.append(events.iloc[11911])    #2012-07-15 01:29:00
#     good_mc = good_mc.append(events.iloc[12110])    #2012-10-08 18:32:00
#     good_mc = good_mc.append(events.iloc[12201])    #2012-11-13 17:23:00
#     good_mc = good_mc.append(events.iloc[12744])    #2013-06-28 08:32:00
#     good_mc = good_mc.append(events.iloc[13249])    #2014-02-18 13:24:00
#     good_mc = good_mc.append(events.iloc[13364])    #2014-04-11 10:06:00
#     good_mc = good_mc.append(events.iloc[13998])    #2015-01-07 07:20:00
#     good_mc = good_mc.append(events.iloc[14176])    #2015-03-17 12:15:00
#     good_mc = good_mc.append(events.iloc[14405])    #2015-06-23 01:36:00
#     good_mc = good_mc.append(events.iloc[14552])    #2015-08-27 08:36:00
#     good_mc = good_mc.append(events.iloc[14707])    #2015-11-07 02:08:00
#     good_mc = good_mc.append(events.iloc[14802])    #2015-12-20 03:11:00
#     good_mc = good_mc.append(events.iloc[14829])    #2015-12-31 18:30:00
#     
#     
#     boundaries = boundaries.append(events.iloc[1563])   #2000-01-21 04:21:00
#     boundaries = boundaries.append(events.iloc[1837])   #2000-05-16 15:22:00
#     boundaries = boundaries.append(events.iloc[2005])   #2000-08-10 00:12:00
#     boundaries = boundaries.append(events.iloc[2096])   #2000-10-04 00:40:00
#     boundaries = boundaries.append(events.iloc[2478])   #2001-04-22 04:26:00
#     boundaries = boundaries.append(events.iloc[3143])   #2002-04-17 14:36:00
#     boundaries = boundaries.append(events.iloc[3424])   #2002-09-06 19:13:00
#     boundaries = boundaries.append(events.iloc[3460])   #2002-10-01 03:35:00 (2)
#     boundaries = boundaries.append(events.iloc[3463])   #2002-10-03 09:47:00
#     boundaries = boundaries.append(events.iloc[4131])   #2003-08-18 01:32:00
#     boundaries = boundaries.append(events.iloc[4400])   #2004-01-24 19:10:00
#     boundaries = boundaries.append(events.iloc[4845])   #2004-07-22 13:26:00
#     boundaries = boundaries.append(events.iloc[4910])   #2004-08-30 01:42:00
#     boundaries = boundaries.append(events.iloc[5545])   #2005-05-29 18:06:00
#     boundaries = boundaries.append(events.iloc[5586])   #2005-06-12 12:24:00
#     boundaries = boundaries.append(events.iloc[6343])   #2006-04-13 23:52:00 (2MCs)
# 
# 
#     icmesheath = icmesheath.append(events.iloc[175])    #1998-05-02 05:11:00
#     icmesheath = icmesheath.append(events.iloc[485])    #1998-09-23 21:28:00
#     icmesheath = icmesheath.append(events.iloc[1449])   #1999-11-12 00:38:00
#     icmesheath = icmesheath.append(events.iloc[826])    #1999-02-18 01:43:00
#     icmesheath = icmesheath.append(events.iloc[1606])   #2000-02-12 04:21:00
#     icmesheath = icmesheath.append(events.iloc[1734])   #2000-04-06 16:43:00
#     icmesheath = icmesheath.append(events.iloc[1856])   #2000-05-24 16:09:00
#     icmesheath = icmesheath.append(events.iloc[2066])   #2000-09-17 20:40:00
#     icmesheath = icmesheath.append(events.iloc[2096])   #2000-10-04 00:40:00
#     icmesheath = icmesheath.append(events.iloc[2430])   #2001-03-31 04:01:00
#     icmesheath = icmesheath.append(events.iloc[2432])   #2001-03-31 14:48:00
#     icmesheath = icmesheath.append(events.iloc[2455])   #2001-04-11 15:07:00
#     icmesheath = icmesheath.append(events.iloc[2456])   #2001-04-11 20:44:00
#     icmesheath = icmesheath.append(events.iloc[2880])   #2001-11-24 11:30:00
#     icmesheath = icmesheath.append(events.iloc[3086])   #2002-03-23 19:28:00
#     icmesheath = icmesheath.append(events.iloc[3087])   #2002-03-24 12:06:00
#     icmesheath = icmesheath.append(events.iloc[3146])   #2002-04-20 02:30:00
#     icmesheath = icmesheath.append(events.iloc[3231])   #2002-05-23 02:38:00
#     icmesheath = icmesheath.append(events.iloc[3372])   #2002-08-01 23:50:00
#     icmesheath = icmesheath.append(events.iloc[3959])   #2003-05-29 21:09:00
#     icmesheath = icmesheath.append(events.iloc[4242])   #2003-10-30 11:13:00
#     icmesheath = icmesheath.append(events.iloc[4579])   #2004-04-03 11:49:00
#     icmesheath = icmesheath.append(events.iloc[5083])   #2004-11-12 14:38:00
#     icmesheath = icmesheath.append(events.iloc[9931])   #2010-04-05 04:09:00
#     icmesheath = icmesheath.append(events.iloc[11110])  #2011-08-05 17:31:00
# 
#     reject = reject.append(events.iloc[2430])   #2001-03-31 04:01:00
#     reject = reject.append(events.iloc[2431])   #2001-03-31 08:47:00
#     reject = reject.append(events.iloc[2432])   #2001-03-31 14:48:00
#     reject = reject.append(events.iloc[3202])   #2002-05-11 00:02:00
#     reject = reject.append(events.iloc[4849])   #2004-07-25 18:01:00
#     reject = reject.append(events.iloc[12093])  #2012-09-30 10:26:00
#     reject = reject.append(events.iloc[12514])  #2013-03-17 04:45:00
#     #reject = reject.append(events.iloc[])
#     #reject = reject.append(events.iloc[])
#     #reject = reject.append(events.iloc[])
#     #reject = reject.append(events.iloc[])
#     #reject = reject.append(events.iloc[])
#     
# 
#==============================================================================

    good_dates = ['1998-10-18 22:14:00',
    '1998-11-07 22:19:00',
    '1998-11-09 03:56:00',
    '1998-11-13 05:01:00',
    '1999-04-16 16:37:00',
    '2000-07-15 15:50:00',
    '2000-08-12 00:19:00',
    '2000-10-13 16:50:00',
    '2000-10-28 21:22:00',
    '2000-11-06 22:25:00',
    '2000-11-28 18:23:00',
    '2001-03-20 02:43:00',
    '2001-03-27 19:18:00',
    '2001-10-03 06:49:00',
    '2001-10-31 18:33:00',
    '2001-11-05 18:00:00',
    '2001-11-06 13:47:00',
    '2002-08-20 18:19:00',
    '2003-05-09 18:34:00',
    '2003-06-18 04:44:00',
    '2003-08-18 01:32:00',
    '2004-07-24 08:36:00',
    '2004-07-27 03:52:00',
    '2004-11-07 20:38:00',
    '2004-11-10 01:52:00',
    '2005-01-07 20:59:00',
    '2005-05-20 03:02:00',
    '2005-07-10 10:40:00',
    '2006-12-14 23:17:00',
    '2009-07-21 08:47:00',
    '2011-10-24 21:17:00',
    '2012-03-09 05:48:00',
    '2012-04-23 16:40:00',
    '2012-07-15 01:29:00',
    '2012-10-08 18:32:00',
    '2012-11-13 17:23:00',
    '2013-06-28 08:32:00',
    '2014-02-18 13:24:00',
    '2014-04-11 10:06:00',
    '2015-01-07 07:20:00',
    '2015-03-17 12:15:00',
    '2015-06-23 01:36:00',
    '2015-08-27 08:36:00',
    '2015-11-07 02:08:00',
    '2015-12-20 03:11:00',
    '2015-12-31 18:30:00']
    
    boundary_dates = ['2000-01-21 04:21:00',
    '2000-05-16 15:22:00',
    '2000-08-10 00:12:00',
    '2000-10-04 00:40:00',
    '2001-04-22 04:26:00',
    '2002-04-17 14:36:00',
    '2002-09-06 19:13:00',
    '2002-10-01 03:35:00',
    '2002-10-03 09:47:00',
    '2003-08-18 01:32:00',
    '2004-01-24 19:10:00',
    '2004-07-22 13:26:00',
    '2004-08-30 01:42:00',
    '2005-05-29 18:06:00',
    '2005-06-12 12:24:00',
    '2006-04-13 23:52:00']
    
    icme_dates = ['1998-05-02 05:11:00',
    '1998-09-23 21:28:00',
    '1999-11-12 00:38:00',
    '1999-02-18 01:43:00',
    '2000-02-12 04:21:00',
    '2000-04-06 16:43:00',
    '2000-05-24 16:09:00',
    '2000-09-17 20:40:00',
    '2000-10-04 00:40:00',
    '2001-03-31 04:01:00',
    '2001-03-31 14:48:00',
    '2001-04-11 15:07:00',
    '2001-04-11 20:44:00',
    '2001-11-24 11:30:00',
    '2002-03-23 19:28:00',
    '2002-03-24 12:06:00',
    '2002-04-20 02:30:00',
    '2002-05-23 02:38:00',
    '2002-08-01 23:50:00',
    '2003-05-29 21:09:00',
    '2003-10-30 11:13:00',
    '2004-04-03 11:49:00',
    '2004-11-12 14:38:00',
    '2010-04-05 04:09:00',
    '2011-08-05 17:31:00']
    
    reject_dates = ['2001-03-31 04:01:00',
    '2001-03-31 08:47:00',
    '2001-03-31 14:48:00',
    '2002-05-11 00:02:00',
    '2004-07-25 18:01:00',
    '2012-09-30 10:26:00',
    '2013-03-17 04:45:00']


    good_ind = []
    good_ind_time_frac = []
    for d in good_dates:
        for dd in events.query('start == "'+d+'" ').index.values.astype(int):
            good_ind.append(dd)
        for dd in events_time_frac.query('start == "'+d+'" ').index.values.astype(int):
            good_ind_time_frac.append(dd)
            
    bound_ind = []
    bound_ind_time_frac = []
    for d in boundary_dates:
        for dd in events.query('start == "'+d+'" ').index.values.astype(int):
            bound_ind.append(dd)
        for dd in events_time_frac.query('start == "'+d+'" ').index.values.astype(int):
            bound_ind_time_frac.append(dd)
            
    icme_ind = []
    icme_ind_time_frac = []
    for d in icme_dates:
        for dd in events.query('start == "'+d+'" ').index.values.astype(int):
            icme_ind.append(dd)            
        for dd in events_time_frac.query('start == "'+d+'" ').index.values.astype(int):
            icme_ind_time_frac.append(dd)

    reject_ind = []
    reject_ind_time_frac = []
    for d in reject_dates:
        for dd in events.query('start == "'+d+'" ').index.values.astype(int):
            reject_ind.append(dd)
        for dd in events_time_frac.query('start == "'+d+'" ').index.values.astype(int):
            reject_ind_time_frac.append(dd)


    #update events with indexing
    #good_ind = good_mc.index
    #bound_ind = boundaries.index
    #icme_ind = icmesheath.index
    #reject_ind = reject.index
    events['geoeff'][good_ind] = 1      #good mc events
    events['geoeff'][bound_ind] = 2     #boundary issues
    events['geoeff'][reject_ind] = 9    #rejected 
    events['geoeff'][icme_ind] = 3      #icme sheath
    
    events_time_frac['geoeff'][good_ind_time_frac] = 1      #good mc events
    events_time_frac['geoeff'][bound_ind_time_frac] = 2     #boundary issues
    events_time_frac['geoeff'][reject_ind_time_frac] = 9    #rejected 
    events_time_frac['geoeff'][icme_ind_time_frac] = 3      #icme sheath
    
    #test_ind = events.query('dstgeoeff == 1.0 and geoeff == 0.0')[['start','end','bzm','tau','dst','dstgeoeff']].index
    #events['geoeff'][test_ind] = 999
    
    #index the events_time_Frac structure
    #geo1_ind = events.query('geoeff == 1').index
    #geo2_ind = events.query('geoeff == 2').index
    #geo3_ind = events.query('geoeff == 3').index
    #geo9_ind = events.query('geoeff == 9').index
    
    
    #for g in geo1_ind:
    #    for gg in (events_time_frac.query('evt_index == ' + g).index).astype('int'):
    #        events_time_frac.geoeff.iloc[gg] = 1
    #for g in geo2_ind:
    #    for gg in (events_time_frac.query('evt_index == ' + g).index).astype('int'):
    #        events_time_frac.geoeff.iloc[gg] = 2
    #for g in geo3_ind:
    #    for gg in (events_time_frac.query('evt_index == ' + g).index).astype('int'):
    #        events_time_frac.geoeff.iloc[gg] = 3
    #for g in geo9_ind:
    #    for gg in (events_time_frac.query('evt_index == ' + g).index).astype('int'):
    #        events_time_frac.geoeff.iloc[gg] = 9


    w_geoeff = np.where(events['geoeff'] == 1.0)[0]
    w_no_geoeff = np.where(events['geoeff'] == 0)[0]
    w_reject = np.where(events['geoeff'] == 9)[0]
    w_bound = np.where(events['geoeff'] == 2)[0]
    w_icme = np.where(events['geoeff'] == 3)[0]
    w_test = np.where(events['geoeff'] == 999)[0]

    #w_geoeff6 = events.query('geoeff == 1.0 and kp >= 5.6 and kp < 6.6')
    #w_geoeff7 = events.query('geoeff == 1.0 and kp >= 6.6 and kp < 7.6')
    #w_geoeff8 = events.query('geoeff == 1.0 and kp >= 8')
    
                           
    #fontP = FontProperties()                #legend
    #fontP.set_size('medium')                       
                                                 
    plt.scatter(events['bzm'].iloc[w_no_geoeff], events['tau'].iloc[w_no_geoeff], c = 'b', label = 'Not Geoeffecive' )   
    plt.scatter(events['bzm'].iloc[w_geoeff], events['tau'].iloc[w_geoeff], c = 'r', label = 'Geoeffecive' )                         
    #plt.scatter(events['bzm'].iloc[w_reject], events['tau'].iloc[w_reject], c = 'g', label = 'Reject')
    #plt.scatter(events['bzm'].iloc[w_bound], events['tau'].iloc[w_bound], c = 'orange', label = 'Bound')    
    #plt.scatter(events['bzm'].iloc[w_test], events['tau'].iloc[w_test], c = 'y', label = 'test')
    plt.ylim(0,150)
    plt.xlim(-75,75)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    #leg = plt.legend(loc='upper right', prop = fontP, fancybox=True, \
    #                 frameon=True, scatterpoints = 1 )
    #leg.get_frame().set_alpha(0.5)
        

    #save the cleaned up dataset   
    pickle.dump(events_time_frac,open("train/events_time_frac_fitall3_train_dst80_kp6_clean2.p", "wb"))
    pickle.dump(events,open("train/events_fitall3_train_dst80_kp6_clean2.p", "wb"))
    
def kfold(events_time_frac):

    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import shuffle
    
    
    events_ss = events_time_frac.query('(geoeff == 1 or geoeff ==0) and frac_est > 0.2')

    #shuffle the events so they are not organized by date
    events_ss = shuffle(events_ss)

    X = np.array(events_ss)      #[['start','bzm_predicted','tau_predicted']])
    y = np.array(events_ss.geoeff)
    skf = StratifiedKFold(n_splits=3, random_state=1002)
    
    train_index = []
    test_index = []
    for train_ind, test_ind in skf.split(X, y):
        train_index.append(train_ind)
        test_index.append(test_ind)
        
    #X_train, X_test = X[train_index[0]], X[test_index[0]]
    #y_train, y_test = y[train_index[0]], y[test_index[0]]

    X_events_train0, X_events_test0 = events_ss.iloc[train_index[0]], events_ss.iloc[test_index[0]]
    y_train0, y_test0 = y[train_index[0]], y[test_index[0]]
    
    X_events_train1, X_events_test1 = events_ss.iloc[train_index[1]], events_ss.iloc[test_index[1]]
    y_train1, y_test1 = y[train_index[1]], y[test_index[1]]
    
    X_events_train2, X_events_test2 = events_ss.iloc[train_index[2]], events_ss.iloc[test_index[2]]
    y_train2, y_test2 = y[train_index[2]], y[test_index[2]]
   
    
    
    ranges=[-100,100,-75,75]
    nbins=[30,60]
    fracs = [0.2,1.0]

    #pdf.create_pdfs(X_events_train0, ranges=ranges, nbins=nbins, fracs = [0.2,1.0], fname='events_time_frac_fitall3_train_dst80_kp6_clean2_ss0')
    
    pdf.create_pdfs(X_events_train1, ranges=ranges, nbins=nbins, fracs = [0.2,1.0], fname='events_time_frac_fitall3_train_dst80_kp6_clean2_ss1')

    pdf.create_pdfs(X_events_train2, ranges=ranges, nbins=nbins, fracs = [0.2,1.0], fname='events_time_frac_fitall3_train_dst80_kp6_clean2_ss2')
        
    
    
    