# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:12:25 2017

@author: hazel.bain
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:05:15 2017

@author: hazel.bain
"""

import read_database as rddb

import plot_ace as pa
import MCpredict as MC
import read_dst as dst

import numpy as np
import pandas as pd
from datetime import timedelta, datetime, date
import calendar

import matplotlib.pyplot as plt

def find_events(start_date, end_date, plotting = 0, csv = 1, livedb = 0):

    #format times
    start_date = datetime.strptime(start_date, "%d-%b-%Y")
    end_date= datetime.strptime(end_date, "%d-%b-%Y")
    
    if (end_date.year - start_date.year) > 0:
        print("*** Dates need to be in the same calander year ***")
        return None
    
    else:
            
        #read in the dst data
        dst_data = dst.read_dst_df()
        
        #get list of week start and end dates - with overlap of one day
        date_list = []
        cal = calendar.Calendar()
        for y in (np.arange(end_date.month - start_date.month + 1)+start_date.month):
            for x in cal.monthdatescalendar(start_date.year, y):
                date_list.append([x[0], x[0]+timedelta(days = 8)])
    
    
        print(date_list)
    
        #get the ace_mag_1m and ace_swepam_1m data for these events
        events = pd.DataFrame()             #observational event characteristics for all MCs
        events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
        errpredict = []
        for i in range(0,len(date_list)):
                    
            #get mc times +/- 24 hours
            st = date_list[i][0]
            et = date_list[i][1]
            
            #format time strings
            stf = datetime.strftime(st, "%Y-%m-%d")
            etf = datetime.strftime(et, "%Y-%m-%d")
            
            try:
               
                pdf = np.zeros((50,50,50,50))
                
                data, events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
                    dst_data[st - timedelta(1):et + timedelta(1)], pdf = pdf, \
                    csv = csv, livedb = livedb, \
                    smooth_num = 100, plotting = plotting,\
                    plt_outfile = 'mcpredict_'+ datetime.strftime(date_list[i][0], "%Y-%m-%d_%H%M") + '.pdf' ,\
                    plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/longterm/')
                
                events = events.append(events_tmp)
                events_frac = events_frac.append(events_frac_tmp)

                
            except:
                print("*** Error getting data ***")
                errpredict.append(i)
                
                
    
        events = events.reset_index() 
    
        #drop duplicate events 
        events_uniq = events.drop_duplicates('start')       
        events_frac_uniq = events_frac.drop_duplicates()       
                
        print("--------Error predict------------")
        print(errpredict)
    
        #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
        #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
        
        return events_uniq, events_frac_uniq




            