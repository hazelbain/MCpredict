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

import matplotlib.pyplot as plt

def MCpredict_all_Richardson(date1, date2, plotting = 0):
        
    #read in the dst data
    dst_data = dst.read_dst_df()
        
    #get the ace_mag_1m and ace_swepam_1m data for these events
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    errpredict = []
    for i in range(0,len(date_list)):
                
        #get mc times +/- 24 hours
        st = date1
        et = mc_list12['mc_start'][i] + timedelta(hours=48)
        
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
        
        try:

            data, events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, dst_data[st - timedelta(1):et + timedelta(1)], smooth_num = 100,\
                line = [mc_list12['mc_start'][i], mc_list12['mc_end'][i]], plotting = plotting,\
                plt_outfile = 'mcpredict_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf' ,\
                plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2_smooth100_bzmfix/')
            
            events = events.append(events_tmp)
            events_frac = events_frac.append(events_frac_tmp)
            
        except:
            errpredict.append(i)

    events = events.reset_index() 

    #drop duplicate events 
    events_uniq = events.drop_duplicates('start')       
            
    print("--------Error predict------------")
    print(errpredict)

    #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
    #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
    
    return events_uniq, events_frac




            