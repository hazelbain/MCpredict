# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:05:15 2017

@author: hazel.bain
"""

import Richardson_ICME_list as icme
import read_database as rddb

import plot_ace as pa
import MCpredict as MC
import read_dst as dst

import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime

import pickle as pickle

def MCpredict_all_Richardson(plotting = 0, csv = 0, livedb = 1, 
                predict = 0, ew = 2, nw = 1, pdf = np.zeros((50,50,50,50)),
                plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_th3/'):
    
    """
    Tests the Chen magnetic cloud fitting model using known magnetic cloud events
    from Richardson and Cane list. 
    
    inputs
    
    plotting: int
        outputs plots of the solar wind plasma and magnetic field data, along 
        with values of the Dst. Indicated whether or not the event is geoeffective
        (red), non geoeffective (green) or ambigous (orange)
    
    """
    
    

    #read in Richardson and Cane ICME list 
    mc_list = icme.read_richardson_icme_list()
    
    #read in the dst data
    dst_data = dst.read_dst_df()
    
    #choose icmes that have MC_flag classification 2 and 1
    #
    # (2) = indicates that a magnetic cloud has been reported in association with
    #the ICME (see (d) above) or (occasionally, or for recent events) that by 
    #our assessment, the ICME has the clear features of a magnetic cloud but a 
    #magnetic cloud may not have been reported. 
    #
    # (1) indicates that the ICME shows evidence of a rotation in field direction
    #but lacks some other characteristics of a magnetic cloud, for example an
    #enhanced magnetic field
    
    mc_list12 = mc_list[np.where(mc_list['MC_flag'] == 2)]

    #run the Chen magnetic cloud fitting routine to obtain fits to solar wind 
    #events to predict durtaion and max/min Bzm. 
    
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    
    errpredict = []                     # keep a note of any events where there were errors
    
    count = 0
    for i in range(0,len(mc_list12['mc_start'])):
        
        if mc_list12['mc_start'][i] == None:
            continue
        
        #get mc times +/- 24 hours
        st = mc_list12['mc_start'][i] - timedelta(hours=24)
        et = mc_list12['mc_start'][i] + timedelta(hours=48)
        
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
        
        #run the MC fit and prediction routine
        try:

            #data,events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, dst_data[st - timedelta(1):et + timedelta(1)], smooth_num = 100,\
            #    line = [mc_list12['mc_start'][i], mc_list12['mc_end'][i]], plotting = plotting,\
            #    plt_outfile = 'mcpredict_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf' ,\
            #    plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2_smooth100_bzmfix/')
         
            
            data, events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
                    dst_data[st - timedelta(1):et + timedelta(1)], pdf = pdf, \
                    csv = csv, livedb = livedb, predict = predict,\
                    smooth_num = 100, plotting = plotting,\
                    plt_outfile = 'mcpredict_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf' ,\
                    plt_outpath = plt_outpath)
                    
            events = events.append(events_tmp)
            events_frac = events_frac.append(events_frac_tmp)
            
            wg = np.where(events_tmp.geoeff == 1.0)
            for j in wg[0]:
                evt_data = data.iloc[events_tmp.istart.iloc[j]:events_tmp.iend.iloc[j]]
                pickle.dump(evt_data,open("C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_evt_data/evt"+str(count)+".p", "wb"))
                count+=1
                
            
            
        except:
            errpredict.append(i)
           
    events = events.reset_index() 

    #drop duplicate events 
    events_uniq = events.drop_duplicates()       
    events_frac_uniq = events_frac.drop_duplicates()         
            
    print("--------Error predict------------")
    print(errpredict)

    #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
    #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
    
    return events_uniq, events_frac_uniq
    
    
def plot_all_Richardson_MC():

    #read in Richardson and Cane ICME list 
    mc_list = icme.read_richardson_icme_list()
    
    #choose icmes that have MC_flag classification 2 and 1
    #
    # (2) = indicates that a magnetic cloud has been reported in association with
    #the ICME (see (d) above) or (occasionally, or for recent events) that by 
    #our assessment, the ICME has the clear features of a magnetic cloud but a 
    #magnetic cloud may not have been reported. 
    #
    # (1) indicates that the ICME shows evidence of a rotation in field direction
    #but lacks some other characteristics of a magnetic cloud, for example an
    #enhanced magnetic field
    
    mc_list12 = mc_list[np.where(mc_list['MC_flag'] > 0)]

    #get the ace_mag_1m and ace_swepam_1m data for these events
    
    errplt = []
    for i in range(0,len(mc_list12['mc_start'])):
        
        if mc_list12['mc_start'][i] == None:
            continue
        
        
        #get mc times +/- 24 hours
        st = mc_list12['mc_start'][i] - timedelta(hours=24)
        et = mc_list12['mc_start'][i] + timedelta(hours=48)
        
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
        
        try:

            pa.plot_ace_dst(stf, etf,\
                line = [mc_list12['mc_start'][i], mc_list12['mc_end'][i]],\
                plt_outfile = 'ace_dst_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf')
            
        except:
            print("\nErrorPlot. recording event index and moving on\n")
            errplt.append(i)
        
        

            